import math
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

try:
    import traci
    import traci.constants as tc
    from sumolib import checkBinary
except ImportError as exc:
    raise ImportError(
        "SUMO Python tools(traci, sumolib)가 필요합니다. "
        "pip install traci sumolib 또는 $SUMO_HOME/tools를 PYTHONPATH에 추가하세요."
    ) from exc

try:
    from sumo_renderer import SumoRenderer
except ImportError:
    from .sumo_renderer import SumoRenderer  # type: ignore[no-redef]


# SUMO 기본 차량 길이(5m) + min_gap(2.5m) 기반 lane 용량 추정 상수
# obs density/queue 정규화에 사용 — 정확치는 아니지만 [0,1] 근사 보장
_VEHICLE_FOOTPRINT_M = 7.5

# ── TraCI subscription 변수 목록 (per-vehicle/lane 폴링 → 구독 일괄수신) ──────
# 과거 _simulate_seconds 가 매 sim-second 차량당 4~5회 개별 TraCI 호출(IPC)을 해
# LOS D(~1,400대)에서 초당 ~7,000 왕복 → rollout 병목의 주범이었다. 구독으로 바꾸면
# simulationStep 직후 클라이언트에 일괄 저장되어 getAllSubscriptionResults() 는
# IPC 없는 로컬 dict 조회가 된다. **수치 산출 로직은 불변** — 데이터 소스만 교체.
_VEH_SUB_VARS = [
    tc.VAR_SPEED,                       # Tier1 속도/정지 감지
    tc.VAR_ACCUMULATED_WAITING_TIME,    # avg_waiting_time 지표 체인
    tc.VAR_WAITING_TIME,                # brt_weight>1 reward 경로 (연속 정지시간)
    tc.VAR_CO2EMISSION,                 # Tier1 CO2
]
# in-lane (controlled): obs density/queue + EW/NS wait + 차량 ID (brt/corridor 경로)
_LANE_IN_SUB_VARS = [
    tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
    tc.VAR_WAITING_TIME,
    tc.LAST_STEP_VEHICLE_NUMBER,
    tc.LAST_STEP_VEHICLE_ID_LIST,
]
# 경계 out-lane (어느 TLS 의 in-lane 도 아닌 네트워크 출구): pressure 의 차량 수만
_LANE_OUT_SUB_VARS = [tc.LAST_STEP_VEHICLE_NUMBER]


class SumoParallelEnv(ParallelEnv):
    """
    PettingZoo Parallel API 기반 SUMO 교차로 신호제어 환경

    단일 교차로: tls_ids=["C"]  → agents=["tl_0"]
    다중 교차로: tls_ids=["C","D",...] → agents=["tl_0","tl_1",...]

    Action / Observation 설계 (SUMO-RL 호환):
      - Action: Discrete(num_green_phases) — 네트워크에서 자동 감지된 green phase 중 선택
                * 단일교차로 (`tls.tll.xml`): 4 green phases → Discrete(4)
                * 2x2grid (sumo-rl): 4 green phases [0,2,4,6] → Discrete(4)
                yellow phase는 정책이 직접 선택할 수 없고, green 전환 시 자동 삽입됨.
      - Observation: [phase_one_hot, min_green_flag, density_per_lane, queue_per_lane]
                shape = num_green + 1 + 2 * num_controlled_lanes
                density/queue는 lane capacity 기반 [0,1] 정규화.

    보상(reward_mode):
      "diff-waiting-time" (기본): (이전 step 누적 대기시간 - 현재) / 10. 차분 기반,
        자기 진입로 대기시간만 본다. BRT 시나리오에서 brt_weight 가 vClass=bus
        차량에만 곱해진다.
      "pressure": Σ#(out_lanes 차량) - Σ#(in_lanes 차량). max-pressure/backpressure
        보상(Varaiya 2013, PressLight 2019). 상태 기반(차분 아님)이라 하류 포화를
        직접 반영해 과포화 스필백·grid lock 을 억제한다. 차량 *수* 기반이라
        brt_weight 는 미적용. 스케일이 작아(±수십) switch_penalty 를 낮추는 게 좋다.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
        "name": "sumo_intersection_v0",
    }

    REWARD_MODES: List[str] = ["diff-waiting-time", "pressure"]

    # CTDE critic wrap 의 고정 이웃 슬롯 (4방향). critic 은 N/E/S/W 이웃 obs 를 모두
    # 받아 value 추정에 활용한다 (actor 는 N/S 만 봄 — _OBS_NBR_DIRS 참조).
    _NEIGHBOR_DIRS: Tuple[str, ...] = ("N", "E", "S", "W")
    # per-agent actor obs 가 방출하는 이웃 방향. 회랑(green wave)은 N/S 축이고 E/W=cross-
    # column 은 진행파에 무관 + 혼잡은 pressure 와 중복이라 컷 → 배포 actor obs 를 lean 하게.
    _OBS_NBR_DIRS: Tuple[str, ...] = ("N", "S")
    # 이웃 한 칸당 혼잡 요약 스칼라 수 (mean_density, mean_queue).
    _NBR_SUMMARY_DIM: int = 2
    # upstream_phase: 이웃 phase 경과시간 정규화 분모(초). 최대 green ~47s, cycle ~100s.
    _PHASE_TIME_NORM: float = 60.0

    def __init__(
        self,
        sumo_cfg: Optional[str] = None,
        use_gui: bool = False,
        delta_time: int = 5,
        yellow_time: int = 3,
        min_green: int = 13,
        max_steps: int = 3600,
        reward_mode: str = "diff-waiting-time",
        tls_ids: Optional[List[str]] = None,
        ctde_mode: bool = False,
        neighbor_obs: bool = False,
        upstream_phase: bool = False,
        progression_coeff: float = 0.0,
        brt_prog_weight: float = 1.0,
        switch_penalty: float = 0.45,
        brt_weight: float = 1.0,
        time_to_teleport: int = 300,
    ):
        self.base_dir = Path(__file__).resolve().parent
        self.sumo_data_dir = self.base_dir / "sumo_data"
        if sumo_cfg:
            self.sumo_cfg = Path(sumo_cfg).resolve()
            if not self.sumo_cfg.exists():
                raise FileNotFoundError(
                    f"SUMO 설정 파일을 찾을 수 없습니다: {self.sumo_cfg}"
                )
        else:
            self.sumo_cfg = self.sumo_data_dir / "single" / "single.sumocfg"
            self._maybe_build_network()

        if reward_mode not in self.REWARD_MODES:
            raise ValueError(
                f"지원하지 않는 reward_mode: '{reward_mode}'. "
                f"선택 가능: {self.REWARD_MODES}"
            )

        self.use_gui = use_gui
        self.delta_time = int(delta_time)
        self.yellow_time = int(yellow_time)
        self.min_green = int(min_green)
        self.max_steps = int(max_steps)
        self.reward_mode = reward_mode
        self.ctde_mode = bool(ctde_mode)
        # Topology-aware 관측/critic 확장 (기본 False = 기존 동작 그대로).
        #   neighbor_obs=True 면:
        #   - (#3) 각 agent actor obs 끝에 N/S 이웃 교차로의 상류 혼잡 요약
        #          [mean_density, mean_queue] × 2방향 = 4 차원을 덧붙여, 분산 실행되는
        #          actor 가 "상류에서 차량이 몰려온다"를 미리 관측(anticipation)한다.
        #          (회랑=N/S 축. E/W=cross 는 컷 — _OBS_NBR_DIRS 참조.)
        #   - (#1) ctde_mode 일 때 critic 입력을 전체 agent naive concat 대신
        #          [own | agent_id one-hot | N/E/S/W 이웃 obs] 로 구성 (critic 은 4방향
        #          이웃 obs 까지 봐 value 추정이 풍부; actor 는 N/S 만 — actor-lean/critic-rich).
        # 인접/방향은 _probe_network_spec 에서 net 연결성으로 자동 도출 (하드코딩 없음).
        self.neighbor_obs = bool(neighbor_obs)
        # upstream_phase: neighbor_obs 위에 N/S 이웃의 위상 시계(phase one-hot + 경과시간)를
        # actor obs 에 추가 → 상류 platoon 도착 타이밍을 관측해 green wave 오프셋을 학습.
        # neighbor_obs 스캐폴딩(인접맵·2-pass·critic wrap)을 요구한다.
        self.upstream_phase = bool(upstream_phase)
        if self.upstream_phase and not self.neighbor_obs:
            raise ValueError("upstream_phase=True 는 neighbor_obs=True 를 요구합니다.")
        # ① 회랑 progression 보상: corridor agent 의 ns-through lane 가중 평균 속도비를
        # progression_coeff 배 가산 (무정차 통과 = green wave). brt_prog_weight>1 면 버스 우대(TSP).
        self.progression_coeff = float(progression_coeff)
        self.brt_prog_weight = float(brt_prog_weight)
        # Phase switch마다 차감해 oscillation을 억제 (모든 reward_mode 공통).
        # 기본 0.45 ≈ yellow 3초 × 1대/초 손실. 튜닝 근거는 CLAUDE.md 참조.
        self.switch_penalty = float(switch_penalty)

        # BRT(vClass=bus) 대기시간 가중치. 1.0=비활성(전 차량 동일 취급),
        # diff-waiting-time 모드에서만 reward에 반영 (BRT 시나리오 권장 1.5~3.0).
        self.brt_weight = float(brt_weight)

        # SUMO 텔레포트 임계(초). 정체에 갇힌 차량을 정체 너머로 이동시켜 grid lock을
        # 자가 회복하는 deadlock 안전밸브. 300=기본, -1=비활성(과포화 시 학습 불가).
        # 공정 비교 위해 전 베이스라인 동일 값 사용. 상세는 CLAUDE.md 참조.
        self.time_to_teleport = int(time_to_teleport)

        # SUMO TLS id → PettingZoo agent id 매핑
        self._tls_ids: List[str] = tls_ids if tls_ids is not None else ["C"]
        self.possible_agents: List[str] = [
            f"tl_{i}" for i in range(len(self._tls_ids))
        ]
        self._agent_to_tls: Dict[str, str] = dict(
            zip(self.possible_agents, self._tls_ids)
        )

        # ── 네트워크 정적 스펙 사전 추출 (probe) ─────────────────────────────
        # obs/act space 결정에 controlled lanes 수 + green phase 인덱스가 필요.
        # SUMO 한 번 띄워 정보만 수집하고 즉시 종료. (reset 시 재시작)
        spec = self._probe_network_spec()
        self._per_agent_lanes: Dict[str, List[str]] = spec["per_agent_lanes"]
        self._per_agent_out_lanes: Dict[str, List[str]] = spec["per_agent_out_lanes"]
        self._per_agent_green_phases: Dict[str, List[int]] = spec["per_agent_green_phases"]
        self._per_agent_yellow_map: Dict[str, Dict[int, int]] = spec["per_agent_yellow_map"]
        self._lane_capacities: Dict[str, float] = spec["lane_capacities"]

        # 전체 입력 lane 합집합 (queue_cumsum 집계용)
        self._lane_ids: List[str] = list(dict.fromkeys(
            ln for agent in self.possible_agents
            for ln in self._per_agent_lanes[agent]
        ))

        # 모든 agent의 obs/act 차원이 동일하다고 가정.
        # num_lanes 는 agent별로 다를 수 있어 (BRT corridor TLS=10, 일반 TLS=8) max 사용.
        # 짧은 agent 의 obs 는 density/queue 부분을 0 패딩으로 채워 shape 일관성 유지.
        first = self.possible_agents[0]
        self._num_green: int = len(self._per_agent_green_phases[first])
        self._num_lanes: int = max(
            len(self._per_agent_lanes[a]) for a in self.possible_agents
        )
        # base local obs: [phase_one_hot, min_green_flag, density, queue]
        self._base_obs_dim: int = self._num_green + 1 + 2 * self._num_lanes
        # upstream_phase 의 이웃 위상 슬롯 차원 = phase_one_hot(num_green) + elapsed_norm(1).
        self._phase_slot_dim: int = self._num_green + 1
        # actor obs 의 이웃 블록: N/S 각 방향당 [혼잡 요약(2)] (+ upstream_phase 면 위상 슬롯).
        #   neighbor_obs only : 2방향 × 2          = 4
        #   + upstream_phase  : 2방향 × (2 + num_green+1) = 14 (3x2-brt: 4+10)
        per_dir_block = self._NBR_SUMMARY_DIM + (
            self._phase_slot_dim if self.upstream_phase else 0
        )
        self._nbr_summary_dim: int = (
            len(self._OBS_NBR_DIRS) * per_dir_block if self.neighbor_obs else 0
        )
        # _obs_dim = actor 가 보는 per-agent local 차원 (= CTDE module 의 local_dim).
        self._obs_dim: int = self._base_obs_dim + self._nbr_summary_dim

        # 인접/방향 맵 (net 연결성 기반 자동 도출). neighbor_obs=False 여도 보관만 함.
        self._neighbor_dir: Dict[str, Dict[str, str]] = spec["neighbor_dir"]
        # 이웃 요약 캐시 — _compute_obs 가 agent 별 [mean_density, mean_queue] 를 채우고
        # _enrich_obs 가 이웃 것을 읽어 붙인다 (2-pass).
        self._agent_dq_summary: Dict[str, np.ndarray] = {
            a: np.zeros(self._NBR_SUMMARY_DIM, dtype=np.float32)
            for a in self.possible_agents
        }
        # upstream_phase 위상 시계 캐시 — _compute_obs 가 [phase_one_hot, elapsed_norm] 를 채움.
        self._agent_phase_summary: Dict[str, np.ndarray] = {
            a: np.zeros(self._phase_slot_dim, dtype=np.float32)
            for a in self.possible_agents
        }

        # ① 회랑(BRT corridor) 자동식별 — BRT 전용차로를 controlled lane 으로 가진 agent.
        # corridor_ns_lanes = 그 agent 의 ns-through lane(진행 보상 대상, BRT 전용차로 포함).
        self._corridor_agents: List[str] = spec["corridor_agents"]
        self._corridor_ns_lanes: Dict[str, List[str]] = spec["corridor_ns_lanes"]

        # obs/act space 캐시 (observation_space()/action_space() 가 lazy 생성)
        self._obs_space: Optional[spaces.Space] = None
        self._act_space: Optional[spaces.Space] = None

        # agents: 현재 에피소드 활성 에이전트
        self.agents: List[str] = []

        # 에이전트별 신호 상태
        # _current_green_idx: green_phases 리스트 내 인덱스 (0..num_green-1)
        # _current_phase   : SUMO 실제 phase index (전환 시 동기화)
        self._current_green_idx: Dict[str, int] = {}
        self._current_phase: Dict[str, int] = {}
        self._elapsed_phase_time: Dict[str, int] = {}
        self._last_obs: Dict[str, np.ndarray] = {
            a: np.zeros(self._obs_dim, dtype=np.float32) for a in self.possible_agents
        }
        # 렌더러 전용 lane별 큐 캐시 (agent → [queue per lane])
        # step()/reset()에서 갱신. SumoRenderer가 obs 의존 없이 정확한 큐 사용.
        self._last_queue_per_lane: Dict[str, List[float]] = {
            a: [0.0] * len(self._per_agent_lanes[a]) for a in self.possible_agents
        }

        # TraCI 연결
        self.conn = None
        self._conn_label: Optional[str] = None
        self.sim_step: int = 0

        # subscription 보조 캐시 — vClass 는 차량당 정적(첫 등장 시 1회 조회, reset 마다
        # clear), lane max speed 는 net 정적(에피소드 간 유지).
        self._vehicle_class: Dict[str, str] = {}
        self._lane_max_speed: Dict[str, float] = {}

        # 에피소드 지표 버퍼
        self._depart_time: Dict[str, int] = {}
        self._latest_waiting: Dict[str, float] = {}
        self._completed_waiting: List[float] = []
        self._completed_travel: List[float] = []
        self._throughput: int = 0
        self._queue_cumsum: float = 0.0

        # Insertion-failure 진단 카운터. lane 포화 시 SUMO 가 차량 출발을 막아도
        # queue/teleport 에 안 잡히므로 loaded - departed = pending insertion 으로 추적.
        self._episode_loaded_total: int = 0
        self._episode_departed_total: int = 0
        self._episode_pending_peak: int = 0
        self._episode_pending_final: int = 0

        # yellow 누적 시간 — oscillation 진단 (yellow_ratio = yellow / total sec).
        self._episode_yellow_seconds: int = 0

        # 진단 카운터 (action_counts 는 동적 num_green 길이)
        self._episode_phase_switches: int = 0
        self._episode_max_queue: float = 0.0
        self._episode_action_counts: np.ndarray = np.zeros(self._num_green, dtype=np.int64)
        self._episode_teleported: int = 0

        # diff-waiting-time 보상 모드 전용
        self._last_wait_measure: Dict[str, float] = {}

        # ── Green Wave 평가 지표 (Tier 1 + Tier 2) ─────────────────────────
        # Tier 1: stop-and-go / 환경부담 / 흐름 직접 측정
        self._episode_co2_sum: float = 0.0           # 총 CO2 배출 (mg)
        self._episode_speed_sum: float = 0.0         # 속도 합 (스텝×차량)
        self._episode_speed_sq_sum: float = 0.0      # 속도 제곱 합 (variance 용)
        self._episode_speed_count: int = 0           # 표본 수 (스텝×차량)
        self._vehicle_stop_count: Dict[str, int] = {}    # 차량별 누적 정지 수
        self._vehicle_prev_speed: Dict[str, float] = {}  # 정지 transition 감지
        self._vehicle_seen: Set[str] = set()         # 에피소드 등장 차량 (정규화 분모)

        # Tier 2: 방향별 대기시간 (E-W vs N-S 코리도어 분석)
        # lane_direction 은 probe 시 1회 결정되어 spec dict 에 포함됨
        self._lane_direction: Dict[str, str] = spec["lane_direction"]
        self._episode_wait_ew_sum: float = 0.0
        self._episode_wait_ns_sum: float = 0.0
        # 누적합 분모 (스텝×lane)
        self._episode_wait_ew_count: int = 0
        self._episode_wait_ns_count: int = 0

        # BRT 우선처리 평가용 — vClass=bus 와 일반 차량 wait/speed 를 분리 누적.
        # _simulate_seconds 의 vehicle loop 에서 분기 1회로 통합 채움.
        # 단위는 기존 episode_speed_sum 과 동일 (스텝×차량 표본).
        self._episode_brt_wait_sum: float = 0.0
        self._episode_brt_wait_count: int = 0
        self._episode_car_wait_sum: float = 0.0
        self._episode_car_wait_count: int = 0
        self._episode_brt_speed_sum: float = 0.0
        self._episode_brt_speed_count: int = 0
        self._episode_car_speed_sum: float = 0.0
        self._episode_car_speed_count: int = 0

        # ① 회랑 진행파 지표 — corridor ns-through lane BRT 평균 속도비 (progression_coeff
        # 와 무관하게 항상 누적; baseline 대비 진행파 정량 비교용). _corridor_step 가 채움.
        self._episode_corridor_brt_ratio_sum: float = 0.0
        self._episode_corridor_brt_count: int = 0

        # 네트워크 시각화 렌더러
        self._renderer = SumoRenderer(self.sumo_cfg)
        # 렌더 타이틀 첫 줄에 표시할 모델 식별 라벨 (record_video 스크립트가 설정).
        # 예: "MAPPO · iter 185" / "CTDE · iter 130" / "FixedTime(Sejong)".
        # None 이면 Step 줄만 표시 (학습/평가에는 영향 없음).
        self.render_label: Optional[str] = None

        # 매 sim step 직후 호출될 콜백 리스트 (frame 캡쳐 등 외부 hook)
        # 학습/평가에는 영향 없음 (등록 안 하면 no-op). record_video 에서 사용.
        self._step_hooks: List[Callable[[int], None]] = []

        # CTDE 단일 교차로 경고 (degenerate: global obs == local obs)
        if self.ctde_mode and len(self._tls_ids) == 1:
            print(
                "[CTDE warning] tls_ids 가 1개입니다. global obs == local obs 가 되어 "
                "centralized critic 의 의미가 사라집니다. 다중 교차로에서 사용하세요."
            )

    def add_step_hook(self, fn: Callable[[int], None]) -> None:
        """매 sim step 후 호출될 콜백 등록.

        fn(sim_step: int) 형태. _simulate_seconds() 안에서 호출되어
        delta_time=5 인 한 번의 env.step() 동안 yellow + green 진행 중
        매초마다 fn 이 호출됨 → 연속 frame 캡쳐 가능.

        예: env.add_step_hook(lambda s: frames.append(env.render()))
        """
        self._step_hooks.append(fn)

    def live_metrics(self) -> dict:
        """현재 시뮬레이션 시점의 실시간 집계 지표 스냅샷 (영상 오버레이용).

        step hook 안에서 호출하면 프레임과 1:1 정렬된 시계열을 얻을 수 있다.
        - co2_kg     : 에피소드 누적 CO2 배출 (kg, 단조 증가)
        - avg_wait   : 완료 차량 평균 대기시간 (s) — 평가 avg_waiting_time 으로 수렴
        - cur_wait   : 현재 도로 위 차량들의 누적 대기시간 총합 (s, 혼잡도 실시간 반영)
        - throughput : 현재까지 통과 완료한 차량 수
        - sim_step   : 현재 시뮬레이션 step
        """
        cw = self._completed_waiting
        avg_wait = (sum(cw) / len(cw)) if cw else 0.0
        cur_wait = sum(self._latest_waiting.values()) if self._latest_waiting else 0.0
        return {
            "co2_kg": self._episode_co2_sum / 1e6,
            "avg_wait": avg_wait,
            "cur_wait": cur_wait,
            "throughput": int(self._throughput),
            "sim_step": int(self.sim_step),
        }

    # ------------------------------------------------------------------
    # PettingZoo 필수 인터페이스 — obs/act space 동적 결정
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        # 모든 agent 동일 space. 인스턴스 속성 캐시 — 과거 @lru_cache(인스턴스 메서드)는
        # 전역 캐시가 self 를 영구 참조해 env 가 GC 되지 않는 누수가 있었음.
        if self._obs_space is None:
            if not self.ctde_mode:
                self._obs_space = spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self._obs_dim,), dtype=np.float32,
                )
            else:
                # CTDE: 단일 flat Box (Dict obs 는 RLlib worker connector 에서 silent fail).
                # 첫 _obs_dim 차원 = 자기 local → CentralizedCriticPPOModule 이 slice.
                n_agents = len(self.possible_agents)
                if self.neighbor_obs:
                    # (#1) [own | agent_id one-hot(N) | N/E/S/W 이웃 obs(_obs_dim × 4)]
                    total_dim = self._obs_dim + n_agents + self._obs_dim * len(self._NEIGHBOR_DIRS)
                else:
                    # legacy: [own | 전체 agent obs concat(_obs_dim × N)]
                    total_dim = self._obs_dim + self._obs_dim * n_agents
                self._obs_space = spaces.Box(
                    low=0.0, high=1.0,
                    shape=(total_dim,), dtype=np.float32,
                )
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Space:
        if self._act_space is None:
            self._act_space = spaces.Discrete(self._num_green)
        return self._act_space

    # ------------------------------------------------------------------
    # Lane 방향 분류 — 평가 지표 per_direction_wait 용
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_direction(shape: list) -> str:
        """lane shape (좌표 리스트) → 'ew' 또는 'ns'.

        start→end 벡터의 각도가 ±45° 이내면 동서, 그 외면 남북.
        2점 미만이면 기본값 'ew'.
        """
        if len(shape) < 2:
            return "ew"
        dx = shape[-1][0] - shape[0][0]
        dy = shape[-1][1] - shape[0][1]
        abs_a = abs(math.degrees(math.atan2(dy, dx)))
        return "ew" if (abs_a <= 45 or abs_a >= 135) else "ns"

    @staticmethod
    def _is_brt_lane(conn, ln: str) -> bool:
        """lane 이 BRT 전용차로인지 — allow 에 bus 포함 & passenger 제외.

        BRT corridor edge 의 전용차로(allow="bus")를 식별해 회랑 agent 를 자동 도출한다
        (하드코딩 없음). getAllowed 가 빈 리스트면 전체 허용(=일반차로) → False.
        """
        try:
            allowed = set(conn.lane.getAllowed(ln))
        except Exception:
            return False
        return ("bus" in allowed) and ("passenger" not in allowed)

    def _corridor_step(self, agent: str) -> float:
        """① corridor agent 의 ns-through lane 가중 평균 속도비 ∈ [0, brt_prog_weight].

        버스가 자유속도로 통과하면 ↑, 정지하면 0 → 무정차(green wave) 유도.
        brt_prog_weight>1 이면 버스 우대(TSP). 회랑 차량 없으면 0.
        분모 정규화로 차량 수 무관·bounded → β 튜닝 안정.

        같은 1-pass 로 BRT-only 속도비 metric 도 누적한다 (progression_coeff 와 무관하게
        항상 기록 → baseline 대비 회랑 진행파 정량 비교용 corridor_brt_speed_ratio).
        """
        lane_res = self.conn.lane.getAllSubscriptionResults()
        veh_res = self.conn.vehicle.getAllSubscriptionResults()
        num = 0.0
        den = 0.0
        for ln in self._corridor_ns_lanes.get(agent, []):
            vmax = self._lane_max_speed.get(ln)
            if vmax is None:  # net 정적 → 최초 1회만 조회
                vmax = self.conn.lane.getMaxSpeed(ln)
                self._lane_max_speed[ln] = vmax
            if vmax <= 1e-6:
                continue
            lr = lane_res.get(ln)
            vids = (lr[tc.LAST_STEP_VEHICLE_ID_LIST] if lr is not None
                    else self.conn.lane.getLastStepVehicleIDs(ln))
            for vid in vids:
                vcls = self._vehicle_class.get(vid)
                if vcls is None:
                    vcls = self.conn.vehicle.getVehicleClass(vid)
                    self._vehicle_class[vid] = vcls
                is_bus = vcls == "bus"
                vr = veh_res.get(vid)
                spd = (float(vr[tc.VAR_SPEED]) if vr is not None
                       else float(self.conn.vehicle.getSpeed(vid)))
                ratio = min(1.0, spd / vmax)
                w = self.brt_prog_weight if is_bus else 1.0
                num += w * ratio
                den += w
                if is_bus:
                    self._episode_corridor_brt_ratio_sum += ratio
                    self._episode_corridor_brt_count += 1
        return num / den if den > 0.0 else 0.0

    # ------------------------------------------------------------------
    # SUMO 유틸리티
    # ------------------------------------------------------------------

    def _maybe_build_network(self) -> None:
        single_dir = self.sumo_data_dir / "single"
        net_file = single_dir / "single_intersection.net.xml"
        if net_file.exists():
            return
        netconvert = shutil.which("netconvert")
        if not netconvert:
            raise FileNotFoundError(
                "single_intersection.net.xml이 없고 netconvert도 찾을 수 없습니다."
            )
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".net.xml", dir=single_dir
        )
        os.close(tmp_fd)
        try:
            cmd = [
                netconvert,
                "--node-files", str(single_dir / "nodes.nod.xml"),
                "--edge-files", str(single_dir / "edges.edg.xml"),
                "--connection-files", str(single_dir / "connections.con.xml"),
                "--tllogic-files", str(single_dir / "tls.tll.xml"),
                "-o", tmp_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"netconvert 네트워크 생성 실패.\n{result.stderr}"
                )
            os.replace(tmp_path, net_file)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def _sumo_binary(self) -> str:
        preferred = "sumo-gui" if self.use_gui else "sumo"
        try:
            return checkBinary(preferred)
        except Exception:
            found = shutil.which(preferred)
            if found:
                return found
            raise FileNotFoundError(f"SUMO binary를 찾을 수 없습니다: {preferred}")

    # ------------------------------------------------------------------
    # 네트워크 스펙 probe — obs/act space 결정에 필요한 정적 정보 추출
    # ------------------------------------------------------------------

    def _probe_network_spec(self) -> dict:
        """임시 SUMO 인스턴스를 띄워 controlled lanes / green phases / lane capacity를 수집.

        TraCI 연결을 즉시 닫고 정보만 메모리에 보관. 이후 reset() 마다 SUMO 재시작.
        """
        binary = self._sumo_binary()
        label = f"probe_{uuid.uuid4().hex[:8]}"
        cmd = [
            binary, "-c", str(self.sumo_cfg),
            "--no-warnings", "true",
            "--no-step-log", "true",
        ]
        # traci.start 가 raise 되면 conn 이 정의되지 않은 채 finally에 진입 →
        # NameError 방지를 위해 사전 초기화
        conn = None
        try:
            traci.start(cmd, label=label)
            conn = traci.getConnection(label)
            known_tls = set(conn.trafficlight.getIDList())
            missing = [tid for tid in self._tls_ids if tid not in known_tls]
            if missing:
                raise ValueError(
                    f"SUMO 네트워크에 존재하지 않는 TLS ID: {missing}. "
                    f"사용 가능한 TLS: {sorted(known_tls)}"
                )

            per_agent_lanes: Dict[str, List[str]] = {}
            per_agent_out_lanes: Dict[str, List[str]] = {}
            per_agent_green_phases: Dict[str, List[int]] = {}
            per_agent_yellow_map: Dict[str, Dict[int, int]] = {}
            lane_capacities: Dict[str, float] = {}
            lane_direction: Dict[str, str] = {}
            # ① 회랑 식별: BRT 전용차로(allow=bus, passenger 차단)를 가진 agent + 그 ns-through lane.
            corridor_ns_lanes: Dict[str, List[str]] = {}

            for agent, tls_id in zip(self.possible_agents, self._tls_ids):
                lanes = list(dict.fromkeys(
                    conn.trafficlight.getControlledLanes(tls_id)
                ))
                per_agent_lanes[agent] = lanes
                out_lanes = list(set(
                    link[0][1]
                    for link in conn.trafficlight.getControlledLinks(tls_id)
                    if link
                ))
                per_agent_out_lanes[agent] = out_lanes

                # lane 용량 추정: lane 길이 / 차량 footprint (5m + 2.5m gap)
                # lane 방향: shape 의 start→end 각도 기반 ew/ns 분류
                for ln in lanes + out_lanes:
                    if ln not in lane_capacities:
                        length = float(conn.lane.getLength(ln))
                        lane_capacities[ln] = max(1.0, length / _VEHICLE_FOOTPRINT_M)
                for ln in lanes:
                    if ln not in lane_direction:
                        try:
                            lane_direction[ln] = self._classify_direction(
                                conn.lane.getShape(ln)
                            )
                        except Exception:
                            lane_direction[ln] = "ew"

                # ① 회랑 식별: BRT 전용차로를 가진 agent 의 ns-through lane 집합.
                # 전용차로 ⟺ allow 에 bus 포함 & passenger 제외 (BRT corridor edge).
                if any(self._is_brt_lane(conn, ln) for ln in lanes):
                    corridor_ns_lanes[agent] = [
                        ln for ln in lanes if lane_direction.get(ln) == "ns"
                    ]

                # green phase 인덱스 + green→다음yellow 매핑
                green_indices, yellow_map = self._extract_phase_structure(conn, tls_id)
                per_agent_green_phases[agent] = green_indices
                per_agent_yellow_map[agent] = yellow_map

            # 첫 agent 기준 num_green 일치 검증 (정사각 grid 가정)
            first_n = len(per_agent_green_phases[self.possible_agents[0]])
            for agent in self.possible_agents:
                if len(per_agent_green_phases[agent]) != first_n:
                    raise ValueError(
                        f"agent별 green phase 수가 다름: "
                        f"{ {a: len(per_agent_green_phases[a]) for a in self.possible_agents} }. "
                        "현재 구현은 동일 구조 교차로만 지원합니다."
                    )

            # ── 인접·방향 자동 도출 (neighbor_obs 확장용) ─────────────────────
            # 하드코딩 없이 net 연결성으로 계산:
            #   A→B 연결 ⟺ A 의 out-lane 이 B 의 controlled in-lane 과 겹침
            #   (같은 edge 의 lane id 는 A 의 out 이자 B 의 in). 양방향 합집합으로
            #   무방향 이웃 집합을 만든 뒤, 교차로 위치 상대좌표로 N/E/S/W 분류.
            agent_pos = self._probe_agent_positions(conn, per_agent_lanes)
            neighbor_dir = self._derive_neighbor_dirs(
                per_agent_lanes, per_agent_out_lanes, agent_pos
            )

            return {
                "per_agent_lanes": per_agent_lanes,
                "per_agent_out_lanes": per_agent_out_lanes,
                "per_agent_green_phases": per_agent_green_phases,
                "per_agent_yellow_map": per_agent_yellow_map,
                "lane_capacities": lane_capacities,
                "lane_direction": lane_direction,
                "neighbor_dir": neighbor_dir,
                "corridor_agents": list(corridor_ns_lanes.keys()),
                "corridor_ns_lanes": corridor_ns_lanes,
            }
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _probe_agent_positions(
        self, conn, per_agent_lanes: Dict[str, List[str]]
    ) -> Dict[str, Tuple[float, float]]:
        """각 agent(교차로)의 대표 (x, y) 좌표를 추정.

        우선 junction.getPosition(tls_id) 을 시도하고, 실패하면 controlled in-lane
        의 junction-쪽 끝점(getShape()[-1]) 평균으로 대체한다. 방향(N/E/S/W) 분류에만
        쓰이므로 정확한 절대좌표가 아니라 상대 배치만 맞으면 충분하다.
        """
        agent_pos: Dict[str, Tuple[float, float]] = {}
        for agent, tls_id in zip(self.possible_agents, self._tls_ids):
            pos: Optional[Tuple[float, float]] = None
            try:
                p = conn.junction.getPosition(tls_id)
                pos = (float(p[0]), float(p[1]))
            except Exception:
                pos = None
            if pos is None:
                ends = []
                for ln in per_agent_lanes.get(agent, []):
                    try:
                        shape = conn.lane.getShape(ln)
                        if shape:
                            ends.append((float(shape[-1][0]), float(shape[-1][1])))
                    except Exception:
                        continue
                if ends:
                    pos = (
                        sum(x for x, _ in ends) / len(ends),
                        sum(y for _, y in ends) / len(ends),
                    )
                else:
                    pos = (0.0, 0.0)
            agent_pos[agent] = pos
        return agent_pos

    def _derive_neighbor_dirs(
        self,
        per_agent_lanes: Dict[str, List[str]],
        per_agent_out_lanes: Dict[str, List[str]],
        agent_pos: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Dict[str, str]]:
        """net 연결성 + 상대좌표로 agent 별 {방향: 이웃agent} 맵을 만든다.

        - 인접 판정: A 의 out-lane 집합과 B 의 in-lane 집합이 겹치면 A→B 연결.
          grid 는 양방향 edge 라 보통 대칭이지만, 한쪽만 잡혀도 무방향 이웃으로 본다.
        - 방향: (B - A) 벡터의 우세 축으로 N/E/S/W 1개 배정 (SUMO y 증가 = 북).
          같은 방향에 둘이 잡히면 더 가까운 쪽을 채택.
        """
        in_set = {a: set(per_agent_lanes.get(a, [])) for a in self.possible_agents}
        out_set = {a: set(per_agent_out_lanes.get(a, [])) for a in self.possible_agents}

        neighbor_dir: Dict[str, Dict[str, str]] = {
            a: {} for a in self.possible_agents
        }
        for a in self.possible_agents:
            ax, ay = agent_pos[a]
            best_dist: Dict[str, float] = {}
            for b in self.possible_agents:
                if b == a:
                    continue
                connected = bool(out_set[a] & in_set[b]) or bool(out_set[b] & in_set[a])
                if not connected:
                    continue
                bx, by = agent_pos[b]
                dx, dy = bx - ax, by - ay
                if abs(dx) >= abs(dy):
                    direction = "E" if dx > 0 else "W"
                else:
                    direction = "N" if dy > 0 else "S"
                dist = dx * dx + dy * dy
                if direction not in best_dist or dist < best_dist[direction]:
                    best_dist[direction] = dist
                    neighbor_dir[a][direction] = b
        return neighbor_dir

    @staticmethod
    def _extract_phase_structure(conn, tls_id: str) -> Tuple[List[int], Dict[int, int]]:
        """TLS의 첫 번째 program logic에서 green phase 인덱스 + green→다음yellow 매핑 추출.

        2x2grid 예: phases = [G, y, G, y, G, y, G, y]
          → green_indices = [0, 2, 4, 6]
          → yellow_map    = {0: 1, 2: 3, 4: 5, 6: 7}

        단일교차로 예: phases = [G, G, G, G, y]
          → green_indices = [0, 1, 2, 3]
          → yellow_map    = {3: 4}  (마지막 green→끝의 yellow만 매핑됨)
            나머지 green→green 전환은 yellow 없이 직접 전환.
        """
        logics = list(conn.trafficlight.getAllProgramLogics(tls_id))
        if not logics:
            raise RuntimeError(f"TLS '{tls_id}'에 program logic이 없습니다.")
        phases = list(getattr(logics[0], "phases", []))
        n = len(phases)
        if n == 0:
            raise RuntimeError(f"TLS '{tls_id}'에 phase가 없습니다.")

        def is_green(state: str) -> bool:
            return 'g' in state.lower()

        def is_yellow(state: str) -> bool:
            s = state.lower()
            return 'y' in s and 'g' not in s

        green_indices: List[int] = []
        yellow_map: Dict[int, int] = {}

        for i, ph in enumerate(phases):
            state = getattr(ph, "state", "")
            if not is_green(state):
                continue
            green_indices.append(i)
            # 다음 phase 순회: yellow를 만나면 매핑, green을 만나면 매핑 없음 (직접 전환)
            for j in range(1, n + 1):
                next_idx = (i + j) % n
                next_state = getattr(phases[next_idx], "state", "")
                if is_green(next_state):
                    break  # yellow 없이 다음 green이 오는 구조 (single intersection)
                if is_yellow(next_state):
                    yellow_map[i] = next_idx
                    break

        if not green_indices:
            raise RuntimeError(
                f"TLS '{tls_id}'에 green phase가 없습니다. "
                "phase state 문자열에 'G'/'g'가 포함된 phase가 있어야 합니다."
            )
        return green_indices, yellow_map

    # ------------------------------------------------------------------
    # SUMO 연결 시작 / 종료
    # ------------------------------------------------------------------

    # 자동 cycle 차단 시 사용하는 매우 큰 phase duration (사실상 영구)
    # 학습 1 episode = 3600 sim sec << 100000 → 자동 phase 진행 발생 안 함
    _PHASE_DURATION_LOCK = 100000.0

    def _set_phase_locked(self, tls_id: str, phase_idx: int) -> None:
        """setPhase + 자동 cycle 차단 (D2 솔루션).

        SUMO TLS program 은 setPhase 호출 후에도 phase duration 만료 시 자동으로
        다음 phase 로 진행한다 (e.g. 2x2grid 의 phase 0 duration=33s → 33초 후
        SUMO 가 자동으로 phase 1 yellow 로 전환). 이는 agent 의 의도와 무관한
        phase 전환을 유발하여 학습 신호를 노이즈화한다.

        해결: setPhase 직후 phase duration 을 100000s 로 강제 → 자동 진행 차단.
              agent action 으로 setPhase 가 다시 호출될 때까지 phase 유지.
        """
        self.conn.trafficlight.setPhase(tls_id, phase_idx)
        self.conn.trafficlight.setPhaseDuration(tls_id, self._PHASE_DURATION_LOCK)

    def _start_sumo(self, seed: Optional[int] = None) -> None:
        if self.conn is not None:
            self.close()
        binary = self._sumo_binary()
        self._conn_label = f"sumo_pz_{uuid.uuid4().hex[:8]}"
        cmd = [
            binary, "-c", str(self.sumo_cfg),
            "--no-warnings", "true",
            "--time-to-teleport", str(self.time_to_teleport),
            # getAccumulatedWaitingTime 의 집계 윈도우. SUMO 기본 100s 는 LOS D 에서
            # 차량당 대기시간을 ~100s 로 포화시켜 avg_waiting_time 지표를 왜곡한다
            # (에피소드 3600s 전체를 덮도록 10000s — SUMO-RL 관례와 동일).
            "--waiting-time-memory", "10000",
            "--seed", str(0 if seed is None else seed),
        ]
        traci.start(cmd, label=self._conn_label)
        self.conn = traci.getConnection(self._conn_label)
        # lane / phase 스펙은 __init__의 probe 결과를 재사용 (재추출 불필요)

    def _subscribe_lanes(self) -> None:
        """lane subscription 설치 (SUMO 프로세스가 reset 마다 새로 떠서 매번 재설치).

        in-lane(controlled 합집합)은 obs/지표/brt·corridor 경로용 4개 변수,
        경계 out-lane(어느 in-lane 도 아님)은 pressure 의 차량 수만 구독.
        같은 lane 을 두 번 subscribe 하면 변수 목록이 *덮어써지므로* 집합을 분리한다.
        """
        in_set = set(self._lane_ids)
        for ln in self._lane_ids:
            self.conn.lane.subscribe(ln, _LANE_IN_SUB_VARS)
        out_only = {
            ln
            for agent in self.possible_agents
            for ln in self._per_agent_out_lanes[agent]
        } - in_set
        for ln in out_only:
            self.conn.lane.subscribe(ln, _LANE_OUT_SUB_VARS)

    # 정지 transition 감지 임계 (m/s). SUMO 기본 차량 모델 기준 정지 직전/후 ≈ 0.
    _STOP_SPEED_THRESHOLD = 0.1

    def _simulate_seconds(self, num_seconds: int) -> Tuple[bool, int]:
        """num_seconds만큼 시뮬레이션 진행; (done, 실제_진행_초) 반환.

        per-vehicle 값은 subscription 일괄수신(getAllSubscriptionResults — IPC 없는
        로컬 조회)으로 읽는다. 순회 집합은 기존과 동일하게 getIDList() 기준
        (텔레포트 중 차량 제외 의미 보존). 구독 결과가 없는 차량(이론상 희귀)은
        기존 개별 호출로 폴백 — 수치 산출 로직은 폴링 시절과 동일.
        """
        for progressed in range(1, num_seconds + 1):
            self.conn.simulationStep()
            self.sim_step += 1

            # 신규 출발 차량 구독 + vClass 1회 캐시 (subscribe 응답이 현재 step 값을
            # 즉시 포함 → 같은 step 에 바로 읽기 가능. bit-identical 게이트로 검증됨.)
            for veh_id in self.conn.simulation.getDepartedIDList():
                self.conn.vehicle.subscribe(veh_id, _VEH_SUB_VARS)
                if veh_id not in self._vehicle_class:
                    self._vehicle_class[veh_id] = (
                        self.conn.vehicle.getVehicleClass(veh_id)
                    )
            veh_res = self.conn.vehicle.getAllSubscriptionResults()

            for veh_id in self.conn.vehicle.getIDList():
                if veh_id not in self._depart_time:
                    self._depart_time[veh_id] = self.sim_step
                r = veh_res.get(veh_id)
                if r is not None:
                    wait_acc = float(r[tc.VAR_ACCUMULATED_WAITING_TIME])
                    speed = float(r[tc.VAR_SPEED])
                    co2 = float(r[tc.VAR_CO2EMISSION])
                else:  # 구독 누락 폴백 (기존 폴링 경로와 동일 호출)
                    wait_acc = float(self.conn.vehicle.getAccumulatedWaitingTime(veh_id))
                    speed = float(self.conn.vehicle.getSpeed(veh_id))
                    co2 = float(self.conn.vehicle.getCO2Emission(veh_id))
                self._latest_waiting[veh_id] = wait_acc
                # Green Wave Tier 1 지표: speed + CO2 + 정지 횟수
                self._episode_speed_sum += speed
                self._episode_speed_sq_sum += speed * speed
                self._episode_speed_count += 1
                self._episode_co2_sum += co2
                self._vehicle_seen.add(veh_id)
                # 정지 transition: prev > 임계 AND 현재 ≤ 임계
                prev = self._vehicle_prev_speed.get(veh_id, speed)
                if prev > self._STOP_SPEED_THRESHOLD and speed <= self._STOP_SPEED_THRESHOLD:
                    self._vehicle_stop_count[veh_id] = (
                        self._vehicle_stop_count.get(veh_id, 0) + 1
                    )
                self._vehicle_prev_speed[veh_id] = speed

                # BRT/일반 차량 분리 누적 (BRT 우선처리 평가 metric).
                # vClass 는 정적이라 첫 등장 시 1회 캐시. 비-BRT 시나리오는 brt_count=0
                # 으로 누적 → info 의 avg_wait_brt 는 0.0 으로 자연 표시.
                vcls = self._vehicle_class.get(veh_id)
                if vcls is None:
                    vcls = self.conn.vehicle.getVehicleClass(veh_id)
                    self._vehicle_class[veh_id] = vcls
                if vcls == "bus":
                    self._episode_brt_wait_sum += wait_acc
                    self._episode_brt_wait_count += 1
                    self._episode_brt_speed_sum += speed
                    self._episode_brt_speed_count += 1
                else:
                    self._episode_car_wait_sum += wait_acc
                    self._episode_car_wait_count += 1
                    self._episode_car_speed_sum += speed
                    self._episode_car_speed_count += 1

            for veh_id in self.conn.simulation.getArrivedIDList():
                self._throughput += 1
                if veh_id in self._depart_time:
                    self._completed_travel.append(
                        float(self.sim_step - self._depart_time.pop(veh_id))
                    )
                if veh_id in self._latest_waiting:
                    self._completed_waiting.append(
                        float(self._latest_waiting.pop(veh_id))
                    )

            # lane별 halting cumsum + max_queue + 방향별 대기시간 — lane 구독 일괄수신.
            lane_res = self.conn.lane.getAllSubscriptionResults()
            halting_per_lane = []
            for ln in self._lane_ids:
                lr = lane_res.get(ln)
                if lr is not None:
                    h = float(lr[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
                    w = float(lr[tc.VAR_WAITING_TIME])
                else:  # 구독 누락 폴백
                    h = float(self.conn.lane.getLastStepHaltingNumber(ln))
                    w = float(self.conn.lane.getWaitingTime(ln))
                halting_per_lane.append(h)
                # Green Wave Tier 2: lane.getWaitingTime 은 그 lane 의 현재 대기 차량 합.
                # EW/NS 분리해 episode 내 평균을 후속에서 계산.
                if self._lane_direction.get(ln, "ew") == "ew":
                    self._episode_wait_ew_sum += w
                    self._episode_wait_ew_count += 1
                else:
                    self._episode_wait_ns_sum += w
                    self._episode_wait_ns_count += 1
            self._queue_cumsum += sum(halting_per_lane)
            if halting_per_lane:
                step_max = max(halting_per_lane)
                if step_max > self._episode_max_queue:
                    self._episode_max_queue = step_max

            self._episode_teleported += int(
                self.conn.simulation.getStartingTeleportNumber()
            )

            # Insertion-failure 추적: 매 step loaded / departed 누적, pending 피크
            # pending = 출발 시각이 지났는데 아직 lane 에 들어가지 못한 차량 수.
            # peak 와 final 두 값 모두 기록 — saturate 패턴 진단용.
            self._episode_loaded_total += int(
                self.conn.simulation.getLoadedNumber()
            )
            self._episode_departed_total += int(
                self.conn.simulation.getDepartedNumber()
            )
            try:
                pending_now = len(self.conn.simulation.getPendingVehicles())
            except (traci.TraCIException, AttributeError):
                pending_now = 0
            if pending_now > self._episode_pending_peak:
                self._episode_pending_peak = pending_now
            self._episode_pending_final = pending_now

            # 외부 hook 호출 (record_video continuous 모드용 frame 캡쳐 등)
            # hook 안에서 env.render() 호출되면 매 sim step 마다 frame 1개 생성됨
            for hook in self._step_hooks:
                try:
                    hook(self.sim_step)
                except Exception:
                    # hook 예외가 시뮬레이션을 중단시키지 않도록 silent 처리
                    pass

            if self.sim_step >= self.max_steps:
                return True, progressed

        return False, num_seconds

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _compute_obs(self, agent: str) -> np.ndarray:
        """SUMO-RL 표준 형태: [phase_one_hot, min_green_flag, density, queue].

        - phase_one_hot : 현재 green phase의 one-hot 벡터 (length num_green)
        - min_green_flag: 현재 phase가 min_green 시간 충족했는지 (0 또는 1)
        - density       : lane별 [현재 차량 수 / lane 용량] (length num_lanes)
        - queue         : lane별 [halting 차량 수 / lane 용량] (length num_lanes)
        """
        lanes = self._per_agent_lanes[agent]
        green_idx = self._current_green_idx[agent]

        phase_one_hot = np.zeros(self._num_green, dtype=np.float32)
        phase_one_hot[green_idx] = 1.0

        min_green_flag = np.float32(
            1.0 if self._elapsed_phase_time[agent] >= self.min_green else 0.0
        )

        # lane별 raw queue/차량수 — lane 구독 결과(로컬 dict, IPC 없음)에서 조회.
        # 직전 simulationStep 의 값 = 기존 개별 호출과 동일 시점.
        lane_res = self.conn.lane.getAllSubscriptionResults()
        raw_queue = [
            float(lane_res[ln][tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
            if ln in lane_res
            else float(self.conn.lane.getLastStepHaltingNumber(ln))
            for ln in lanes
        ]
        self._last_queue_per_lane[agent] = raw_queue

        density = np.array([
            min(1.0, (lane_res[ln][tc.LAST_STEP_VEHICLE_NUMBER]
                      if ln in lane_res
                      else self.conn.lane.getLastStepVehicleNumber(ln))
                / self._lane_capacities[ln])
            for ln in lanes
        ], dtype=np.float32)
        queue = np.array([
            min(1.0, q / self._lane_capacities[ln])
            for ln, q in zip(lanes, raw_queue)
        ], dtype=np.float32)

        # 이웃 요약 캐시: 실제 lane(패딩 전) 평균 density/queue. _enrich_obs 가 읽는다.
        if self.neighbor_obs:
            self._agent_dq_summary[agent] = np.array(
                [float(density.mean()) if density.size else 0.0,
                 float(queue.mean()) if queue.size else 0.0],
                dtype=np.float32,
            )
            # upstream_phase: 이웃이 읽을 위상 시계 [phase_one_hot, elapsed_norm].
            if self.upstream_phase:
                ps = np.zeros(self._phase_slot_dim, dtype=np.float32)
                ps[green_idx] = 1.0
                ps[-1] = min(1.0, self._elapsed_phase_time[agent] / self._PHASE_TIME_NORM)
                self._agent_phase_summary[agent] = ps

        # mixed-topology (BRT corridor TLS vs 일반 TLS) 시 agent별 lane 수가 다름.
        # shape 일관성 위해 max(_num_lanes) 까지 0 패딩.
        pad = self._num_lanes - len(lanes)
        if pad > 0:
            density = np.concatenate([density, np.zeros(pad, dtype=np.float32)])
            queue   = np.concatenate([queue,   np.zeros(pad, dtype=np.float32)])

        obs = np.concatenate([phase_one_hot, [min_green_flag], density, queue])
        self._last_obs[agent] = obs
        return obs

    # ------------------------------------------------------------------
    # PettingZoo Parallel API
    # ------------------------------------------------------------------

    def _enrich_obs(
        self, local_obs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """neighbor_obs 면 각 agent actor obs 끝에 N/S 이웃 상류 요약을 덧붙인다.

        방향당 블록 = [mean_density, mean_queue] (혼잡, #3) + (upstream_phase 면)
        [phase_one_hot, elapsed_norm] (위상 시계, ②). 직전 _compute_obs 가 채운
        _agent_dq_summary / _agent_phase_summary 캐시를 읽는다. 없는 방향은 0.
        방출 방향은 _OBS_NBR_DIRS=(N,S) — 회랑(green wave)은 N/S 축이라 E/W 는 컷.
        결과 길이 = _obs_dim (29 또는 39). 2-pass 인 이유: 이웃 요약은 다른 agent 의 obs
        계산 결과를 참조하므로 모든 _compute_obs 가 끝난 뒤에야 안전히 조립 가능.
        """
        if not self.neighbor_obs:
            return local_obs
        zero2 = np.zeros(self._NBR_SUMMARY_DIM, dtype=np.float32)
        zero_ps = np.zeros(self._phase_slot_dim, dtype=np.float32)
        enriched: Dict[str, np.ndarray] = {}
        for a, base in local_obs.items():
            dirs = self._neighbor_dir.get(a, {})
            parts = [base]
            for d in self._OBS_NBR_DIRS:
                nbr = dirs.get(d)
                parts.append(self._agent_dq_summary.get(nbr, zero2) if nbr else zero2)
                if self.upstream_phase:
                    parts.append(self._agent_phase_summary.get(nbr, zero_ps) if nbr else zero_ps)
            obs = np.concatenate(parts).astype(np.float32)
            enriched[a] = obs
            self._last_obs[a] = obs  # 결측 agent fallback 용 (enriched 길이로 갱신)
        return enriched

    def _wrap_obs(self, local_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """ctde_mode 면 per-agent obs 를 단일 flat Box 로 평탄화 (critic 입력).

        두 가지 critic 레이아웃 (둘 다 첫 _obs_dim 차원 = 자기 local → module 이 slice):
        - neighbor_obs=True  (#1): [own | agent_id one-hot(N) | N/E/S/W 이웃 obs(_obs_dim)]
              무관한 비이웃 agent 를 빼고 방향성·정체성을 부여한 agent-aware critic.
              없는 방향 이웃은 0 패딩.
        - neighbor_obs=False (legacy): [own | 전체 agent obs concat(_obs_dim × N)]
              기존 naive 방식 (옛 체크포인트 재현용).

        concat 순서는 항상 `possible_agents` 기준 (stable). `self.agents` 는 에피소드
        종료 시 빈 리스트로 mutate 되므로 사용 금지. 결측 agent 는 0 벡터로 채운다.
        (Dict obs 는 RLlib worker connector 에서 silent fail → 항상 flat Box.)
        """
        if not self.ctde_mode:
            return local_obs
        zero = np.zeros(self._obs_dim, dtype=np.float32)

        if self.neighbor_obs:
            n_agents = len(self.possible_agents)
            out: Dict[str, np.ndarray] = {}
            for a in local_obs:
                idx = self.possible_agents.index(a)
                agent_id = np.zeros(n_agents, dtype=np.float32)
                agent_id[idx] = 1.0
                parts = [local_obs[a], agent_id]
                dirs = self._neighbor_dir.get(a, {})
                for d in self._NEIGHBOR_DIRS:
                    nbr = dirs.get(d)
                    if nbr is not None:
                        parts.append(local_obs.get(nbr, self._last_obs.get(nbr, zero)))
                    else:
                        parts.append(zero)
                out[a] = np.concatenate(parts).astype(np.float32)
            return out

        # legacy: 전체 agent naive concat
        global_vec = np.concatenate([
            local_obs.get(a, self._last_obs.get(a, zero))
            for a in self.possible_agents
        ]).astype(np.float32)
        return {
            a: np.concatenate([local_obs[a], global_vec]).astype(np.float32)
            for a in local_obs
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.agents = list(self.possible_agents)
        self._start_sumo(seed=seed)
        # lane subscription 은 SUMO 프로세스마다 재설치, vClass 캐시는 에피소드 단위.
        # (_lane_max_speed 는 net 정적이라 유지.)
        self._subscribe_lanes()
        self._vehicle_class.clear()

        self.sim_step = 0
        self._depart_time.clear()
        self._latest_waiting.clear()
        self._completed_waiting.clear()
        self._completed_travel.clear()
        self._throughput = 0
        self._queue_cumsum = 0.0
        self._last_wait_measure.clear()

        # 진단 카운터 초기화
        self._episode_phase_switches = 0
        self._episode_max_queue = 0.0
        self._episode_action_counts.fill(0)
        self._episode_teleported = 0
        self._episode_loaded_total = 0
        self._episode_departed_total = 0
        self._episode_pending_peak = 0
        self._episode_pending_final = 0
        self._episode_yellow_seconds = 0

        # Green Wave 지표 초기화 (Tier 1 + 2)
        self._episode_co2_sum = 0.0
        self._episode_speed_sum = 0.0
        self._episode_speed_sq_sum = 0.0
        self._episode_speed_count = 0
        self._vehicle_stop_count.clear()
        self._vehicle_prev_speed.clear()
        self._vehicle_seen.clear()
        self._episode_wait_ew_sum = 0.0
        self._episode_wait_ns_sum = 0.0
        self._episode_wait_ew_count = 0
        self._episode_wait_ns_count = 0

        # BRT/car 분리 누적 초기화
        self._episode_brt_wait_sum = 0.0
        self._episode_brt_wait_count = 0
        self._episode_car_wait_sum = 0.0
        self._episode_car_wait_count = 0
        self._episode_brt_speed_sum = 0.0
        self._episode_brt_speed_count = 0
        self._episode_car_speed_sum = 0.0
        self._episode_car_speed_count = 0
        self._episode_corridor_brt_ratio_sum = 0.0
        self._episode_corridor_brt_count = 0

        # 초기 phase: 각 agent의 green_phases[0] 으로 강제 설정 (자동 cycle 차단)
        for agent in self.agents:
            initial_phase = self._per_agent_green_phases[agent][0]
            self._current_green_idx[agent] = 0
            self._current_phase[agent] = initial_phase
            self._elapsed_phase_time[agent] = 0
            self._set_phase_locked(self._agent_to_tls[agent], initial_phase)

        local_obs = {a: self._compute_obs(a) for a in self.agents}
        local_obs = self._enrich_obs(local_obs)
        observations = self._wrap_obs(local_obs)
        infos = {a: {"phase": self._current_phase[a]} for a in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]):
        # 1. 정책 출력 분포 추적 — min_green 미충족이라도 출력 자체는 카운트
        for agent in self.agents:
            self._episode_action_counts[int(actions[agent])] += 1

        # 2. action → 목표 green phase 인덱스 매핑
        target_green_idx = {a: int(actions[a]) for a in self.agents}

        # 3. switching 판정: 다른 green을 요청 + min_green 충족
        switching = {
            a: (
                target_green_idx[a] != self._current_green_idx[a]
                and self._elapsed_phase_time[a] >= self.min_green
            )
            for a in self.agents
        }
        self._episode_phase_switches += sum(switching.values())

        done = False

        # 4. yellow 전환: 현재 green→다음yellow 가 매핑된 agent만 yellow 적용
        yellow_applied = False
        for agent in self.agents:
            if not switching[agent]:
                continue
            current_phase_idx = self._current_phase[agent]
            yellow_idx = self._per_agent_yellow_map[agent].get(current_phase_idx)
            if yellow_idx is not None:
                self._set_phase_locked(self._agent_to_tls[agent], yellow_idx)
                yellow_applied = True

        if yellow_applied:
            done, yellow_progressed = self._simulate_seconds(self.yellow_time)
            # 옵션 C: 실제로 yellow phase 로 시뮬레이션된 sec 누적
            self._episode_yellow_seconds += int(yellow_progressed)
            # 전환하지 않은 agent 는 이 yellow 구간에도 자기 green 을 유지했으므로
            # 경과시간에 산입 (min_green 게이팅 + upstream_phase 의 elapsed_norm 위상
            # 시계 정확화). 전환 agent 는 아래에서 0 으로 리셋되므로 가산해도 무관하나
            # 의미 명확성을 위해 비전환 agent 에만 더한다.
            for agent in self.agents:
                if not switching[agent]:
                    self._elapsed_phase_time[agent] += int(yellow_progressed)

        # 5. 목표 green phase 적용
        if not done:
            for agent in self.agents:
                if switching[agent]:
                    new_green_idx = target_green_idx[agent]
                    new_phase = self._per_agent_green_phases[agent][new_green_idx]
                    self._current_green_idx[agent] = new_green_idx
                    self._current_phase[agent] = new_phase
                    self._elapsed_phase_time[agent] = 0
                    self._set_phase_locked(self._agent_to_tls[agent], new_phase)

        # 6. delta_time 진행
        if not done:
            done, progressed = self._simulate_seconds(self.delta_time)
            for agent in self.agents:
                self._elapsed_phase_time[agent] += progressed

        # 7. 결과 계산
        avg_wait = (
            float(np.mean(self._completed_waiting)) if self._completed_waiting else 0.0
        )
        std_wait = (
            float(np.std(self._completed_waiting)) if self._completed_waiting else 0.0
        )
        avg_travel = (
            float(np.mean(self._completed_travel)) if self._completed_travel else 0.0
        )
        local_obs: Dict[str, np.ndarray] = {}
        rewards, terminations, truncations, infos = {}, {}, {}, {}
        for agent in self.agents:
            local_obs[agent] = self._compute_obs(agent)

            # ── 보상 함수 ─────────────────────────────────────────────
            lanes = self._per_agent_lanes[agent]

            if self.reward_mode == "pressure":
                # max-pressure: 하류 차량 수 - 상류 차량 수 (상태 기반).
                # 하류가 포화일수록 reward 가 낮아져 스필백을 억제한다.
                # lane 구독 결과 사용 (경계 out-lane 도 VEHICLE_NUMBER 구독됨).
                lane_res = self.conn.lane.getAllSubscriptionResults()

                def _veh_num(ln):
                    lr = lane_res.get(ln)
                    if lr is not None:
                        return lr[tc.LAST_STEP_VEHICLE_NUMBER]
                    return self.conn.lane.getLastStepVehicleNumber(ln)

                out_lanes = self._per_agent_out_lanes[agent]
                n_out = sum(_veh_num(ln) for ln in out_lanes)
                n_in = sum(_veh_num(ln) for ln in lanes)
                reward = float(n_out - n_in)
            else:
                # diff-waiting-time: (이전 step 대기시간 - 현재) / 10.
                # 측정치는 두 경로 모두 *연속 정지시간*(vehicle.getWaitingTime; 움직이면
                # 리셋) — lane.getWaitingTime == Σ vehicle.getWaitingTime 이므로
                # brt_weight == 1.0 이면 fast path 와 수학적으로 동일하다.
                # (과거 w>1 경로는 getAccumulatedWaitingTime(100s 윈도우 누적)을 써서
                #  가중치 토글이 측정치 정의까지 바꾸는 버그가 있었음 — 통일 완료.)
                if self.brt_weight == 1.0:
                    current_wait = sum(
                        self.conn.lane.getWaitingTime(ln) for ln in lanes
                    ) / 10.0
                else:
                    lane_res = self.conn.lane.getAllSubscriptionResults()
                    veh_res = self.conn.vehicle.getAllSubscriptionResults()
                    weighted = 0.0
                    for ln in lanes:
                        lr = lane_res.get(ln)
                        vids = (lr[tc.LAST_STEP_VEHICLE_ID_LIST] if lr is not None
                                else self.conn.lane.getLastStepVehicleIDs(ln))
                        for vid in vids:
                            vcls = self._vehicle_class.get(vid)
                            if vcls is None:
                                vcls = self.conn.vehicle.getVehicleClass(vid)
                                self._vehicle_class[vid] = vcls
                            w = self.brt_weight if vcls == "bus" else 1.0
                            vr = veh_res.get(vid)
                            wt = (float(vr[tc.VAR_WAITING_TIME]) if vr is not None
                                  else float(self.conn.vehicle.getWaitingTime(vid)))
                            weighted += w * wt
                    current_wait = weighted / 10.0
                reward = self._last_wait_measure.get(agent, current_wait) - current_wait
                self._last_wait_measure[agent] = current_wait

            # switch_penalty: phase oscillation 억제 (모든 reward_mode 공통)
            if switching[agent] and self.switch_penalty != 0.0:
                reward -= self.switch_penalty

            # ① 회랑 progression: corridor agent 의 ns-through lane 무정차 통과를 가산
            # (dense 가중 평균 속도비). green wave 유도. metric 은 항상 누적, reward 가산은
            # progression_coeff != 0 일 때만. 비-corridor agent 는 no-op.
            if agent in self._corridor_agents:
                r_prog = self._corridor_step(agent)
                if self.progression_coeff != 0.0:
                    reward += self.progression_coeff * r_prog

            rewards[agent] = reward
            terminations[agent] = False
            truncations[agent] = done

        # ── Green Wave 평가 지표 집계 (Tier 1 + 2) ─────────────────────────
        n_seen = max(1, len(self._vehicle_seen))
        n_speed = max(1, self._episode_speed_count)
        avg_speed = self._episode_speed_sum / n_speed
        # 표본분산 = E[X²] − (E[X])²
        speed_var = max(0.0, self._episode_speed_sq_sum / n_speed - avg_speed ** 2)
        speed_std = math.sqrt(speed_var)
        speed_cv = speed_std / avg_speed if avg_speed > 1e-6 else 0.0
        avg_stops = sum(self._vehicle_stop_count.values()) / n_seen
        avg_co2 = self._episode_co2_sum / n_seen
        wait_ew = (
            self._episode_wait_ew_sum / self._episode_wait_ew_count
            if self._episode_wait_ew_count > 0 else 0.0
        )
        wait_ns = (
            self._episode_wait_ns_sum / self._episode_wait_ns_count
            if self._episode_wait_ns_count > 0 else 0.0
        )

        # BRT 우선처리 평가용 — vClass=bus 와 일반 차량 분리 평균.
        # 표본이 없으면 0.0 (예: single/2x2/3x2 시나리오 → brt_count=0).
        avg_wait_brt = (
            self._episode_brt_wait_sum / self._episode_brt_wait_count
            if self._episode_brt_wait_count > 0 else 0.0
        )
        avg_wait_car = (
            self._episode_car_wait_sum / self._episode_car_wait_count
            if self._episode_car_wait_count > 0 else 0.0
        )
        avg_speed_brt = (
            self._episode_brt_speed_sum / self._episode_brt_speed_count
            if self._episode_brt_speed_count > 0 else 0.0
        )
        avg_speed_car = (
            self._episode_car_speed_sum / self._episode_car_speed_count
            if self._episode_car_speed_count > 0 else 0.0
        )
        # ① 회랑 진행파 — corridor ns-through lane BRT 평균 속도비 ∈ [0,1] (높을수록 무정차).
        corridor_brt_speed_ratio = (
            self._episode_corridor_brt_ratio_sum / self._episode_corridor_brt_count
            if self._episode_corridor_brt_count > 0 else 0.0
        )

        # info dict — 모든 agent 가 동일한 에피소드 지표를 받음 (callback/eval 호환)
        for agent in self.agents:
            infos[agent] = {
                "phase": self._current_phase[agent],
                "phase_changed": switching[agent],
                "avg_waiting_time": avg_wait,
                "std_waiting_time": std_wait,
                "avg_travel_time": avg_travel,
                "total_queue_length": float(self._queue_cumsum),
                "throughput": self._throughput,
                "phase_switches": int(self._episode_phase_switches),
                "max_queue": float(self._episode_max_queue),
                "teleported": int(self._episode_teleported),
                # Insertion-failure 가시화: lane saturation 으로 출발 못 한 차량 수
                # loaded - departed = 누적 insertion 실패, pending_peak/final 은 즉시 대기수
                "vehicles_loaded":       int(self._episode_loaded_total),
                "vehicles_departed":     int(self._episode_departed_total),
                "vehicles_lost_insert":  int(self._episode_loaded_total - self._episode_departed_total),
                "pending_insert_peak":   int(self._episode_pending_peak),
                "pending_insert_final":  int(self._episode_pending_final),
                # yellow 비율 — phase oscillation 정도의 단일 수치 진단.
                # 균형값(단일교차로 기본 ≈ 0.07)보다 크게 높으면 oscillation 의심.
                "yellow_seconds":        int(self._episode_yellow_seconds),
                "yellow_ratio":          (float(self._episode_yellow_seconds) / float(self.sim_step)
                                          if self.sim_step > 0 else 0.0),
                "action_counts": self._episode_action_counts.tolist(),
                # 렌더러/디버깅용 — agent별 controlled lane 순서대로의 halting 차량 수
                "queue_per_lane": list(self._last_queue_per_lane[agent]),
                # Green Wave Tier 1 직접 지표
                "avg_stops_per_vehicle": float(avg_stops),
                "avg_co2_per_vehicle":   float(avg_co2),
                "avg_speed":             float(avg_speed),
                # Green Wave Tier 2 코리도어 지표
                "speed_cv":              float(speed_cv),
                "per_direction_wait_ew": float(wait_ew),
                "per_direction_wait_ns": float(wait_ns),
                # BRT 우선처리 평가용 — vClass=bus vs 일반 차량 분리 metric.
                # brt_weight 적용 여부와 무관하게 항상 기록 (baseline 비교 용이).
                # 비-BRT 시나리오는 brt_seen=0 / avg_wait_brt=0.0 으로 자연 표시.
                "avg_wait_brt":  float(avg_wait_brt),
                "avg_wait_car":  float(avg_wait_car),
                "avg_speed_brt": float(avg_speed_brt),
                "avg_speed_car": float(avg_speed_car),
                "brt_seen":      int(self._episode_brt_wait_count),
                "car_seen":      int(self._episode_car_wait_count),
                # ① 회랑 진행파 직접 지표 (corridor ns-through BRT 평균 속도비).
                "corridor_brt_speed_ratio": float(corridor_brt_speed_ratio),
                "corridor_brt_seen":        int(self._episode_corridor_brt_count),
            }

        local_obs = self._enrich_obs(local_obs)
        observations = self._wrap_obs(local_obs)

        if done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self) -> np.ndarray:
        return self._renderer.render(
            agent_to_tls=self._agent_to_tls,
            current_phase=self._current_phase,
            last_obs=self._last_obs,
            sim_step=self.sim_step,
            max_steps=self.max_steps,
            # 정확한 큐 시각화를 위한 lane별 raw halting 차량 수 + lane id 매핑
            # obs 가 [phase_one_hot, density, queue] 구조로 바뀐 후 obs[:4] 가
            # 큐가 아니게 되어 추가됨 (SumoRenderer 옵션 B 큐 막대 정확화)
            queue_per_lane=self._last_queue_per_lane,
            lane_ids=self._per_agent_lanes,
            title_label=self.render_label,
        )

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None


if __name__ == "__main__":
    env = SumoParallelEnv()
    info: dict = {}
    step = 0
    try:
        obs, infos = env.reset(seed=42)
        print(f"agents: {env.agents}")
        print(f"obs_dim: {env._obs_dim} (num_green={env._num_green}, num_lanes={env._num_lanes})")
        print(f"obs shapes: {[(a, o.shape) for a, o in obs.items()]}")
        print(f"green_phases per agent: {env._per_agent_green_phases}")
        print(f"yellow_map per agent:   {env._per_agent_yellow_map}")

        while env.agents:
            actions = {a: env.action_space(a).sample() for a in env.agents}
            obs, rew, term, trunc, info = env.step(actions)
            step += 1
    finally:
        env.close()
    print(f"done after {step} steps | info: {info}")
