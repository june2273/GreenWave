import functools
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

try:
    import traci
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
      "queue"            : -(mean(queues)+max(queues))/10 + throughput×0.5
      "diff-waiting-time": 누적 대기시간 변화량 (SUMO-RL 검증 방식)
      "pressure"         : 출력 lane 차량 수 - 입력 lane 차량 수
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
        "name": "sumo_intersection_v0",
    }

    REWARD_MODES: List[str] = ["queue", "diff-waiting-time", "pressure"]

    def __init__(
        self,
        sumo_cfg: Optional[str] = None,
        use_gui: bool = False,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 10,
        max_steps: int = 3600,
        reward_mode: str = "queue",
        tls_ids: Optional[List[str]] = None,
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
            self.sumo_cfg = self.sumo_data_dir / "single.sumocfg"
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

        # 모든 agent의 obs/act 차원이 동일하다고 가정 (정사각 grid). 첫 agent 기준.
        # 다른 형태가 필요하면 max+padding 방식으로 확장 가능.
        first = self.possible_agents[0]
        self._num_green: int = len(self._per_agent_green_phases[first])
        self._num_lanes: int = len(self._per_agent_lanes[first])
        self._obs_dim: int = self._num_green + 1 + 2 * self._num_lanes

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

        # 에피소드 지표 버퍼
        self._depart_time: Dict[str, int] = {}
        self._latest_waiting: Dict[str, float] = {}
        self._completed_waiting: List[float] = []
        self._completed_travel: List[float] = []
        self._throughput: int = 0
        self._queue_cumsum: float = 0.0

        # 진단 카운터 (action_counts는 동적 num_green 길이)
        self._episode_phase_switches: int = 0
        self._episode_max_queue: float = 0.0
        self._episode_action_counts: np.ndarray = np.zeros(self._num_green, dtype=np.int64)
        self._episode_teleported: int = 0

        # diff-waiting-time 보상 모드 전용
        self._last_wait_measure: Dict[str, float] = {}

        # 네트워크 시각화 렌더러
        self._renderer = SumoRenderer(self.sumo_cfg)

    # ------------------------------------------------------------------
    # PettingZoo 필수 인터페이스 — obs/act space 동적 결정
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Discrete(self._num_green)

    # ------------------------------------------------------------------
    # SUMO 유틸리티
    # ------------------------------------------------------------------

    def _maybe_build_network(self) -> None:
        net_file = self.sumo_data_dir / "single_intersection.net.xml"
        if net_file.exists():
            return
        netconvert = shutil.which("netconvert")
        if not netconvert:
            raise FileNotFoundError(
                "single_intersection.net.xml이 없고 netconvert도 찾을 수 없습니다."
            )
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".net.xml", dir=self.sumo_data_dir
        )
        os.close(tmp_fd)
        try:
            cmd = [
                netconvert,
                "--node-files", str(self.sumo_data_dir / "nodes.nod.xml"),
                "--edge-files", str(self.sumo_data_dir / "edges.edg.xml"),
                "--connection-files", str(self.sumo_data_dir / "connections.con.xml"),
                "--tllogic-files", str(self.sumo_data_dir / "tls.tll.xml"),
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
                for ln in lanes + out_lanes:
                    if ln not in lane_capacities:
                        length = float(conn.lane.getLength(ln))
                        lane_capacities[ln] = max(1.0, length / _VEHICLE_FOOTPRINT_M)

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

            return {
                "per_agent_lanes": per_agent_lanes,
                "per_agent_out_lanes": per_agent_out_lanes,
                "per_agent_green_phases": per_agent_green_phases,
                "per_agent_yellow_map": per_agent_yellow_map,
                "lane_capacities": lane_capacities,
            }
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

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
            "--time-to-teleport", "-1",
            "--seed", str(0 if seed is None else seed),
        ]
        traci.start(cmd, label=self._conn_label)
        self.conn = traci.getConnection(self._conn_label)
        # lane / phase 스펙은 __init__의 probe 결과를 재사용 (재추출 불필요)

    def _simulate_seconds(self, num_seconds: int) -> Tuple[bool, int]:
        """num_seconds만큼 시뮬레이션 진행; (done, 실제_진행_초) 반환"""
        for progressed in range(1, num_seconds + 1):
            self.conn.simulationStep()
            self.sim_step += 1

            for veh_id in self.conn.vehicle.getIDList():
                if veh_id not in self._depart_time:
                    self._depart_time[veh_id] = self.sim_step
                self._latest_waiting[veh_id] = (
                    self.conn.vehicle.getAccumulatedWaitingTime(veh_id)
                )

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

            # lane별 halting cumsum + max_queue 추적
            halting_per_lane = [
                float(self.conn.lane.getLastStepHaltingNumber(ln))
                for ln in self._lane_ids
            ]
            self._queue_cumsum += sum(halting_per_lane)
            if halting_per_lane:
                step_max = max(halting_per_lane)
                if step_max > self._episode_max_queue:
                    self._episode_max_queue = step_max

            self._episode_teleported += int(
                self.conn.simulation.getStartingTeleportNumber()
            )

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

        # lane별 raw queue 1회 조회 → 캐시 + obs 정규화 양쪽에 재사용 (traci 호출 절약)
        raw_queue = [
            float(self.conn.lane.getLastStepHaltingNumber(ln)) for ln in lanes
        ]
        self._last_queue_per_lane[agent] = raw_queue

        density = np.array([
            min(1.0, self.conn.lane.getLastStepVehicleNumber(ln) / self._lane_capacities[ln])
            for ln in lanes
        ], dtype=np.float32)
        queue = np.array([
            min(1.0, q / self._lane_capacities[ln])
            for ln, q in zip(lanes, raw_queue)
        ], dtype=np.float32)

        obs = np.concatenate([phase_one_hot, [min_green_flag], density, queue])
        self._last_obs[agent] = obs
        return obs

    # ------------------------------------------------------------------
    # PettingZoo Parallel API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.agents = list(self.possible_agents)
        self._start_sumo(seed=seed)

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

        # 초기 phase: 각 agent의 green_phases[0] 으로 강제 설정 (자동 cycle 차단)
        for agent in self.agents:
            initial_phase = self._per_agent_green_phases[agent][0]
            self._current_green_idx[agent] = 0
            self._current_phase[agent] = initial_phase
            self._elapsed_phase_time[agent] = 0
            self._set_phase_locked(self._agent_to_tls[agent], initial_phase)

        observations = {a: self._compute_obs(a) for a in self.agents}
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
        throughput_before = self._throughput

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
            done, _ = self._simulate_seconds(self.yellow_time)

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
        step_throughput = self._throughput - throughput_before

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for agent in self.agents:
            obs = self._compute_obs(agent)
            observations[agent] = obs

            # ── 보상 함수 ─────────────────────────────────────────────
            lanes = self._per_agent_lanes[agent]

            if self.reward_mode == "diff-waiting-time":
                # B1: 정규화 /100 → /10 — reward magnitude 10배 ↑로 VF 학습 신호 강화
                # (free flow 에선 ~0.5, 정체에선 ~5+ 범위가 되어 VF variance 확보)
                current_wait = sum(
                    self.conn.lane.getWaitingTime(ln) for ln in lanes
                ) / 10.0
                reward = self._last_wait_measure.get(agent, current_wait) - current_wait
                self._last_wait_measure[agent] = current_wait

            elif self.reward_mode == "pressure":
                out_lanes = self._per_agent_out_lanes.get(agent, [])
                reward = float(
                    sum(self.conn.lane.getLastStepVehicleNumber(ln) for ln in out_lanes)
                    - sum(self.conn.lane.getLastStepVehicleNumber(ln) for ln in lanes)
                )

            else:  # "queue" (default) — obs 구조가 바뀌어 직접 lane 조회
                queues = np.array([
                    self.conn.lane.getLastStepHaltingNumber(ln) for ln in lanes
                ], dtype=np.float32)
                queue_penalty = (float(queues.mean()) + float(queues.max())) / 10.0
                throughput_bonus = step_throughput * 0.5
                reward = -queue_penalty + throughput_bonus

            rewards[agent] = reward
            terminations[agent] = False
            truncations[agent] = done
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
                "action_counts": self._episode_action_counts.tolist(),
                # 렌더러/디버깅용 — agent별 controlled lane 순서대로의 halting 차량 수
                "queue_per_lane": list(self._last_queue_per_lane[agent]),
            }

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
