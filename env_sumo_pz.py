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


class SumoParallelEnv(ParallelEnv):
    """
    PettingZoo Parallel API 기반 SUMO 교차로 신호제어 환경

    단일 교차로: tls_ids=["C"]  → agents=["tl_0"]
    다중 교차로: tls_ids=["C","D",...] → agents=["tl_0","tl_1",...]

    행동(Discrete 4): 0=North / 1=South / 2=East / 3=West 단독 green
    관측(shape 10): [queue×4, speed×4, phase_norm, elapsed_norm]  (동적 lane 감지, 최대 4 lane)
    보상(reward_mode 선택):
      "queue"            : -(mean(queues)+max(queues))/10 + throughput×0.5  — 기아 방향 추가 패널티
      "diff-waiting-time": -(누적대기 변화량)/100  — 대기 감소 시 양의 보상 (SUMO-RL 방식)
      "pressure"         : out_vehicles - in_vehicles  — 처리량 직접 최대화
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
        "name": "sumo_intersection_v0",
    }

    # 보상 모드 선택지
    # "queue"            : -(mean(queues) + max(queues)) / 10  — 기아 방향 추가 패널티
    # "diff-waiting-time": 누적 대기시간 변화량 (SUMO-RL 검증 방식)
    # "pressure"         : 출력 lane 차량 수 - 입력 lane 차량 수 (처리량 직접 최대화)
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
            # 외부 네트워크 사용 시 GreenWave 기본 net 자동생성 불필요
        else:
            self.sumo_cfg = self.sumo_data_dir / "single.sumocfg"
            # 기본 단일교차로 net.xml이 없을 때만 netconvert로 자동 생성
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

        # 동적 lane id 테이블 (reset 시 _start_sumo() 에서 getControlledLanes() 로 채워짐)
        # _per_agent_lanes  : agent별 입력 lane id 목록 (관측·보상 계산)
        # _per_agent_out_lanes: agent별 출력 lane id 목록 (pressure 보상 전용)
        # _lane_ids         : 전체 에이전트 입력 lane 합집합 (queue_cumsum 집계)
        self._per_agent_lanes: Dict[str, List[str]] = {}
        self._per_agent_out_lanes: Dict[str, List[str]] = {}
        self._lane_ids: List[str] = []

        # agents: 현재 에피소드에서 활성 에이전트 (reset 시 복원, done 시 빈 리스트)
        self.agents: List[str] = []

        # 에이전트별 신호 상태
        self._current_phase: Dict[str, int] = {}
        self._elapsed_phase_time: Dict[str, int] = {}
        self._yellow_phase_index: Dict[str, int] = {}
        self._last_obs: Dict[str, np.ndarray] = {
            a: np.zeros(10, dtype=np.float32) for a in self.possible_agents
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

        # 진단 지표 (mode collapse / network saturation 감지용)
        # phase_switches    : 에피소드 누적 phase 전환 횟수 (agent 합산)
        # max_queue         : 에피소드 중 단일 lane이 도달한 최대 halting 차량 수
        # action_counts     : agent×action 누적 선택 횟수 (정책 분포 추적)
        # teleported        : SUMO가 교착 해소를 위해 텔레포트한 차량 수
        self._episode_phase_switches: int = 0
        self._episode_max_queue: float = 0.0
        self._episode_action_counts: np.ndarray = np.zeros(4, dtype=np.int64)
        self._episode_teleported: int = 0

        # diff-waiting-time 보상 모드 전용: 직전 스텝의 agent별 누적 대기시간 합
        self._last_wait_measure: Dict[str, float] = {}

        # 네트워크 시각화 렌더러 (matplotlib + sumolib 기반)
        self._renderer = SumoRenderer(self.sumo_cfg)

    # ------------------------------------------------------------------
    # PettingZoo 필수 인터페이스
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(10,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Discrete(4)

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
        # 임시 파일에 먼저 생성 후 atomic rename → 병렬 worker 동시 시작 시 TOCTOU 방지
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
            os.replace(tmp_path, net_file)  # POSIX atomic
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

        # TLS 존재 얼리 페일: 잘못된 tls_ids는 _find_yellow_phase_index에서 불명확하게
        # 실패하므로 TraCI 연결 직후 명시적으로 검증
        known_tls = set(self.conn.trafficlight.getIDList())
        missing = [tid for tid in self._tls_ids if tid not in known_tls]
        if missing:
            raise ValueError(
                f"SUMO 네트워크에 존재하지 않는 TLS ID: {missing}. "
                f"사용 가능한 TLS: {sorted(known_tls)}"
            )

        # 동적 lane 감지 (SUMO-RL 패턴): 네트워크 위상에서 직접 읽어
        # 하드코딩 없이 임의 SUMO 네트워크(2x2 그리드 등)에서도 동작
        self._per_agent_lanes = {}
        self._per_agent_out_lanes = {}
        for agent in self.possible_agents:
            tls_id = self._agent_to_tls[agent]
            # 입력 lane: TLS가 제어하는 모든 lane (중복 제거, 순서 보존)
            self._per_agent_lanes[agent] = list(dict.fromkeys(
                self.conn.trafficlight.getControlledLanes(tls_id)
            ))
            # 출력 lane: TLS 연결에서 목적지 lane 추출 (pressure 보상 전용)
            self._per_agent_out_lanes[agent] = list(set(
                link[0][1]
                for link in self.conn.trafficlight.getControlledLinks(tls_id)
                if link
            ))
        # 전체 입력 lane 합집합 (queue_cumsum 집계용, 에이전트 간 중복 제거)
        self._lane_ids = list(dict.fromkeys(
            ln for agent in self.possible_agents
            for ln in self._per_agent_lanes[agent]
        ))

        for agent in self.possible_agents:
            tls_id = self._agent_to_tls[agent]
            self._yellow_phase_index[agent] = self._find_yellow_phase_index(tls_id)

    def _find_yellow_phase_index(self, tls_id: str) -> int:
        for logic in self.conn.trafficlight.getAllProgramLogics(tls_id):
            for idx, phase in enumerate(getattr(logic, "phases", [])):
                state = getattr(phase, "state", "").lower()
                if state and set(state).issubset({"y", "r"}) and "y" in state:
                    return idx
        raise RuntimeError(
            f"TLS '{tls_id}'에서 all-yellow phase를 찾지 못했습니다. "
            "tls.tll.xml의 phase 구성을 확인하세요."
        )

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

            # lane별 halting 차량 수: cumsum 누적 + 단일 lane 최대값 추적
            # (max_queue 는 1 step당 1번만 계산하면 충분 — 동일 루프에서 처리)
            halting_per_lane = [
                float(self.conn.lane.getLastStepHaltingNumber(ln))
                for ln in self._lane_ids
            ]
            self._queue_cumsum += sum(halting_per_lane)
            if halting_per_lane:
                step_max = max(halting_per_lane)
                if step_max > self._episode_max_queue:
                    self._episode_max_queue = step_max

            # teleport 누적: getStartingTeleportNumber는 "이번 step에 시작된 텔레포트 수"
            # 네트워크 교착(grid lock) 발생 시 SUMO가 차량을 강제 이동시키는 신호
            self._episode_teleported += int(
                self.conn.simulation.getStartingTeleportNumber()
            )

            if self.sim_step >= self.max_steps:
                return True, progressed

        return False, num_seconds

    def _compute_obs(self, agent: str) -> np.ndarray:
        # agent별 입력 lane 최대 4개 사용 (shape=10 고정 유지)
        # 4x4 격자처럼 lane이 4개 이상인 경우 앞 4개만, 미만이면 0으로 패딩
        lanes = self._per_agent_lanes.get(agent, self._lane_ids)[:4]
        queues = [float(self.conn.lane.getLastStepHaltingNumber(ln)) for ln in lanes]
        speeds = [max(0.0, float(self.conn.lane.getLastStepMeanSpeed(ln))) for ln in lanes]
        # 4 lane 미만인 경우 0 패딩
        queues += [0.0] * (4 - len(queues))
        speeds += [0.0] * (4 - len(speeds))
        phase_norm = self._current_phase[agent] / 3.0
        elapsed_norm = self._elapsed_phase_time[agent] / max(1.0, float(self.min_green))
        obs = np.array(queues + speeds + [phase_norm, elapsed_norm], dtype=np.float32)
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

        # 진단 카운터 초기화 (reset마다 0으로)
        self._episode_phase_switches = 0
        self._episode_max_queue = 0.0
        self._episode_action_counts.fill(0)
        self._episode_teleported = 0

        for agent in self.agents:
            self._current_phase[agent] = 0
            self._elapsed_phase_time[agent] = 0
            self.conn.trafficlight.setPhase(self._agent_to_tls[agent], 0)

        observations = {a: self._compute_obs(a) for a in self.agents}
        infos = {a: {"phase": 0} for a in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]):
        # 정책 출력 분포 추적 — min_green 미충족으로 실제 적용 안 된 action도 카운트
        # (실제 적용된 phase는 _episode_phase_switches로 별도 측정)
        for agent in self.agents:
            self._episode_action_counts[int(actions[agent])] += 1

        # min_green 조건 만족 + phase 변경 요청인 에이전트 식별
        switching = {
            a: (
                int(actions[a]) != self._current_phase[a]
                and self._elapsed_phase_time[a] >= self.min_green
            )
            for a in self.agents
        }
        self._episode_phase_switches += sum(switching.values())

        done = False
        throughput_before = self._throughput  # 스텝 시작 시점 처리량 스냅샷

        # 전환 에이전트의 TLS에 yellow 적용
        for agent in self.agents:
            if switching[agent]:
                self.conn.trafficlight.setPhase(
                    self._agent_to_tls[agent], self._yellow_phase_index[agent]
                )

        # yellow 유지 시간 진행
        if any(switching.values()):
            done, _ = self._simulate_seconds(self.yellow_time)
            if not done:
                for agent in self.agents:
                    if switching[agent]:
                        new_phase = int(actions[agent])
                        self._current_phase[agent] = new_phase
                        self._elapsed_phase_time[agent] = 0
                        self.conn.trafficlight.setPhase(
                            self._agent_to_tls[agent], new_phase
                        )

        # delta_time 진행
        if not done:
            done, progressed = self._simulate_seconds(self.delta_time)
            for agent in self.agents:
                self._elapsed_phase_time[agent] += progressed

        # 에이전트별 결과 계산
        avg_wait = (
            float(np.mean(self._completed_waiting)) if self._completed_waiting else 0.0
        )
        std_wait = (
            float(np.std(self._completed_waiting)) if self._completed_waiting else 0.0
        )
        avg_travel = (
            float(np.mean(self._completed_travel)) if self._completed_travel else 0.0
        )

        # 이번 스텝에서 완료된 차량 수 (delta): _simulate_seconds 내부에서 누적됨
        step_throughput = self._throughput - throughput_before

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for agent in self.agents:
            obs = self._compute_obs(agent)
            observations[agent] = obs

            # ── 보상 함수 (reward_mode 별 분기) ──────────────────────────────
            if self.reward_mode == "diff-waiting-time":
                # 누적 대기시간 변화량: 감소 시 양의 보상 (SUMO-RL 검증 방식)
                lanes = self._per_agent_lanes.get(agent, self._lane_ids)
                current_wait = sum(
                    self.conn.lane.getWaitingTime(ln) for ln in lanes
                ) / 100.0
                reward = self._last_wait_measure.get(agent, current_wait) - current_wait
                self._last_wait_measure[agent] = current_wait

            elif self.reward_mode == "pressure":
                # 처리량 압력: 출력 lane 차량 수 - 입력 lane 차량 수
                in_lanes  = self._per_agent_lanes.get(agent, self._lane_ids)
                out_lanes = self._per_agent_out_lanes.get(agent, [])
                reward = float(
                    sum(self.conn.lane.getLastStepVehicleNumber(ln) for ln in out_lanes)
                    - sum(self.conn.lane.getLastStepVehicleNumber(ln) for ln in in_lanes)
                )

            else:  # "queue" (default)
                # mean+max 조합: 기아 방향에 추가 패널티 + 처리량 보너스
                queue_arr = obs[:4]
                queue_penalty = (float(np.mean(queue_arr)) + float(np.max(queue_arr))) / 10.0
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
                # 진단 지표 (mode collapse / grid lock 감지)
                "phase_switches": int(self._episode_phase_switches),
                "max_queue": float(self._episode_max_queue),
                "teleported": int(self._episode_teleported),
                "action_counts": self._episode_action_counts.tolist(),
            }

        if done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self) -> np.ndarray:
        """
        matplotlib + sumolib 기반 네트워크 시각화.
        sumo_renderer.SumoRenderer에 현재 신호 상태를 전달해 RGB 배열 반환.
        net.xml 로드에 실패한 경우 단일교차로 numpy fallback 사용.
        """
        return self._renderer.render(
            agent_to_tls=self._agent_to_tls,
            current_phase=self._current_phase,
            last_obs=self._last_obs,
            sim_step=self.sim_step,
            max_steps=self.max_steps,
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
        print("agents:", env.agents)
        print("obs shapes:", {a: o.shape for a, o in obs.items()})

        while env.agents:
            actions = {a: env.action_space(a).sample() for a in env.agents}
            obs, rew, term, trunc, info = env.step(actions)
            step += 1
    finally:
        env.close()
    print(f"done after {step} steps | info: {info}")
