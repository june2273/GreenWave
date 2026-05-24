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


class SumoParallelEnv(ParallelEnv):
    """
    PettingZoo Parallel API 기반 SUMO 교차로 신호제어 환경

    단일 교차로: tls_ids=["C"]  → agents=["tl_0"]
    다중 교차로: tls_ids=["C","D",...] → agents=["tl_0","tl_1",...]

    행동(Discrete 4): 0=North / 1=South / 2=East / 3=West 단독 green
    관측(shape 10): [queue×4, speed×4, phase_norm, elapsed_norm]
    보상: -(queue_length / 10) + (step_arrivals × 0.5)  — 대기열 패널티 + 처리량 보너스
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 5,
        "name": "sumo_intersection_v0",
    }

    # 단일 교차로 기준 진입 edge (다중 교차로 확장 시 tls별 per-agent 설정으로 교체)
    _DEFAULT_INCOMING_EDGES: List[str] = ["n2c", "s2c", "e2c", "w2c"]

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

        # 단일 교차로 기준 lane ids (확장 시 per-agent로 분리)
        self._lane_ids: List[str] = [
            f"{e}_0" for e in self._DEFAULT_INCOMING_EDGES
        ]

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

            self._queue_cumsum += sum(
                float(self.conn.lane.getLastStepHaltingNumber(ln))
                for ln in self._lane_ids
            )

            if self.sim_step >= self.max_steps:
                return True, progressed

        return False, num_seconds

    def _compute_obs(self, agent: str) -> np.ndarray:
        queues = [
            float(self.conn.lane.getLastStepHaltingNumber(ln))
            for ln in self._lane_ids
        ]
        speeds = [
            max(0.0, float(self.conn.lane.getLastStepMeanSpeed(ln)))
            for ln in self._lane_ids
        ]
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

        for agent in self.agents:
            self._current_phase[agent] = 0
            self._elapsed_phase_time[agent] = 0
            self.conn.trafficlight.setPhase(self._agent_to_tls[agent], 0)

        observations = {a: self._compute_obs(a) for a in self.agents}
        infos = {a: {"phase": 0} for a in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]):
        # min_green 조건 만족 + phase 변경 요청인 에이전트 식별
        switching = {
            a: (
                int(actions[a]) != self._current_phase[a]
                and self._elapsed_phase_time[a] >= self.min_green
            )
            for a in self.agents
        }

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
        avg_travel = (
            float(np.mean(self._completed_travel)) if self._completed_travel else 0.0
        )

        # 이번 스텝에서 완료된 차량 수 (delta): _simulate_seconds 내부에서 누적됨
        step_throughput = self._throughput - throughput_before

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for agent in self.agents:
            obs = self._compute_obs(agent)
            observations[agent] = obs
            queue_penalty = float(np.sum(obs[:4])) / 10.0
            throughput_bonus = step_throughput * 0.5
            rewards[agent] = -queue_penalty + throughput_bonus
            terminations[agent] = False
            truncations[agent] = done
            infos[agent] = {
                "phase": self._current_phase[agent],
                "phase_changed": switching[agent],
                "avg_waiting_time": avg_wait,
                "avg_travel_time": avg_travel,
                "total_queue_length": float(self._queue_cumsum),
                "throughput": self._throughput,
            }

        if done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """첫 번째 에이전트(tl_0) 기준 간단 2D 시각화"""
        agent = self.possible_agents[0]
        obs = self._last_obs.get(agent, np.zeros(10, dtype=np.float32))
        phase = self._current_phase.get(agent, 0)

        h, w = 480, 640
        frame = np.full((h, w, 3), 245, dtype=np.uint8)
        frame[180:300, :] = 210
        frame[:, 260:380] = 210
        frame[190:290, 270:370] = 175

        active = np.array([50, 180, 75], dtype=np.uint8)
        inactive = np.array([200, 80, 70], dtype=np.uint8)
        qn = min(int(obs[0] * 6), 150)
        qs = min(int(obs[1] * 6), 150)
        qe = min(int(obs[2] * 6), 150)
        qw = min(int(obs[3] * 6), 150)

        frame[max(20, 170 - qn):170, 300:340] = active if phase == 0 else inactive
        frame[310:min(460, 310 + qs), 300:340] = active if phase == 1 else inactive
        frame[220:260, 390:min(620, 390 + qe)] = active if phase == 2 else inactive
        frame[220:260, max(20, 250 - qw):250] = active if phase == 3 else inactive

        return frame

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
