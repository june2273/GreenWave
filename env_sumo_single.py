import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import traci
    from sumolib import checkBinary
except ImportError as exc:
    raise ImportError(
        "SUMO Python tools(traci, sumolib)가 필요합니다. "
        "SUMO 설치 후 PYTHONPATH에 $SUMO_HOME/tools를 추가하세요."
    ) from exc


@dataclass
class EpisodeMetrics:
    """에피소드 종료 시 보고용 핵심 지표"""
    avg_waiting_time: float
    avg_travel_time: float
    total_queue_length: float
    throughput: int


class SumoSingleIntersectionEnv(gym.Env):
    """
    SUMO 단일 교차로 신호제어용 Gymnasium 환경

    구분: 클래스 docstring
    행동(action): 0~3 (N/S/E/W 단독 green)
    관측(obs): [4개 queue, 4개 평균속도, 현재 phase, 현재 phase 유지시간]
    보상(reward): - total_queue_length
    """

    # render()가 이미지 배열(rgb_array)을 반환하고, 기본 영상 fps는 5라는 표시
    # 학습에는 직접 필요하지 않지만, render/video 저장용 환경 정보
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(
        self,
        sumo_cfg: Optional[str] = None,
        use_gui: bool = False,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 10,
        max_steps: int = 3600,
        reward_mode: str = "queue",
    ):
        super().__init__()
        # 프로젝트 상대 경로 기준 SUMO 파일 위치 고정
        self.base_dir = Path(__file__).resolve().parent
        self.sumo_data_dir = self.base_dir / "sumo_data"
        self.sumo_cfg = Path(sumo_cfg) if sumo_cfg else self.sumo_data_dir / "single.sumocfg"

        # net 파일 부재 시 node/edge/connection/tls 파일 기반 자동 생성
        self._maybe_build_network()

        self.use_gui = use_gui
        self.delta_time = int(delta_time)
        self.yellow_time = int(yellow_time)
        self.min_green = int(min_green)
        self.max_steps = int(max_steps)
        self.reward_mode = reward_mode

        # 4개 방향 중 1개 방향만 green 선택
        self.action_space = spaces.Discrete(4)
        # [queue(4), speed(4), phase(1), elapsed(1)] 총 10차원
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(10,),
            dtype=np.float32,
        )

        # 관측/지표 계산용 진입방향 edge, lane id
        self.incoming_edges = ["n2c", "s2c", "e2c", "w2c"]
        self.lane_ids = [f"{edge}_0" for edge in self.incoming_edges]

        # 신호 제어 상태
        self.current_phase = 0
        self.elapsed_phase_time = 0
        self.sim_step = 0
        self._last_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # TraCI 연결 상태
        self._conn_label: Optional[str] = None
        self.conn = None
        self.tls_id = "C"
        self.yellow_phase_index: Optional[int] = None

        # 에피소드 지표 계산용 누적 버퍼
        self._depart_time: Dict[str, int] = {}
        self._latest_waiting: Dict[str, float] = {}
        self._completed_waiting: List[float] = []
        self._completed_travel: List[float] = []
        self._throughput = 0
        self._queue_cumsum = 0.0

    def _maybe_build_network(self) -> None:
        """
        구분: 새 메서드(Custom)
        SUMO 네트워크 파일(.net.xml) 부재 시 netconvert 생성
        """
        net_file = self.sumo_data_dir / "single_intersection.net.xml"
        if net_file.exists():
            return

        netconvert = shutil.which("netconvert")
        if not netconvert:
            raise FileNotFoundError(
                "single_intersection.net.xml이 없고 netconvert도 찾을 수 없습니다. "
                "SUMO 설치 후 netconvert를 사용할 수 있게 설정하세요."
            )

        cmd = [
            netconvert,
            "--node-files",
            str(self.sumo_data_dir / "nodes.nod.xml"),
            "--edge-files",
            str(self.sumo_data_dir / "edges.edg.xml"),
            "--connection-files",
            str(self.sumo_data_dir / "connections.con.xml"),
            "--tllogic-files",
            str(self.sumo_data_dir / "tls.tll.xml"),
            "-o",
            str(net_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "netconvert로 네트워크 생성에 실패했습니다.\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

    def _sumo_binary(self) -> str:
        """
        구분: 새 메서드(Custom)
        실행 대상 SUMO 바이너리(sumo/sumo-gui) 경로 조회
        """
        preferred = "sumo-gui" if self.use_gui else "sumo"
        try:
            return checkBinary(preferred)
        except Exception:
            found = shutil.which(preferred)
            if found:
                return found
            raise FileNotFoundError(f"SUMO binary를 찾을 수 없습니다: {preferred}")

    def _start_sumo(self, seed: Optional[int] = None) -> None:
        """
        구분: 새 메서드(Custom)
        에피소드 시작용 SUMO 실행 및 TraCI 연결 오픈
        """
        if self.conn is not None:
            self.close()

        binary = self._sumo_binary()
        self._conn_label = f"sumo_env_{uuid.uuid4().hex[:8]}"

        cmd = [
            binary,
            "-c",
            str(self.sumo_cfg),
            "--no-warnings",
            "true",
            "--time-to-teleport",
            "-1",
            "--seed",
            str(0 if seed is None else seed),
        ]

        traci.start(cmd, label=self._conn_label)
        self.conn = traci.getConnection(self._conn_label)

        # 단일 교차로 실습 기준 첫 번째 traffic light id 사용
        tls_ids = self.conn.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError("트래픽 라이트 ID를 찾지 못했습니다.")
        self.tls_id = tls_ids[0]
        self.yellow_phase_index = self._find_yellow_phase_index()

    def _find_yellow_phase_index(self) -> int:
        """
        구분: 새 메서드(Custom)
        현재 신호 프로그램에서 all-yellow 전환 phase 인덱스를 탐색
        """
        logics = self.conn.trafficlight.getAllProgramLogics(self.tls_id)
        for logic in logics:
            phases = getattr(logic, "phases", [])
            for idx, phase in enumerate(phases):
                state = getattr(phase, "state", "")
                if not state:
                    continue
                lowered = state.lower()
                state_set = set(lowered)
                if "y" in state_set and state_set.issubset({"y", "r"}):
                    return idx

        raise RuntimeError(
            "all-yellow phase를 찾지 못했습니다. tls.tll.xml의 phase 구성을 확인하세요."
        )

    def _simulate_seconds(self, num_seconds: int) -> Tuple[bool, int]:
        """
        구분: 새 메서드(Custom)
        num_seconds 동안 SUMO 진행 + 지표 버퍼 업데이트

        반환값
            done: max_steps 도달 여부
            progressed: 실제 진행된 초 수
        """
        progressed = 0
        done = False
        for _ in range(num_seconds):
            self.conn.simulationStep()
            self.sim_step += 1
            progressed += 1

            # 네트워크 내 차량 출발시각/누적대기시간 추적
            veh_ids = self.conn.vehicle.getIDList()
            for veh_id in veh_ids:
                if veh_id not in self._depart_time:
                    self._depart_time[veh_id] = self.sim_step
                self._latest_waiting[veh_id] = self.conn.vehicle.getAccumulatedWaitingTime(veh_id)

            # 도착 차량 기준 travel time, waiting time, throughput 집계
            arrived = self.conn.simulation.getArrivedIDList()
            for veh_id in arrived:
                self._throughput += 1
                if veh_id in self._depart_time:
                    travel_t = self.sim_step - self._depart_time.pop(veh_id)
                    self._completed_travel.append(float(travel_t))
                if veh_id in self._latest_waiting:
                    self._completed_waiting.append(float(self._latest_waiting.pop(veh_id)))

            # 시뮬레이션 매 초 전체 queue 누적
            queue_now = sum(float(self.conn.lane.getLastStepHaltingNumber(lane)) for lane in self.lane_ids)
            self._queue_cumsum += queue_now

            if self.sim_step >= self.max_steps:
                done = True
                break

        return done, progressed

    def _compute_observation(self) -> np.ndarray:
        """
        구분: 새 메서드(Custom)
        SUMO 상태 -> RL 관측 벡터(10차원) 변환
        """
        queues = [float(self.conn.lane.getLastStepHaltingNumber(lane)) for lane in self.lane_ids]
        speeds = [max(0.0, float(self.conn.lane.getLastStepMeanSpeed(lane))) for lane in self.lane_ids]
        phase_norm = float(self.current_phase) / 3.0
        # min_green 대비 현재 phase 유지 비율(1.0 초과 가능)
        elapsed_norm = float(self.elapsed_phase_time) / max(1.0, float(self.min_green))
        obs = np.array(queues + speeds + [phase_norm, elapsed_norm], dtype=np.float32)
        self._last_obs = obs
        return obs

    def _episode_metrics(self) -> EpisodeMetrics:
        """
        구분: 새 메서드(Custom)
        누적 버퍼 기반 에피소드 지표 계산
        """
        avg_wait = float(np.mean(self._completed_waiting)) if self._completed_waiting else 0.0
        avg_travel = float(np.mean(self._completed_travel)) if self._completed_travel else 0.0
        return EpisodeMetrics(
            avg_waiting_time=avg_wait,
            avg_travel_time=avg_travel,
            total_queue_length=float(self._queue_cumsum),
            throughput=int(self._throughput),
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        구분: 오버라이딩(Override, gym.Env.reset)
        Gymnasium reset: SUMO 재시작 + 내부 버퍼 초기화
        """
        super().reset(seed=seed)
        self._start_sumo(seed=seed)

        self.current_phase = 0
        self.elapsed_phase_time = 0
        self.sim_step = 0

        self._depart_time.clear()
        self._latest_waiting.clear()
        self._completed_waiting.clear()
        self._completed_travel.clear()
        self._throughput = 0
        self._queue_cumsum = 0.0

        # 에피소드 시작 phase = North-only 고정
        self.conn.trafficlight.setPhase(self.tls_id, self.current_phase)
        obs = self._compute_observation()
        info = {"phase": self.current_phase}
        return obs, info

    def step(self, action: int):
        """
        구분: 오버라이딩(Override, gym.Env.step)
        Gymnasium step

        1) action이 현재 phase와 다르고 min_green을 만족하면 yellow를 거쳐 전환
        2) delta_time 만큼 시뮬레이션 진행
        3) 관측/보상/지표(info) 반환
        """
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        done = False
        phase_changed = False

        # min_green 조건 만족 시에만 phase 변경 허용
        can_switch = action != self.current_phase and self.elapsed_phase_time >= self.min_green

        if can_switch:
            phase_changed = True
            if self.yellow_phase_index is None:
                raise RuntimeError("yellow phase 인덱스가 초기화되지 않았습니다.")
            self.conn.trafficlight.setPhase(self.tls_id, self.yellow_phase_index)
            done, _ = self._simulate_seconds(self.yellow_time)
            if not done:
                self.current_phase = action
                self.elapsed_phase_time = 0
                self.conn.trafficlight.setPhase(self.tls_id, self.current_phase)

        if not done:
            done, progressed = self._simulate_seconds(self.delta_time)
            self.elapsed_phase_time += progressed

        obs = self._compute_observation()
        # obs 앞 4개 원소 = 각 방향 queue length
        total_queue = float(np.sum(obs[:4]))

        if self.reward_mode == "queue":
            reward = -total_queue
        else:
            # 실습 기본값 queue 보상, 다른 모드 확장 시 분기 지점
            reward = -total_queue

        metrics = self._episode_metrics()
        info = {
            "phase": self.current_phase,
            "phase_changed": phase_changed,
            "avg_waiting_time": metrics.avg_waiting_time,
            "avg_travel_time": metrics.avg_travel_time,
            "total_queue_length": metrics.total_queue_length,
            "throughput": metrics.throughput,
        }

        terminated = False
        truncated = done
        return obs, reward, terminated, truncated, info

    def render(self):
        """
        구분: 오버라이딩(Override, gym.Env.render)
        간단한 2D 시각화 프레임(rgb_array) 생성
        """
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = np.array([245, 245, 245], dtype=np.uint8)

        # roads
        frame[180:300, :] = np.array([210, 210, 210], dtype=np.uint8)
        frame[:, 260:380] = np.array([210, 210, 210], dtype=np.uint8)

        # intersection box
        frame[190:290, 270:370] = np.array([175, 175, 175], dtype=np.uint8)

        queues = self._last_obs[:4]
        bar_color_active = np.array([50, 180, 75], dtype=np.uint8)
        bar_color_inactive = np.array([200, 80, 70], dtype=np.uint8)

        # N, S, E, W bars
        qn = min(int(queues[0] * 6), 150)
        qs = min(int(queues[1] * 6), 150)
        qe = min(int(queues[2] * 6), 150)
        qw = min(int(queues[3] * 6), 150)

        frame[max(20, 170 - qn):170, 300:340] = bar_color_active if self.current_phase == 0 else bar_color_inactive
        frame[310:min(460, 310 + qs), 300:340] = bar_color_active if self.current_phase == 1 else bar_color_inactive
        frame[220:260, 390:min(620, 390 + qe)] = bar_color_active if self.current_phase == 2 else bar_color_inactive
        frame[220:260, max(20, 250 - qw):250] = bar_color_active if self.current_phase == 3 else bar_color_inactive

        return frame

    def close(self):
        """
        구분: 오버라이딩(Override, gym.Env.close)
        TraCI 연결 종료
        """
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None


if __name__ == "__main__":
    env = SumoSingleIntersectionEnv()
    info: dict = {}
    try:
        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            act = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated
    finally:
        env.close()
    print("done", info)
