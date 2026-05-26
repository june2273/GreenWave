"""
SUMO 네트워크 시각화 렌더러 (v2)

각 TLS 교차로에 N/S/E/W 방향별 신호등 + 큐 막대 시각화.
matplotlib + sumolib 기반 headless 렌더링 (macOS 포함).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _queue_color(q: float) -> str:
    """큐 길이 → 색상 (초록 ≤3 / 주황 ≤7 / 빨강 ≥8). 단순 단계별 강조."""
    if q <= 3:
        return "#2ecc71"
    if q <= 7:
        return "#f39c12"
    return "#e74c3c"


# SUMO connection 의 dir 속성 → 사람이 읽기 쉬운 movement 약어
# s = straight, l = left, r = right, t = u-turn(반환)
# 직진(Th) vs 좌회전(Lt) 구분이 사용자 인지에 핵심 → 라벨에 함께 표시
_DIR_TO_MOVEMENT = {"s": "Th", "l": "Lt", "r": "Rt", "t": "Tn"}


class SumoRenderer:
    """
    matplotlib + sumolib 기반 SUMO 네트워크 렌더러

    render() 반환값: np.ndarray shape=(H, W, 3) uint8 RGB (imageio 호환)

    v2 개선:
    - 교차로별 N/S/E/W 방향 신호등 (초록 글로우 / 어두운 빨강)
    - 큐 막대: 도로 위에 채워지는 컬러 막대 (초록/주황/빨강)
    - 축 눈금 제거, 범례 추가, 고해상도 1100×1100
    """

    FIG_SIZE = 10.0
    FIG_DPI  = 110

    BG_COLOR     = "#0d1117"
    ROAD_COLOR   = "#2c3047"
    BRT_LANE_CLR = "#3498db"      # BRT 전용 차선 표시 (파란색)
    NODE_COLOR   = "#4a4a6a"
    ACTIVE_CLR   = "#00e676"      # green light
    YELLOW_CLR   = "#ffca28"      # yellow light (P4 신규)
    INACTIVE_CLR = "#c62828"      # red light
    CTR_BG_CLR   = "#1a1a3a"

    DIRS    = ["N", "S", "E", "W"]
    DIR_VEC = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    DIR_ARR = ["^", "v", ">", "<"]    # 진입 방향 화살표 (차가 그 방향에서 옴)

    # 좌회전 시 출구 방향 (진입 → 좌회전 후 어디로 가는가, 우교차로 기준)
    # N에서 진입 후 좌회전 → W로 감, 등등
    _LEFT_TURN_EXIT = {"N": "W", "S": "E", "E": "N", "W": "S"}
    # 방향 → marker (가려는 방향 화살표)
    _DIR_TO_MARKER  = {"N": "^",  "S": "v",  "E": ">",  "W": "<"}

    def __init__(self, sumo_cfg_path: str | Path) -> None:
        self.sumo_cfg = Path(sumo_cfg_path).resolve()
        self._net = None
        self._load_net()
        # lane → SUMO movement(Th/Lt/Rt/Tn) 매핑 (connection dir 속성 기반)
        # _build_phase_directions() 내에서 채워짐 — phase 구조 분석과 1회 처리
        self._lane_to_movement: Dict[Tuple[str, str], str] = {}
        # (tls_id, phase_idx) → {"dirs": set, "movement": str} 사전 구축
        # movement: "Th"(직진) / "Lt"(좌회전) / "Rt"(우회전) / "Tn"(U-turn)
        #          / "Mix"(한 phase에 여러 movement 혼재) / "Yel"(yellow phase)
        self._phase_info: Dict[str, Dict[int, Dict]] = self._build_phase_directions()
        # lane → 진입 방향 캐시 (큐 막대 렌더링 시 매 frame 재계산 회피)
        self._lane_dir_cache: Dict[Tuple[str, str], str] = {}
        # BRT 전용 lane id set — net.xml 의 `allow="bus"` 속성으로 식별.
        # 도로 색 (파란 보조선) + 큐 카테고리 분류에 사용.
        self._brt_lanes: set = self._extract_brt_lanes()

    # ── net 로드 ───────────────────────────────────────────────────────────────

    def _get_net_file(self) -> Optional[str]:
        try:
            elem = ET.parse(str(self.sumo_cfg)).find(".//net-file")
            if elem is None:
                return None
            p = Path(elem.get("value", ""))
            if not p.is_absolute():
                p = self.sumo_cfg.parent / p
            return str(p.resolve())
        except Exception:
            return None

    def _load_net(self) -> None:
        try:
            import sumolib
            net_file = self._get_net_file()
            if net_file and Path(net_file).exists():
                self._net = sumolib.net.readNet(net_file, withInternal=False)
        except Exception:
            self._net = None

    def _extract_brt_lanes(self) -> set:
        """net.xml 에서 `allow="bus"` 인 lane id 추출 → BRT 차선 set.

        SUMO 의 vClass 시스템에서 BRT 전용 차선은 `allow="bus"` 로 표기.
        BRT 가 없는 map (single, 2x2, 3x2) 에서는 빈 set 반환.
        internal junction lane (id 가 `:` 로 시작) 은 제외.
        """
        if self._net is None:
            return set()
        net_file = self._get_net_file()
        if not net_file or not Path(net_file).exists():
            return set()
        try:
            tree = ET.parse(net_file)
        except Exception:
            return set()
        brt = set()
        for lane in tree.findall(".//lane"):
            if "bus" in (lane.get("allow") or ""):
                lid = lane.get("id", "")
                if lid and not lid.startswith(":"):
                    brt.add(lid)
        return brt

    # ── 스케일 파라미터 ────────────────────────────────────────────────────────

    def _scale(self):
        (xmin, ymin), (xmax, ymax) = self._net.getBBoxXY()
        sz = max(xmax - xmin, ymax - ymin)
        # IO 축소 (0.065 → 0.05): 신호등 + 큐 막대를 junction 가까이 모아
        # 인접 교차로 사이에 시각적 여백 확보.
        IO   = sz * 0.05
        RA   = sz * 0.020   # 활성 신호등 반지름 (약간 축소)
        RI   = sz * 0.014
        # 큐 막대 길이 축소 (0.18 → 0.10): 단일 막대로 복귀했으므로 짧게도
        # 충분히 인지 가능. 인접 교차로 간 시각 분리 강화.
        BLEN = sz * 0.10
        # 단일 막대 → 다소 두껍게 (이전 평행 3개 sz×0.006 → 단일 sz×0.014).
        BW   = sz * 0.014
        return IO, RA, RI, BLEN, BW, (xmin, ymin, xmax, ymax)

    # ── Phase → 활성 방향 매핑 (옵션 B: net.xml 직접 파싱) ─────────────────────

    def _build_phase_directions(self) -> Dict[str, Dict[int, Dict]]:
        """{tls_id: {phase_idx: {"dirs": set("N","S",...), "movement": str}}} 사전 구축.

        net.xml 의 tlLogic + connection 요소를 직접 파싱하여:
          1. 각 TLS의 phase state 문자열 추출
          2. (tls_id, linkIndex) → (from_lane, movement_dir) 매핑
             movement_dir 은 SUMO connection 의 dir 속성 (s/l/r/t)
          3. lane → 진입 방향 (N/S/E/W) 변환
          4. phase state 의 'G'/'g' 위치를 lane → 방향 + movement 로 환산
          5. phase 별 dominant movement 결정 (단일이면 그것, 혼합이면 "Mix")

        부산물로 self._lane_to_movement 캐시도 채움 (큐 라벨 분리 시 사용).
        실패 시 빈 dict 반환 → render() 가 legacy fallback 동작.
        """
        if self._net is None:
            return {}
        net_file = self._get_net_file()
        if not net_file or not Path(net_file).exists():
            return {}

        try:
            tree = ET.parse(net_file)
        except Exception:
            return {}

        # 1. TLS 별 phase state 문자열
        phases_by_tls: Dict[str, list] = {}
        for tlLogic in tree.findall(".//tlLogic"):
            tls_id = tlLogic.get("id")
            if not tls_id:
                continue
            phases_by_tls[tls_id] = [
                ph.get("state", "") for ph in tlLogic.findall("phase")
            ]

        # 2. TLS 별 (link_idx → (lane_id, sumo_dir)) 매핑
        # sumo_dir 은 connection 의 dir 속성: s/l/r/t/? (없으면 "?")
        lanes_by_tls: Dict[str, Dict[int, Tuple[str, str]]] = {}
        for conn in tree.findall(".//connection"):
            tl = conn.get("tl")
            if not tl:
                continue
            try:
                link_idx = int(conn.get("linkIndex", "-1"))
            except ValueError:
                continue
            if link_idx < 0:
                continue
            from_edge = conn.get("from", "")
            from_lane = conn.get("fromLane", "")
            if not from_edge or not from_lane:
                continue
            sumo_dir = conn.get("dir", "?")  # "s"/"l"/"r"/"t"/"?"
            lanes_by_tls.setdefault(tl, {})[link_idx] = (
                f"{from_edge}_{from_lane}", sumo_dir
            )

        # 3. 각 TLS의 phase별 활성 방향 + movement
        result: Dict[str, Dict[int, Dict]] = {}
        for tls_id, phase_states in phases_by_tls.items():
            link_to_lane_mov = lanes_by_tls.get(tls_id, {})
            if not link_to_lane_mov:
                continue

            # link_idx 순서대로 (direction, movement) 사전 계산
            max_idx = max(link_to_lane_mov.keys())
            lane_info: List[Tuple[str, str]] = []
            for i in range(max_idx + 1):
                lane_id, sumo_dir = link_to_lane_mov.get(i, ("", "?"))
                direction = (
                    self._lane_to_direction(lane_id, tls_id) if lane_id else "?"
                )
                movement = _DIR_TO_MOVEMENT.get(sumo_dir, "?")
                lane_info.append((direction, movement))
                # 부산물: lane → movement 캐시 (큐 라벨 분리에 사용)
                if lane_id:
                    self._lane_to_movement[(lane_id, tls_id)] = movement

            phase_dict: Dict[int, Dict] = {}
            for phase_idx, state in enumerate(phase_states):
                green_dirs: set = set()
                yellow_dirs: set = set()
                active_movements: set = set()
                for i, ch in enumerate(state):
                    if i >= len(lane_info):
                        break
                    d, m = lane_info[i]
                    # 'G'/'g' → green (우선권 있음/양보 모두 활성)
                    if ch in ("G", "g"):
                        if d in self.DIRS:
                            green_dirs.add(d)
                        if m != "?":
                            active_movements.add(m)
                    # 'y'/'Y' → yellow (전환 phase 시 active dir 추적)
                    elif ch in ("y", "Y"):
                        if d in self.DIRS:
                            yellow_dirs.add(d)
                # movement 라벨 (P4 시각화에서 좌회전/직진 구분에 사용)
                if not green_dirs and not active_movements:
                    movement_label = "Yel"
                elif active_movements:
                    movement_label = "+".join(sorted(active_movements))
                else:
                    movement_label = "?"
                # 키 명명: 기존 "dirs" → "green_dirs" (yellow_dirs 와 명확 구분)
                # _active_dirs_for 등 호출자가 green_dirs 키 사용하도록 동기화 필요
                phase_dict[phase_idx] = {
                    "green_dirs":  green_dirs,
                    "yellow_dirs": yellow_dirs,
                    "movement":    movement_label,
                }
            result[tls_id] = phase_dict

        return result

    def _lane_to_direction(self, lane_id: str, tls_id: str) -> str:
        """lane_id (= edge_id + '_' + lane_index) → 진입 방향 N/S/E/W 추정.

        edge 의 from_node 위치를 junction 기준으로 비교. _get_approach_dirs 와
        동일한 방식이지만 lane 단위로 호출 가능하도록 분리.
        """
        try:
            edge_id = lane_id.rsplit("_", 1)[0]
            edge = self._net.getEdge(edge_id)
            junction = self._net.getNode(tls_id)
            jx, jy = junction.getCoord()
            fx, fy = edge.getFromNode().getCoord()
            if abs(fy - jy) > abs(fx - jx):
                return "N" if fy > jy else "S"
            return "E" if fx > jx else "W"
        except Exception:
            return "?"

    def _cached_lane_dir(self, lane_id: str, tls_id: str) -> str:
        """lane → 진입 방향 (캐시). 큐 막대 렌더링에서 매 frame 호출되므로 캐시."""
        key = (lane_id, tls_id)
        if key not in self._lane_dir_cache:
            self._lane_dir_cache[key] = self._lane_to_direction(lane_id, tls_id)
        return self._lane_dir_cache[key]

    def _cached_lane_movement(self, lane_id: str, tls_id: str) -> str:
        """lane → movement (Th/Lt/Rt/Tn/?). _build_phase_directions 부산물 활용."""
        return self._lane_to_movement.get((lane_id, tls_id), "?")

    def _active_dirs_for(self, agent: str, agent_to_tls: Dict[str, str],
                         current_phase: Dict[str, int]) -> set:
        """주어진 agent의 현재 phase에서 활성 green 방향 set 반환.

        매핑이 없으면 (legacy fallback) phase index % 4 → DIRS 단일 방향 반환.
        """
        tls_id = agent_to_tls.get(agent, "")
        phase = current_phase.get(agent, 0)
        info = self._phase_info.get(tls_id, {}).get(phase)
        if info is not None:
            return info["green_dirs"]
        # fallback: 단일교차로 호환 (phase 0/1/2/3 → DIRS[0/1/2/3])
        return {self.DIRS[phase % 4]}

    def _yellow_dirs_for(self, agent: str, agent_to_tls: Dict[str, str],
                         current_phase: Dict[str, int]) -> set:
        """현재 phase 의 yellow 진입 방향 set. yellow phase 가 아니면 빈 set."""
        tls_id = agent_to_tls.get(agent, "")
        phase = current_phase.get(agent, 0)
        info = self._phase_info.get(tls_id, {}).get(phase)
        if info is not None:
            return info["yellow_dirs"]
        return set()

    def _phase_movement_for(self, agent: str, agent_to_tls: Dict[str, str],
                            current_phase: Dict[str, int]) -> str:
        """현재 phase의 movement 약어 (Th/Lt/Rt/Tn/Mix/Yel/"")."""
        tls_id = agent_to_tls.get(agent, "")
        phase = current_phase.get(agent, 0)
        info = self._phase_info.get(tls_id, {}).get(phase)
        return info["movement"] if info else ""

    # ── 메인 렌더 ──────────────────────────────────────────────────────────────

    def render(
        self,
        agent_to_tls: Dict[str, str],
        current_phase: Dict[str, int],
        last_obs: Dict[str, np.ndarray],
        sim_step: int,
        max_steps: int,
        queue_per_lane: Optional[Dict[str, List[float]]] = None,
        lane_ids: Optional[Dict[str, List[str]]] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        queue_per_lane : {agent: [halting_count per controlled lane]}
            env가 매 step 캐시하여 전달하는 lane별 raw 큐 수치.
            제공 시 obs[:4] 의존 없이 정확한 큐 막대 시각화.
        lane_ids : {agent: [lane_id, ...]} — queue_per_lane 의 lane id 순서.
            lane_id 가 controlled lane 순서와 1:1 매칭되어야 함.
        """
        if self._net is None:
            return self._render_fallback(
                current_phase, last_obs, list(agent_to_tls.keys())
            )

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D
        import matplotlib.patches as mp

        fig = Figure(figsize=(self.FIG_SIZE, self.FIG_SIZE), dpi=self.FIG_DPI)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.set_facecolor(self.BG_COLOR)
        fig.patch.set_facecolor(self.BG_COLOR)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_visible(False)

        IO, RA, RI, BLEN, BW, (xmin, ymin, xmax, ymax) = self._scale()

        # ── 도로 ──────────────────────────────────────────────────────────────
        # edge 마다 base 도로 선 + (BRT lane 포함 시) 파란 보조선 overlay.
        # BRT lane.getShape() 는 SUMO 의 정확한 lane geometry 라 평행 표시 가능.
        for edge in self._net.getEdges():
            pts = edge.getShape()
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=self.ROAD_COLOR,
                    linewidth=7, zorder=1, solid_capstyle="round")
            # 이 edge 의 BRT lane (있다면) 별도 파란 선으로 강조
            for lane in edge.getLanes():
                if lane.getID() not in self._brt_lanes:
                    continue
                lpts = lane.getShape()
                if len(lpts) < 2:
                    continue
                lxs, lys = zip(*lpts)
                ax.plot(lxs, lys, color=self.BRT_LANE_CLR,
                        linewidth=3.5, alpha=0.85,
                        zorder=1.5, solid_capstyle="round")

        # ── 교차로 ────────────────────────────────────────────────────────────
        tls_ids = set(agent_to_tls.values())

        for node in self._net.getNodes():
            cx, cy = node.getCoord()
            nid = node.getID()

            if nid not in tls_ids:
                ax.plot(cx, cy, "o", color=self.NODE_COLOR, markersize=6, zorder=2)
                continue

            agent = next(a for a, t in agent_to_tls.items() if t == nid)
            # P3/P4: green/yellow/red 3단계 + movement 별 marker 분리
            green_dirs  = self._active_dirs_for(agent, agent_to_tls, current_phase)
            yellow_dirs = self._yellow_dirs_for(agent, agent_to_tls, current_phase)
            movement    = self._phase_movement_for(agent, agent_to_tls, current_phase)
            # 좌회전 허용 phase: "Lt" 포함 시 표시.
            # 단일 교차로 (single.sumocfg) 는 모든 phase 가 "Lt+Th" 라서 이전의
            # `"+" not in movement` 조건이 한 번도 만족 안 됨 → 좌회전 마커 미표시 버그.
            # 2x2grid 처럼 phase 가 분리된 경우 ("Lt", "Rt+Th" 등) 에서는
            # Lt 가 active 한 phase 만 마커가 뜨므로 영향 없음.
            is_left_turn_phase = "Lt" in movement
            obs = last_obs.get(agent, np.zeros(10, dtype=np.float32))

            # 중심 배경 원 + 에이전트 레이블
            ax.add_patch(mp.Circle((cx, cy), RA * 1.3, color=self.CTR_BG_CLR, zorder=2))
            ax.text(cx, cy, agent, color="white", fontsize=9,
                    ha="center", va="center", fontweight="bold", zorder=10)

            # ── ① 신호등 인디케이터 (옵션 B 디자인) ──────────────────────────
            # green + Lt phase → 가려는 방향(좌회전 출구) 화살표
            # green + Th/Rt/Mix → 동그라미만 (직진은 marker 없음)
            # yellow → 노란 동그라미 + 작은 진입 화살표
            # red → 작은 빨간 동그라미만
            for i, (dname, (dx, dy), entry_arrow) in enumerate(
                zip(self.DIRS, self.DIR_VEC, self.DIR_ARR)
            ):
                tx = cx + dx * IO
                ty = cy + dy * IO

                if dname in green_dirs:
                    state = "green"
                    sig_clr = self.ACTIVE_CLR
                    radius  = RA
                elif dname in yellow_dirs:
                    state = "yellow"
                    sig_clr = self.YELLOW_CLR
                    radius  = RA * 0.85   # green 보다 약간 작게 (전환 중 표시)
                else:
                    state = "red"
                    sig_clr = self.INACTIVE_CLR
                    radius  = RI

                # 글로우: green/yellow 만 (활성 표시)
                if state in ("green", "yellow"):
                    for gr, ga in [(3.5, 0.04), (2.5, 0.11), (1.6, 0.24)]:
                        ax.add_patch(mp.Circle(
                            (tx, ty), radius * gr, color=sig_clr, alpha=ga, zorder=3
                        ))

                # 동그라미 본체
                ax.add_patch(mp.Circle(
                    (tx, ty), radius,
                    color=sig_clr, alpha=1.0 if state != "red" else 0.65, zorder=4
                ))

                # 동그라미 안 marker 결정 (옵션 B)
                marker = None
                if state == "green" and is_left_turn_phase:
                    # 좌회전 phase → 가려는 방향(좌회전 출구) 화살표
                    exit_dir = self._LEFT_TURN_EXIT.get(dname)
                    if exit_dir:
                        marker = self._DIR_TO_MARKER[exit_dir]
                elif state == "yellow":
                    # yellow → 진입 방향 작은 화살표
                    marker = entry_arrow
                # red, green 직진/Mix → marker 없음 (동그라미만)

                if marker:
                    ax.plot(tx, ty, marker=marker, color="white",
                            markersize=13 if state == "green" else 9,
                            zorder=5, markeredgewidth=0)

            # ── ② 큐 막대 ────────────────────────────────────────────────────
            # 방향별 총 큐 합산 → 단일 막대 (이전 3-카테고리 평행 분리는 가독성
            # 문제로 원복). 막대 색은 큐 길이 임계값 기반 (초록/주황/빨강).
            _DVEC = {"N": (0,1), "S": (0,-1), "E": (1,0), "W": (-1,0)}
            dir_queue: Dict[str, float] = {}

            if queue_per_lane is not None and agent in queue_per_lane:
                qs  = queue_per_lane[agent]
                lns = lane_ids[agent] if (lane_ids and agent in lane_ids) else []
                for ln, q in zip(lns, qs):
                    adir = self._cached_lane_dir(ln, nid)
                    if adir not in self.DIRS:
                        continue
                    dir_queue[adir] = dir_queue.get(adir, 0.0) + float(q)

            for adir, queue in dir_queue.items():
                adx, ady = _DVEC.get(adir, (0, 0))
                if adx == 0 and ady == 0:
                    continue
                if queue <= 0:
                    continue

                tx_q = cx + adx * IO
                ty_q = cy + ady * IO

                # 정규화 상수 10: 10대 큐면 풀 길이 (실측 max_queue ~ 18 적정)
                bar_len = min(queue / 10.0, 1.0) * BLEN
                qclr    = _queue_color(queue)

                if adx == 0:   # N / S (수직 막대)
                    y_edge = ty_q + ady * RI
                    y0 = y_edge if ady > 0 else y_edge - bar_len
                    ax.add_patch(mp.Rectangle(
                        (tx_q - BW / 2, y0), BW, max(bar_len, 0.3),
                        color=qclr, alpha=0.9, zorder=6
                    ))
                    lx, ly = tx_q + RI + IO * 0.22, ty_q
                    ha, va = "left", "center"
                else:          # E / W (수평 막대)
                    x_edge = tx_q + adx * RI
                    x0 = x_edge if adx > 0 else x_edge - bar_len
                    ax.add_patch(mp.Rectangle(
                        (x0, ty_q - BW / 2), max(bar_len, 0.3), BW,
                        color=qclr, alpha=0.9, zorder=6
                    ))
                    lx, ly = tx_q, ty_q + RI + IO * 0.22
                    ha, va = "center", "bottom"

                is_active_approach = (adir in green_dirs)
                text_clr = (self.ACTIVE_CLR if is_active_approach
                            else qclr if queue > 3 else "#aaaaaa")
                ax.text(lx, ly, f"{adir}: {int(queue)}",
                        color=text_clr,
                        fontsize=10 if is_active_approach else 8,
                        ha=ha, va=va,
                        fontweight="bold" if is_active_approach else "normal",
                        zorder=9)

        # ── 타이틀 ─────────────────────────────────────────────────────────────
        # phase index + 활성 방향 set + movement 약어 함께 표시
        # 예: "tl_0: p0 [N+S Th]" (직진), "tl_1: p2 [N+S Lt]" (좌회전)
        # yellow phase 는 "[-- Yel]" 로 표시
        def _fmt_phase(ag: str) -> str:
            ph = current_phase.get(ag, 0)
            dirs = self._active_dirs_for(ag, agent_to_tls, current_phase)
            mov  = self._phase_movement_for(ag, agent_to_tls, current_phase)
            dirs_str = "+".join(sorted(dirs)) if dirs else "--"
            mov_str  = mov or "?"
            return f"{ag}: p{ph} [{dirs_str} {mov_str}]"

        # agent 가 많으면 (3x2 grid → 6 agent) 한 줄 가로 길이가 figure 폭 초과해
        # 좌우 텍스트가 잘림. 줄당 최대 PER_ROW agent 로 자동 줄바꿈.
        PER_ROW = 4
        agents_sorted = sorted(agent_to_tls.keys())
        title_chunks = [
            "   ".join(_fmt_phase(ag) for ag in agents_sorted[i:i + PER_ROW])
            for i in range(0, len(agents_sorted), PER_ROW)
        ]
        title_body = "\n".join(title_chunks)
        ax.set_title(
            f"Step {sim_step} / {max_steps}\n{title_body}",
            color="white", fontsize=11, fontweight="bold", pad=10,
        )

        # ── 범례 ───────────────────────────────────────────────────────────────
        legend_elems = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=self.ACTIVE_CLR, markersize=12,
                   label="Green (Through=circle / Left=exit arrow)"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=self.YELLOW_CLR, markersize=10,
                   label="Yellow (transition)"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=self.INACTIVE_CLR, markersize=8,
                   label="Red (stop)"),
            mp.Patch(color="#2ecc71", label="Queue ≤ 3 veh"),
            mp.Patch(color="#f39c12", label="Queue ≤ 7 veh"),
            mp.Patch(color="#e74c3c", label="Queue ≥ 8 veh"),
            Line2D([0], [0], color=self.BRT_LANE_CLR, linewidth=3,
                   label="BRT lane"),
        ]
        ax.legend(
            handles=legend_elems,
            loc="lower right",
            facecolor="#1c1c3e", edgecolor="#3a3a6a",
            labelcolor="white", fontsize=9,
        )

        # ── 뷰 범위 ────────────────────────────────────────────────────────────
        margin = max(xmax - xmin, ymax - ymin) * 0.22
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)

        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = canvas.get_width_height()
        return buf.reshape(h, w, 4)[:, :, :3].copy()

    # ── Fallback (net 로드 실패 시 빈 frame + 에러 메시지) ─────────────────────

    def _render_fallback(
        self,
        current_phase: Dict[str, int],
        last_obs: Dict[str, np.ndarray],
        agents: List[str],
    ) -> np.ndarray:
        """net.xml 로드 실패 시 단순 안내 frame 반환 (sumolib 미설치 등 예외 상황)."""
        h, w = 480, 640
        frame = np.full((h, w, 3), 30, dtype=np.uint8)  # 어두운 회색 배경
        # 빨간 X 표시 (대각선)
        for i in range(min(h, w)):
            y, x = i * h // min(h, w), i * w // min(h, w)
            if 0 <= y < h and 0 <= x < w:
                frame[max(0, y-1):y+2, max(0, x-1):x+2] = [200, 50, 50]
                frame[max(0, y-1):y+2, max(0, w-x-1):w-x+2] = [200, 50, 50]
        return frame


# ── 단독 테스트 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else "sumo_data/single.sumocfg"
    renderer = SumoRenderer(cfg)

    if renderer._net is None:
        print(f"[WARN] net 로드 실패 ({cfg})")
    else:
        nodes = list(renderer._net.getNodes())
        edges = list(renderer._net.getEdges())
        print(f"[OK] {len(nodes)} nodes, {len(edges)} edges")

    if "2x2" in cfg:
        a2t = {"tl_0": "1", "tl_1": "2", "tl_2": "5", "tl_3": "6"}
        # 4개 교차로에 4개 green phase 모두 다르게 배치 → 옵션B 라벨링 검증
        # phase 0 = 수평 직진, 2 = 수평 좌회전, 4 = 수직 직진, 6 = 수직 좌회전
        cp  = {"tl_0": 0, "tl_1": 2, "tl_2": 4, "tl_3": 6}
        lo  = {
            "tl_0": np.array([2, 0, 9, 1, 5, 5, 5, 5, 0.0, 0.3], dtype=np.float32),
            "tl_1": np.array([0, 5, 1, 4, 5, 5, 5, 5, 0.75, 0.5], dtype=np.float32),
            "tl_2": np.array([5, 1, 0, 10, 5, 5, 5, 5, 0.5, 0.8], dtype=np.float32),
            "tl_3": np.array([1, 0, 4, 2, 5, 5, 5, 5, 0.25, 0.2], dtype=np.float32),
        }
    else:
        a2t = {"tl_0": "C"}
        cp  = {"tl_0": 2}
        lo  = {"tl_0": np.array([3, 1, 7, 0, 5, 5, 5, 5, 0.5, 0.3], dtype=np.float32)}

    frame = renderer.render(a2t, cp, lo, sim_step=300, max_steps=3600)
    print(f"frame: {frame.shape} {frame.dtype}")

    out = Path("sumo_renderer_test.png")
    try:
        import imageio.v2 as iio
        iio.imwrite(out, frame)
        print(f"저장: {out}")
    except ImportError:
        print("imageio 없음")
