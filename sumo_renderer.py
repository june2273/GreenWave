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
    """큐 길이 → 색상 (초록 / 주황 / 빨강)"""
    if q <= 3:
        return "#2ecc71"
    elif q <= 7:
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
    NODE_COLOR   = "#4a4a6a"
    ACTIVE_CLR   = "#00e676"
    INACTIVE_CLR = "#c62828"
    CTR_BG_CLR   = "#1a1a3a"

    DIRS    = ["N", "S", "E", "W"]
    DIR_VEC = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    DIR_ARR = ["^", "v", ">", "<"]

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

    # ── 스케일 파라미터 ────────────────────────────────────────────────────────

    def _scale(self):
        (xmin, ymin), (xmax, ymax) = self._net.getBBoxXY()
        sz = max(xmax - xmin, ymax - ymin)
        IO   = sz * 0.065   # 중심 → 신호등 거리
        RA   = sz * 0.022   # 활성 신호등 반지름
        RI   = sz * 0.015   # 비활성 신호등 반지름
        # 큐 막대 최대 길이: 교차로 사이 거리의 약 절반까지 뻗도록 sz × 0.30
        # (이전 IO * 0.90 ≈ sz × 0.058 보다 ~5배 길어짐 — 작은 큐도 잘 보임)
        BLEN = sz * 0.30
        BW   = sz * 0.013   # 큐 막대 너비
        return IO, RA, RI, BLEN, BW, (xmin, ymin, xmax, ymax)

    # ── 실제 접근 방향 계산 (sumolib geometry) ─────────────────────────────────

    def _get_approach_dirs(self, agent_to_tls: Dict[str, str]) -> Dict[str, list]:
        """
        각 TLS의 입력 edge 위치에서 접근 방향 추정.
        from_node가 junction 기준 어느 방향에 있는지로 N/S/E/W 판별.

        Returns: {agent: ['N','S','E','W'순서_리스트]} (obs[0..3] 에 대응)
        인식 불가능한 경우 fallback ['?','?','?','?'] 반환.
        """
        result: Dict[str, list] = {}
        if self._net is None:
            return result
        _DIR_VEC = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}

        for agent, tls_id in agent_to_tls.items():
            try:
                node = self._net.getNode(tls_id)
                jx, jy = node.getCoord()
                dirs: list = []
                for edge in node.getIncoming():
                    if edge.getFunction() == "internal":
                        continue
                    fx, fy = edge.getFromNode().getCoord()
                    if abs(fy - jy) > abs(fx - jx):
                        d = "N" if fy > jy else "S"
                    else:
                        d = "E" if fx > jx else "W"
                    for _ in edge.getLanes():   # lane 수만큼 반복
                        dirs.append(d)
                while len(dirs) < 4:
                    dirs.append("?")
                result[agent] = dirs[:4]
            except Exception:
                result[agent] = ["?", "?", "?", "?"]
        return result

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
                active_dirs: set = set()
                active_movements: set = set()
                for i, ch in enumerate(state):
                    if i >= len(lane_info):
                        break
                    # 'G'(우선권 있음) + 'g'(우선권 양보) 모두 활성으로 간주
                    if ch in ("G", "g"):
                        d, m = lane_info[i]
                        if d in self.DIRS:
                            active_dirs.add(d)
                        if m != "?":
                            active_movements.add(m)
                # movement 라벨: 단일이면 그것, 여러개면 "+" join (사용자 인지)
                # 예: phase 0 (직진 s + 우회전 r) → "Th+Rt"  (사용자가 직진 phase로 식별)
                #     phase 2 (좌회전 l only)    → "Lt"
                #     phase 1/3/5/7 (yellow)    → "Yel"
                if not active_dirs and not active_movements:
                    movement_label = "Yel"
                elif active_movements:
                    movement_label = "+".join(sorted(active_movements))
                else:
                    movement_label = "?"
                phase_dict[phase_idx] = {
                    "dirs": active_dirs,
                    "movement": movement_label,
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
            return info["dirs"]
        # fallback: 단일교차로 호환 (phase 0/1/2/3 → DIRS[0/1/2/3])
        return {self.DIRS[phase % 4]}

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

        # obs[i] → 실제 접근 방향 계산 (sumolib geometry)
        approach_dirs = self._get_approach_dirs(agent_to_tls)

        # ── 도로 ──────────────────────────────────────────────────────────────
        for edge in self._net.getEdges():
            pts = edge.getShape()
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=self.ROAD_COLOR,
                    linewidth=4, zorder=1, solid_capstyle="round")

        # ── 교차로 ────────────────────────────────────────────────────────────
        tls_ids = set(agent_to_tls.values())

        for node in self._net.getNodes():
            cx, cy = node.getCoord()
            nid = node.getID()

            if nid not in tls_ids:
                ax.plot(cx, cy, "o", color=self.NODE_COLOR, markersize=6, zorder=2)
                continue

            agent = next(a for a, t in agent_to_tls.items() if t == nid)
            # 실제 활성 방향 set (양방향 동시 green 지원, % 4 매핑 제거)
            active_dirs = self._active_dirs_for(agent, agent_to_tls, current_phase)
            obs = last_obs.get(agent, np.zeros(10, dtype=np.float32))

            # 중심 배경 원 + 에이전트 레이블
            ax.add_patch(mp.Circle((cx, cy), RA * 1.3, color=self.CTR_BG_CLR, zorder=2))
            ax.text(cx, cy, agent, color="white", fontsize=9,
                    ha="center", va="center", fontweight="bold", zorder=10)

            # ── ① 신호등 인디케이터 (N/S/E/W 고정 위치, active_dirs 기준) ─────
            for i, (dname, (dx, dy), arrow) in enumerate(
                zip(self.DIRS, self.DIR_VEC, self.DIR_ARR)
            ):
                tx = cx + dx * IO
                ty = cy + dy * IO
                is_active = (dname in active_dirs)
                radius    = RA if is_active else RI
                sig_clr   = self.ACTIVE_CLR if is_active else self.INACTIVE_CLR

                # 글로우 링 (활성 방향만)
                if is_active:
                    for gr, ga in [(3.5, 0.04), (2.5, 0.11), (1.6, 0.24)]:
                        ax.add_patch(mp.Circle(
                            (tx, ty), radius * gr, color=sig_clr, alpha=ga, zorder=3
                        ))
                # 신호등 원 + 방향 화살표
                ax.add_patch(mp.Circle(
                    (tx, ty), radius,
                    color=sig_clr, alpha=1.0 if is_active else 0.65, zorder=4
                ))
                ax.plot(tx, ty, marker=arrow, color="white",
                        markersize=13 if is_active else 8,
                        zorder=5, markeredgewidth=0)

            # ── ② 큐 막대 ────────────────────────────────────────────────────
            # 우선순위 1: env 가 전달한 lane별 raw 큐 (정확) → controlled lanes
            #             전체 (8 또는 12개) 모두 반영
            # 우선순위 2: legacy fallback (obs[:4]) — 단일교차로 구버전 호환용
            #
            # 같은 방향에 직진/좌회전 lane이 모두 있는 경우 라벨에 분리 표시.
            #   dir_queue       : {dir: total_q}                  (막대 길이)
            #   dir_by_movement : {dir: {movement: q}}            (라벨 분리)
            _DVEC = {"N": (0,1), "S": (0,-1), "E": (1,0), "W": (-1,0)}
            dir_queue: Dict[str, float] = {}
            dir_by_movement: Dict[str, Dict[str, float]] = {}

            if queue_per_lane is not None and agent in queue_per_lane:
                # 정확 경로: lane id 와 큐를 1:1 매칭
                qs    = queue_per_lane[agent]
                lns   = lane_ids[agent] if (lane_ids and agent in lane_ids) else []
                for ln, q in zip(lns, qs):
                    adir = self._cached_lane_dir(ln, nid)
                    if adir not in self.DIRS:
                        continue
                    amov = self._cached_lane_movement(ln, nid)
                    dir_queue[adir] = dir_queue.get(adir, 0.0) + float(q)
                    bucket = dir_by_movement.setdefault(adir, {})
                    bucket[amov] = bucket.get(amov, 0.0) + float(q)
            else:
                # legacy fallback: obs[:4] = 첫 4개 lane의 queue (구버전 obs 구조)
                lane_dirs = approach_dirs.get(agent, [])
                for obs_i, q in enumerate(obs[:4].tolist()):
                    adir = lane_dirs[obs_i] if obs_i < len(lane_dirs) else "?"
                    if adir != "?":
                        dir_queue[adir] = dir_queue.get(adir, 0.0) + float(q)

            for adir, queue in dir_queue.items():
                adx, ady = _DVEC.get(adir, (0, 0))
                if adx == 0 and ady == 0:
                    continue

                tx_q = cx + adx * IO   # 해당 방향 인디케이터 위치
                ty_q = cy + ady * IO

                # 큐 막대: 인디케이터 원 바깥쪽에서 도로 방향으로 뻗음
                # 정규화 상수 15→10: 10대 정도 큐도 풀 길이로 표시
                # (실측상 max_queue ~ 18 정도라 적절. 사용자 요청: 작은 큐도 시각화)
                bar_len = min(queue / 10.0, 1.0) * BLEN
                qclr    = _queue_color(queue)

                if adx == 0:   # N / S (수직 막대)
                    y_edge = ty_q + ady * RI
                    y0 = y_edge if ady > 0 else y_edge - bar_len
                    ax.add_patch(mp.Rectangle(
                        (tx_q - BW / 2, y0), BW, max(bar_len, 0.3),
                        color=qclr, alpha=0.85, zorder=6
                    ))
                    lx, ly = tx_q + RI + IO * 0.22, ty_q
                    ha, va = "left", "center"
                else:          # E / W (수평 막대)
                    x_edge = tx_q + adx * RI
                    x0 = x_edge if adx > 0 else x_edge - bar_len
                    ax.add_patch(mp.Rectangle(
                        (x0, ty_q - BW / 2), max(bar_len, 0.3), BW,
                        color=qclr, alpha=0.85, zorder=6
                    ))
                    lx, ly = tx_q, ty_q + RI + IO * 0.22
                    ha, va = "center", "bottom"

                # 레이블: 이 접근 방향이 현재 green을 받고 있는지 확인
                # (양방향 동시 green 지원: 예 N+S 가 둘 다 active 일 수 있음)
                is_active_approach = (adir in active_dirs)
                q_text_clr = (self.ACTIVE_CLR if is_active_approach
                              else _queue_color(queue) if queue > 3
                              else "#888888")
                # 라벨 형식 — 직진/좌회전 분리:
                #   단일 movement 인 경우 "N Th: 4"
                #   여러 movement 혼합 "N Th:4 Lt:1"
                #   movement 정보 없으면 legacy "N: 4"
                mov_breakdown = dir_by_movement.get(adir, {})
                meaningful = {m: q for m, q in mov_breakdown.items()
                              if q > 0 and m != "?"}
                if not meaningful:
                    label_text = f"{adir}: {int(queue)}"
                elif len(meaningful) == 1:
                    m, q = next(iter(meaningful.items()))
                    label_text = f"{adir} {m}: {int(q)}"
                else:
                    parts = " ".join(f"{m}:{int(q)}"
                                     for m, q in sorted(meaningful.items()))
                    label_text = f"{adir} {parts}"

                ax.text(lx, ly, label_text,
                        color=q_text_clr,
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

        green_dirs = "   ".join(
            _fmt_phase(ag) for ag in sorted(agent_to_tls.keys())
        )
        ax.set_title(
            f"Step {sim_step} / {max_steps}\n{green_dirs}",
            color="white", fontsize=11, fontweight="bold", pad=10,
        )

        # ── 범례 ───────────────────────────────────────────────────────────────
        legend_elems = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=self.ACTIVE_CLR, markersize=12,
                   label="Green light (active)"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=self.INACTIVE_CLR, markersize=8,
                   label="Red light (stop)"),
            mp.Patch(color="#2ecc71", label="Queue ≤ 3 veh"),
            mp.Patch(color="#f39c12", label="Queue ≤ 7 veh"),
            mp.Patch(color="#e74c3c", label="Queue ≥ 8 veh"),
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

    # ── Fallback ───────────────────────────────────────────────────────────────

    def _render_fallback(
        self,
        current_phase: Dict[str, int],
        last_obs: Dict[str, np.ndarray],
        agents: List[str],
    ) -> np.ndarray:
        agent = agents[0] if agents else "tl_0"
        obs   = last_obs.get(agent, np.zeros(10, dtype=np.float32))
        phase = current_phase.get(agent, 0)
        h, w  = 480, 640
        frame = np.full((h, w, 3), 245, dtype=np.uint8)
        frame[180:300, :]        = 210
        frame[:, 260:380]        = 210
        frame[190:290, 270:370]  = 175
        active   = np.array([50, 180, 75],  dtype=np.uint8)
        inactive = np.array([200, 80, 70],  dtype=np.uint8)
        qn = min(int(obs[0] * 6), 150)
        qs = min(int(obs[1] * 6), 150)
        qe = min(int(obs[2] * 6), 150)
        qw = min(int(obs[3] * 6), 150)
        frame[max(20, 170 - qn):170,         300:340] = active if phase == 0 else inactive
        frame[310:min(460, 310 + qs),         300:340] = active if phase == 1 else inactive
        frame[220:260, 390:min(620, 390 + qe)]         = active if phase == 2 else inactive
        frame[220:260, max(20, 250 - qw):250]          = active if phase == 3 else inactive
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
