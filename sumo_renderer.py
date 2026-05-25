"""
SUMO 네트워크 시각화 렌더러 (v2)

각 TLS 교차로에 N/S/E/W 방향별 신호등 + 큐 막대 시각화.
matplotlib + sumolib 기반 headless 렌더링 (macOS 포함).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _queue_color(q: float) -> str:
    """큐 길이 → 색상 (초록 / 주황 / 빨강)"""
    if q <= 3:
        return "#2ecc71"
    elif q <= 7:
        return "#f39c12"
    return "#e74c3c"


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
        BLEN = IO * 0.90    # 큐 막대 최대 길이 (인디케이터 바깥으로 뻗음)
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

    # ── 메인 렌더 ──────────────────────────────────────────────────────────────

    def render(
        self,
        agent_to_tls: Dict[str, str],
        current_phase: Dict[str, int],
        last_obs: Dict[str, np.ndarray],
        sim_step: int,
        max_steps: int,
    ) -> np.ndarray:
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
            phase = current_phase.get(agent, 0) % 4
            obs   = last_obs.get(agent, np.zeros(10, dtype=np.float32))

            # 중심 배경 원 + 에이전트 레이블
            ax.add_patch(mp.Circle((cx, cy), RA * 1.3, color=self.CTR_BG_CLR, zorder=2))
            ax.text(cx, cy, agent, color="white", fontsize=9,
                    ha="center", va="center", fontweight="bold", zorder=10)

            # ── ① 신호등 인디케이터 (N/S/E/W 고정 위치, phase action 기준) ─────
            for i, (dname, (dx, dy), arrow) in enumerate(
                zip(self.DIRS, self.DIR_VEC, self.DIR_ARR)
            ):
                tx = cx + dx * IO
                ty = cy + dy * IO
                is_active = (i == phase)
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

            # ── ② 큐 막대 (sumolib geometry로 계산한 실제 접근 방향에 배치) ──
            # 신호등 인디케이터와 분리: 큐 위치 = 실제 차량이 대기하는 도로 방향
            # 대기 차량은 교차로 바깥쪽에 쌓이므로 막대가 외부로 뻗어야 올바름
            lane_dirs  = approach_dirs.get(agent, [])
            _DVEC = {"N": (0,1), "S": (0,-1), "E": (1,0), "W": (-1,0)}

            # 같은 접근 방향 lane이 여러 개일 경우 큐 합산
            dir_queue: dict = {}
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
                bar_len = min(queue / 15.0, 1.0) * BLEN
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
                is_active_approach = (adir in self.DIRS and
                                      self.DIRS.index(adir) == phase)
                q_text_clr = (self.ACTIVE_CLR if is_active_approach
                              else _queue_color(queue) if queue > 3
                              else "#888888")
                ax.text(lx, ly, f"{adir}: {int(queue)}",
                        color=q_text_clr,
                        fontsize=10 if is_active_approach else 8,
                        ha=ha, va=va,
                        fontweight="bold" if is_active_approach else "normal",
                        zorder=9)

        # ── 타이틀 ─────────────────────────────────────────────────────────────
        green_dirs = "   ".join(
            f"{ag}: {self.DIRS[current_phase.get(ag, 0) % 4]}=green"
            for ag in sorted(agent_to_tls.keys())
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
        cp  = {"tl_0": 0, "tl_1": 3, "tl_2": 2, "tl_3": 1}
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
