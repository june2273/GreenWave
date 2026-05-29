"""
3-way 비교 영상 합성기 — FixedTime / MAPPO / CTDE 를 가로로 나란히 + 실시간 지표 오버레이.

설계
----
1. record_video_*.py 가 --dump-metrics 로 저장한 프레임별 JSON (co2_kg / avg_wait /
   throughput) 을 읽어 프레임과 1:1 정렬.
2. matplotlib(Agg) 로 매 프레임 1920x1080 다크 캔버스를 그린다:
     - 상단 배너: GreenWave 브랜드(좌) + 타이틀(중앙) + ITERATION(우)
     - 3개 패널(각 560x560)을 가로로 배치 (입력 영상 프레임 그대로 imshow)
     - 패널별 메트릭 카드(AVG WAIT / CO2 TOTAL / throughput) — 실시간 갱신
   생성한 프레임을 imageio(libx264) 로 인코딩.
3. 디자인: 영상의 다크 네이비(#0d1116) + 형광 그린(#2ecc71) 톤에 맞춘 모던 카드 UI.
   정책별 액센트: FixedTime=amber / MAPPO=blue / CTDE=green.

왜 ffmpeg 가 아니라 Python 인가
--------------------------------
로컬 Homebrew ffmpeg 8.1.1 bottle 은 libass/libfreetype/fontconfig 없이 빌드되어
`subtitles`/`drawtext` 필터가 존재하지 않는다 → ffmpeg 단독 텍스트 burn-in 불가.
matplotlib 으로 직접 합성하면 시스템 의존성 없이 동일 결과 + 완전한 디자인 제어.

사용 예
-------
python build_3way_compare.py \
  --base-dir samples/videos/3-way \
  --iter 100 \
  --out samples/videos/3-way/compare_3way_1080p.mp4

(개별 경로 직접 지정도 가능 — --videos / --metrics / --labels)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import imageio.v2 as imageio


# ── 색상 (matplotlib hex) ────────────────────────────────────────────
C_WHITE = "#f0f6fc"
C_GRAY  = "#8b949e"
C_GREEN = "#2ecc71"   # CTDE / 브랜드 액센트
C_BLUE  = "#58a6ff"   # MAPPO
C_AMBER = "#e8a33d"   # FixedTime

BG_CANVAS = "#0d1116"
BG_BANNER = "#11161d"
BG_CARD   = "#161b22"

ACCENT = {
    "FixedTime": C_AMBER,
    "MAPPO":     C_BLUE,
    "CTDE":      C_GREEN,
}

# ── 레이아웃 (1920x1080, top-left origin 픽셀) ──────────────────────
CANVAS_W, CANVAS_H = 1920, 1080
BANNER_H = 96
PANEL_W = 560
PANEL_Y = 150          # 패널 상단 y
GAP = 40
MARGIN_X = 80
PANEL_X = [MARGIN_X + i * (PANEL_W + GAP) for i in range(3)]   # [80, 680, 1280]
PANEL_CX = [x + PANEL_W // 2 for x in PANEL_X]                  # [360, 960, 1560]
PANEL_BOTTOM = PANEL_Y + PANEL_W                               # 710
CARD_Y = 730
CARD_H = 300

# 폰트 크기 (pt)
FS_BRAND = 30
FS_TITLE = 19
FS_ITER  = 17
FS_POLICY = 24
FS_CARDLBL = 14
FS_VALUE = 36
FS_THRU  = 13


def _fmt_wait(sample):
    if sample.get("throughput", 0) < 20:
        return "--"
    return f"{sample.get('avg_wait', 0):.1f} s"


def _fmt_co2(sample):
    kg = sample.get("co2_kg", 0.0)
    return f"{kg:,.0f} kg" if kg >= 10 else f"{kg:.1f} kg"


def _add_panel_axes(fig, i):
    """패널 i 의 figure-normalized 좌표 axes 생성 (top-left 픽셀 → 좌하단 원점)."""
    left = PANEL_X[i] / CANVAS_W
    width = PANEL_W / CANVAS_W
    bottom = (CANVAS_H - PANEL_BOTTOM) / CANVAS_H
    height = PANEL_W / CANVAS_H
    ax = fig.add_axes([left, bottom, width, height], zorder=2)
    ax.axis("off")
    return ax


def build_figure(labels, iter_num, title, first_frames):
    """정적 요소를 모두 그린 figure + (panel_im_artists, dyn_text_artists) 반환."""
    fig = plt.figure(figsize=(CANVAS_W / 100, CANVAS_H / 100), dpi=100)
    fig.patch.set_facecolor(BG_CANVAS)

    # 배경 오버레이 axes (전체, y축 반전 → top-left 픽셀 좌표 사용)
    bg = fig.add_axes([0, 0, 1, 1], zorder=1)
    bg.set_xlim(0, CANVAS_W)
    bg.set_ylim(CANVAS_H, 0)
    bg.axis("off")

    # 배너 + 그린 라인
    bg.add_patch(Rectangle((0, 0), CANVAS_W, BANNER_H, facecolor=BG_BANNER, edgecolor="none"))
    bg.add_patch(Rectangle((0, BANNER_H), CANVAS_W, 3, facecolor=C_GREEN, edgecolor="none"))

    # 브랜드 / 타이틀 / iter
    bg.text(48, BANNER_H / 2, "GreenWave", color=C_GREEN, fontsize=FS_BRAND,
            fontweight="bold", ha="left", va="center")
    bg.text(CANVAS_W / 2, BANNER_H / 2, title, color=C_WHITE, fontsize=FS_TITLE,
            ha="center", va="center")
    bg.text(CANVAS_W - 48, BANNER_H / 2, f"ITERATION  {iter_num}", color=C_GRAY,
            fontsize=FS_ITER, fontweight="bold", ha="right", va="center")

    panel_axes = []
    dyn = []  # (wait_artist, co2_artist, thru_artist) per panel
    for i, label in enumerate(labels):
        cx = PANEL_CX[i]
        acc = ACCENT.get(label, C_WHITE)

        # 정책 라벨 (패널 위)
        suffix = "  (ours)" if label == "CTDE" else (
            "  (current)" if label == "FixedTime" else "")
        bg.text(cx, 120, label.upper() + suffix, color=acc, fontsize=FS_POLICY,
                fontweight="bold", ha="center", va="center")

        # 카드 배경 + 상단 액센트 바
        bg.add_patch(Rectangle((PANEL_X[i], CARD_Y), PANEL_W, CARD_H,
                               facecolor=BG_CARD, edgecolor="none"))
        bg.add_patch(Rectangle((PANEL_X[i], CARD_Y), PANEL_W, 4,
                               facecolor=acc, edgecolor="none"))

        # 정적 라벨
        bg.text(cx, CARD_Y + 40, "AVG WAIT", color=C_GRAY, fontsize=FS_CARDLBL,
                fontweight="bold", ha="center", va="center")
        bg.text(cx, CARD_Y + 173, "CO2 TOTAL", color=C_GRAY, fontsize=FS_CARDLBL,
                fontweight="bold", ha="center", va="center")

        # 동적 값 (초기 텍스트는 비워둠 → 루프에서 set_text)
        a_wait = bg.text(cx, CARD_Y + 90, "", color=C_WHITE, fontsize=FS_VALUE,
                         fontweight="bold", ha="center", va="center")
        a_co2 = bg.text(cx, CARD_Y + 223, "", color=acc, fontsize=FS_VALUE,
                        fontweight="bold", ha="center", va="center")
        a_thru = bg.text(cx, CARD_Y + 280, "", color=C_GRAY, fontsize=FS_THRU,
                         ha="center", va="center")
        dyn.append((a_wait, a_co2, a_thru))

        # 패널 영상 axes
        ax = _add_panel_axes(fig, i)
        im = ax.imshow(first_frames[i], aspect="auto")
        panel_axes.append(im)

    return fig, panel_axes, dyn


def main():
    args = parse_args()
    base = Path(args.base_dir)
    videos = args.videos or [str(base / f"cmp_{n}.mp4") for n in ("fixed", "mappo", "ctde")]
    metrics = args.metrics or [str(base / f"cmp_{n}.json") for n in ("fixed", "mappo", "ctde")]
    out_path = args.out or str(base / f"compare_3way_{args.res}p.mp4")

    for v in videos + metrics:
        if not Path(v).exists():
            raise FileNotFoundError(
                f"입력 없음: {v}\n먼저 record_video_*.py --dump-metrics 로 생성하세요.")

    # 지표 로드
    data = [json.loads(Path(m).read_text()) for m in metrics]
    metrics_list = [d["samples"] for d in data]
    fps = data[0].get("fps", args.fps)

    # 영상 리더
    readers = [imageio.get_reader(v, "ffmpeg") for v in videos]
    iters = [r.iter_data() for r in readers]

    # 첫 프레임으로 figure 초기화
    first_frames = [next(it) for it in iters]
    fig, panel_ims, dyn = build_figure(args.labels, args.iter, args.title, first_frames)

    if args.print_only:
        print(f"[plan] out={out_path} res={args.res} fps={fps} "
              f"frames(min)={min(len(s) for s in metrics_list)}")
        plt.close(fig)
        return

    # 출력 writer
    ff_params = ["-crf", "18", "-movflags", "+faststart"]
    if args.res == "720":
        ff_params += ["-vf", "scale=1280:720"]
    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264", format="ffmpeg",
        pixelformat="yuv420p", macro_block_size=None, ffmpeg_params=ff_params,
    )

    total = min(len(s) for s in metrics_list)

    def _render_current(frame_idx):
        for p in range(3):
            s = metrics_list[p][min(frame_idx, len(metrics_list[p]) - 1)]
            w, c, t = dyn[p]
            w.set_text(_fmt_wait(s))
            c.set_text(_fmt_co2(s))
            t.set_text(f"throughput  {s.get('throughput', 0):,}")
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    try:
        # frame 0 (이미 first_frames 로딩됨)
        writer.append_data(_render_current(0))

        idx = 1
        while idx < total:
            try:
                frames_now = [next(it) for it in iters]
            except StopIteration:
                break
            for p in range(3):
                panel_ims[p].set_data(frames_now[p])
            writer.append_data(_render_current(idx))
            if idx % 200 == 0:
                print(f"  ... {idx}/{total} frames")
            idx += 1
    finally:
        writer.close()
        for r in readers:
            try:
                r.close()
            except Exception:
                pass
        plt.close(fig)

    dur = idx / max(1, fps)
    print(f"\n[완료] {out_path}  ({idx} frames @ {fps}fps, ~{dur:.1f}s, {args.res}p)")


def parse_args():
    p = argparse.ArgumentParser(description="Build 3-way comparison video with live metrics")
    p.add_argument("--base-dir", type=str, default="samples/videos/3-way",
                   help="cmp_fixed/cmp_mappo/cmp_ctde 의 .mp4 + .json 이 있는 폴더")
    p.add_argument("--videos", nargs=3, default=None,
                   help="패널 영상 3개 (Fixed MAPPO CTDE 순). 미지정 시 base-dir/cmp_*.mp4")
    p.add_argument("--metrics", nargs=3, default=None,
                   help="지표 JSON 3개 (순서 동일). 미지정 시 base-dir/cmp_*.json")
    p.add_argument("--labels", nargs=3, default=["FixedTime", "MAPPO", "CTDE"])
    p.add_argument("--iter", type=int, default=100)
    p.add_argument("--title", type=str,
                   default="2x2 BRT Corridor - Adaptive Signal Control")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--step", type=int, default=1,
                   help="(현재 미사용 — 매 프레임 갱신)")
    p.add_argument("--res", type=str, default="1080", choices=["1080", "720"])
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--print-only", action="store_true",
                   help="합성 없이 계획만 출력")
    return p.parse_args()


if __name__ == "__main__":
    main()
