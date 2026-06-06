"""
stats_eval.py — paired 통계 분석기 (evaluate_mappo.py 산출 CSV 후처리)

evaluate_mappo.py 가 만든 `results/eval_metrics_mappo_N.csv` 를 입력받아
MAPPO vs CTDE vs FixedTime 비교의 **신뢰성(reliability)** 증거를 생성한다.

설계 배경 (왜 이 검정들인가):
  - 평가 루프가 세 알고리즘을 매 episode 동일 seed 로 실행 → 환경 수요 변동이
    상쇄된 **paired/blocked 설계**다. 따라서 쌍체(paired) 검정이 검정력이 높다.
  - 강화학습 평가는 소표본·비정규·이상치가 흔해 t-test/ANOVA 의 정규성 가정이
    약하다 → **비모수 검정 + 부트스트랩 CI** 가 표준.

파이프라인 (지표마다):
  1. Friedman (omnibus 게이트, 알고리즘 ≥3) — "셋 중 어딘가 차이가 있나?"
  2. 쌍별 Wilcoxon signed-rank + Holm 보정 (사후) — "어느 쌍이 다른가?"
  3. Vargha–Delaney A12 효과크기 — "좋은 쪽이 이길 확률" (방향맵 반영)
  4. 부트스트랩 CI — mean 과 IQM(이상치 강건, Agarwal식) 둘 다

산출물 (out-dir):
  stats_tests.csv / stats_ci.csv / stats_summary.txt / figs/{box,forest}_*.png

SUMO/Ray 비의존 → 일반 종료 (os._exit 불필요).

사용:
  .venv/bin/python stats_eval.py --csv results/eval_metrics_mappo_1.csv
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")  # headless 렌더링
import matplotlib.pyplot as plt


# ── 지표 방향 맵 ──────────────────────────────────────────────────────────────
# "클수록 좋음" 인 지표만 명시. 나머지는 전부 "작을수록 좋음"(기본).
# 효과크기 A12 와 "승리" 판정, 헤드라인 %감소 계산이 방향에 따라 뒤집히지 않게 함.
HIGHER_BETTER = {
    "throughput",
    "avg_speed", "avg_speed_brt", "avg_speed_car",
}

# 기본 분석 지표 (CSV 에 존재하는 것만 자동 필터). 헤드라인은 avg_waiting_time.
DEFAULT_METRICS = [
    "avg_waiting_time",        # ← 헤드라인
    "avg_travel_time",
    "total_queue_length",
    "throughput",
    "teleported",
    "avg_stops_per_vehicle",
    "avg_co2_per_vehicle",
    "avg_speed",
]
HEADLINE_METRIC = "avg_waiting_time"

# 알고리즘 표시 순서 선호도 (작을수록 먼저). 베이스라인(FixedTime*)은 항상 뒤로.
def _algo_rank(name: str) -> int:
    if name.startswith("FixedTime"):
        return 9
    return {"MAPPO": 0, "CTDE": 1}.get(name, 5)


# ── 통계 유틸 ─────────────────────────────────────────────────────────────────
def iqm(a) -> float:
    """Interquartile Mean = 양끝 25% 절사 평균 (이상치 강건, Agarwal 2021)."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    if a.size < 4:
        return float(np.mean(a))  # 소표본은 절사 무의미 → 평균
    return float(stats.trim_mean(a, 0.25))


def bootstrap_ci(values, stat_fn, n_boot, alpha, rng):
    """percentile 부트스트랩 신뢰구간. 반환 (point, lo, hi)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    point = stat_fn(values)
    if values.size < 2:
        return point, float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boots = np.array([stat_fn(values[i]) for i in idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def bootstrap_rel_reduction(better, ref, n_boot, alpha, rng):
    """paired 상대 개선율(%) 의 부트스트랩 CI.

    better/ref 는 동일 episode 로 정렬된 쌍체 배열. IQM 기준.
    개선율 = (iqm(ref) - iqm(better)) / iqm(ref) * 100  (값이 작을수록 좋은 지표 기준).
    episode 인덱스를 재표본추출(쌍 유지).
    반환 (point%, lo%, hi%).
    """
    better = np.asarray(better, float)
    ref = np.asarray(ref, float)
    mask = ~(np.isnan(better) | np.isnan(ref))
    better, ref = better[mask], ref[mask]
    n = better.size

    def _rel(b, r):
        ir = iqm(r)
        if ir == 0 or np.isnan(ir):
            return float("nan")
        return (ir - iqm(b)) / ir * 100.0

    point = _rel(better, ref)
    if n < 2:
        return point, float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([_rel(better[i], ref[i]) for i in idx])
    boots = boots[~np.isnan(boots)]
    if boots.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def vargha_delaney_a12(x, y, higher_better: bool) -> float:
    """A12 = '무작위로 뽑은 x 가 y 보다 *더 좋을* 확률' (방향맵 반영).

    원 정의 A12_raw = P(x>y) + 0.5 P(x==y). higher_better 면 그대로,
    lower_better 면 1 - A12_raw 로 뒤집어 항상 ">0.5 = x 가 더 우수" 로 통일.
    0.5=무차이, 0.56/0.64/0.71 = small/medium/large (Vargha & Delaney 2000).
    """
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    gt = sum((xi > y).sum() for xi in x)
    eq = sum((xi == y).sum() for xi in x)
    a12_raw = (gt + 0.5 * eq) / (x.size * y.size)
    return a12_raw if higher_better else 1.0 - a12_raw


def a12_magnitude(a12: float) -> str:
    """A12 효과크기 라벨 (0.5 대칭)."""
    if np.isnan(a12):
        return "n/a"
    d = abs(a12 - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


def holm_correction(pvals):
    """Holm–Bonferroni step-down 보정. nan 은 nan 으로 통과."""
    pvals = np.asarray(pvals, float)
    adj = np.full(pvals.shape, np.nan)
    valid = np.where(~np.isnan(pvals))[0]
    m = valid.size
    if m == 0:
        return adj
    order = valid[np.argsort(pvals[valid])]
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvals[idx], 1.0)
        prev = max(prev, val)
        adj[idx] = prev
    return adj


def paired_wilcoxon(x, y):
    """쌍체 Wilcoxon signed-rank. 반환 (statistic, pvalue, n).

    모든 차이가 0(완전 동일)이면 scipy 가 ValueError → (nan, nan, n) 로 graceful.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = x.size
    if n < 1 or np.allclose(x, y):
        return float("nan"), float("nan"), n
    try:
        res = stats.wilcoxon(x, y)
        return float(res.statistic), float(res.pvalue), n
    except ValueError:
        return float("nan"), float("nan"), n


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_raw(csv_path: str):
    """CSV → raw row DataFrame + 알고리즘 리스트(표시 순서 정렬).

    raw row = algorithm 이 '_mean'/'_std' summary 가 아닌 행.
    """
    df = pd.read_csv(csv_path)
    if "algorithm" not in df.columns:
        raise ValueError(f"'algorithm' 컬럼이 없습니다: {csv_path}")
    raw = df[~df["algorithm"].astype(str).str.contains("_mean|_std", regex=True)].copy()
    raw["episode"] = pd.to_numeric(raw["episode"], errors="coerce")
    algos = sorted(raw["algorithm"].unique().tolist(), key=_algo_rank)
    return raw, algos


def pivot_metric(raw: pd.DataFrame, metric: str, algos):
    """raw → wide 테이블 (index=episode, columns=algorithm). complete block 만 유지."""
    sub = raw[raw["algorithm"].isin(algos)]
    wide = sub.pivot_table(index="episode", columns="algorithm",
                           values=metric, aggfunc="first")
    wide = wide.reindex(columns=algos)
    wide = wide.dropna(axis=0, how="any")  # 한 episode 라도 결측이면 쌍체에서 제외
    return wide


# ── 분석 본체 ─────────────────────────────────────────────────────────────────
def analyze(csv_path, metrics, out_dir, n_boot, alpha, seed):
    raw, algos = load_raw(csv_path)
    metrics = [m for m in metrics if m in raw.columns]
    if not metrics:
        raise ValueError("분석할 지표 컬럼이 CSV 에 하나도 없습니다.")

    baselines = [a for a in algos if a.startswith("FixedTime")]
    baseline = baselines[0] if baselines else None
    rl_algos = [a for a in algos if not a.startswith("FixedTime")]

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    test_rows = []
    ci_rows = []
    summary_lines = []
    headline_lines = []

    summary_lines.append(f"입력 CSV : {csv_path}")
    summary_lines.append(f"알고리즘 : {algos}  (baseline={baseline})")
    summary_lines.append(f"부트스트랩 : n_boot={n_boot}, alpha={alpha}, seed={seed}")
    summary_lines.append("")

    for metric in metrics:
        higher_better = metric in HIGHER_BETTER
        wide = pivot_metric(raw, metric, algos)
        n_blocks = len(wide)
        direction = "↑클수록 우수" if higher_better else "↓작을수록 우수"
        summary_lines.append(f"━━━ {metric}  ({direction}, n={n_blocks} paired blocks) ━━━")

        if n_blocks < 2:
            summary_lines.append("  (paired block 부족 → 검정 생략)\n")
            continue

        # ── 부트스트랩 CI (알고리즘별 mean & IQM) ───────────────────────────
        for a in algos:
            vals = wide[a].to_numpy()
            m_pt, m_lo, m_hi = bootstrap_ci(vals, np.mean, n_boot, alpha, rng)
            i_pt, i_lo, i_hi = bootstrap_ci(vals, iqm, n_boot, alpha, rng)
            ci_rows.append(dict(metric=metric, algorithm=a, n=n_blocks,
                                mean=m_pt, mean_lo=m_lo, mean_hi=m_hi,
                                iqm=i_pt, iqm_lo=i_lo, iqm_hi=i_hi))
            summary_lines.append(
                f"  {a:18s} mean={m_pt:9.3f} [{m_lo:8.3f},{m_hi:8.3f}]  "
                f"IQM={i_pt:9.3f} [{i_lo:8.3f},{i_hi:8.3f}]")

        # ── 상수 지표 가드 (예: teleported=0 전역) ──────────────────────────
        # 모든 값이 동일하면 분산 0 → Friedman/Wilcoxon 정의 불가(=검정 무의미).
        # teleported=0 같은 경우는 그 자체가 "텔레포트 인공물 아님" 검증 결과이므로
        # CI(전부 동일)만 남기고 검정은 생략한다.
        if np.nanstd(wide.to_numpy()) == 0:
            summary_lines.append(
                "  (모든 값 동일 → 검정 생략; teleported=0 이면 텔레포트 인공물 아님을 의미)")
            summary_lines.append("")
            direction_ascii = "higher=better" if higher_better else "lower=better"
            _plot_box(wide, algos, metric, direction_ascii, fig_dir)
            _plot_forest(ci_rows, algos, metric, direction_ascii, fig_dir)
            continue

        # ── Friedman omnibus (알고리즘 ≥3) ──────────────────────────────────
        if len(algos) >= 3:
            try:
                fr = stats.friedmanchisquare(*[wide[a].to_numpy() for a in algos])
                fr_stat, fr_p = float(fr.statistic), float(fr.pvalue)
            except ValueError:
                fr_stat, fr_p = float("nan"), float("nan")
            test_rows.append(dict(metric=metric, test="Friedman", pair="ALL",
                                  n=n_blocks, statistic=fr_stat,
                                  p_raw=fr_p, p_holm=fr_p,
                                  a12="", magnitude="",
                                  significant=(not np.isnan(fr_p) and fr_p < alpha)))
            summary_lines.append(
                f"  Friedman χ²={fr_stat:.3f}  p={fr_p:.4g}  "
                f"{'유의' if (not np.isnan(fr_p) and fr_p < alpha) else 'n.s.'}")
            gate_ok = (not np.isnan(fr_p) and fr_p < alpha)
        else:
            gate_ok = True  # 2-way: Friedman 생략, 바로 Wilcoxon
            summary_lines.append("  (알고리즘 2개 → Friedman 생략, 단일 Wilcoxon)")

        # ── 쌍별 Wilcoxon + Holm ────────────────────────────────────────────
        pairs = list(itertools.combinations(algos, 2))  # algos 는 이미 정렬됨
        raw_p, pair_meta = [], []
        for x_a, y_a in pairs:
            stat_w, p_w, n_w = paired_wilcoxon(wide[x_a].to_numpy(), wide[y_a].to_numpy())
            a12 = vargha_delaney_a12(wide[x_a].to_numpy(), wide[y_a].to_numpy(), higher_better)
            raw_p.append(p_w)
            pair_meta.append((x_a, y_a, stat_w, n_w, a12))
        p_holm = holm_correction(raw_p)

        for (x_a, y_a, stat_w, n_w, a12), p_w, p_h in zip(pair_meta, raw_p, p_holm):
            sig = (not np.isnan(p_h)) and (p_h < alpha)
            test_rows.append(dict(metric=metric, test="Wilcoxon", pair=f"{x_a} vs {y_a}",
                                  n=n_w, statistic=stat_w,
                                  p_raw=p_w, p_holm=p_h,
                                  a12=a12, magnitude=a12_magnitude(a12),
                                  significant=sig))
            summary_lines.append(
                f"  {x_a} vs {y_a:18s} W={stat_w if not np.isnan(stat_w) else float('nan'):>7.1f} "
                f"p_holm={p_h:.4g} A12={a12:.3f}({a12_magnitude(a12)}) "
                f"{'유의' if sig else 'n.s.'}  →  {x_a if a12>0.5 else y_a} 우세")

        # ── 헤드라인 지표: RL vs baseline 문장 ──────────────────────────────
        if metric == HEADLINE_METRIC and baseline is not None:
            for rl in rl_algos:
                if rl not in wide.columns:
                    continue
                b = wide[baseline].to_numpy()
                r = wide[rl].to_numpy()
                # 승리 episode 수 (헤드라인 지표는 lower better)
                if higher_better:
                    wins = int(np.sum(r > b))
                else:
                    wins = int(np.sum(r < b))
                # 해당 쌍의 Holm p, A12 회수
                p_h = next((tr["p_holm"] for tr in test_rows
                            if tr["test"] == "Wilcoxon" and tr["metric"] == metric
                            and tr["pair"] in (f"{rl} vs {baseline}", f"{baseline} vs {rl}")),
                           float("nan"))
                a12 = vargha_delaney_a12(r, b, higher_better)
                pct, plo, phi = bootstrap_rel_reduction(r, b, n_boot, alpha, rng)
                line = (f"{rl} 는 {baseline} 대비 동일 {n_blocks}개 시나리오 중 {wins}개 우월 "
                        f"(Wilcoxon+Holm p={p_h:.3g}, A12={a12:.2f}, "
                        f"평균대기 IQM {pct:.1f}% 개선 [95% CI {plo:.1f}~{phi:.1f}%]).")
                headline_lines.append(line)

        summary_lines.append("")

        # ── 그림: 박스플롯 + forest(IQM±CI) ────────────────────────────────
        # 그림 텍스트는 ASCII (matplotlib 기본 폰트에 한글 글리프 없음)
        direction_ascii = "higher=better" if higher_better else "lower=better"
        _plot_box(wide, algos, metric, direction_ascii, fig_dir)
        _plot_forest(ci_rows, algos, metric, direction_ascii, fig_dir)

    # ── 파일 저장 ──────────────────────────────────────────────────────────
    tests_df = pd.DataFrame(test_rows)
    ci_df = pd.DataFrame(ci_rows)
    tests_df.to_csv(out_dir / "stats_tests.csv", index=False)
    ci_df.to_csv(out_dir / "stats_ci.csv", index=False)

    if headline_lines:
        summary_lines.insert(0, "")
        for hl in reversed(headline_lines):
            summary_lines.insert(0, "  ★ " + hl)
        summary_lines.insert(0, "=== 발표용 헤드라인 ===")
    summary_text = "\n".join(summary_lines)
    (out_dir / "stats_summary.txt").write_text(summary_text + "\n", encoding="utf-8")

    return out_dir, summary_text, tests_df, ci_df


# ── 플롯 ──────────────────────────────────────────────────────────────────────
def _plot_box(wide, algos, metric, direction, fig_dir):
    fig, ax = plt.subplots(figsize=(1.6 * len(algos) + 2, 4.2))
    data = [wide[a].dropna().to_numpy() for a in algos]
    ax.boxplot(data, tick_labels=algos, showmeans=True)
    # 개별 점 오버레이 (jitter)
    for i, d in enumerate(data, start=1):
        jit = np.random.default_rng(0).normal(0, 0.04, size=len(d))
        ax.scatter(np.full(len(d), i) + jit, d, alpha=0.45, s=18, color="tab:blue")
    ax.set_title(f"{metric}  ({direction})")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"box_{metric}.png", dpi=130)
    plt.close(fig)


def _plot_forest(ci_rows, algos, metric, direction, fig_dir):
    sub = [r for r in ci_rows if r["metric"] == metric]
    if not sub:
        return
    order = {a: i for i, a in enumerate(algos)}
    sub = sorted(sub, key=lambda r: order.get(r["algorithm"], 99))
    ys = np.arange(len(sub))
    pts = [r["iqm"] for r in sub]
    los = [r["iqm"] - r["iqm_lo"] for r in sub]
    his = [r["iqm_hi"] - r["iqm"] for r in sub]
    labels = [r["algorithm"] for r in sub]
    fig, ax = plt.subplots(figsize=(6.2, 1.0 * len(sub) + 1.6))
    ax.errorbar(pts, ys, xerr=[los, his], fmt="o", color="tab:red",
                capsize=5, markersize=7, lw=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric}  (IQM ± 95% bootstrap CI)")
    ax.set_title(f"{metric}  ({direction})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"forest_{metric}.png", dpi=130)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="evaluate_mappo.py CSV → paired 통계(Friedman/Wilcoxon+Holm/A12/부트스트랩 CI)")
    p.add_argument("--csv", required=True, help="eval_metrics_mappo_N.csv 경로")
    p.add_argument("--metrics", nargs="*", default=None,
                   help=f"분석 지표 (기본: {DEFAULT_METRICS}). CSV 에 있는 것만 사용.")
    p.add_argument("--out-dir", default=None,
                   help="출력 폴더 (기본: results/stats_<csv파일명>/)")
    p.add_argument("--n-boot", type=int, default=10000, help="부트스트랩 재표본 수")
    p.add_argument("--alpha", type=float, default=0.05, help="유의수준 (CI = 1-alpha)")
    p.add_argument("--seed", type=int, default=0, help="부트스트랩 재현성 시드")
    return p.parse_args()


def main():
    args = parse_args()
    metrics = args.metrics if args.metrics else DEFAULT_METRICS
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = f"results/stats_{Path(args.csv).stem}"

    out_path, summary_text, tests_df, ci_df = analyze(
        csv_path=args.csv, metrics=metrics, out_dir=out_dir,
        n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)

    print(summary_text)
    print("\n" + "=" * 60)
    print(f"Saved → {out_path}/")
    print(f"  stats_tests.csv   ({len(tests_df)} rows)")
    print(f"  stats_ci.csv      ({len(ci_df)} rows)")
    print(f"  stats_summary.txt")
    print(f"  figs/box_*.png, figs/forest_*.png")


if __name__ == "__main__":
    main()
