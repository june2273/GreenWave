"""
MAPPO 평가 — RLlib 체크포인트 기반

MAPPO vs Fixed-time 을 동일 SUMO seed로 쌍대 비교(paired comparison)하여
지표 CSV를 생성. CSV는 두 가지 섹션을 포함:

  1. raw rows     : 에피소드별 측정값 (algorithm × episode)
  2. summary rows : 알고리즘별 mean / std (algorithm == "MAPPO_summary" 등)

또한 모델 메타데이터(hyperparameters, train_iter)가 있으면 모든 행에
복제 기록하여 캡쳐만 보고도 LLM이 출처 모델을 식별 가능.
"""
import argparse
import json
import re
from pathlib import Path


def _versioned_output(model_path: str, template: str, ext: str) -> str:
    """모델 경로에서 버전 번호를 추출해 기본 출력 경로를 생성.

    예) model_path="models/MAPPO_sumo_4", template="results/eval_metrics_mappo"
        → "results/eval_metrics_mappo_4.csv"
    버전 번호를 찾지 못하면 template + ext 를 그대로 반환.
    """
    m = re.search(r"_(\d+)$", Path(model_path).name)
    suffix = f"_{m.group(1)}" if m else ""
    return f"{template}{suffix}{ext}"


def _load_train_metadata(model_path: str) -> dict:
    """모델 디렉터리의 train_metadata.json 을 읽어 dict 반환 (없으면 빈 dict).

    train_mappo.py 가 학습 종료 시 저장하는 파일이며, CSV 모든 행에 복제되어
    캡쳐만 봐도 어떤 모델·하이퍼파라미터인지 LLM이 식별할 수 있게 한다.
    """
    meta_path = Path(model_path) / "train_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


import numpy as np
import pandas as pd
import ray
import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

try:
    from .env_sumo_pz import SumoParallelEnv
except ImportError:
    from env_sumo_pz import SumoParallelEnv


def _make_env(config: dict) -> ParallelPettingZooEnv:
    return ParallelPettingZooEnv(SumoParallelEnv(**config))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MAPPO vs Fixed-time baseline")
    p.add_argument("--model", type=str, required=True,
                   help="RLlib 체크포인트 디렉터리 (예: models/MAPPO_sumo_1)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--csv-out", type=str, default=None,
                   help="저장 경로 (미지정 시 --model 버전 번호로 자동 생성, "
                        "예: results/eval_metrics_mappo_4.csv)")
    p.add_argument("--baseline-phase-steps", type=int, default=3,
                   help="Fixed-time baseline: N 스텝마다 phase 순환")
    p.add_argument("--max-steps", type=int, default=3600)
    p.add_argument("--delta-time", type=int, default=5)
    p.add_argument("--min-green", type=int, default=10)
    p.add_argument("--yellow-time", type=int, default=2)
    p.add_argument("--tls-ids", nargs="+", default=["C"],
                   help="학습 시 사용한 TLS id 목록 (train_mappo.py와 일치해야 함)")
    p.add_argument("--reward-mode", type=str, default="queue",
                   choices=["queue", "diff-waiting-time", "pressure"],
                   help="학습 시 사용한 보상 모드 (train_mappo.py와 일치해야 함)")
    p.add_argument("--sumo-cfg", type=str, default=None,
                   help="SUMO 설정 파일 경로 (학습 시와 동일하게 지정)")
    p.add_argument("--traffic", type=str, default="default",
                   choices=["default", "high"],
                   help="2x2grid 트래픽 강도 사전셋 (train_mappo.py 와 동일 의미). "
                        "high 선택 시 --sumo-cfg 미지정이면 2x2grid_dense.sumocfg 자동 사용.")
    return p.parse_args()


def _action_ratios(action_counts: list) -> dict:
    """action_counts 리스트 → 비율 dict {action_k_ratio: ...}.

    길이가 num_green 만큼 가변. mode collapse 진단: 균등 분포에 가까우면
    1/num_green 씩, 한쪽 쏠림이면 1.0 근처.
    """
    counts = np.asarray(action_counts, dtype=np.float64)
    n = len(counts)
    total = counts.sum()
    if total <= 0:
        return {f"action_{i}_ratio": 0.0 for i in range(n)}
    ratios = counts / total
    return {f"action_{i}_ratio": float(ratios[i]) for i in range(n)}


def _row_from_info(algorithm: str, ep: int, seed: int, info: dict) -> dict:
    """env.step() 마지막 info → CSV row dict 변환 (진단 컬럼 평탄화)."""
    row = {
        "algorithm": algorithm,
        "episode":   ep,
        "seed":      seed,
        "avg_waiting_time":   info.get("avg_waiting_time",   np.nan),
        "std_waiting_time":   info.get("std_waiting_time",   np.nan),
        "avg_travel_time":    info.get("avg_travel_time",    np.nan),
        "total_queue_length": info.get("total_queue_length", np.nan),
        "throughput":         info.get("throughput",         np.nan),
        "phase_switches":     info.get("phase_switches",     np.nan),
        "max_queue":          info.get("max_queue",          np.nan),
        "teleported":         info.get("teleported",         np.nan),
    }
    row.update(_action_ratios(info.get("action_counts", [])))
    return row


def run_episode(env: SumoParallelEnv, action_fn, seed: int) -> dict:
    """에피소드 1회 실행 후 마지막 info 반환"""
    obs_dict, _ = env.reset(seed=seed)
    step_idx = 0
    last_infos: dict = {}
    while env.agents:
        actions = action_fn(obs_dict, step_idx, env.agents)
        obs_dict, _, _, _, last_infos = env.step(actions)
        step_idx += 1
    # 단일·다중 교차로 공통: 첫 번째 에이전트 info 기준 (에피소드 지표는 동일)
    return next(iter(last_infos.values())) if last_infos else {}


def main():
    args = parse_args()

    register_env("sumo_pz", _make_env)
    ray.init(ignore_reinit_error=True)
    algo = PPO.from_checkpoint(str(Path(args.model).resolve()))
    module = algo.get_module("shared_policy")

    # 모델 학습 메타데이터 (없으면 빈 dict) — CSV 모든 행에 prefix로 부착
    train_meta = _load_train_metadata(args.model)

    # --traffic high + --sumo-cfg 미지정 → dense sumocfg 자동 사용
    sumo_cfg_effective = args.sumo_cfg
    if args.traffic == "high" and not sumo_cfg_effective:
        sumo_cfg_effective = str(
            (Path(__file__).resolve().parent
             / "sumo_data" / "2x2grid_dense.sumocfg").resolve()
        )
        print(f"[traffic=high] sumo_cfg 자동 사용: {sumo_cfg_effective}")

    env_kwargs = dict(
        use_gui=False,
        delta_time=args.delta_time,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        max_steps=args.max_steps,
        tls_ids=args.tls_ids,
        reward_mode=args.reward_mode,
    )
    if sumo_cfg_effective:
        env_kwargs["sumo_cfg"] = sumo_cfg_effective

    rows = []

    # 두 환경을 미리 생성해 인터리브 루프에서 공유
    env_mappo = SumoParallelEnv(**env_kwargs)
    env_fix   = SumoParallelEnv(**env_kwargs)

    # 동적 METRIC_COLS: num_green에 맞춰 action_*_ratio 컬럼 수 조정
    # (단일·2x2grid 모두 num_green=4 이지만 다른 네트워크 대응)
    num_green = env_mappo._num_green
    metric_cols = [
        "avg_waiting_time", "std_waiting_time", "avg_travel_time",
        "total_queue_length", "throughput",
        "phase_switches", "max_queue", "teleported",
        *[f"action_{i}_ratio" for i in range(num_green)],
    ]

    def mappo_action(obs_dict, step_idx, agents):
        return {
            agent: int(torch.argmax(
                module.forward_inference(
                    {"obs": torch.tensor(obs_dict[agent][None], dtype=torch.float32)}
                )["action_dist_inputs"],
                dim=-1,
            ).item())
            for agent in agents
        }

    def fixed_action(obs_dict, step_idx, agents):
        phase = int((step_idx // args.baseline_phase_steps) % 4)
        return {agent: phase for agent in agents}

    try:
        # ── 쌍대 비교: 동일 ep에서 동일 seed 사용 ────────────────────────────
        # 같은 seed → SUMO가 동일한 차량 수요를 생성하므로
        # 알고리즘 성능 차이만 측정 가능 (교통 수요 샘플 효과 제거)
        for ep in range(args.episodes):
            seed = args.seed + ep

            # ── MAPPO ────────────────────────────────────────────────────────
            info = run_episode(env_mappo, mappo_action, seed=seed)
            rows.append(_row_from_info("MAPPO", ep, seed, info))
            r = rows[-1]
            act_dist_str = ",".join(f"{r.get(f'action_{j}_ratio', 0.0):.2f}"
                                    for j in range(num_green))
            print(f"[MAPPO]     ep={ep} seed={seed} | "
                  f"wait={r['avg_waiting_time']:.1f}s | "
                  f"queue={r['total_queue_length']:.0f} | "
                  f"switches={r['phase_switches']:.0f} | "
                  f"act_dist=[{act_dist_str}] | "
                  f"teleport={r['teleported']:.0f}")

            # ── FixedTime (동일 seed) ─────────────────────────────────────────
            info = run_episode(env_fix, fixed_action, seed=seed)
            rows.append(_row_from_info("FixedTime", ep, seed, info))
            r = rows[-1]
            print(f"[FixedTime] ep={ep} seed={seed} | "
                  f"wait={r['avg_waiting_time']:.1f}s | "
                  f"queue={r['total_queue_length']:.0f} | "
                  f"teleport={r['teleported']:.0f}")
    finally:
        # 각 cleanup 독립 실행 — Ray 2.10+ algo.stop()/ray.shutdown() hang 회피
        for cleanup_fn, name in (
            (env_mappo.close, "env_mappo.close"),
            (env_fix.close,   "env_fix.close"),
            (algo.stop,       "algo.stop"),
            (ray.shutdown,    "ray.shutdown"),
        ):
            try:
                cleanup_fn()
            except Exception as e:
                print(f"[cleanup warning] {name}: {type(e).__name__}: {e}")

    # ── 요약 행 추가 (algorithm별 mean / std) ─────────────────────────────
    df_raw = pd.DataFrame(rows)
    summary_rows = []
    for algo_name in ("MAPPO", "FixedTime"):
        sub = df_raw[df_raw["algorithm"] == algo_name]
        if sub.empty:
            continue
        means = sub[metric_cols].mean(numeric_only=True)
        stds  = sub[metric_cols].std(numeric_only=True, ddof=0)
        summary_rows.append({
            "algorithm": f"{algo_name}_mean",
            "episode": "", "seed": "",
            **{c: means[c] for c in metric_cols},
        })
        summary_rows.append({
            "algorithm": f"{algo_name}_std",
            "episode": "", "seed": "",
            **{c: stds[c] for c in metric_cols},
        })
    df = pd.concat([df_raw, pd.DataFrame(summary_rows)], ignore_index=True)

    # ── 메타데이터 prefix 컬럼 (모든 행에 동일값) ─────────────────────────
    # 캡쳐만 보고도 LLM이 어느 모델·hyperparameter 결과인지 식별 가능
    meta_prefix = {
        "model_path":        str(Path(args.model).resolve()),
        "train_iter":        train_meta.get("train_iter", ""),
        "train_total_steps": train_meta.get("train_total_steps", ""),
        "reward_mode":       train_meta.get("reward_mode", args.reward_mode),
        "lr":                train_meta.get("lr", ""),
        "entropy_coeff":     train_meta.get("entropy_coeff", ""),
        "vf_clip_param":     train_meta.get("vf_clip_param", ""),
        "train_seed":        train_meta.get("seed", ""),
        "tls_ids":           "_".join(args.tls_ids),
        "num_workers":       train_meta.get("num_workers", ""),
        "traffic":           train_meta.get("traffic", args.traffic),
    }
    for k, v in meta_prefix.items():
        df.insert(0, k, v)

    # ── 저장 및 출력 ──────────────────────────────────────────────────────
    csv_out = args.csv_out or _versioned_output(args.model, "results/eval_metrics_mappo", ".csv")
    out_path = Path(csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Train metadata: {'loaded' if train_meta else 'not found (older model)'}")
    print("\n=== Summary (mean ± std) ===")
    for algo_name in ("MAPPO", "FixedTime"):
        sub = df_raw[df_raw["algorithm"] == algo_name][metric_cols]
        if sub.empty:
            continue
        print(f"\n[{algo_name}]")
        for col in ("avg_waiting_time", "avg_travel_time",
                    "total_queue_length", "throughput",
                    "phase_switches", "teleported"):
            m, s = sub[col].mean(), sub[col].std(ddof=0)
            print(f"  {col:22s}: {m:>10.2f} ± {s:.2f}")


if __name__ == "__main__":
    # Ray 2.10+ 의 잔존 worker actor 가 Python interpreter 종료를 지연시키는
    # 문제 방지 — main() finally 의 cleanup 후 os._exit 으로 강제 종료
    import os, sys
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\n평가 중단 (KeyboardInterrupt)")
        exit_code = 130
    except Exception as e:
        import traceback
        print(f"\n평가 실패: {type(e).__name__}: {e}")
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(exit_code)
