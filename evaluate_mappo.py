"""
MAPPO 평가 — RLlib 체크포인트 기반

evaluate.py 와 동일한 지표(avg_waiting_time, avg_travel_time,
total_queue_length, throughput)를 MAPPO vs Fixed-time 으로 비교해 CSV 저장.
"""
import argparse
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
    return p.parse_args()


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

    env_kwargs = dict(
        use_gui=False,
        delta_time=args.delta_time,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        max_steps=args.max_steps,
        tls_ids=args.tls_ids,
    )

    rows = []

    # 두 환경을 미리 생성해 인터리브 루프에서 공유
    env_mappo = SumoParallelEnv(**env_kwargs)
    env_fix   = SumoParallelEnv(**env_kwargs)

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
            rows.append({
                "algorithm": "MAPPO",
                "episode":   ep,
                "seed":      seed,
                "avg_waiting_time":   info.get("avg_waiting_time",   np.nan),
                "avg_travel_time":    info.get("avg_travel_time",    np.nan),
                "total_queue_length": info.get("total_queue_length", np.nan),
                "throughput":         info.get("throughput",         np.nan),
            })
            print(f"[MAPPO]     ep={ep} seed={seed} | "
                  f"wait={rows[-1]['avg_waiting_time']:.1f}s | "
                  f"queue={rows[-1]['total_queue_length']:.0f}")

            # ── FixedTime (동일 seed) ─────────────────────────────────────────
            info = run_episode(env_fix, fixed_action, seed=seed)
            rows.append({
                "algorithm": "FixedTime",
                "episode":   ep,
                "seed":      seed,
                "avg_waiting_time":   info.get("avg_waiting_time",   np.nan),
                "avg_travel_time":    info.get("avg_travel_time",    np.nan),
                "total_queue_length": info.get("total_queue_length", np.nan),
                "throughput":         info.get("throughput",         np.nan),
            })
            print(f"[FixedTime] ep={ep} seed={seed} | "
                  f"wait={rows[-1]['avg_waiting_time']:.1f}s | "
                  f"queue={rows[-1]['total_queue_length']:.0f}")
    finally:
        env_mappo.close()
        env_fix.close()
        algo.stop()
        ray.shutdown()

    # ── 저장 및 출력 ──────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_out = args.csv_out or _versioned_output(args.model, "results/eval_metrics_mappo", ".csv")
    out_path = Path(csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print("\n=== Mean Metrics by Algorithm ===")
    print(df.groupby("algorithm")[[
        "avg_waiting_time", "avg_travel_time",
        "total_queue_length", "throughput"
    ]].mean())


if __name__ == "__main__":
    main()
