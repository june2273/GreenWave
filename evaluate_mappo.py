"""
MAPPO 평가 — RLlib 체크포인트 기반

evaluate.py 와 동일한 지표(avg_waiting_time, avg_travel_time,
total_queue_length, throughput)를 MAPPO vs Fixed-time 으로 비교해 CSV 저장.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ray
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
    p.add_argument("--csv-out", type=str, default="results/eval_metrics_mappo.csv")
    p.add_argument("--baseline-phase-steps", type=int, default=3,
                   help="Fixed-time baseline: N 스텝마다 phase 순환")
    p.add_argument("--max-steps", type=int, default=3600)
    p.add_argument("--delta-time", type=int, default=5)
    p.add_argument("--min-green", type=int, default=10)
    p.add_argument("--yellow-time", type=int, default=2)
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

    env_kwargs = dict(
        use_gui=False,
        delta_time=args.delta_time,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        max_steps=args.max_steps,
    )

    rows = []

    try:
        # ── MAPPO 평가 ────────────────────────────────────────────────────────
        env_mappo = SumoParallelEnv(**env_kwargs)

        def mappo_action(obs_dict, step_idx, agents):
            return {
                agent: int(algo.compute_single_action(
                    obs_dict[agent], policy_id="shared_policy"
                ))
                for agent in agents
            }

        try:
            for ep in range(args.episodes):
                info = run_episode(env_mappo, mappo_action, seed=args.seed + ep)
                rows.append({
                    "algorithm": "MAPPO",
                    "episode": ep,
                    "avg_waiting_time":  info.get("avg_waiting_time",  np.nan),
                    "avg_travel_time":   info.get("avg_travel_time",   np.nan),
                    "total_queue_length": info.get("total_queue_length", np.nan),
                    "throughput":        info.get("throughput",         np.nan),
                })
                print(f"[MAPPO]     ep={ep} | "
                      f"wait={rows[-1]['avg_waiting_time']:.1f}s | "
                      f"queue={rows[-1]['total_queue_length']:.0f}")
        finally:
            env_mappo.close()

        # ── Fixed-time baseline ───────────────────────────────────────────────
        env_fix = SumoParallelEnv(**env_kwargs)

        def fixed_action(obs_dict, step_idx, agents):
            phase = int((step_idx // args.baseline_phase_steps) % 4)
            return {agent: phase for agent in agents}

        try:
            for ep in range(args.episodes):
                info = run_episode(env_fix, fixed_action, seed=args.seed + 1000 + ep)
                rows.append({
                    "algorithm": "FixedTime",
                    "episode": ep,
                    "avg_waiting_time":  info.get("avg_waiting_time",  np.nan),
                    "avg_travel_time":   info.get("avg_travel_time",   np.nan),
                    "total_queue_length": info.get("total_queue_length", np.nan),
                    "throughput":        info.get("throughput",         np.nan),
                })
                print(f"[FixedTime] ep={ep} | "
                      f"wait={rows[-1]['avg_waiting_time']:.1f}s | "
                      f"queue={rows[-1]['total_queue_length']:.0f}")
        finally:
            env_fix.close()
    finally:
        algo.stop()
        ray.shutdown()

    # ── 저장 및 출력 ──────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_path = Path(args.csv_out)
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
