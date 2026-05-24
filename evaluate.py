import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

try:
    from .env_sumo_single import SumoSingleIntersectionEnv
except ImportError:
    from env_sumo_single import SumoSingleIntersectionEnv


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DQN vs Fixed-time baseline")
    p.add_argument("--model", type=str, default="models/dqn_sumo_single.zip")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--csv-out", type=str, default="results/eval_metrics.csv")
    p.add_argument("--baseline-phase-steps", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=3600)
    p.add_argument("--delta-time", type=int, default=5)
    p.add_argument("--min-green", type=int, default=10)
    p.add_argument("--yellow-time", type=int, default=2)
    return p.parse_args()


def run_episode(env, action_fn, seed: int):
    obs, _ = env.reset(seed=seed)
    done = False
    step_idx = 0
    info = {}
    while not done:
        action = action_fn(obs, step_idx)
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step_idx += 1
    return info


def main():
    args = parse_args()

    rows = []

    # DQN evaluation
    model = DQN.load(args.model)
    env_dqn = SumoSingleIntersectionEnv(
        use_gui=False,
        delta_time=args.delta_time,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        max_steps=args.max_steps,
    )

    def dqn_action(obs, step_idx):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    for ep in range(args.episodes):
        seed = args.seed + ep
        info = run_episode(env_dqn, dqn_action, seed=seed)
        rows.append(
            {
                "algorithm": "DQN",
                "episode": ep,
                "avg_waiting_time": info.get("avg_waiting_time", np.nan),
                "avg_travel_time": info.get("avg_travel_time", np.nan),
                "total_queue_length": info.get("total_queue_length", np.nan),
                "throughput": info.get("throughput", np.nan),
            }
        )

    env_dqn.close()

    # Fixed-time baseline (4-phase cyclic)
    env_fix = SumoSingleIntersectionEnv(
        use_gui=False,
        delta_time=args.delta_time,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        max_steps=args.max_steps,
    )

    def fixed_action(obs, step_idx):
        return int((step_idx // args.baseline_phase_steps) % 4)

    for ep in range(args.episodes):
        seed = args.seed + 1000 + ep
        info = run_episode(env_fix, fixed_action, seed=seed)
        rows.append(
            {
                "algorithm": "FixedTime",
                "episode": ep,
                "avg_waiting_time": info.get("avg_waiting_time", np.nan),
                "avg_travel_time": info.get("avg_travel_time", np.nan),
                "total_queue_length": info.get("total_queue_length", np.nan),
                "throughput": info.get("throughput", np.nan),
            }
        )

    env_fix.close()

    df = pd.DataFrame(rows)
    out_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("\n=== Mean Metrics by Algorithm ===")
    print(df.groupby("algorithm")[["avg_waiting_time", "avg_travel_time", "total_queue_length", "throughput"]].mean())


if __name__ == "__main__":
    main()
