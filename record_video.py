import argparse
from pathlib import Path

import imageio.v2 as imageio
from stable_baselines3 import DQN

try:
    from .env_sumo_single import SumoSingleIntersectionEnv
except ImportError:
    from env_sumo_single import SumoSingleIntersectionEnv


def parse_args():
    """비디오 녹화 실행 인자 파싱"""
    p = argparse.ArgumentParser(description="Record DQN policy rollout as mp4")
    p.add_argument("--model", type=str, default="models/dqn_sumo_single.zip")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--output", type=str, default="videos/dqn_policy_rollout.mp4")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--delta-time", type=int, default=5)
    p.add_argument("--min-green", type=int, default=10)
    p.add_argument("--yellow-time", type=int, default=2)
    return p.parse_args()


def main():
    """DQN 모델의 정책을 실행하고 결과를 비디오로 저장"""
    args = parse_args()

    env = SumoSingleIntersectionEnv(
        use_gui=False,
        delta_time=args.delta_time,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        max_steps=args.max_steps,
    )
    model = DQN.load(args.model)

    obs, _ = env.reset(seed=args.seed)
    done = False
    frames = [env.render()]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = bool(terminated or truncated)
        frames.append(env.render())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=args.fps)

    env.close()
    print(f"Saved video: {out_path}")


if __name__ == "__main__":
    main()
