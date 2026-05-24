"""
MAPPO 정책 롤아웃 영상 저장 — RLlib 체크포인트 기반

record_video.py 의 MAPPO 버전.
SumoParallelEnv.render() 프레임을 모아 mp4 인코딩.
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
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
    p = argparse.ArgumentParser(description="Record MAPPO policy rollout as mp4")
    p.add_argument("--model", type=str, required=True,
                   help="RLlib 체크포인트 디렉터리 (예: models/MAPPO_sumo_1)")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--output", type=str, default="videos/mappo_policy_rollout.mp4")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--delta-time", type=int, default=5)
    p.add_argument("--min-green", type=int, default=10)
    p.add_argument("--yellow-time", type=int, default=2)
    p.add_argument("--tls-ids", nargs="+", default=["C"],
                   help="SUMO 네트워크 내 TLS id 목록 (단일: C)")
    return p.parse_args()


def main():
    args = parse_args()

    register_env("sumo_pz", _make_env)
    ray.init(ignore_reinit_error=True)
    algo = PPO.from_checkpoint(str(Path(args.model).resolve()))
    module = algo.get_module("shared_policy")

    env = SumoParallelEnv(
        use_gui=False,
        delta_time=args.delta_time,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        max_steps=args.max_steps,
        tls_ids=args.tls_ids,
    )

    try:
        obs_dict, _ = env.reset(seed=args.seed)
        frames = [env.render()]  # 초기 상태 프레임 포함

        while env.agents:
            actions = {
                agent: int(torch.argmax(
                    module.forward_inference(
                        {"obs": torch.tensor(obs_dict[agent][None], dtype=torch.float32)}
                    )["action_dist_inputs"],
                    dim=-1,
                ).item())
                for agent in env.agents
            }
            obs_dict, _, _, _, _ = env.step(actions)
            frames.append(env.render())
    finally:
        env.close()
        algo.stop()
        ray.shutdown()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=args.fps)
    print(f"Saved video: {out_path}  ({len(frames)} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
