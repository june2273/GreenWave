"""
Ray RLlib MAPPO — SUMO 교차로 신호제어 학습

단일 교차로(tl_0 하나)에서도 Parallel PettingZoo + shared policy 구조를 사용하므로
다중 교차로(tl_0, tl_1, ...) 확장 시 tls_ids 인자만 추가하면 됩니다.
"""
import argparse
from pathlib import Path

import numpy as np
import ray
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

try:
    from .env_sumo_pz import SumoParallelEnv
except ImportError:
    from env_sumo_pz import SumoParallelEnv


def _make_env(config: dict) -> ParallelPettingZooEnv:
    """RLlib 환경 팩토리: SumoParallelEnv → PettingZooEnv(RLlib MultiAgentEnv)"""
    return ParallelPettingZooEnv(SumoParallelEnv(**config))


def parse_args():
    p = argparse.ArgumentParser(description="Train MAPPO on SUMO intersection(s)")
    p.add_argument("--num-iters", type=int, default=200,
                   help="학습 반복 횟수 (1 iter = train_batch_size 스텝 수집 후 업데이트)")
    p.add_argument("--num-workers", type=int, default=1,
                   help="RLlib rollout worker 수 (SUMO 병렬 인스턴스 수)")
    p.add_argument("--checkpoint-out", type=str, default="models/mappo_sumo_single")
    p.add_argument("--checkpoint-freq", type=int, default=20,
                   help="체크포인트 저장 주기 (iter 단위)")
    p.add_argument("--max-steps", type=int, default=3600)
    p.add_argument("--delta-time", type=int, default=5)
    p.add_argument("--min-green", type=int, default=10)
    p.add_argument("--yellow-time", type=int, default=2)
    # 다중 교차로 확장 시: --tls-ids C D E ...
    p.add_argument("--tls-ids", nargs="+", default=["C"],
                   help="SUMO 네트워크 내 TLS id 목록 (단일: C)")
    return p.parse_args()


def main():
    args = parse_args()

    register_env("sumo_pz", _make_env)
    ray.init(ignore_reinit_error=True)

    env_config = {
        "use_gui": False,
        "delta_time": args.delta_time,
        "min_green": args.min_green,
        "yellow_time": args.yellow_time,
        "max_steps": args.max_steps,
        "tls_ids": args.tls_ids,
    }

    # shared policy 스펙 정의 (obs/act space는 모든 에이전트 동일)
    obs_space = spaces.Box(
        low=0.0, high=np.finfo(np.float32).max, shape=(10,), dtype=np.float32
    )
    act_space = spaces.Discrete(4)

    config = (
        PPOConfig()
        .environment("sumo_pz", env_config=env_config)
        .framework("torch")
        # num_env_runners=0: driver process에서 직접 rollout 수행 (디버깅 편의)
        # num_env_runners≥1: 별도 worker process 사용 (안정적 병렬 수집)
        .env_runners(num_env_runners=args.num_workers)
        .resources(num_gpus=0)  # M4 Mac: RLlib은 MPS 미지원, CPU 학습
        .multi_agent(
            # MAPPO 핵심: 모든 에이전트가 하나의 shared policy를 공유
            # 다중 교차로 확장 시에도 이 구조 그대로 사용
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            train_batch_size=4000,
            num_epochs=10,        # Ray 2.10+: num_sgd_iter → num_epochs
            minibatch_size=128,   # Ray 2.10+: sgd_minibatch_size → minibatch_size
            lambda_=0.95,         # GAE λ
            clip_param=0.2,       # PPO clip ε
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
    )

    algo = config.build_algo()
    checkpoint_dir = Path(args.checkpoint_out)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"학습 시작 | iters={args.num_iters} | workers={args.num_workers} "
          f"| tls={args.tls_ids}")

    for i in range(1, args.num_iters + 1):
        result = algo.train()

        mean_rew = result.get("episode_reward_mean", float("nan"))
        ep_len = result.get("episode_len_mean", float("nan"))
        print(f"Iter {i:4d}/{args.num_iters} | "
              f"mean_reward={mean_rew:8.2f} | ep_len={ep_len:.0f}")

        if i % args.checkpoint_freq == 0:
            ckpt = algo.save(str(checkpoint_dir))
            print(f"  → checkpoint: {ckpt}")

    final_ckpt = algo.save(str(checkpoint_dir))
    print(f"\n학습 완료. 최종 checkpoint: {final_ckpt}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
