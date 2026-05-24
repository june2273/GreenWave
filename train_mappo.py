"""
Ray RLlib MAPPO — SUMO 교차로 신호제어 학습

단일 교차로(tl_0 하나)에서도 Parallel PettingZoo + shared policy 구조를 사용하므로
다중 교차로(tl_0, tl_1, ...) 확장 시 tls_ids 인자만 추가하면 됩니다.
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import ray
from torch.utils.tensorboard import SummaryWriter
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


def _next_out_path(algo: str = "MAPPO", model_dir: str = "models") -> str:
    """models/ 에서 {algo}_sumo_N 패턴의 다음 버전 경로를 반환.

    기존에 MAPPO_sumo_1, MAPPO_sumo_2 가 있으면 MAPPO_sumo_3 을 반환.
    RLlib 체크포인트는 디렉터리 형식으로 저장되므로 확장자 없이 사용.
    (SB3의 .zip 과 달리 RLlib 은 디렉터리 기반 체크포인트)
    """
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(algo)}_sumo_(\d+)$")
    versions = [
        int(m.group(1))
        for entry in d.iterdir()
        if (m := pat.match(entry.name))
    ]
    return str(d / f"{algo}_sumo_{max(versions, default=0) + 1}")


def parse_args():
    p = argparse.ArgumentParser(description="Train MAPPO on SUMO intersection(s)")
    p.add_argument("--num-iters", type=int, default=200,
                   help="학습 반복 횟수 (1 iter = train_batch_size 스텝 수집 후 업데이트)")
    p.add_argument("--num-workers", type=int, default=1,
                   help="RLlib rollout worker 수 (SUMO 병렬 인스턴스 수)")
    p.add_argument("--out", type=str, default=None,
                   help="저장 경로 (미지정 시 models/MAPPO_sumo_N 자동 버전 생성)")
    p.add_argument("--checkpoint-freq", type=int, default=20,
                   help="중간 체크포인트 저장 주기 (iter 단위)")
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
            vf_loss_coeff=1.0,    # 보상 스케일 다운 후 VF loss 가중치 높임
            entropy_coeff=0.01,
            vf_clip_param=500.0,  # reward scale 조정 후 discounted return 범위 -300~-500, 여유 있게 설정
        )
    )

    algo = config.build_algo()
    # algo.save()는 절대 경로 필요 (pyarrow URI 파싱 때문에 상대 경로 불가)
    out_path = str(Path(args.out if args.out else _next_out_path()).resolve())
    Path(out_path).mkdir(parents=True, exist_ok=True)

    run_name = Path(out_path).name
    tb_writer = SummaryWriter(log_dir=f"results/tb_mappo/{run_name}")
    print(f"학습 시작 | iters={args.num_iters} | workers={args.num_workers} "
          f"| tls={args.tls_ids} | out={out_path}")
    print(f"TensorBoard: tensorboard --logdir results/tb_mappo")

    try:
        for i in range(1, args.num_iters + 1):
            result = algo.train()

            # Ray 2.10+: 지표 경로 변경 (env_runner_results→env_runners, learner_results→learners)
            env_stats    = result.get("env_runners", {})
            mean_rew     = env_stats.get("episode_return_mean", float("nan"))
            ep_len       = env_stats.get("episode_len_mean",    float("nan"))
            total_steps  = result.get("num_env_steps_sampled_lifetime", 0)
            tb_writer.add_scalar("reward/mean",       mean_rew,    i)
            tb_writer.add_scalar("episode/len_mean",  ep_len,      i)
            tb_writer.add_scalar("train/total_steps", total_steps, i)

            # 손실 지표 (Ray 2.10+: learners 하위)
            policy_stats = result.get("learners", {}).get("shared_policy", {})
            for tag, key in (
                ("loss/total",   "total_loss"),
                ("loss/policy",  "policy_loss"),
                ("loss/value",   "vf_loss"),
                ("loss/entropy", "entropy"),
            ):
                val = policy_stats.get(key)
                if val is not None:
                    tb_writer.add_scalar(tag, val, i)

            ep_str = f"{ep_len:.0f}" if not math.isnan(ep_len) else "nan"
            print(f"Iter {i:4d}/{args.num_iters} | "
                  f"mean_reward={mean_rew:8.2f} | ep_len={ep_str} | steps={int(total_steps)}")

            if i % args.checkpoint_freq == 0:
                ckpt = algo.save(str(out_path))
                print(f"  → checkpoint: {ckpt}")

        final_ckpt = algo.save(str(out_path))
        print(f"\n학습 완료. 최종 checkpoint: {final_ckpt}")
    finally:
        tb_writer.close()
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
