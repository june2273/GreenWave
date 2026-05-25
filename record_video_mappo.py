"""
MAPPO 정책 롤아웃 영상 저장 — RLlib 체크포인트 기반

record_video.py 의 MAPPO 버전.
SumoParallelEnv.render() 프레임을 모아 mp4 인코딩.
"""
import argparse
import re
from pathlib import Path


def _versioned_output(model_path: str, template: str, ext: str) -> str:
    """모델 경로에서 버전 번호를 추출해 기본 출력 경로를 생성.

    예) model_path="models/MAPPO_sumo_4", template="videos/mappo_policy_rollout"
        → "videos/mappo_policy_rollout_4.mp4"
    버전 번호를 찾지 못하면 template + ext 를 그대로 반환.
    """
    m = re.search(r"_(\d+)$", Path(model_path).name)
    suffix = f"_{m.group(1)}" if m else ""
    return f"{template}{suffix}{ext}"


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
    p.add_argument("--output", type=str, default=None,
                   help="저장 경로 (미지정 시 --model 버전 번호로 자동 생성, "
                        "예: videos/mappo_policy_rollout_4.mp4)")
    p.add_argument("--fps", type=int, default=30,
                   help="비디오 재생 fps. continuous 모드는 30fps 권장 "
                        "(1 sim sec → 1 frame → 30 배속). short 모드는 5~10fps.")
    p.add_argument("--mode", type=str, default="continuous",
                   choices=["short", "continuous"],
                   help="short: env.step() 마다 1 frame (요약, 720 frame). "
                        "continuous: 매 sim step 마다 1 frame (자연스러운 흐름, "
                        "yellow phase 포착, ~3600 frame).")
    p.add_argument("--max-steps", type=int, default=1200)
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
                   help="2x2grid 트래픽 강도 사전셋 (default/high). "
                        "high 선택 시 --sumo-cfg 미지정이면 2x2grid_dense.sumocfg 자동 사용.")
    return p.parse_args()


def main():
    args = parse_args()

    register_env("sumo_pz", _make_env)
    ray.init(ignore_reinit_error=True)
    algo = PPO.from_checkpoint(str(Path(args.model).resolve()))
    module = algo.get_module("shared_policy")

    # --traffic high + --sumo-cfg 미지정 → dense sumocfg 자동
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
    env = SumoParallelEnv(**env_kwargs)

    try:
        obs_dict, _ = env.reset(seed=args.seed)
        frames = [env.render()]  # 초기 상태 프레임 포함

        # continuous 모드: env._simulate_seconds 가 매 sim step 후 호출하는 hook 등록
        # → env.step() 한 번에 (yellow_time + delta_time) sim step 동안 매초 frame 캡쳐
        # → yellow phase, 큐 누적/감소, 신호 전환이 모두 자연스러운 흐름으로 보임
        if args.mode == "continuous":
            env.add_step_hook(lambda sim_step: frames.append(env.render()))

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
            # short 모드만 env.step() 후 명시적 frame 추가 (continuous 는 hook 이 담당)
            if args.mode == "short":
                frames.append(env.render())
    finally:
        # 각 cleanup 독립 실행 — Ray 2.10+ algo.stop()/ray.shutdown() hang 회피
        for cleanup_fn, name in (
            (env.close,    "env.close"),
            (algo.stop,    "algo.stop"),
            (ray.shutdown, "ray.shutdown"),
        ):
            try:
                cleanup_fn()
            except Exception as e:
                print(f"[cleanup warning] {name}: {type(e).__name__}: {e}")

    output = args.output or _versioned_output(args.model, "videos/mappo_policy_rollout", ".mp4")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=args.fps)
    duration = len(frames) / max(1, args.fps)
    print(f"Saved video: {out_path}  ({len(frames)} frames @ {args.fps}fps, "
          f"~{duration:.1f}s, mode={args.mode})")


if __name__ == "__main__":
    # Ray 2.10+ 의 잔존 worker actor / SUMO 좀비 프로세스가 Python interpreter
    # 종료를 지연시켜 "비디오 저장 후 무한 hang" 으로 보이는 문제 방지
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\n비디오 녹화 중단 (KeyboardInterrupt)")
        exit_code = 130
    except Exception as e:
        print(f"\n비디오 녹화 실패: {type(e).__name__}: {e}")
        exit_code = 1
    finally:
        import os
        os._exit(exit_code)
