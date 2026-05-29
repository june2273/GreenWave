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
from ray.rllib.core.rl_module.rl_module import RLModule

try:
    from .env_sumo_pz import SumoParallelEnv
    from .map_presets import MAP_CHOICES, resolve_map_args
except ImportError:
    from env_sumo_pz import SumoParallelEnv
    from map_presets import MAP_CHOICES, resolve_map_args


def _load_rl_module(model_path: str) -> RLModule:
    """체크포인트에서 shared_policy RLModule 가중치만 로드 (env runner 없이).

    PPO.from_checkpoint()는 학습 당시 num_env_runners 설정을 복원하며
    SUMO 환경을 다시 스핀업하려다 실패(IndexError: list index out of range)함.
    RLModule.from_checkpoint()는 가중치 파일만 읽으므로 SUMO/Ray worker 불필요.
    """
    module_path = Path(model_path) / "learner_group" / "learner" / "rl_module" / "shared_policy"
    if not module_path.exists():
        raise FileNotFoundError(
            f"RLModule 체크포인트를 찾을 수 없습니다: {module_path}\n"
            f"모델 경로가 올바른지 확인하세요: {model_path}"
        )
    return RLModule.from_checkpoint(str(module_path))


def parse_args():
    p = argparse.ArgumentParser(description="Record MAPPO policy rollout as mp4")
    p.add_argument("--model", type=str, required=True,
                   help="RLlib 체크포인트 디렉터리 (예: models/MAPPO_sumo_1)")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--output", type=str, default=None,
                   help="저장 경로 (미지정 시 --model 버전 번호로 자동 생성, "
                        "예: videos/mappo_policy_rollout_4.mp4)")
    p.add_argument("--fps", type=int, default=10,
                   help="비디오 재생 fps. continuous 모드 기본 10fps "
                        "(1 sim sec → 1 frame → 10 배속, ~2분 영상). "
                        "더 빠르게 보려면 20~30, 느리게 보려면 5~8 권장. "
                        "short 모드는 5fps 권장.")
    p.add_argument("--mode", type=str, default="continuous",
                   choices=["short", "continuous"],
                   help="short: env.step() 마다 1 frame (요약, 720 frame). "
                        "continuous: 매 sim step 마다 1 frame (자연스러운 흐름, "
                        "yellow phase 포착, ~3600 frame).")
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--delta-time", type=int, default=5)
    p.add_argument("--min-green", type=int, default=13,
                   help="학습 시와 동일해야 함 (train_mappo.py 와 default 일치)")
    p.add_argument("--yellow-time", type=int, default=3)
    p.add_argument("--map", type=str, default="single", choices=MAP_CHOICES,
                   help="시나리오 사전셋. 학습 시 사용한 --map 과 일치해야 함.")
    p.add_argument("--reward-mode", type=str, default="diff-waiting-time",
                   choices=["diff-waiting-time"],
                   help="보상 모드. 현재 diff-waiting-time 단일 모드만 지원.")
    p.add_argument("--sumo-cfg", type=str, default=None,
                   help="SUMO 설정 파일 경로 (학습 시와 동일하게 지정)")
    p.add_argument("--traffic", type=str, default="default",
                   choices=["default", "high"],
                   help="2x2grid 트래픽 강도 사전셋 (default/high). "
                        "high 선택 시 --sumo-cfg 미지정이면 2x2grid_dense.sumocfg 자동 사용.")
    p.add_argument("--brt-weight", type=float, default=1.0,
                   help="env 에 전달할 BRT 가중치 (학습 시와 동일 권장). "
                        "영상 자체에는 영향 없음 (정책 추론만), reward 진단용.")
    p.add_argument("--dump-metrics", type=str, default=None,
                   help="프레임별 실시간 지표 (co2_kg/avg_wait/cur_wait/throughput) 를 "
                        "JSON 으로 저장. frames 와 1:1 정렬됨 (3-way 비교 영상 오버레이용).")
    p.add_argument("--sample", action="store_true",
                   help="argmax 대신 Categorical sampling 으로 행동 선택 "
                        "(train 과 동일, 확률 mass 그대로 반영). "
                        "기본 argmax 는 mode collapse 시 비주류 action 을 절대 선택 안 함 — "
                        "--sample 로 정책의 진짜 행동 분포 시각화 가능. "
                        "재현성 위해 --seed 고정 필수.")
    return p.parse_args()


def main():
    args = parse_args()

    # CTDE 체크포인트 역직렬화에 CentralizedCriticPPOModule 이 필요 — 항상 선임포트
    try:
        from ctde_module import CentralizedCriticPPOModule  # noqa: F401
    except ImportError:
        pass

    ray.init(ignore_reinit_error=True)
    module = _load_rl_module(str(Path(args.model).resolve()))

    # --sample 모드: torch 난수 시드 고정으로 Categorical sampling 재현성 확보
    if args.sample:
        torch.manual_seed(args.seed)

    # --map 으로부터 sumo_cfg / tls_ids 결정 (preset 사용)
    sumo_cfg_effective, tls_ids_effective = resolve_map_args(
        map_name=args.map,
        sumo_cfg_arg=args.sumo_cfg,
        tls_ids_arg=None,
        traffic=args.traffic,
    )
    print(f"[map={args.map}] sumo_cfg={sumo_cfg_effective} tls_ids={tls_ids_effective}")

    env_kwargs = dict(
        use_gui=False,
        delta_time=args.delta_time,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        max_steps=args.max_steps,
        tls_ids=tls_ids_effective,
        reward_mode=args.reward_mode,
        brt_weight=args.brt_weight,
    )
    if sumo_cfg_effective:
        env_kwargs["sumo_cfg"] = sumo_cfg_effective
    env = SumoParallelEnv(**env_kwargs)

    try:
        obs_dict, _ = env.reset(seed=args.seed)
        frames = [env.render()]  # 초기 상태 프레임 포함
        metrics = [env.live_metrics()] if args.dump_metrics else None

        # continuous 모드: env._simulate_seconds 가 매 sim step 후 호출하는 hook 등록
        # → env.step() 한 번에 (yellow_time + delta_time) sim step 동안 매초 frame 캡쳐
        # → yellow phase, 큐 누적/감소, 신호 전환이 모두 자연스러운 흐름으로 보임
        if args.mode == "continuous":
            def _capture(sim_step):
                frames.append(env.render())
                if metrics is not None:
                    metrics.append(env.live_metrics())
            env.add_step_hook(_capture)

        def _select_action(logits):
            """args.sample=True: Categorical sampling (train 동일 분포).
            False: argmax (deterministic, 기본).
            """
            if args.sample:
                return int(torch.distributions.Categorical(logits=logits).sample().item())
            return int(torch.argmax(logits, dim=-1).item())

        while env.agents:
            actions = {
                agent: _select_action(
                    module.forward_inference(
                        {"obs": torch.tensor(obs_dict[agent][None], dtype=torch.float32)}
                    )["action_dist_inputs"]
                )
                for agent in env.agents
            }
            obs_dict, _, _, _, _ = env.step(actions)
            # short 모드만 env.step() 후 명시적 frame 추가 (continuous 는 hook 이 담당)
            if args.mode == "short":
                frames.append(env.render())
                if metrics is not None:
                    metrics.append(env.live_metrics())
    finally:
        # 각 cleanup 독립 실행 — Ray 2.10+ ray.shutdown() hang 회피
        for cleanup_fn, name in (
            (env.close,    "env.close"),
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

    if args.dump_metrics and metrics is not None:
        import json
        mpath = Path(args.dump_metrics)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with open(mpath, "w") as f:
            json.dump({"fps": args.fps, "frames": len(frames), "samples": metrics}, f)
        print(f"Saved metrics: {mpath}  ({len(metrics)} samples, "
              f"frames={len(frames)} → {'aligned' if len(metrics)==len(frames) else 'MISALIGNED!'})")


if __name__ == "__main__":
    # Ray 2.10+ 의 잔존 worker actor / SUMO 좀비 프로세스가 Python interpreter
    # 종료를 지연시켜 "비디오 저장 후 무한 hang" 으로 보이는 문제 방지
    import os, sys
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\n비디오 녹화 중단 (KeyboardInterrupt)")
        exit_code = 130
    except Exception as e:
        import traceback
        print(f"\n비디오 녹화 실패: {type(e).__name__}: {e}")
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(exit_code)
