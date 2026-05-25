"""
Ray RLlib MAPPO — SUMO 교차로 신호제어 학습

단일 교차로(tl_0 하나)에서도 Parallel PettingZoo + shared policy 구조를 사용하므로
다중 교차로(tl_0, tl_1, ...) 확장 시 tls_ids 인자만 추가하면 됩니다.
"""
import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

try:
    import psutil  # ray 의 transitive dep — 항상 설치되어 있음
except ImportError:
    psutil = None  # 메모리/프로세스 모니터링은 best-effort 로 동작

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


# ──────────────────────────────────────────────────────────────────────────────
# 시스템 모니터링 헬퍼 — 메모리 누수 / SUMO 프로세스 누수 감지
# ──────────────────────────────────────────────────────────────────────────────

def _process_rss_mb() -> float:
    """현재 Python 프로세스 RSS (MiB). psutil 미설치 시 NaN.

    iter 별로 기록해 우상향 그래프가 보이면 메모리 누수 의심.
    """
    if psutil is None:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _sumo_process_count() -> int:
    """현재 머신에서 실행 중인 SUMO 프로세스 수.

    reset()에서 traci.start() / close() 사이 누수가 발생하면 증가.
    Ray worker 수보다 많으면 좀비 프로세스 의심.
    """
    if psutil is None:
        return -1
    count = 0
    for p in psutil.process_iter(attrs=["name"]):
        try:
            name = (p.info.get("name") or "").lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if name.startswith("sumo"):
            count += 1
    return count


# ──────────────────────────────────────────────────────────────────────────────
# 커스텀 콜백 — env info에 노출된 진단 지표를 RLlib 메트릭으로 끌어올림
# ──────────────────────────────────────────────────────────────────────────────

def _build_callback():
    """RLlib 콜백 클래스를 동적으로 생성.

    Ray 버전별 콜백 API가 다르므로 (RLlibCallback ↔ DefaultCallbacks),
    import 가능한 것을 골라 on_episode_end 핸들러를 안전하게 등록한다.
    """
    try:
        from ray.rllib.callbacks.callbacks import RLlibCallback as _Base
    except ImportError:
        from ray.rllib.algorithms.callbacks import DefaultCallbacks as _Base

    class GreenWaveMetricsCallback(_Base):
        """에피소드 종료 시점에 env info에서 진단 지표를 추출해 metrics에 기록.

        기록 지표:
          - policy/phase_switches : 에피소드 누적 phase 전환 수
          - policy/teleported     : SUMO grid lock 텔레포트 차량 수
          - policy/max_queue      : 단일 lane 최대 halting 차량 수
          - policy/action_{0..3}_ratio : 정책 행동 분포 (mode collapse 진단)
        """

        def on_episode_end(self, *, episode=None, metrics_logger=None, **kwargs):
            # 콜백 실패가 학습을 중단시키지 않도록 전체를 try로 감쌈
            try:
                last_info = self._extract_last_info(episode)
                if not last_info:
                    return

                self._log_metric(episode, metrics_logger,
                                 "phase_switches", float(last_info.get("phase_switches", 0)))
                self._log_metric(episode, metrics_logger,
                                 "teleported", float(last_info.get("teleported", 0)))
                self._log_metric(episode, metrics_logger,
                                 "max_queue", float(last_info.get("max_queue", 0)))

                counts = last_info.get("action_counts", [])
                total = sum(counts) or 1
                # 4 하드코딩 대신 실제 action 개수 사용 (num_green 가변 대응)
                for i, c in enumerate(counts):
                    self._log_metric(episode, metrics_logger,
                                     f"action_{i}_ratio", c / total)
            except Exception:
                # 학습 진행이 최우선 — 메트릭 수집 실패는 silent하게 무시
                return

        @staticmethod
        def _extract_last_info(episode):
            """MultiAgentEpisode / EpisodeV2 양쪽에서 안전하게 마지막 info 추출.

            Ray 2.10+ MultiAgentEpisode.get_infos() 의 첫 positional 인자는
            agent_id가 아니라 indices(int) 임. indices=-1 로 마지막 step의 info를
            {agent_id: info} dict로 받아 첫 non-empty 값을 사용.
            """
            if episode is None:
                return None
            # New API stack
            try:
                infos = episode.get_infos(indices=-1)
                if isinstance(infos, dict):
                    for info in infos.values():
                        if info:
                            return info
                elif isinstance(infos, list) and infos:
                    return infos[0]
            except (AssertionError, AttributeError, TypeError, IndexError, KeyError):
                pass
            # Old API stack
            try:
                for aid in ("tl_0", "tl_1", "tl_2", "tl_3"):
                    info = episode.last_info_for(aid)
                    if info:
                        return info
            except (AttributeError, TypeError):
                pass
            return None

        @staticmethod
        def _log_metric(episode, metrics_logger, key, value):
            """new API: metrics_logger.log_value / old API: episode.custom_metrics."""
            if metrics_logger is not None:
                try:
                    metrics_logger.log_value(key, value)
                    return
                except Exception:
                    pass
            if episode is not None and hasattr(episode, "custom_metrics"):
                episode.custom_metrics[key] = value

    return GreenWaveMetricsCallback


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
                   help="SUMO 네트워크 내 TLS id 목록 (단일: C, 2x2: 1 2 5 6)")
    p.add_argument("--seed", type=int, default=42,
                   help="전역 랜덤 시드 (random/numpy/torch/SUMO 일괄 설정)")
    p.add_argument("--reward-mode", type=str, default="queue",
                   choices=["queue", "diff-waiting-time", "pressure"],
                   help="보상 함수 모드 (queue: mean+max 패널티, "
                        "diff-waiting-time: 대기시간 변화량, pressure: 처리량 차이)")
    p.add_argument("--sumo-cfg", type=str, default=None,
                   help="SUMO 설정 파일 경로 (미지정 시 기본 단일교차로 사용)")
    p.add_argument("--traffic", type=str, default="default",
                   choices=["default", "high"],
                   help="2x2grid 트래픽 강도 사전셋 ("
                        "default: 원본 sumo-rl routes (~0.4대/초, free flow), "
                        "high: routes_2x2_dense (~1.4대/초, 정체). "
                        "high 선택 시 --sumo-cfg 미지정이면 2x2grid_dense.sumocfg 자동 사용. "
                        "--sumo-cfg 명시 시 그것이 우선.)")
    return p.parse_args()


def main():
    args = parse_args()

    # 전역 시드 설정 (재현성 확보)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    register_env("sumo_pz", _make_env)
    ray.init(ignore_reinit_error=True)

    # --traffic high + --sumo-cfg 미지정 시 자동으로 dense sumocfg 사용
    # --sumo-cfg 명시 시 그것이 우선 (사용자가 직접 routes 지정한 경우 존중)
    sumo_cfg_effective = args.sumo_cfg
    if args.traffic == "high" and not sumo_cfg_effective:
        sumo_cfg_effective = str(
            (Path(__file__).resolve().parent
             / "sumo_data" / "2x2grid_dense.sumocfg").resolve()
        )
        print(f"[traffic=high] sumo_cfg 자동 사용: {sumo_cfg_effective}")

    env_config = {
        "use_gui": False,
        "delta_time": args.delta_time,
        "min_green": args.min_green,
        "yellow_time": args.yellow_time,
        "max_steps": args.max_steps,
        "tls_ids": args.tls_ids,
        "reward_mode": args.reward_mode,
    }
    if sumo_cfg_effective:
        env_config["sumo_cfg"] = sumo_cfg_effective

    # ── obs/act space 동적 결정 ─────────────────────────────────────────
    # 네트워크별로 controlled lanes 수와 green phase 수가 다르므로 (단일=8 lanes,
    # 2x2grid=12 lanes) 임시 probe env를 생성해 실제 space를 추출.
    # SumoParallelEnv.__init__ 가 내부적으로 1회 SUMO probe를 수행하므로
    # 여기서 환경 객체만 만들면 즉시 spec을 얻을 수 있음.
    probe_env = SumoParallelEnv(**env_config)
    probe_agent = probe_env.possible_agents[0]
    obs_space = probe_env.observation_space(probe_agent)
    act_space = probe_env.action_space(probe_agent)
    # 후속 TB 로깅에서 action_{i}_ratio 키 가변 생성에 사용
    num_green = probe_env._num_green
    print(f"환경 spec | obs={obs_space.shape} (num_green={num_green}, "
          f"num_lanes={probe_env._num_lanes}) | act=Discrete({num_green})")
    probe_env.close()

    # 하이퍼파라미터 — 한 곳에 모아두고 메타데이터에도 동일하게 기록
    # 변경 이력 (MAPPO_sumo_11 부터):
    #   (B4) entropy_coeff   0.02  → 0.005  — reward magnitude 와 균형 맞춤
    #                                          (이전엔 entropy bonus 가 reward 신호와
    #                                          동일 스케일이라 exploration 편향 과도)
    #   (A3) vf_clip_param   500.0 → 10.0   — diff-waiting-time reward 스케일에 맞춤
    #                                          (이전 500 은 사실상 clip 작동 안 함)
    #   (C4) train_batch_size 4000 → 8000   — gradient noise variance 감소 (∝ 1/N)
    hparams = dict(
        lr=1e-4,
        gamma=0.99,
        train_batch_size=8000,    # C4: 4000 → 8000
        num_epochs=6,
        minibatch_size=128,
        lambda_=0.95,
        clip_param=0.2,
        vf_loss_coeff=0.5,
        entropy_coeff=0.005,      # B4: 0.02 → 0.005
        vf_clip_param=10.0,       # A3: 500 → 10 (reward 스케일에 맞춤)
    )

    config = (
        PPOConfig()
        .environment("sumo_pz", env_config=env_config)
        .framework("torch")
        # num_env_runners=0: driver process에서 직접 rollout 수행 (디버깅 편의)
        # num_env_runners≥1: 별도 worker process 사용 (안정적 병렬 수집)
        .env_runners(num_env_runners=args.num_workers)
        .resources(num_gpus=0)  # M4 Mac: RLlib은 MPS 미지원, CPU 학습
        .callbacks(callbacks_class=_build_callback())  # 진단 지표 RLlib 메트릭 등록
        .multi_agent(
            # MAPPO 핵심: 모든 에이전트가 하나의 shared policy를 공유
            # 다중 교차로 확장 시에도 이 구조 그대로 사용
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
        .training(**hparams)
    )

    algo = config.build_algo()
    # algo.save()는 절대 경로 필요 (pyarrow URI 파싱 때문에 상대 경로 불가)
    out_path = str(Path(args.out if args.out else _next_out_path()).resolve())
    Path(out_path).mkdir(parents=True, exist_ok=True)

    run_name = Path(out_path).name
    tb_writer = SummaryWriter(log_dir=f"results/tb_mappo/{run_name}")

    # 학습 메타데이터 (평가 시 CSV에 복제되어 출처 추적 가능)
    train_meta = {
        "model_name":   run_name,
        "model_path":   out_path,
        "tls_ids":      args.tls_ids,
        "reward_mode":  args.reward_mode,
        "seed":         args.seed,
        "num_workers":  args.num_workers,
        "max_steps":    args.max_steps,
        "delta_time":   args.delta_time,
        "min_green":    args.min_green,
        "yellow_time":  args.yellow_time,
        "sumo_cfg":     sumo_cfg_effective,
        "traffic":      args.traffic,
        # 학습 중 갱신됨
        "train_iter":        0,
        "train_total_steps": 0,
        **hparams,
    }

    def _save_metadata():
        (Path(out_path) / "train_metadata.json").write_text(
            json.dumps(train_meta, indent=2, ensure_ascii=False)
        )

    print(f"학습 시작 | iters={args.num_iters} | workers={args.num_workers} "
          f"| tls={args.tls_ids} | reward={args.reward_mode} | seed={args.seed} | out={out_path}")
    print(f"TensorBoard: tensorboard --logdir results/tb_mappo")

    try:
        for i in range(1, args.num_iters + 1):
            t0 = time.time()
            result = algo.train()
            iter_time = time.time() - t0

            # Ray 2.10+: 지표 경로 변경 (env_runner_results→env_runners, learner_results→learners)
            env_stats    = result.get("env_runners", {})
            mean_rew     = env_stats.get("episode_return_mean", float("nan"))
            ep_len       = env_stats.get("episode_len_mean",    float("nan"))
            total_steps  = result.get("num_env_steps_sampled_lifetime", 0)
            tb_writer.add_scalar("reward/mean",       mean_rew,    i)
            tb_writer.add_scalar("episode/len_mean",  ep_len,      i)
            tb_writer.add_scalar("train/total_steps", total_steps, i)

            # 손실 + grad_norm 지표 (Ray 2.10+: learners 하위)
            policy_stats = result.get("learners", {}).get("shared_policy", {})
            for tag, key in (
                ("loss/total",       "total_loss"),
                ("loss/policy",      "policy_loss"),
                ("loss/value",       "vf_loss"),
                ("loss/entropy",     "entropy"),
                # grad_norm: RLlib 2.10+ 가 자동 기록하는 키들 중 발견되는 것을 사용
                ("learner/grad_norm_global",      "gradients_default_optimizer_global_norm"),
                ("learner/grad_norm_policy",      "gradients_policy_default_optimizer_global_norm"),
            ):
                val = policy_stats.get(key)
                if val is not None:
                    tb_writer.add_scalar(tag, val, i)

            # 콜백이 등록한 커스텀 메트릭 (action_dist, phase_switches, teleported, max_queue)
            # new API stack: env_stats 하위에 평탄화되어 들어옴
            # old API stack: env_stats["custom_metrics"]["<key>_mean"] 형태
            custom = env_stats.get("custom_metrics", {})
            action_keys = tuple(f"action_{j}_ratio" for j in range(num_green))
            for k in ("phase_switches", "teleported", "max_queue") + action_keys:
                # 후보 키 순회: 정확한 키 우선, 이후 _mean 변형
                val = env_stats.get(k, env_stats.get(f"{k}_mean",
                       custom.get(f"{k}_mean", custom.get(k))))
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    if k.startswith("action_"):
                        tb_writer.add_scalar(f"policy/action_dist/{k.replace('_ratio','')}", val, i)
                    else:
                        tb_writer.add_scalar(f"policy/{k}", val, i)

            # 시스템 지표 — 메모리 누수 / SUMO 좀비 프로세스 / iter time
            tb_writer.add_scalar("system/process_rss_mb",     _process_rss_mb(),  i)
            tb_writer.add_scalar("system/sumo_process_count", _sumo_process_count(), i)
            tb_writer.add_scalar("system/iter_time_sec",      iter_time, i)

            ep_str = f"{ep_len:.0f}" if not math.isnan(ep_len) else "nan"
            print(f"Iter {i:4d}/{args.num_iters} | "
                  f"mean_reward={mean_rew:8.2f} | ep_len={ep_str} | "
                  f"steps={int(total_steps)} | "
                  f"rss={_process_rss_mb():.0f}MB | iter_t={iter_time:.1f}s")

            # 메타데이터 진행도 갱신 (checkpoint 시 같이 저장)
            train_meta["train_iter"]        = i
            train_meta["train_total_steps"] = int(total_steps)

            if i % args.checkpoint_freq == 0:
                ckpt = algo.save(str(out_path))
                _save_metadata()
                print(f"  → checkpoint: {ckpt}")

        final_ckpt = algo.save(str(out_path))
        _save_metadata()
        print(f"\n학습 완료. 최종 checkpoint: {final_ckpt}")
        print(f"메타데이터: {out_path}/train_metadata.json")
    finally:
        # 각 cleanup 을 독립적으로 try/except — 하나가 hang/실패해도 다음 진행
        for cleanup_fn, name in (
            (tb_writer.close,  "tb_writer.close"),
            (algo.stop,        "algo.stop"),
            (ray.shutdown,     "ray.shutdown"),
        ):
            try:
                cleanup_fn()
            except Exception as e:
                print(f"[cleanup warning] {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # Ray 2.10+ 의 잔존 worker actor / SUMO 좀비 프로세스가 Python interpreter
    # 종료를 지연시켜 "학습 완료 후 무한 hang" 으로 보이는 문제를 방지.
    # main() 의 finally 에서 정상 cleanup 시도 완료 후, os._exit() 로
    # 모든 자식 프로세스 및 background thread 를 즉시 종료.
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\n학습 중단 (KeyboardInterrupt)")
        exit_code = 130  # 128 + SIGINT(2) — POSIX 관례
    except Exception as e:
        print(f"\n학습 실패: {type(e).__name__}: {e}")
        exit_code = 1
    finally:
        import os
        os._exit(exit_code)
