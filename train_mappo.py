"""
Ray RLlib MAPPO — SUMO 교차로 신호제어 학습

단일 교차로(tl_0 하나)에서도 Parallel PettingZoo + shared policy 구조를 사용하므로
다중 교차로 확장은 --map preset 으로 처리됩니다 (single / 2x2 / 2x2-brt / 3x2 / 3x2-brt).
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
    from .map_presets import MAP_CHOICES, resolve_map_args
except ImportError:
    from env_sumo_pz import SumoParallelEnv
    from map_presets import MAP_CHOICES, resolve_map_args


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
                # 옵션 C: oscillation 진단용 yellow_ratio + insertion-failure 가시화
                self._log_metric(episode, metrics_logger,
                                 "yellow_ratio", float(last_info.get("yellow_ratio", 0)))
                self._log_metric(episode, metrics_logger,
                                 "vehicles_lost_insert",
                                 float(last_info.get("vehicles_lost_insert", 0)))

                # BRT 우선처리 metric (BRT 시나리오에서만 의미. 그 외는 0.0)
                for k in ("avg_wait_brt", "avg_wait_car",
                          "avg_speed_brt", "avg_speed_car"):
                    if k in last_info:
                        self._log_metric(episode, metrics_logger,
                                         k, float(last_info.get(k, 0.0)))

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
    p.add_argument("--min-green", type=int, default=13,
                   help="phase 최소 유지 sec. delta_time=5 기준 3 env step floor. "
                        "oscillation 억제를 위해 13 권장.")
    p.add_argument("--yellow-time", type=int, default=3)
    # 시나리오 선택 — sumo_cfg + default tls_ids 자동 결정
    p.add_argument("--map", type=str, default="single", choices=MAP_CHOICES,
                   help="시나리오 사전셋. single=단일교차로, 2x2=sumo-rl 2x2, "
                        "2x2-brt=좌측 BRT corridor, 3x2=2col x 3row, "
                        "3x2-brt=3x2+좌측 BRT corridor. "
                        "--sumo-cfg 명시 시 그것이 우선.")
    p.add_argument("--seed", type=int, default=42,
                   help="전역 랜덤 시드 (random/numpy/torch/SUMO 일괄 설정)")
    p.add_argument("--reward-mode", type=str, default="diff-waiting-time",
                   choices=["diff-waiting-time"],
                   help="보상 함수 모드 (diff-waiting-time: 이전 step 누적대기시간 - 현재) / 10. "
                        "현재 단일 모드만 지원 (queue/pressure 는 cleanup 으로 제거됨).")
    p.add_argument("--switch-penalty", type=float, default=0.45,
                   help="phase switch 마다 reward 에서 빼는 페널티 (oscillation 억제). "
                        "0 = 비활성, 0.45 = yellow 3sec × 1대/sec 손실 추정 (yellow_time=3 기본값에 맞춤).")
    p.add_argument("--brt-weight", type=float, default=1.0,
                   help="BRT(vClass=bus) 차량 waiting time 가중치. "
                        "1.0 = 가중치 비활성 (baseline, 기존 동작과 bit-identical). "
                        "BRT 시나리오(2x2-brt, 3x2-brt) 학습 시 1.5~3.0 권장. "
                        "reward_mode='diff-waiting-time' 에서만 reward 에 반영됨. "
                        "평가 metric (avg_wait_brt/car, avg_speed_brt/car) 분리는 "
                        "이 값과 무관하게 항상 기록.")
    p.add_argument("--sumo-cfg", type=str, default=None,
                   help="SUMO 설정 파일 경로 (미지정 시 기본 단일교차로 사용)")
    p.add_argument("--traffic", type=str, default="default",
                   choices=["default", "high"],
                   help="2x2grid 트래픽 강도 사전셋 ("
                        "default: 원본 sumo-rl routes (~0.4대/초, free flow), "
                        "high: routes_2x2_dense (~1.4대/초, 정체). "
                        "high 선택 시 --sumo-cfg 미지정이면 2x2grid_dense.sumocfg 자동 사용. "
                        "--sumo-cfg 명시 시 그것이 우선.)")
    p.add_argument("--ctde", action="store_true",
                   help="Centralized critic (CTDE) 모드 활성화. "
                        "Actor 는 local obs, Critic 은 전체 agent obs concat 을 봄. "
                        "Green Wave 형 협조 학습 유도. 체크포인트는 MAPPO_CTDE_sumo_N 에 저장.")
    p.add_argument("--ctde-reward", type=str, default="shared",
                   choices=["shared", "local"],
                   help="--ctde 와 함께 사용. shared: 모든 agent 가 mean(local rewards) 받음 "
                        "(global coordination). local: 기존 per-agent reward 유지.")
    return p.parse_args()


def main():
    args = parse_args()

    # 전역 시드 설정 (재현성 확보)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    register_env("sumo_pz", _make_env)
    ray.init(ignore_reinit_error=True)

    # --map 으로부터 sumo_cfg / tls_ids 결정 (preset 사용)
    sumo_cfg_effective, tls_ids_effective = resolve_map_args(
        map_name=args.map,
        sumo_cfg_arg=args.sumo_cfg,
        tls_ids_arg=None,
        traffic=args.traffic,
    )
    print(f"[map={args.map}] sumo_cfg={sumo_cfg_effective} tls_ids={tls_ids_effective}")

    env_config = {
        "use_gui": False,
        "delta_time": args.delta_time,
        "min_green": args.min_green,
        "yellow_time": args.yellow_time,
        "max_steps": args.max_steps,
        "tls_ids": tls_ids_effective,
        "reward_mode": args.reward_mode,
        "ctde_mode": bool(args.ctde),
        "ctde_shared_reward": (args.ctde_reward == "shared"),
        "switch_penalty": args.switch_penalty,
        "brt_weight": args.brt_weight,
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
    # CTDE module 이 flat Box obs 에서 local 부분만 slice 하려면 local_dim 필요.
    # ctde_mode 일 때 obs_space.shape = (local + global concat) 이라 raw _obs_dim 사용.
    local_dim_for_module = probe_env._obs_dim
    print(f"환경 spec | obs={obs_space.shape} (num_green={num_green}, "
          f"num_lanes={probe_env._num_lanes}, local_dim={local_dim_for_module}) "
          f"| act=Discrete({num_green})")
    probe_env.close()

    # 하이퍼파라미터 — 한 곳에 모아두고 메타데이터에도 동일하게 기록
    # 변경 이력 (MAPPO_sumo_11 부터):
    #   (B4) entropy_coeff   0.02  → 0.005  — reward magnitude 와 균형 맞춤
    #                                          (이전엔 entropy bonus 가 reward 신호와
    #                                          동일 스케일이라 exploration 편향 과도)
    #   (B5) entropy_coeff   0.005 → 0.03  — dense traffic 의 좌회전 phase 가 학습
    #                                          초기 부정 reward → 확률 0 으로 collapse
    #                                          후 영구 미선택 되는 mode collapse 진단됨
    #                                          (eval_metrics_mappo_17 에서 모든 ep act_dist
    #                                          이 NS·EW 직진 phase 에만 집중, 좌회전 ratio
    #                                          ~0). entropy bonus 6배 강화 (0.005 → 0.03)
    #                                          로 4개 phase 모두 충분히 탐색하게 함.
    #                                          (RLlib 버전 안전성 위해 schedule 대신 scalar.
    #                                          collapse 가 풀린 뒤 fine-tune 시 0.005 로
    #                                          낮춰 exploit 단계 별도 진행 권장.)
    #   (A3) vf_clip_param   500.0 → 10.0 → 1000.0
    #                                          10.0 으로 낮췄더니 vf_loss_unclipped=175189
    #                                          대비 vf_loss=9.89 로 VF 신호 거의 전부 소실
    #                                          → diff-waiting-time 실제 reward 스케일에 맞춰
    #                                            1000.0 으로 복원
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
        entropy_coeff=0.03,       # B5: 0.005 → 0.03 (mode collapse 방지, 좌회전 phase 탐색)
        vf_clip_param=1000.0,     # A3: 500 → 10 → 1000 (VF signal 복원)
    )

    config = (
        PPOConfig()
        .environment("sumo_pz", env_config=env_config)
        .framework("torch")
        # num_env_runners=0: driver process에서 직접 rollout 수행 (디버깅 편의)
        # num_env_runners≥1: 별도 worker process 사용 (안정적 병렬 수집)
        # sample_timeout_s=1800: 다중 교차로(2x2-brt, 3x2 등) 1 iter rollout이
        #   600s를 초과해 빈 sample 반환 → train_total_steps=0/NaN 증상 발생.
        #   30분으로 상향 (driver mode num_workers=0 은 무관)
        .env_runners(
            num_env_runners=args.num_workers,
            sample_timeout_s=1800.0,
        )
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

    # ── CTDE: centralized critic 모듈 주입 ───────────────────────────────
    # obs 는 flat Box [local(D) | global(D*N)]. module 이 forward 시 slice.
    # Dict obs 는 RLlib worker process 의 connector pipeline 에서 silent fail
    # 하므로 사용하지 않음 (total_steps=0 증상). PPOCatalog 도 우회 (분리 encoder).
    if args.ctde:
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from ctde_module import CentralizedCriticPPOModule
        config = config.rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": RLModuleSpec(
                        module_class=CentralizedCriticPPOModule,
                        observation_space=obs_space,
                        action_space=act_space,
                        model_config={
                            "hidden_dim": 128,
                            # module 이 flat Box 에서 local 부분 slice 할 때 사용
                            "local_dim": local_dim_for_module,
                        },
                    )
                }
            )
        )

    algo = config.build_algo()
    # algo.save()는 절대 경로 필요 (pyarrow URI 파싱 때문에 상대 경로 불가)
    # CTDE 모드면 MAPPO_CTDE_sumo_N, 아니면 기존 MAPPO_sumo_N
    algo_prefix = "MAPPO_CTDE" if args.ctde else "MAPPO"
    out_path = str(Path(
        args.out if args.out else _next_out_path(algo=algo_prefix)
    ).resolve())
    Path(out_path).mkdir(parents=True, exist_ok=True)

    run_name = Path(out_path).name
    tb_writer = SummaryWriter(log_dir=f"results/tb_mappo/{run_name}")

    # 학습 메타데이터 (평가 시 CSV에 복제되어 출처 추적 가능)
    train_meta = {
        "model_name":   run_name,
        "model_path":   out_path,
        "map":          args.map,
        "tls_ids":      tls_ids_effective,
        "reward_mode":  args.reward_mode,
        "seed":         args.seed,
        "num_workers":  args.num_workers,
        "max_steps":    args.max_steps,
        "delta_time":   args.delta_time,
        "min_green":    args.min_green,
        "yellow_time":  args.yellow_time,
        "sumo_cfg":     sumo_cfg_effective,
        "traffic":      args.traffic,
        "switch_penalty": args.switch_penalty,
        "brt_weight":   args.brt_weight,
        "ctde_mode":    bool(args.ctde),
        "ctde_reward":  args.ctde_reward if args.ctde else None,
        # 학습 중 갱신됨
        "train_iter":        0,
        "train_total_steps": 0,
        **hparams,
    }

    def _save_metadata():
        (Path(out_path) / "train_metadata.json").write_text(
            json.dumps(train_meta, indent=2, ensure_ascii=False)
        )

    ctde_tag = (
        f"CTDE (reward={args.ctde_reward})" if args.ctde else "MAPPO (decentralized critic)"
    )
    print(f"학습 시작 [{ctde_tag}] | iters={args.num_iters} | workers={args.num_workers} "
          f"| tls={tls_ids_effective} | reward={args.reward_mode} | seed={args.seed} | out={out_path}")
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
                # PPO KL divergence — policy update 과격함 진단 (>0.05 면 lr/clip 조정)
                ("policy/kl_mean",   "mean_kl_loss"),
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
            for k in ("phase_switches", "teleported", "max_queue",
                      "yellow_ratio", "vehicles_lost_insert") + action_keys:
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
    import os, sys
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\n학습 중단 (KeyboardInterrupt)")
        exit_code = 130  # 128 + SIGINT(2) — POSIX 관례
    except Exception as e:
        import traceback
        print(f"\n학습 실패: {type(e).__name__}: {e}")
        traceback.print_exc()
        exit_code = 1
    finally:
        # os._exit() 는 stdio buffer 를 flush 하지 않으므로 명시적으로 flush
        # (이전엔 학습 로그 print 가 buffer 에 남은 채 종료되어 보이지 않았음)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(exit_code)
