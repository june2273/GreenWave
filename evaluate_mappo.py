"""
MAPPO 평가 — RLlib 체크포인트 기반

MAPPO vs Fixed-time 을 동일 SUMO seed로 쌍대 비교(paired comparison)하여
지표 CSV를 생성. CSV는 두 가지 섹션을 포함:

  1. raw rows     : 에피소드별 측정값 (algorithm × episode)
  2. summary rows : 알고리즘별 mean / std (algorithm == "MAPPO_summary" 등)

또한 모델 메타데이터(hyperparameters, train_iter)가 있으면 모든 행에
복제 기록하여 캡쳐만 보고도 LLM이 출처 모델을 식별 가능.
"""
import argparse
import json
import re
from pathlib import Path


def _versioned_output(model_path: str, template: str, ext: str) -> str:
    """모델 경로에서 버전 번호를 추출해 기본 출력 경로를 생성.

    예) model_path="models/MAPPO_sumo_4", template="results/eval_metrics_mappo"
        → "results/eval_metrics_mappo_4.csv"
    버전 번호를 찾지 못하면 template + ext 를 그대로 반환.
    """
    m = re.search(r"_(\d+)$", Path(model_path).name)
    suffix = f"_{m.group(1)}" if m else ""
    return f"{template}{suffix}{ext}"


def _load_train_metadata(model_path: str) -> dict:
    """모델 디렉터리의 train_metadata.json 을 읽어 dict 반환 (없으면 빈 dict).

    train_mappo.py 가 학습 종료 시 저장하는 파일이며, CSV 모든 행에 복제되어
    캡쳐만 봐도 어떤 모델·하이퍼파라미터인지 LLM이 식별할 수 있게 한다.
    """
    meta_path = Path(model_path) / "train_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


import numpy as np
import pandas as pd
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
    p = argparse.ArgumentParser(description="Evaluate MAPPO vs Fixed-time baseline")
    p.add_argument("--model", type=str, default=None,
                   help="MAPPO 체크포인트 디렉터리 (예: models/MAPPO_sumo_1). "
                        "단순 MAPPO (decentralized critic) 가 학습된 모델. "
                        "미지정 시 --model-ctde 단독 평가 (CTDE vs FixedTime 2-way).")
    p.add_argument("--model-ctde", type=str, default=None,
                   help="(선택) CTDE-MAPPO 체크포인트 디렉터리 (예: models/MAPPO_CTDE_sumo_1). "
                        "--model 과 함께 지정 시 3-way 비교 (FixedTime vs MAPPO vs CTDE) 수행.")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--csv-out", type=str, default=None,
                   help="저장 경로 (미지정 시 --model 또는 --model-ctde 버전 번호로 자동 생성, "
                        "예: results/eval_metrics_mappo_4.csv)")
    p.add_argument("--baseline-phase-steps", type=int, default=3,
                   help="Fixed-time baseline: N 스텝마다 phase 순환")
    p.add_argument("--max-steps", type=int, default=3600)
    p.add_argument("--delta-time", type=int, default=5)
    p.add_argument("--min-green", type=int, default=13,
                   help="학습 시와 동일해야 함 (train_mappo.py 와 default 일치)")
    p.add_argument("--yellow-time", type=int, default=2)
    p.add_argument("--map", type=str, default="single", choices=MAP_CHOICES,
                   help="시나리오 사전셋. 학습 시 사용한 --map 과 일치해야 함.")
    p.add_argument("--reward-mode", type=str, default="diff-waiting-time",
                   choices=["diff-waiting-time"],
                   help="보상 모드. 현재 diff-waiting-time 단일 모드만 지원.")
    p.add_argument("--sumo-cfg", type=str, default=None,
                   help="SUMO 설정 파일 경로 (학습 시와 동일하게 지정)")
    p.add_argument("--traffic", type=str, default="default",
                   choices=["default", "high"],
                   help="2x2grid 트래픽 강도 사전셋 (train_mappo.py 와 동일 의미). "
                        "high 선택 시 --sumo-cfg 미지정이면 2x2grid_dense.sumocfg 자동 사용.")
    p.add_argument("--brt-weight", type=float, default=None,
                   help="env 에 전달할 BRT 가중치. "
                        "미지정 시 model 의 train_metadata.json 에서 자동 로드 (없으면 1.0). "
                        "명시 시 metadata 보다 우선. baseline 비교 시 1.0 명시 권장.")
    p.add_argument("--sample", action="store_true",
                   help="argmax 대신 Categorical sampling 으로 행동 선택 "
                        "(train 과 동일, 확률 mass 그대로 반영). "
                        "기본 argmax 는 mode collapse 시 비주류 action 을 절대 선택 안 함 — "
                        "--sample 로 정책의 진짜 분포를 반영한 평가 가능. "
                        "재현성 위해 episode 별 seed 가 torch 시드로도 사용됨.")
    args = p.parse_args()
    if args.model is None and args.model_ctde is None:
        p.error("--model 또는 --model-ctde 중 하나 이상을 지정해야 합니다.")
    return args


def _action_ratios(action_counts: list) -> dict:
    """action_counts 리스트 → 비율 dict {action_k_ratio: ...}.

    길이가 num_green 만큼 가변. mode collapse 진단: 균등 분포에 가까우면
    1/num_green 씩, 한쪽 쏠림이면 1.0 근처.
    """
    counts = np.asarray(action_counts, dtype=np.float64)
    n = len(counts)
    total = counts.sum()
    if total <= 0:
        return {f"action_{i}_ratio": 0.0 for i in range(n)}
    ratios = counts / total
    return {f"action_{i}_ratio": float(ratios[i]) for i in range(n)}


def _row_from_info(algorithm: str, ep: int, seed: int, info: dict) -> dict:
    """env.step() 마지막 info → CSV row dict 변환 (진단 컬럼 평탄화)."""
    row = {
        "algorithm": algorithm,
        "episode":   ep,
        "seed":      seed,
        "avg_waiting_time":   info.get("avg_waiting_time",   np.nan),
        "std_waiting_time":   info.get("std_waiting_time",   np.nan),
        "avg_travel_time":    info.get("avg_travel_time",    np.nan),
        "total_queue_length": info.get("total_queue_length", np.nan),
        "throughput":         info.get("throughput",         np.nan),
        "phase_switches":     info.get("phase_switches",     np.nan),
        "max_queue":          info.get("max_queue",          np.nan),
        "teleported":         info.get("teleported",         np.nan),
        # Insertion-failure 가시화 — teleport 비활성 시 silent drop 추적용
        "vehicles_loaded":     info.get("vehicles_loaded",     np.nan),
        "vehicles_departed":   info.get("vehicles_departed",   np.nan),
        "vehicles_lost_insert":info.get("vehicles_lost_insert",np.nan),
        "pending_insert_peak": info.get("pending_insert_peak", np.nan),
        "pending_insert_final":info.get("pending_insert_final",np.nan),
        # 옵션 C: yellow_ratio = yellow / total sim sec — oscillation 진단
        "yellow_seconds":      info.get("yellow_seconds",      np.nan),
        "yellow_ratio":        info.get("yellow_ratio",        np.nan),
        # Green Wave Tier 1 직접 지표
        "avg_stops_per_vehicle": info.get("avg_stops_per_vehicle", np.nan),
        "avg_co2_per_vehicle":   info.get("avg_co2_per_vehicle",   np.nan),
        "avg_speed":             info.get("avg_speed",             np.nan),
        # Green Wave Tier 2 코리도어 지표
        "speed_cv":              info.get("speed_cv",              np.nan),
        "per_direction_wait_ew": info.get("per_direction_wait_ew", np.nan),
        "per_direction_wait_ns": info.get("per_direction_wait_ns", np.nan),
        # BRT 우선처리 평가용 — vClass=bus 와 일반 차량 분리 metric.
        # 비-BRT 시나리오는 brt_seen=0 / avg_wait_brt=0.0 (자연 표시).
        "avg_wait_brt":  info.get("avg_wait_brt",  np.nan),
        "avg_wait_car":  info.get("avg_wait_car",  np.nan),
        "avg_speed_brt": info.get("avg_speed_brt", np.nan),
        "avg_speed_car": info.get("avg_speed_car", np.nan),
        "brt_seen":      info.get("brt_seen",      np.nan),
        "car_seen":      info.get("car_seen",      np.nan),
    }
    row.update(_action_ratios(info.get("action_counts", [])))
    return row


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

    # CTDE 체크포인트 역직렬화에 CentralizedCriticPPOModule 이 필요 — 항상 선임포트
    try:
        from ctde_module import CentralizedCriticPPOModule  # noqa: F401
    except ImportError:
        pass

    ray.init(ignore_reinit_error=True)

    module = None
    train_meta: dict = {}
    if args.model is not None:
        module = _load_rl_module(str(Path(args.model).resolve()))
        # 모델 학습 메타데이터 (없으면 빈 dict) — CSV 모든 행에 prefix로 부착
        train_meta = _load_train_metadata(args.model)

    module_ctde = None
    train_meta_ctde: dict = {}
    if args.model_ctde:
        module_ctde = _load_rl_module(str(Path(args.model_ctde).resolve()))
        train_meta_ctde = _load_train_metadata(args.model_ctde)
        if not train_meta_ctde.get("ctde_mode"):
            print(
                "[warning] --model-ctde 로 지정된 체크포인트의 train_metadata.json 에 "
                "ctde_mode=True 가 없습니다. CTDE 체크포인트가 맞는지 확인하세요."
            )

    # --map 으로부터 sumo_cfg / tls_ids 결정 (preset 사용)
    sumo_cfg_effective, tls_ids_effective = resolve_map_args(
        map_name=args.map,
        sumo_cfg_arg=args.sumo_cfg,
        tls_ids_arg=None,
        traffic=args.traffic,
    )
    print(f"[map={args.map}] sumo_cfg={sumo_cfg_effective} tls_ids={tls_ids_effective}")

    # 학습 시 사용한 map 과 eval map 이 다르면 경고
    if train_meta.get("map") and train_meta["map"] != args.map:
        print(f"[warning] train map='{train_meta['map']}' vs eval map='{args.map}' 불일치")

    # BRT 가중치 결정: CLI 명시값 > train_metadata > default(1.0)
    # 학습 시와 동일한 reward 환경에서 평가하기 위함. 베이스라인 비교 시
    # --brt-weight 1.0 명시 권장.
    if args.brt_weight is not None:
        brt_weight_effective = float(args.brt_weight)
    else:
        brt_weight_effective = float(train_meta.get("brt_weight", 1.0))
    print(f"[brt_weight={brt_weight_effective}]")

    env_kwargs = dict(
        use_gui=False,
        delta_time=args.delta_time,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        max_steps=args.max_steps,
        tls_ids=tls_ids_effective,
        reward_mode=args.reward_mode,
        brt_weight=brt_weight_effective,
    )
    if sumo_cfg_effective:
        env_kwargs["sumo_cfg"] = sumo_cfg_effective

    rows = []

    # 환경 인스턴스 — MAPPO/FixedTime 은 ctde_mode=False, CTDE 는 True (Dict obs).
    # 모든 환경이 같은 seed 로 reset 되어 SUMO 차량 수요가 동일하게 재현됨.
    env_mappo = SumoParallelEnv(**env_kwargs, ctde_mode=False) if module is not None else None
    env_fix   = SumoParallelEnv(**env_kwargs, ctde_mode=False)
    env_ctde = (
        SumoParallelEnv(**env_kwargs, ctde_mode=True, ctde_shared_reward=True)
        if module_ctde is not None else None
    )

    # 동적 METRIC_COLS: num_green에 맞춰 action_*_ratio 컬럼 수 조정
    # (단일·2x2grid 모두 num_green=4 이지만 다른 네트워크 대응)
    _ref_env = env_mappo if env_mappo is not None else (env_ctde if env_ctde is not None else env_fix)
    num_green = _ref_env._num_green
    metric_cols = [
        "avg_waiting_time", "std_waiting_time", "avg_travel_time",
        "total_queue_length", "throughput",
        "phase_switches", "max_queue", "teleported",
        # Insertion-failure 컬럼 — dense traffic 시 좌회전 lane saturation 가시화
        "vehicles_loaded", "vehicles_departed", "vehicles_lost_insert",
        "pending_insert_peak", "pending_insert_final",
        # 옵션 C: oscillation 진단
        "yellow_seconds", "yellow_ratio",
        # Green Wave 지표 — 모든 algorithm 에 공통 기록
        "avg_stops_per_vehicle", "avg_co2_per_vehicle", "avg_speed",
        "speed_cv", "per_direction_wait_ew", "per_direction_wait_ns",
        # BRT 우선처리 metric — BRT 시나리오에서 BRT vs 일반 차량 분리 분석
        "avg_wait_brt", "avg_wait_car", "avg_speed_brt", "avg_speed_car",
        "brt_seen", "car_seen",
        *[f"action_{i}_ratio" for i in range(num_green)],
    ]

    def _select_action(logits):
        """args.sample=True: Categorical sampling (train 분포 그대로).
        False: argmax (deterministic, 기본).
        """
        if args.sample:
            return int(torch.distributions.Categorical(logits=logits).sample().item())
        return int(torch.argmax(logits, dim=-1).item())

    def mappo_action(obs_dict, step_idx, agents):
        return {
            agent: _select_action(
                module.forward_inference(
                    {"obs": torch.tensor(obs_dict[agent][None], dtype=torch.float32)}
                )["action_dist_inputs"]
            )
            for agent in agents
        }

    def ctde_action(obs_dict, step_idx, agents):
        """CTDE 모델 — obs 는 flat Box [local | global concat].

        모듈이 forward 시 첫 local_dim 차원만 slice 해 actor 에 사용
        (decentralized execution). 호출자는 평소처럼 ndarray 만 전달하면 됨.
        """
        return {
            agent: _select_action(
                module_ctde.forward_inference(
                    {"obs": torch.tensor(obs_dict[agent][None], dtype=torch.float32)}
                )["action_dist_inputs"]
            )
            for agent in agents
        }

    def fixed_action(obs_dict, step_idx, agents):
        phase = int((step_idx // args.baseline_phase_steps) % 4)
        return {agent: phase for agent in agents}

    try:
        # ── 쌍대 비교: 동일 ep에서 동일 seed 사용 ────────────────────────────
        # 같은 seed → SUMO가 동일한 차량 수요를 생성하므로
        # 알고리즘 성능 차이만 측정 가능 (교통 수요 샘플 효과 제거)
        for ep in range(args.episodes):
            seed = args.seed + ep

            # --sample 모드: episode 별 torch 시드 고정 → Categorical sampling 재현 가능
            # (env seed 와 별도로 torch 난수 stream 을 episode 마다 reset)
            if args.sample:
                torch.manual_seed(seed)

            # ── MAPPO (--model 지정 시에만) ───────────────────────────────────
            if env_mappo is not None:
                info = run_episode(env_mappo, mappo_action, seed=seed)
                rows.append(_row_from_info("MAPPO", ep, seed, info))
                r = rows[-1]
                act_dist_str = ",".join(f"{r.get(f'action_{j}_ratio', 0.0):.2f}"
                                        for j in range(num_green))
                print(f"[MAPPO]     ep={ep} seed={seed} | "
                      f"wait={r['avg_waiting_time']:.1f}s | "
                      f"queue={r['total_queue_length']:.0f} | "
                      f"stops/veh={r['avg_stops_per_vehicle']:.2f} | "
                      f"co2/veh={r['avg_co2_per_vehicle']:.0f} | "
                      f"speed={r['avg_speed']:.2f} | "
                      f"act_dist=[{act_dist_str}] | "
                      f"teleport={r['teleported']:.0f} | "
                      f"lost_insert={r['vehicles_lost_insert']:.0f} "
                      f"(pending_peak={r['pending_insert_peak']:.0f}) | "
                      f"yellow={r['yellow_ratio']*100:.1f}% "
                      f"(switches={r['phase_switches']:.0f})")

            # ── CTDE (선택, 동일 seed) ────────────────────────────────────────
            if env_ctde is not None:
                info = run_episode(env_ctde, ctde_action, seed=seed)
                rows.append(_row_from_info("CTDE", ep, seed, info))
                r = rows[-1]
                print(f"[CTDE]      ep={ep} seed={seed} | "
                      f"wait={r['avg_waiting_time']:.1f}s | "
                      f"queue={r['total_queue_length']:.0f} | "
                      f"stops/veh={r['avg_stops_per_vehicle']:.2f} | "
                      f"co2/veh={r['avg_co2_per_vehicle']:.0f} | "
                      f"speed={r['avg_speed']:.2f} | "
                      f"teleport={r['teleported']:.0f} | "
                      f"lost_insert={r['vehicles_lost_insert']:.0f} "
                      f"(pending_peak={r['pending_insert_peak']:.0f})")

            # ── FixedTime (동일 seed) ─────────────────────────────────────────
            info = run_episode(env_fix, fixed_action, seed=seed)
            rows.append(_row_from_info("FixedTime", ep, seed, info))
            r = rows[-1]
            print(f"[FixedTime] ep={ep} seed={seed} | "
                  f"wait={r['avg_waiting_time']:.1f}s | "
                  f"queue={r['total_queue_length']:.0f} | "
                  f"stops/veh={r['avg_stops_per_vehicle']:.2f} | "
                  f"co2/veh={r['avg_co2_per_vehicle']:.0f} | "
                  f"speed={r['avg_speed']:.2f} | "
                  f"teleport={r['teleported']:.0f} | "
                  f"lost_insert={r['vehicles_lost_insert']:.0f} "
                  f"(pending_peak={r['pending_insert_peak']:.0f}) | "
                  f"yellow={r['yellow_ratio']*100:.1f}% "
                  f"(switches={r['phase_switches']:.0f})")
    finally:
        # 각 cleanup 독립 실행 — Ray 2.10+ ray.shutdown() hang 회피
        cleanup_targets = []
        if env_mappo is not None:
            cleanup_targets.append((env_mappo.close, "env_mappo.close"))
        cleanup_targets.append((env_fix.close, "env_fix.close"))
        if env_ctde is not None:
            cleanup_targets.append((env_ctde.close, "env_ctde.close"))
        cleanup_targets.append((ray.shutdown, "ray.shutdown"))
        for cleanup_fn, name in cleanup_targets:
            try:
                cleanup_fn()
            except Exception as e:
                print(f"[cleanup warning] {name}: {type(e).__name__}: {e}")

    # ── 요약 행 추가 (algorithm별 mean / std) ─────────────────────────────
    # CTDE 는 --model-ctde 지정 시에만 rows 에 존재하므로 동적으로 포함
    df_raw = pd.DataFrame(rows)
    algo_names = []
    if module is not None:
        algo_names.append("MAPPO")
    if module_ctde is not None:
        algo_names.append("CTDE")
    algo_names.append("FixedTime")
    summary_rows = []
    for algo_name in algo_names:
        sub = df_raw[df_raw["algorithm"] == algo_name]
        if sub.empty:
            continue
        means = sub[metric_cols].mean(numeric_only=True)
        stds  = sub[metric_cols].std(numeric_only=True, ddof=0)
        summary_rows.append({
            "algorithm": f"{algo_name}_mean",
            "episode": "", "seed": "",
            **{c: means[c] for c in metric_cols},
        })
        summary_rows.append({
            "algorithm": f"{algo_name}_std",
            "episode": "", "seed": "",
            **{c: stds[c] for c in metric_cols},
        })
    df = pd.concat([df_raw, pd.DataFrame(summary_rows)], ignore_index=True)

    # ── 메타데이터 prefix 컬럼 (모든 행에 동일값) ─────────────────────────
    # 캡쳐만 보고도 LLM이 어느 모델·hyperparameter 결과인지 식별 가능
    meta_prefix = {
        "model_path":        (str(Path(args.model).resolve()) if args.model else ""),
        "model_path_ctde":   (str(Path(args.model_ctde).resolve())
                              if args.model_ctde else ""),
        "train_iter":        train_meta.get("train_iter", ""),
        "train_iter_ctde":   train_meta_ctde.get("train_iter", ""),
        "train_total_steps": train_meta.get("train_total_steps", ""),
        "reward_mode":       train_meta.get("reward_mode", args.reward_mode),
        "ctde_reward":       train_meta_ctde.get("ctde_reward", ""),
        "lr":                train_meta.get("lr", ""),
        "entropy_coeff":     train_meta.get("entropy_coeff", ""),
        "vf_clip_param":     train_meta.get("vf_clip_param", ""),
        "switch_penalty":    train_meta.get("switch_penalty", ""),
        "brt_weight":        brt_weight_effective,
        "min_green":         train_meta.get("min_green", args.min_green),
        "map":               train_meta.get("map", args.map),
        "train_seed":        train_meta.get("seed", ""),
        "tls_ids":           "_".join(tls_ids_effective),
        "num_workers":       train_meta.get("num_workers", ""),
        "traffic":           train_meta.get("traffic", args.traffic),
    }
    for k, v in meta_prefix.items():
        df.insert(0, k, v)

    # ── 저장 및 출력 ──────────────────────────────────────────────────────
    _version_src = args.model or args.model_ctde or "results/eval_metrics_mappo"
    csv_out = args.csv_out or _versioned_output(_version_src, "results/eval_metrics_mappo", ".csv")
    out_path = Path(csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    if args.model:
        print(f"Train metadata: {'loaded' if train_meta else 'not found (older model)'}")
    if env_ctde is not None:
        print(f"CTDE metadata:  {'loaded' if train_meta_ctde else 'not found'}")
    print("\n=== Summary (mean ± std) ===")
    for algo_name in algo_names:
        sub = df_raw[df_raw["algorithm"] == algo_name][metric_cols]
        if sub.empty:
            continue
        print(f"\n[{algo_name}]")
        for col in ("avg_waiting_time", "avg_travel_time",
                    "total_queue_length", "throughput",
                    "phase_switches", "teleported",
                    # Insertion-failure — silent vehicle drop 진단
                    "vehicles_loaded", "vehicles_departed",
                    "vehicles_lost_insert", "pending_insert_peak",
                    # 옵션 C: oscillation 진단
                    "yellow_ratio",
                    # Green Wave 핵심 지표 — CTDE 우월성 검증용
                    "avg_stops_per_vehicle", "avg_co2_per_vehicle",
                    "avg_speed", "speed_cv",
                    "per_direction_wait_ew", "per_direction_wait_ns"):
            m, s = sub[col].mean(), sub[col].std(ddof=0)
            print(f"  {col:24s}: {m:>10.3f} ± {s:.3f}")


if __name__ == "__main__":
    # Ray 2.10+ 의 잔존 worker actor 가 Python interpreter 종료를 지연시키는
    # 문제 방지 — main() finally 의 cleanup 후 os._exit 으로 강제 종료
    import os, sys
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\n평가 중단 (KeyboardInterrupt)")
        exit_code = 130
    except Exception as e:
        import traceback
        print(f"\n평가 실패: {type(e).__name__}: {e}")
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(exit_code)
