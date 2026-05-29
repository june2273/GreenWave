"""
Fixed-Time 베이스라인 신호 영상 저장 — RL 모델 없이 고정 사이클 신호로 SUMO 시뮬레이션 녹화

record_video_mappo.py 의 자매 스크립트. RLModule 추론 대신 고정신호 정책으로
SumoParallelEnv 를 굴려 mp4 인코딩.

두 가지 베이스라인:
  --baseline symmetric  : 3-way 비교용 균등 N-step 사이클 (모든 agent 동기, 4×15s)
  --baseline sejong     : 세종시 실측 per-TLS 비대칭 (SEJONG_PER_TLS_PHASE_SECONDS, 6 TLS)

evaluate_mappo.py 의 fixed_action / fixed_action_sejong 와 동일한 phase 매핑 사용 →
비디오와 평가 CSV 가 같은 행동을 묘사 (재현성 보장).
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio

try:
    from .env_sumo_pz import SumoParallelEnv
    from .map_presets import MAP_CHOICES, resolve_map_args
except ImportError:
    from env_sumo_pz import SumoParallelEnv
    from map_presets import MAP_CHOICES, resolve_map_args


# ─ 세종시 실측 신호 타이밍 (evaluate_mappo.py 와 동일 — 1:1 복제) ──────────
# 출처:
#   세담터 (세종특별자치시_신호등.csv, 2026.04, 1순위)
#   공공데이터포털 (세종특별자치시_교차로 신호 표준데이터, 2023, 2순위)
#   미확인 교차로 (도움4로): 같은 corridor 내 실측 교차로의 신호 사용 (추정)
#
# 4-phase 매핑 [NS_SR, NS_L, EW_SR, EW_L] (단위: 초):
SEJONG_PER_TLS_PHASE_SECONDS = {
    "tl_0": [45, 25, 45, 25],  # 성금 (한누리/BRT)
    "tl_1": [46, 20, 31, 23],  # 청사 (갈매)
    "tl_2": [45, 25, 45, 25],  # 도움4로(한누리), 성금 추정
    "tl_3": [46, 20, 31, 23],  # 도움4로(갈매), 청사 추정
    "tl_4": [47, 20, 47, 28],  # 어진 (한누리/BRT)
    "tl_5": [45, 20, 35, 20],  # 가름로 (갈매, 세담터2026)
}


def _phase_from_secs(phase_secs, step_idx, dt):
    """[NS_SR, NS_L, EW_SR, EW_L] (초) + 현재 step_idx → phase 인덱스 (0~3).
    evaluate_mappo.py 의 동명 함수와 동일. 초→step 환산은 floor + 최소 1 step.
    """
    phase_steps = [max(1, s // dt) for s in phase_secs]
    cycle_len = sum(phase_steps)
    boundaries, acc = [], 0
    for s in phase_steps:
        boundaries.append(acc)
        acc += s
    cycle_pos = step_idx % cycle_len
    phase = 0
    for i in range(len(boundaries) - 1, -1, -1):
        if cycle_pos >= boundaries[i]:
            phase = i
            break
    return phase


def parse_args():
    p = argparse.ArgumentParser(
        description="Record Fixed-Time baseline signal control as mp4 (no RL model)"
    )
    # ── 베이스라인 종류 ───────────────────────────────────────────
    p.add_argument("--baseline", type=str, default="symmetric",
                   choices=["symmetric", "sejong"],
                   help="symmetric: N-step 균등 사이클 (3-way 비교 영상용). "
                        "sejong: 세종시 실측 per-TLS 비대칭 (3x2-brt 최종 영상용).")
    p.add_argument("--baseline-phase-steps", type=int, default=3,
                   help="symmetric 전용: N 스텝마다 phase 순환 "
                        "(default 3 × delta_time 5 = 15s/phase).")
    p.add_argument("--sejong-phase-seconds", type=int, nargs=4, default=None,
                   help="sejong 전용 단일 set 강제: [NS_SR NS_L EW_SR EW_L]. "
                        "미명시 시 SEJONG_PER_TLS_PHASE_SECONDS (6 TLS 각각 다른 cycle).")

    # ── 출력 / 시각화 ───────────────────────────────────────────
    p.add_argument("--output", type=str, default=None,
                   help="저장 경로 (미지정 시 baseline+map 으로 자동: "
                        "videos/fixed_symmetric_<map>.mp4 또는 videos/fixed_sejong_<map>.mp4).")
    p.add_argument("--fps", type=int, default=10,
                   help="continuous 기본 10fps (~2분 영상). short 모드는 5fps 권장.")
    p.add_argument("--mode", type=str, default="continuous",
                   choices=["short", "continuous"],
                   help="short: env.step() 마다 1 frame. "
                        "continuous: 매 sim step 마다 1 frame (yellow phase 포착).")
    p.add_argument("--seed", type=int, default=777,
                   help="MAPPO/CTDE 비디오와 동일 seed 사용 권장 (같은 trip).")
    p.add_argument("--dump-metrics", type=str, default=None,
                   help="프레임별 실시간 지표 (co2_kg/avg_wait/cur_wait/throughput) 를 "
                        "JSON 으로 저장. frames 와 1:1 정렬됨 (3-way 비교 영상 오버레이용).")

    # ── 환경 파라미터 (학습/평가 default 와 일치) ─────────────
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--delta-time", type=int, default=5)
    p.add_argument("--min-green", type=int, default=13)
    p.add_argument("--yellow-time", type=int, default=3)
    p.add_argument("--brt-weight", type=float, default=1.0,
                   help="env 에 전달할 BRT 가중치 (영상 자체에는 영향 없음, info 진단용).")

    # ── 시나리오 ────────────────────────────────────────────────
    p.add_argument("--map", type=str, default="2x2-brt", choices=MAP_CHOICES,
                   help="시나리오 사전셋. 3-way 비교: 2x2-brt. 세종 최종: 3x2-brt.")
    p.add_argument("--sumo-cfg", type=str, default=None,
                   help="preset 무시하고 직접 지정 (학습/평가 시와 동일하게).")
    p.add_argument("--traffic", type=str, default="default",
                   choices=["default", "high"],
                   help="3x2-brt sejong 비디오는 --traffic high 필수.")

    return p.parse_args()


def _default_output(baseline: str, map_name: str) -> str:
    tag = map_name.replace("-", "_")
    return f"videos/fixed_{baseline}_{tag}.mp4"


def main():
    args = parse_args()

    # ── 정책 함수 정의 ───────────────────────────────────────────
    if args.baseline == "symmetric":
        def policy_fn(step_idx, agents):
            phase = int((step_idx // args.baseline_phase_steps) % 4)
            return {agent: phase for agent in agents}
        label = "FixedTime(Symmetric)"
    else:  # sejong
        dt = args.delta_time

        if args.sejong_phase_seconds is not None:
            # 단일 set 강제 — 모든 agent 동일
            phase_secs = args.sejong_phase_seconds
            def policy_fn(step_idx, agents):
                phase = _phase_from_secs(phase_secs, step_idx, dt)
                return {agent: phase for agent in agents}
        else:
            # per-TLS 실측 — agent 별 다른 cycle
            def policy_fn(step_idx, agents):
                result = {}
                for agent in agents:
                    secs = SEJONG_PER_TLS_PHASE_SECONDS.get(agent, [33, 20, 33, 20])
                    result[agent] = _phase_from_secs(secs, step_idx, dt)
                return result
        label = "FixedTime(Sejong)"

    # ── 환경 셋업 ────────────────────────────────────────────────
    sumo_cfg_effective, tls_ids_effective = resolve_map_args(
        map_name=args.map,
        sumo_cfg_arg=args.sumo_cfg,
        tls_ids_arg=None,
        traffic=args.traffic,
    )
    print(f"[{label}] map={args.map} sumo_cfg={sumo_cfg_effective} tls_ids={tls_ids_effective}")

    env_kwargs = dict(
        use_gui=False,
        delta_time=args.delta_time,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        max_steps=args.max_steps,
        tls_ids=tls_ids_effective,
        reward_mode="diff-waiting-time",
        brt_weight=args.brt_weight,
    )
    if sumo_cfg_effective:
        env_kwargs["sumo_cfg"] = sumo_cfg_effective
    env = SumoParallelEnv(**env_kwargs)

    # ── 롤아웃 + 프레임 캡쳐 ─────────────────────────────────────
    try:
        obs_dict, _ = env.reset(seed=args.seed)
        frames = [env.render()]
        metrics = [env.live_metrics()] if args.dump_metrics else None

        if args.mode == "continuous":
            def _capture(sim_step):
                frames.append(env.render())
                if metrics is not None:
                    metrics.append(env.live_metrics())
            env.add_step_hook(_capture)

        step_idx = 0
        last_info = {}
        while env.agents:
            actions = policy_fn(step_idx, env.agents)
            obs_dict, _, _, _, info_dict = env.step(actions)
            if args.mode == "short":
                frames.append(env.render())
                if metrics is not None:
                    metrics.append(env.live_metrics())
            if info_dict:
                # 임의 agent info 보관 (env-wide 키만 사용)
                last_info = next(iter(info_dict.values()))
            step_idx += 1
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[cleanup warning] env.close: {type(e).__name__}: {e}")

    # ── mp4 인코딩 ───────────────────────────────────────────────
    output = args.output or _default_output(args.baseline, args.map)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=args.fps)
    duration = len(frames) / max(1, args.fps)
    print(f"Saved video: {out_path}  ({len(frames)} frames @ {args.fps}fps, "
          f"~{duration:.1f}s, mode={args.mode}, baseline={args.baseline})")

    if args.dump_metrics and metrics is not None:
        import json
        mpath = Path(args.dump_metrics)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with open(mpath, "w") as f:
            json.dump({"fps": args.fps, "frames": len(frames), "samples": metrics}, f)
        print(f"Saved metrics: {mpath}  ({len(metrics)} samples, "
              f"frames={len(frames)} → {'aligned' if len(metrics)==len(frames) else 'MISALIGNED!'})")

    # 진단 정보 (info 키 일부 출력)
    if last_info:
        diag_keys = ("vehicles_loaded", "vehicles_departed", "phase_switches",
                     "yellow_seconds", "max_queue", "avg_wait_brt", "avg_wait_car")
        diag = {k: last_info.get(k) for k in diag_keys if k in last_info}
        if diag:
            print(f"  diag: {diag}")


if __name__ == "__main__":
    # SUMO 좀비 프로세스가 interpreter 종료를 지연시키는 문제 방지
    # (record_video_mappo.py 동일 패턴)
    import os, sys, traceback
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\n비디오 녹화 중단 (KeyboardInterrupt)")
        exit_code = 130
    except Exception as e:
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
