import argparse
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

try:
    from .env_sumo_single import SumoSingleIntersectionEnv
except ImportError:
    from env_sumo_single import SumoSingleIntersectionEnv


def parse_args():
    """학습 실행 인자 파싱"""
    p = argparse.ArgumentParser(description="Train DQN on SUMO single intersection")
    # 총 학습 스텝 수
    p.add_argument("--timesteps", type=int, default=200_000)
    # 랜덤 시드
    p.add_argument("--seed", type=int, default=42)
    # 저장 파일 경로 확장자 없이 지정 시 SB3가 .zip 생성
    p.add_argument("--model-out", type=str, default="models/dqn_sumo_single")
    # 에피소드 최대 시뮬레이션 스텝
    p.add_argument("--max-steps", type=int, default=3600)
    # action 1회당 진행할 시뮬레이션 시간(초)
    p.add_argument("--delta-time", type=int, default=5)
    # phase 최소 유지 시간(초)
    p.add_argument("--min-green", type=int, default=10)
    # phase 전환 시 yellow 유지 시간(초)
    p.add_argument("--yellow-time", type=int, default=2)
    return p.parse_args()


def main():
    """DQN 학습 실행 엔트리 포인트"""
    args = parse_args()

    # SUMO 단일 교차로 환경 생성
    env = SumoSingleIntersectionEnv(
        use_gui=False,
        delta_time=args.delta_time,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        max_steps=args.max_steps,
    )
    # Monitor 래퍼: episode return/length 등 학습 로그 기록
    env = Monitor(env)

    # DQN 모델 구성
    # MlpPolicy: 벡터 관측 입력용 기본 MLP 정책 네트워크
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,  # optimizer 학습률
        buffer_size=50_000,  # 리플레이 버퍼 최대 크기
        learning_starts=1_000,  # 버퍼가 일정량 쌓인 후 학습 시작
        batch_size=64,  # 한 번 업데이트에 사용하는 샘플 수
        gamma=0.99,  # 미래 보상 할인율
        train_freq=4,  # 환경 스텝 기준 학습 주기
        target_update_interval=500,  # 타깃 Q 네트워크 동기화 주기
        exploration_fraction=0.3,  # epsilon-greedy 탐험 비율 스케줄
        exploration_final_eps=0.05,
        verbose=1,
        seed=args.seed,
        tensorboard_log="./results/tb_dqn",  # TensorBoard 로그 저장 경로
    )

    # 학습 실행
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    # 학습 모델 저장(.zip)
    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))

    # TraCI 연결 정리
    env.close()
    print(f"Saved model to: {out_path}.zip")


if __name__ == "__main__":
    main()
