# GreenWave — SUMO 교차로 신호제어 강화학습

단일 교차로에서 DQN(SB3)과 MAPPO(RLlib)를 실험하고,
다중 교차로 MAPPO로 확장할 수 있는 실습 프레임워크.

---

## 시스템 요구사항

- macOS (Apple Silicon M-series 권장)
- Python 3.10+
- SUMO 1.20+

---

## 설치

### 1. SUMO 바이너리 (Homebrew)

```bash
brew install sumo
```

설치 확인

```bash
sumo --version && netconvert --version
```

### 2. Python 패키지

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

설치 확인

```bash
python -c "import traci, sumolib; print('SUMO Python OK')"
python -c "import ray, pettingzoo; print('RLlib OK')"
```

---

## 파일 구조

```
GreenWave/
├── env_sumo_single.py   # Gymnasium Env — SB3 DQN용 단일 에이전트 환경
├── env_sumo_pz.py       # PettingZoo ParallelEnv — RLlib MAPPO용 멀티에이전트 환경
├── train_dqn.py         # SB3 DQN 학습
├── train_mappo.py       # RLlib MAPPO 학습 (다중 교차로 확장 가능)
├── evaluate.py          # DQN vs Fixed-time 성능 비교 → CSV
├── record_video.py      # DQN 정책 롤아웃 영상 저장
├── sumo_data/           # SUMO 네트워크 XML 파일
│   ├── nodes.nod.xml
│   ├── edges.edg.xml
│   ├── connections.con.xml
│   ├── tls.tll.xml
│   ├── routes.rou.xml
│   └── single.sumocfg
├── models/              # 저장된 모델 (DQN .zip, MAPPO 디렉터리)
├── results/             # 평가 CSV, TensorBoard 로그
│   ├── eval_metrics.csv
│   ├── tb_dqn/
│   └── tb_mappo/
└── videos/              # 롤아웃 영상 (mp4)
```

---

## 환경 설계

### 행동 · 관측 · 보상 (공통)

| 항목 | 내용 |
|------|------|
| 행동 `Discrete(4)` | 0=North / 1=South / 2=East / 3=West 단독 green |
| 관측 `shape=(10,)` | `[queue×4, speed×4, phase_norm, elapsed_norm]` |
| 보상 | `−total_queue_length` (진입 차선 대기 차량 합) |
| 제약 | phase 변경 시 yellow 삽입, minimum green time 적용 |

### 두 환경 클래스 비교

| | `env_sumo_single.py` | `env_sumo_pz.py` |
|---|---|---|
| 기반 | `gymnasium.Env` | `pettingzoo.ParallelEnv` |
| 에이전트 | 단일 | `tl_0`, `tl_1`, ... |
| 알고리즘 | SB3 DQN | RLlib MAPPO |
| 다중 교차로 | ✗ | ✅ `tls_ids` 인자 추가 |

### 시스템 구조도

```
┌─────────────────────────────────────────────────────────────┐
│  DQN (SB3)                    MAPPO (RLlib)                 │
│                                                              │
│  SB3 DQN Agent                RLlib PPO (shared policy)     │
│       ↓ action (Discrete 4)        ↓ {agent: action}        │
│  SumoSingleIntersectionEnv    SumoParallelEnv (PettingZoo)  │
│       ↓ TraCI                      ↓ TraCI                  │
│       └──────── SUMO Simulator ────┘                        │
│                      ↑                                       │
│               sumo_data/ XML                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 실행 순서

### DQN (SB3 / 단일 에이전트)

```bash
# 1. 학습 (모델: models/dqn_sumo_single.zip)
python train_dqn.py --timesteps 120000

# 2. DQN vs Fixed-time 비교 평가 (결과: results/eval_metrics.csv)
python evaluate.py --model models/dqn_sumo_single.zip --episodes 5

# 3. 롤아웃 영상 저장 (결과: videos/dqn_policy_rollout.mp4)
python record_video.py --model models/dqn_sumo_single.zip
```

### MAPPO (RLlib / 멀티에이전트)

저장 경로는 `models/MAPPO_sumo_N` 형식으로 자동 버전 생성됩니다.

```bash
# 빠른 테스트 (20 iter ≈ 80k steps, ~5분)
python train_mappo.py --num-iters 20 --num-workers 0

# 본 학습 (DQN 300k 스텝 수준 비교)
python train_mappo.py --num-iters 100 --num-workers 1

# 저장 경로 직접 지정
python train_mappo.py --num-iters 100 --out models/MAPPO_sumo_exp1

# 다중 교차로 확장 (추후)
python train_mappo.py --tls-ids C D E --num-workers 3
```

**주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--num-iters` | 200 | 학습 반복 횟수 (1 iter = 4000 스텝 수집 후 10 epoch 업데이트) |
| `--num-workers` | 1 | Ray env runner 수 (0=driver 직접 수행, 디버깅 권장) |
| `--out` | 자동 | 체크포인트 저장 경로 |
| `--checkpoint-freq` | 20 | 중간 체크포인트 주기 (iter 단위) |
| `--tls-ids` | `["C"]` | SUMO TLS id 목록 (다중 교차로 확장 시 추가) |

**스텝 수 계산**

```
train_batch_size=4000 × num_iters = 총 env steps
에피소드 1개 ≈ max_steps(3600) ÷ delta_time(5) ≈ 720 steps
iter 1회 ≈ 5~6 에피소드
```

**체크포인트 복원**

```python
from ray.rllib.algorithms.ppo import PPO
algo = PPO.from_checkpoint("models/MAPPO_sumo_1")
obs, _ = env.reset()
action = algo.compute_single_action(obs, policy_id="shared_policy")
```

---

## TensorBoard

학습 중 또는 학습 후 별도 터미널에서 실행 후 `http://localhost:6006` 접속.

```bash
# DQN + MAPPO 동시 비교
tensorboard --logdir results/

# MAPPO만
tensorboard --logdir results/tb_mappo

# DQN만
tensorboard --logdir results/tb_dqn
```

| 알고리즘 | 로그 경로 | 주요 지표 |
|----------|-----------|-----------|
| DQN | `results/tb_dqn/` | `rollout/ep_rew_mean`, `train/loss` |
| MAPPO | `results/tb_mappo/MAPPO_sumo_N/` | `reward/mean`, `loss/total`, `loss/policy`, `loss/value`, `loss/entropy` |

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| `avg_waiting_time` | 완료 차량 평균 누적 대기 시간 (초) |
| `avg_travel_time` | 완료 차량 평균 통행 시간 (초) |
| `total_queue_length` | 에피소드 전체 queue 누적합 |
| `throughput` | 에피소드 내 목적지 도착 차량 수 |

---

## 참고

- SUMO 네트워크 파일(`single_intersection.net.xml`)이 없으면 `netconvert`로 자동 생성
- SB3 모델은 `.zip` 아카이브, RLlib 체크포인트는 디렉터리 형식으로 저장
- `--num-workers 0`: driver process에서 직접 롤아웃 (디버깅 용이, 단일 SUMO 인스턴스)
- `--num-workers N≥1`: 별도 worker process로 SUMO N개 병렬 실행 (학습 속도 향상)
- SUMO 시뮬레이션은 CPU 전용 / RLlib(PyTorch)은 CPU 학습 (M4 Mac MPS 미지원)
