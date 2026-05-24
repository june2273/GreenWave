# GreenWave — SUMO 교차로 신호제어 강화학습 (MAPPO)

RLlib MAPPO(Multi-Agent PPO)로 SUMO 단일·다중 교차로 신호를 제어하는 실습 프레임워크.  
단일 교차로(`tls_ids=["C"]`)에서 시작해 `--tls-ids` 인자 하나로 다중 교차로로 확장됩니다.

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
├── env_sumo_pz.py          # PettingZoo ParallelEnv — MAPPO 멀티에이전트 환경
├── train_mappo.py          # RLlib MAPPO 학습 (단일/다중 교차로)
├── evaluate_mappo.py       # MAPPO vs Fixed-time 성능 비교 → CSV
├── record_video_mappo.py   # MAPPO 정책 롤아웃 영상 저장
├── sumo_data/              # SUMO 네트워크 XML 파일
│   ├── nodes.nod.xml
│   ├── edges.edg.xml
│   ├── connections.con.xml
│   ├── tls.tll.xml
│   ├── routes.rou.xml
│   └── single.sumocfg
│   # single_intersection.net.xml — 런타임에 netconvert로 자동 생성 (gitignored)
├── models/                 # 학습 체크포인트 (gitignored)
├── results/                # 평가 CSV, TensorBoard 로그 (gitignored)
├── videos/                 # 롤아웃 영상 mp4 (gitignored)
└── samples/                # 커밋된 참조용 샘플 산출물 (README 참조)
```

---

## 환경 설계 (`env_sumo_pz.py`)

### 행동 · 관측 · 보상

| 항목 | 내용 |
|------|------|
| 행동 `Discrete(4)` | 0=North / 1=South / 2=East / 3=West 단독 green |
| 관측 `shape=(10,)` | `[queue×4, speed×4, phase_norm, elapsed_norm]` |
| 보상 | `-(queue_length/10) + (step_throughput × 0.5)` |
| 제약 | phase 변경 시 yellow 삽입, minimum green time 적용 |

### 시스템 구조도

```
RLlib PPO (shared policy)
        ↓ {agent_id: action}
SumoParallelEnv (PettingZoo ParallelEnv)
        ↓ TraCI
   SUMO Simulator
        ↑
 sumo_data/ XML
```

에이전트 매핑: `tl_0 → TLS "C"`, `tl_1 → TLS "D"`, ...  
모든 에이전트가 하나의 shared policy를 공유 → 다중 교차로로 자연스럽게 확장.

---

## 실행 순서

### 1. 학습

```bash
# 빠른 테스트 (20 iter ≈ 80k steps, ~5분)
python train_mappo.py --num-iters 20 --num-workers 0

# 본 학습 (200 iter)
python train_mappo.py --num-iters 200 --num-workers 1

# 저장 경로 직접 지정
python train_mappo.py --num-iters 200 --out models/MAPPO_sumo_exp1

# 다중 교차로 확장
python train_mappo.py --tls-ids C D E --num-workers 3
```

**주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--num-iters` | 200 | 학습 반복 횟수 (1 iter = 4000 스텝 수집 후 10 epoch 업데이트) |
| `--num-workers` | 1 | Ray env runner 수 (0=driver 직접 수행, 디버깅 권장) |
| `--out` | 자동 | 체크포인트 저장 경로 (`models/MAPPO_sumo_N` 자동 버전) |
| `--checkpoint-freq` | 20 | 중간 체크포인트 주기 (iter 단위) |
| `--seed` | 42 | 전역 랜덤 시드 (재현성) |
| `--tls-ids` | `["C"]` | SUMO TLS id 목록 (다중 교차로 확장 시 추가) |

**스텝 수 계산**

```
train_batch_size=4000 × num_iters = 총 env steps
에피소드 1개 ≈ max_steps(3600) ÷ delta_time(5) ≈ 720 steps
iter 1회 ≈ 5~6 에피소드
```

### 2. 평가

```bash
# MAPPO vs Fixed-time 비교 (결과: results/eval_metrics_mappo.csv)
python evaluate_mappo.py --model models/MAPPO_sumo_1 --episodes 5
```

### 3. 롤아웃 영상 저장

```bash
# 영상 저장 (결과: videos/mappo_policy_rollout.mp4)
python record_video_mappo.py --model models/MAPPO_sumo_1
```

---

## TensorBoard

학습 중 또는 학습 후 별도 터미널에서 실행 후 `http://localhost:6006` 접속.

```bash
tensorboard --logdir results/tb_mappo
```

| 지표 | 설명 |
|------|------|
| `reward/mean` | 에피소드 평균 누적 보상 |
| `episode/len_mean` | 평균 에피소드 길이 (steps) |
| `loss/total` | 전체 손실 |
| `loss/policy` | 정책 손실 |
| `loss/value` | 가치함수 손실 |
| `loss/entropy` | 엔트로피 보너스 |

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| `avg_waiting_time` | 완료 차량 평균 누적 대기 시간 (초) |
| `avg_travel_time` | 완료 차량 평균 통행 시간 (초) |
| `total_queue_length` | 에피소드 전체 queue 누적합 |
| `throughput` | 에피소드 내 목적지 도착 차량 수 |

---

## 체크포인트 복원

```python
from ray.rllib.algorithms.ppo import PPO
algo = PPO.from_checkpoint("models/MAPPO_sumo_1")
obs, _ = env.reset()
action = algo.compute_single_action(obs, policy_id="shared_policy")
```

---

## 참고

- `sumo_data/single_intersection.net.xml` 이 없으면 첫 실행 시 `netconvert`로 자동 생성
- RLlib 체크포인트는 디렉터리 형식으로 저장 (`models/MAPPO_sumo_N/`)
- `--num-workers 0`: driver process에서 직접 롤아웃 (디버깅 용이, SUMO 1개)
- `--num-workers N≥1`: 별도 worker process로 SUMO N개 병렬 실행 (학습 속도 향상)
- SUMO 시뮬레이션은 CPU 전용 / RLlib(PyTorch)은 CPU 학습 (M4 Mac MPS 미지원)
- 주요 산출물(`models/`, `results/`, `videos/`)은 gitignored — 참조용 샘플은 `samples/` 참조
