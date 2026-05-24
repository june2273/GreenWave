# GreenWave — SUMO 교차로 신호제어 강화학습 (MAPPO)

RLlib MAPPO(Multi-Agent PPO)로 SUMO 단일·다중 교차로 신호를 제어하는 실습 프레임워크.  
단일 교차로(`tls_ids=["C"]`)에서 시작해 `--tls-ids` + `--sumo-cfg` 인자로 임의 SUMO 네트워크(2×2 격자 등)로 확장됩니다.

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
│   ├── single.sumocfg      # 기본 단일교차로 설정
│   └── 2x2grid.sumocfg     # SUMO-RL 2x2 격자 교차로 설정 (TLS: "1","2","5","6")
│   # single_intersection.net.xml — 첫 실행 시 netconvert로 자동 생성 (gitignored)
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
| 관측 `shape=(10,)` | `[queue×4, speed×4, phase_norm, elapsed_norm]` (동적 lane 감지, 최대 4 lane) |
| 보상 `--reward-mode` | `queue`: `−(mean+max)/10 + throughput×0.5` (기본, 기아 방향 추가 패널티) |
| | `diff-waiting-time`: 누적 대기시간 변화량 (SUMO-RL 검증 방식) |
| | `pressure`: 출력 lane − 입력 lane 차량 수 (처리량 직접 최대화) |
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

에이전트 매핑: `tl_0 → TLS "C"`, `tl_1 → TLS "D"`, ...  (2×2 격자: `tl_0→"1"`, `tl_1→"2"`, `tl_2→"5"`, `tl_3→"6"`)  
모든 에이전트가 하나의 shared policy를 공유 → 다중 교차로로 자연스럽게 확장.  
Lane은 `trafficlight.getControlledLanes()` 로 동적 감지 — 임의 SUMO 네트워크에서 동작.

---

## 실행 순서

### 1. 학습

```bash
# 빠른 테스트 (20 iter ≈ 80k steps, ~5분)
python train_mappo.py --num-iters 20 --num-workers 0

# 본 학습 (200 iter, 재현 가능)
python train_mappo.py --num-iters 200 --num-workers 1 --seed 42

# 저장 경로 직접 지정
python train_mappo.py --num-iters 200 --out models/MAPPO_sumo_exp1

# 다중 교차로 확장
python train_mappo.py --tls-ids C D E --num-workers 3

# 보상 함수 변경 (SUMO-RL 방식)
python train_mappo.py --reward-mode diff-waiting-time --num-iters 100
python train_mappo.py --reward-mode pressure --num-iters 100

# 2x2 격자 교차로 (SUMO-RL 네트워크 사용)
python train_mappo.py --tls-ids 1 2 5 6 --sumo-cfg sumo_data/2x2grid.sumocfg --num-workers 4
```

**주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--num-iters` | 200 | 학습 반복 횟수 (1 iter = 4000 스텝 수집 후 6 epoch 업데이트) |
| `--num-workers` | 1 | Ray env runner 수 (0=driver 직접 수행, 디버깅 권장) |
| `--out` | 자동 | 체크포인트 저장 경로 (`models/MAPPO_sumo_N` 자동 버전) |
| `--checkpoint-freq` | 20 | 중간 체크포인트 주기 (iter 단위) |
| `--tls-ids` | `["C"]` | SUMO TLS id 목록 (2x2 격자: `1 2 5 6`) |
| `--seed` | 42 | 전역 랜덤 시드 (random / numpy / torch 일괄 설정) |
| `--reward-mode` | `queue` | 보상 함수: `queue` / `diff-waiting-time` / `pressure` |
| `--sumo-cfg` | 자동 | SUMO 설정 파일 경로 (미지정 시 기본 단일교차로 사용) |

**스텝 수 계산**

```
train_batch_size=4000 × num_iters = 총 env steps
에피소드 1개 ≈ max_steps(3600) ÷ delta_time(5) ≈ 720 steps
iter 1회 ≈ 5~6 에피소드
```

**PPO 하이퍼파라미터 (현재 설정)**

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `lr` | `1e-4` | policy 진동 억제 (이전: 3e-4) |
| `num_epochs` | `6` | 배치당 SGD 업데이트 (이전: 10) |
| `vf_loss_coeff` | `0.5` | VF 급학습 시 advantage 급변 완화 (이전: 1.0) |
| `entropy_coeff` | `0.02` | 조기 수렴 방지 (이전: 0.01) |
| `train_batch_size` | `4000` | 수집 스텝 수 |
| `minibatch_size` | `128` | SGD 미니배치 크기 |
| `gamma` | `0.99` | 할인율 |
| `lambda_` | `0.95` | GAE λ |
| `clip_param` | `0.2` | PPO clip ε |
| `vf_clip_param` | `500.0` | VF clip (discounted return 범위 ≈ −300~−500) |

### 2. 평가

```bash
# MAPPO vs Fixed-time 쌍대 비교
# 출력: results/eval_metrics_mappo_N.csv  (N = 모델 버전 번호 자동)
python evaluate_mappo.py --model models/MAPPO_sumo_1 --episodes 5

# 학습 시 tls-ids / reward-mode / sumo-cfg 를 변경했다면 동일하게 전달
python evaluate_mappo.py --model models/MAPPO_sumo_1 --tls-ids C --reward-mode queue

# 2x2 격자 평가
python evaluate_mappo.py --model models/MAPPO_sumo_7 \
  --tls-ids 1 2 5 6 --sumo-cfg sumo_data/2x2grid.sumocfg
```

> **쌍대 비교**: 동일 에피소드 인덱스에 동일 SUMO seed를 사용해  
> 교통 수요 차이가 아닌 정책 자체의 성능 차이만 측정합니다.  
> CSV에 `seed` 컬럼이 포함되어 쌍 확인이 가능합니다.

### 3. 롤아웃 영상 저장

```bash
# 출력: videos/mappo_policy_rollout_N.mp4  (N = 모델 버전 번호 자동)
python record_video_mappo.py --model models/MAPPO_sumo_1

# 학습 시 tls-ids / reward-mode / sumo-cfg 를 변경했다면 동일하게 전달
python record_video_mappo.py --model models/MAPPO_sumo_1 --tls-ids C --reward-mode queue
```

### 4. 2×2 격자 교차로 (SUMO-RL 네트워크)

SUMO-RL 라이브러리의 2×2 격자 네트워크를 GreenWave에서 그대로 사용할 수 있습니다.

**네트워크 정보**

| 항목 | 내용 |
|------|------|
| 네트워크 | `sumo_rl/nets/2x2grid/2x2.net.xml` (SUMO-RL) |
| TLS IDs | `"1"` (좌상), `"2"` (우상), `"5"` (좌하), `"6"` (우하) |
| 트래픽 수요 | 4방향 통과 flow (확률 0.1/step) |
| 설정 파일 | `sumo_data/2x2grid.sumocfg` (GreenWave 내 포함) |

```bash
# 2×2 격자 학습 (4 에이전트, 4 workers 권장)
python train_mappo.py \
  --tls-ids 1 2 5 6 \
  --sumo-cfg sumo_data/2x2grid.sumocfg \
  --reward-mode diff-waiting-time \
  --num-workers 4 \
  --num-iters 100 \
  --seed 42

# 2×2 격자 평가
python evaluate_mappo.py \
  --model models/MAPPO_sumo_N \
  --tls-ids 1 2 5 6 \
  --sumo-cfg sumo_data/2x2grid.sumocfg \
  --reward-mode diff-waiting-time \
  --episodes 5
```

> **주의**: `sumo_data/2x2grid.sumocfg`는 SUMO-RL 설치 경로의 절대경로를 참조합니다.  
> 경로가 다르다면 파일 내 `net-file` / `route-files` 값을 실제 경로로 수정하세요.

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
import torch
from ray.rllib.algorithms.ppo import PPO

algo = PPO.from_checkpoint("models/MAPPO_sumo_1")
module = algo.get_module("shared_policy")   # RLModule API (Ray 2.10+)

obs, _ = env.reset()
action = int(torch.argmax(
    module.forward_inference(
        {"obs": torch.tensor(obs[None], dtype=torch.float32)}
    )["action_dist_inputs"],
    dim=-1,
).item())
```

---

## 참고

- `sumo_data/single_intersection.net.xml` 이 없으면 기본 단일교차로 첫 실행 시 `netconvert`로 자동 생성 (임시 파일 → atomic rename, 병렬 worker 충돌 방지). `--sumo-cfg` 지정 시 자동 생성 불필요 — 호출하지 않음
- RLlib 체크포인트는 디렉터리 형식으로 저장 (`models/MAPPO_sumo_N/`)
- `--num-workers 0`: driver process에서 직접 롤아웃 (디버깅 용이, SUMO 1개)
- `--num-workers N≥1`: 별도 worker process로 SUMO N개 병렬 실행 (학습 속도 향상)
- SUMO 시뮬레이션은 CPU 전용 / RLlib(PyTorch)은 CPU 학습 (M4 Mac MPS 미지원)
- `--seed` 로 random / numpy / torch 시드를 일괄 설정해 실험 재현성 확보
- 잘못된 `--tls-ids` 전달 시 TraCI 연결 직후 명확한 에러 메시지로 조기 종료
- 평가 출력 파일은 모델 버전을 자동 반영 (`eval_metrics_mappo_N.csv`, `mappo_policy_rollout_N.mp4`)
- 주요 산출물(`models/`, `results/`, `videos/`)은 gitignored — 참조용 샘플은 `samples/` 참조
- `--reward-mode diff-waiting-time`: 누적 대기시간 변화량 기반 보상 (SUMO-RL 검증 방식, 방향 기아 방지)
- `--reward-mode pressure`: 출력−입력 차량 수 차이 (처리량 직접 최대화)
- 학습·평가·영상 스크립트 모두 `--reward-mode` / `--sumo-cfg` 인자 일치 필요
- **2x2 격자**: `sumo_data/2x2grid.sumocfg` → SUMO-RL 네트워크 파일 참조 (TLS: 1,2,5,6)
  lane은 `trafficlight.getControlledLanes()` 로 동적 감지 — 임의 SUMO 네트워크 지원
