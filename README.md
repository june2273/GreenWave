# GreenWave — SUMO 교차로 신호제어 강화학습 (MAPPO)

RLlib MAPPO(Multi-Agent PPO)로 SUMO 단일·다중 교차로 신호를 제어하는 실습 프레임워크.
단일 교차로(`tls_ids=["C"]`)에서 시작해 `--tls-ids` + `--sumo-cfg`(또는 `--traffic`) 인자로 임의 SUMO 네트워크(2×2 격자 등)로 확장됩니다.

**핵심 설계**:
- SUMO TLS phase 자동 감지 → action space `Discrete(num_green)` 동적 결정 (단일/2x2grid/임의 네트워크 자동 호환)
- SUMO program 자동 cycle 차단으로 agent action 만이 phase 전환을 결정
- 양방향 동시 green / 좌회전 / yellow 신호를 정확히 시각화 (matplotlib)
- 학습/평가 CSV에 메타데이터·진단 지표 자동 기록 (mode collapse, memory leak, grid lock 진단)

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

환경 변수 (TraCI/sumolib 사용에 필수):

```bash
export SUMO_HOME=$(brew --prefix sumo)/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

설치 확인:

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

설치 확인:

```bash
python -c "import traci, sumolib; print('SUMO Python OK')"
python -c "import ray, pettingzoo, matplotlib; print('RLlib + Renderer OK')"
```

> **참고**: `psutil` 은 Ray transitive dependency 로 자동 설치되며, train_mappo 의 메모리/프로세스 모니터링에 사용됩니다.

---

## 파일 구조

```
GreenWave/
├── env_sumo_pz.py             # PettingZoo ParallelEnv — MAPPO 멀티에이전트 환경
├── train_mappo.py             # RLlib MAPPO 학습 (단일/다중 교차로)
├── evaluate_mappo.py          # MAPPO vs Fixed-time 성능 비교 → CSV
├── record_video_mappo.py      # MAPPO 정책 롤아웃 영상 저장 (short/continuous)
├── sumo_renderer.py           # matplotlib 기반 SUMO 네트워크 시각화
├── sumo_data/
│   ├── nodes.nod.xml          # 단일교차로 노드
│   ├── edges.edg.xml          # 단일교차로 엣지
│   ├── connections.con.xml    # 단일교차로 connection
│   ├── tls.tll.xml            # 단일교차로 TLS (4 green + 1 yellow phase)
│   ├── routes.rou.xml         # 단일교차로 traffic
│   ├── single.sumocfg         # 단일교차로 SUMO 설정
│   ├── 2x2grid.sumocfg        # 2x2grid 기본 (sumo-rl routes, ~0.4대/초)
│   ├── 2x2grid_dense.sumocfg  # 2x2grid 정체 (~1.4대/초, --traffic high)
│   └── routes_2x2_dense.rou.xml  # 양방향 + 좌회전 cross flow
│   # single_intersection.net.xml — 첫 실행 시 netconvert로 자동 생성 (gitignored)
├── models/                    # 학습 체크포인트 (gitignored)
│   └── MAPPO_sumo_N/train_metadata.json  # 학습 메타데이터 (자동 저장)
├── results/
│   ├── eval_metrics_mappo_N.csv      # 평가 결과
│   └── tb_mappo/MAPPO_sumo_N/        # TensorBoard 로그
├── videos/                    # 롤아웃 영상 mp4 (gitignored)
└── samples/                   # 커밋된 참조용 샘플 산출물
```

---

## 환경 설계 (`env_sumo_pz.py`)

### Action Space — `Discrete(num_green)` 자동 결정

네트워크의 TLS phase 중 **green phase 만** 자동 감지해 action 으로 매핑:

| 네트워크 | green phases | Discrete | action → phase 매핑 |
|----------|-------------|----------|---------------------|
| 단일교차로 | `[0, 1, 2, 3]` (모든 phase가 green) | `Discrete(4)` | action 0~3 → phase 0~3 |
| 2x2grid (sumo-rl) | `[0, 2, 4, 6]` (8 phase 중 4개) | `Discrete(4)` | action 0~3 → phase 0/2/4/6 |

phase 전환 시 SUMO connection 의 `dir` 속성 (s/l/r/t) 기반으로 자동 yellow 삽입.

### Observation Space — SUMO-RL 표준 호환

```
[phase_one_hot(num_green), min_green_flag(1), density_per_lane(L), queue_per_lane(L)]
shape = num_green + 1 + 2 * L
```

| 네트워크 | num_green | controlled lanes (L) | shape |
|----------|-----------|----------------------|-------|
| 단일교차로 | 4 | 4 (또는 8) | (13,) |
| 2x2grid | 4 | 8 | (21,) |

- `phase_one_hot`: 현재 green phase 인덱스 one-hot
- `min_green_flag`: 현재 phase가 min_green 시간 충족했는지 (0/1)
- `density_per_lane`: lane별 차량 수 / lane 용량 (lane_length / 7.5m), [0, 1] 정규화
- `queue_per_lane`: lane별 halting 차량 수 / lane 용량, [0, 1] 정규화

### Reward Modes (`--reward-mode`)

| 모드 | 공식 | 특성 |
|------|------|------|
| `queue` | `−(mean(queues)+max(queues))/10 + throughput×0.5` | 기아 방향 패널티 + 처리량 보너스 |
| `diff-waiting-time` | `prev_wait − current_wait` (스케일 `/10.0`) | 대기시간 감소 = + 보상 (SUMO-RL 검증) |
| `pressure` | `out_vehicles − in_vehicles` | 처리량 직접 최대화 |

> **(B1) Reward Scaling 변경**: diff-waiting-time 정규화가 `/100` → `/10` 으로 강화됨 (이전 magnitude 가 너무 작아 `loss/value` 가 0 근처에 정체되던 문제 해결).

### SUMO Program 자동 cycle 차단 (D2)

SUMO TLS program 은 `setPhase()` 호출 후에도 phase duration 만료 시 자동으로 다음 phase 로 진행하는데, 이는 agent 의 의도와 무관한 phase 전환을 유발해 학습 신호를 노이즈화합니다.

**해결**: `_set_phase_locked()` 헬퍼가 `setPhase()` 직후 `setPhaseDuration(tls_id, 100000s)` 를 호출하여 자동 진행을 차단. **agent action 만이 phase 전환의 유일한 결정 요소**가 됩니다.

### 시스템 구조도

```
   RLlib PPO (shared policy)
            ↓ {agent_id: action}
   SumoParallelEnv (PettingZoo ParallelEnv)
       ↓ traci.setPhase + setPhaseDuration(lock)
       SUMO Simulator
            ↑
      sumo_data/ XML
```

- 에이전트 매핑: `tl_0 → TLS "C"`, `tl_1 → TLS "D"`, ... (2×2 격자: `tl_0→"1"`, `tl_1→"2"`, `tl_2→"5"`, `tl_3→"6"`)
- 모든 에이전트가 **하나의 shared policy** 공유 → 다중 교차로로 자연 확장
- Lane은 `trafficlight.getControlledLanes()` 로 동적 감지 — 임의 SUMO 네트워크 자동 호환

---

## 실행 순서

### 1. 학습

```bash
# 빠른 검증 (10 iter, ~10분, 단일교차로)
python train_mappo.py --num-iters 10 --reward-mode diff-waiting-time --seed 42

# 본 학습 — 단일교차로 (150 iter)
python train_mappo.py --num-iters 150 --num-workers 1 --seed 42

# 다중 교차로 — 2x2grid 기본 트래픽
python train_mappo.py --tls-ids 1 2 5 6 \
  --sumo-cfg sumo_data/2x2grid.sumocfg \
  --reward-mode diff-waiting-time \
  --num-iters 150 --num-workers 4 --seed 42

# 다중 교차로 — 2x2grid 정체 시나리오 (--traffic high)
python train_mappo.py --tls-ids 1 2 5 6 \
  --traffic high \
  --reward-mode diff-waiting-time \
  --num-iters 150 --num-workers 4 --seed 42
```

#### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--num-iters` | 200 | 학습 반복 횟수 (1 iter = `train_batch_size` 스텝 수집 후 업데이트) |
| `--num-workers` | 1 | Ray env runner 수 (0=driver 직접, 디버깅용) |
| `--out` | 자동 | 체크포인트 경로 (미지정 시 `models/MAPPO_sumo_N` 자동 버전) |
| `--checkpoint-freq` | 20 | 중간 체크포인트 주기 (iter 단위) |
| `--tls-ids` | `["C"]` | SUMO TLS id 목록 (2x2 격자: `1 2 5 6`) |
| `--seed` | 42 | 전역 랜덤 시드 (random/numpy/torch 일괄) |
| `--reward-mode` | `queue` | `queue` / `diff-waiting-time` / `pressure` |
| `--sumo-cfg` | 자동 | SUMO 설정 파일 (미지정 + `--traffic default` 시 단일교차로) |
| `--traffic` | `default` | `default`: 원본 routes / `high`: 정체 시나리오 (2x2grid_dense.sumocfg 자동 사용) |
| `--max-steps` | 3600 | 에피소드 최대 sim 초 |
| `--delta-time` | 5 | env step당 진행 sim 초 |
| `--min-green` | 10 | 최소 green 유지 시간 (초) |
| `--yellow-time` | 2 | green→green 전환 시 yellow 시간 (초) |

#### `--traffic` 사전셋 비교

| 옵션 | sumocfg | 도착률 | 특성 |
|------|---------|--------|------|
| `default` | `2x2grid.sumocfg` (sumo-rl) | ~0.4대/초 (4 flow × 0.1) | free flow |
| `high` | `2x2grid_dense.sumocfg` | ~1.4대/초 (양방향 + 좌회전 cross) | 정체 시나리오 |

> `--sumo-cfg` 명시 시 `--traffic` 보다 우선. high 모드에 다른 sumocfg 쓰려면 `--sumo-cfg` 로 override.

#### PPO 하이퍼파라미터 (현재 설정)

| 파라미터 | 값 | 이전 → 변경 이유 |
|---------|-----|------|
| `lr` | `1e-4` | (그대로) policy 진동 억제 |
| `train_batch_size` | **`8000`** | 4000 → 8000 (C4: gradient noise 감소) |
| `num_epochs` | `6` | (그대로) |
| `minibatch_size` | `128` | (그대로) |
| `gamma` | `0.99` | (그대로) |
| `lambda_` | `0.95` | (그대로) GAE λ |
| `clip_param` | `0.2` | (그대로) PPO clip ε |
| `vf_loss_coeff` | `0.5` | (그대로) |
| `entropy_coeff` | **`0.005`** | 0.02 → 0.005 (B4: reward 스케일과 균형) |
| `vf_clip_param` | **`10.0`** | 500 → 10 (A3: diff-waiting-time reward 스케일 적합) |

### 2. 평가

```bash
# MAPPO vs Fixed-time 쌍대 비교
# 출력: results/eval_metrics_mappo_N.csv  (N = 모델 버전 번호 자동)
python evaluate_mappo.py --model models/MAPPO_sumo_1 --episodes 5

# 학습 시와 동일한 tls-ids / reward-mode / traffic 전달
python evaluate_mappo.py --model models/MAPPO_sumo_11 \
  --tls-ids 1 2 5 6 --traffic high \
  --reward-mode diff-waiting-time --episodes 5
```

> **쌍대 비교**: 동일 에피소드 인덱스에 동일 SUMO seed 를 사용해 교통 수요 차이가 아닌 정책 성능 차이만 측정.

#### CSV 출력 형식

평가 CSV 는 3가지 섹션을 포함:

1. **메타데이터 prefix 컬럼** (모든 행에 동일값):
   `model_path`, `train_iter`, `train_total_steps`, `reward_mode`, `lr`, `entropy_coeff`, `vf_clip_param`, `train_seed`, `tls_ids`, `num_workers`, `traffic`
2. **raw 행** (에피소드별): `algorithm` (MAPPO/FixedTime), `episode`, `seed`, 메트릭들
3. **요약 행**: `MAPPO_mean`, `MAPPO_std`, `FixedTime_mean`, `FixedTime_std`

#### 평가 메트릭

| 컬럼 | 설명 |
|------|------|
| `avg_waiting_time` | 완료 차량 평균 누적 대기 시간 (초) |
| `std_waiting_time` | 대기 시간 표준편차 |
| `avg_travel_time` | 완료 차량 평균 통행 시간 (초) |
| `total_queue_length` | 에피소드 전체 queue 누적합 |
| `throughput` | 에피소드 내 도착 차량 수 |
| `phase_switches` | 에피소드 누적 phase 전환 횟수 |
| `max_queue` | 단일 lane 최대 halting 차량 수 |
| `teleported` | SUMO grid lock 텔레포트 차량 수 (0 이상이면 정체 한계 초과) |
| `action_{0,1,2,3}_ratio` | 정책 action 선택 비율 (**mode collapse 진단** — 균등이면 0.25씩) |

### 3. 롤아웃 영상 저장

```bash
# 기본: continuous 모드 (sim sec 당 1 frame, 자연스러운 흐름)
# 출력: videos/mappo_policy_rollout_N.mp4
python record_video_mappo.py --model models/MAPPO_sumo_11 \
  --tls-ids 1 2 5 6 --traffic high \
  --reward-mode diff-waiting-time

# 짧은 요약 영상 (short 모드 — env step 당 1 frame)
python record_video_mappo.py --model models/MAPPO_sumo_11 \
  --mode short --fps 5
```

#### 비디오 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--mode` | `continuous` | `continuous`: 매 sim step 1 frame (자연스러운 흐름, yellow 캡쳐) / `short`: env step 당 1 frame (요약) |
| `--fps` | 30 | continuous 권장 30, short 권장 5~10 |
| `--max-steps` | 1200 | 에피소드 최대 sim 초 |
| `--traffic` | `default` | 학습 시와 동일하게 지정 |

#### 시각화 디자인 (옵션 B)

| 신호 상태 | 동그라미 색 | 안의 마커 |
|----------|------------|----------|
| **Green + 직진/우회전** (`Th`/`Rt`/`Th+Rt`) | 초록 | 없음 (동그라미만) |
| **Green + 좌회전** (`Lt`) | 초록 | **가려는 방향 화살표** (N 진입→`◀`, S→`▶`, E→`▲`, W→`▼`) |
| **Yellow** | 노란 | 진입 방향 화살표 (작게) |
| **Red** | 빨간 (작은) | 없음 |

phase 라벨 형식: `tl_0: p4 [E+W Rt+Th]` (phase index + 활성 방향 + movement)

큐 막대:
- 도로 절반까지 뻗는 길이 (이전 IO×0.9 → sz×0.30)
- 같은 방향 lane 들의 큐를 (방향, movement) 별로 분리 집계
- 라벨: `W Th: 12` (서쪽 직진 lane 12대) / `N Th:4 Lt:1` (혼합)

---

## TensorBoard

학습 중 또는 후 별도 터미널에서 실행 후 `http://localhost:6006` 접속.

```bash
tensorboard --logdir results/tb_mappo
```

| 지표 카테고리 | 설명 |
|--------------|------|
| **reward/mean** | 에피소드 평균 누적 보상 |
| **episode/len_mean** | 평균 에피소드 길이 |
| **loss/{total,policy,value,entropy}** | PPO 학습 손실 |
| **learner/grad_norm_{global,policy}** | gradient norm (explosion 감지) |
| **policy/action_dist/action_{0,1,2,3}** | action 선택 비율 (**mode collapse 진단**) |
| **policy/phase_switches** | 에피소드 누적 phase 전환 |
| **policy/max_queue** | 단일 lane 최대 큐 |
| **policy/teleported** | grid lock 텔레포트 |
| **system/process_rss_mb** | 프로세스 메모리 (누수 감지) |
| **system/sumo_process_count** | SUMO 프로세스 수 (좀비 감지) |
| **system/iter_time_sec** | iter 소요 시간 |

---

## 학습 진단 가이드

**Regression 통과 기준** (10 iter 학습 후):

1. ✅ `reward/mean` 상승 추세 (초기 음수 → 점진 증가)
2. ✅ `policy/action_dist/action_{0,1,2,3}` 모두 **5% 이상** (한쪽 0% 이면 mode collapse)
3. ✅ `loss/entropy` 점진 감소 (0 근처 붕괴 X)
4. ✅ `policy/teleported` 0~소수 (grid lock 없음)
5. ✅ `system/process_rss_mb` 안정 (지속 증가 X)
6. ✅ `system/sumo_process_count` 일정 (좀비 없음)

**문제 진단**:

| 증상 | 원인 후보 | 대응 |
|------|----------|------|
| action 한쪽 0% / 99% | mode collapse | `entropy_coeff` ↑, traffic 다양화 |
| `loss/value` 거의 0 | reward magnitude 부족 | `--traffic high`, reward `/10` 정규화 확인 |
| `phase_switches` > 1000 | SUMO 자동 cycle (구버전 코드?) | D2 패치 적용 확인 |
| `teleported` > 0 | 정체 한계 초과 (grid lock) | `--traffic` 낮추기, network 수정 |
| `process_rss_mb` 우상향 | 메모리 누수 | env close() 누락 확인 |

---

## 체크포인트 복원

```python
import torch
from ray.rllib.algorithms.ppo import PPO

algo = PPO.from_checkpoint("models/MAPPO_sumo_11")
module = algo.get_module("shared_policy")  # RLModule API (Ray 2.10+)

obs, _ = env.reset()
action = int(torch.argmax(
    module.forward_inference(
        {"obs": torch.tensor(obs[None], dtype=torch.float32)}
    )["action_dist_inputs"],
    dim=-1,
).item())
```

학습 시 자동 저장된 메타데이터:

```python
import json
meta = json.loads((Path("models/MAPPO_sumo_11") / "train_metadata.json").read_text())
# {"model_name", "tls_ids", "reward_mode", "seed", "lr", "entropy_coeff",
#  "vf_clip_param", "train_iter", "train_total_steps", "traffic", ...}
```

---

## 참고 / 팁

### SUMO 연동
- `sumo_data/single_intersection.net.xml` 이 없으면 단일교차로 첫 실행 시 `netconvert` 로 자동 생성 (임시 파일 → atomic rename, 병렬 worker 충돌 방지)
- `--sumo-cfg` 명시 또는 `--traffic high` 시 자동 생성 호출 안 함
- 잘못된 `--tls-ids` 전달 시 TraCI 연결 직후 명확한 에러로 조기 종료
- RLlib 체크포인트는 디렉터리 형식 (`models/MAPPO_sumo_N/`)

### 병렬 학습
- `--num-workers 0`: driver process 직접 롤아웃 (디버깅, SUMO 1개)
- `--num-workers N≥1`: 별도 worker process로 SUMO N개 병렬
- num_workers 는 **병렬 SUMO 인스턴스 수**이지 교차로 수가 아님 (2x2grid 의 4 교차로 = 1 SUMO 인스턴스의 4 agent)

### 학습/평가/영상 호출 일관성
- `--reward-mode`, `--sumo-cfg`/`--traffic`, `--tls-ids` 는 학습·평가·영상에서 **동일하게 지정** 필요
- 평가 CSV 의 메타 prefix 컬럼이 학습 메타데이터 (`train_metadata.json`) 를 자동 로드해 출처 추적 가능

### 트래픽 시나리오 선택
- **연구/개발 단계**: `--traffic default` (빠른 학습, 약 5분/iter)
- **본 평가**: `--traffic high` (현실적 정체, reward magnitude ↑, VF 학습 신호 풍부)

### 시스템
- SUMO 시뮬레이션은 CPU 전용
- RLlib(PyTorch) CPU 학습 (M4 Mac MPS 미지원)
- `--seed` 로 random/numpy/torch 시드 일괄 설정해 재현성 확보

### 산출물 관리
- `models/`, `results/`, `videos/` 는 gitignored
- 참조용 샘플은 `samples/` 참조
- 평가 출력 파일은 모델 버전 자동 반영 (`eval_metrics_mappo_N.csv`, `mappo_policy_rollout_N.mp4`)
