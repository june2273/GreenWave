# GreenWave 🌱

> MAPPO 기반 다중 교차로 신호제어 강화학습 프레임워크  
> Multi-Agent PPO · SUMO Traffic Simulator · RLlib · PettingZoo

---

## Overview

**Green Wave**란, 일련의 신호등을 연동해 차량이 일정한 속도로 주행하면
멈추지 않고 연속해서 교차로를 통과할 수 있게 만드는 제어 방식입니다.

GreenWave는 SUMO 시뮬레이터 위에서 **여러 교차로 신호를 동시에 최적화**하는 MAPPO 프레임워크입니다.
단일 교차로부터 3×2 격자망·BRT 우선 신호까지, `--map` 하나로 시나리오를 전환할 수 있습니다.

---

## Features

| 특징 | 설명 |
|------|------|
| **CTDE-MAPPO** | Actor는 자기 교차로만 보고 실행, Critic은 전체 교차로 정보를 보고 학습 → 교차로 간 협조(Green Wave) 유도 |
| **BRT 우선처리 보상함수** | 세종시 BRT 노선 특화 — vClass=bus 차량의 대기시간을 `w`배 가중해 정책이 버스 우선 신호를 자연스럽게 학습 |
| **동적 시나리오 전환** | `--map` 하나로 단일·격자·BRT 시나리오 전환, TLS 위상 자동 감지 |
| **다양한 평가 모드** | 2-way (단일 모델 vs Fixed) / 3-way (MAPPO vs CTDE vs Fixed) / 세종 (세종 실측 신호 베이스라인) — 동일 시드 쌍대 평가 → CSV 자동 저장 |
| **정책 롤아웃 영상** | matplotlib 기반 실시간 신호 시각화, mp4 출력 |

---

## Architecture

### 시스템 흐름

```
RLlib PPO (shared policy)
        ↓ {agent_id: action}
SumoParallelEnv  (PettingZoo ParallelEnv)
        ↓ traci.setPhase + setPhaseDuration(lock)
    SUMO Simulator  ←→  sumo_data/ XML
        ↓ lane states
SumoParallelEnv  →  {agent_id: obs, reward, done}
        ↑
RLlib PPO update
```

- **shared policy**: 모든 교차로가 하나의 정책 공유 → 시나리오 확장 용이
- **phase lock**: `setPhaseDuration(100000s)` 로 SUMO 자동 cycle 차단 — agent action만이 phase 전환 결정
- **dynamic action space**: TLS state string 파싱으로 green phase 자동 감지 (`Discrete(4)`, 전 시나리오 공통)

### MAPPO — Actor-Critic 구조와 CTDE

각 교차로(agent)는 **하나의 shared policy**를 공유합니다.
Critic은 학습 시에만 사용되며, 실행(inference)은 actor만으로 완전 분산 동작합니다.

**CTDE (Centralized Training, Decentralized Execution)**:
- **학습 시**: Critic이 모든 교차로의 obs를 concat한 global state를 보며 Advantage를 추정
  → 교차로 A의 신호가 교차로 B의 대기시간에 미치는 영향을 학습
- **실행 시**: Actor는 자기 교차로 obs만 사용 → 현장 분산 배포 가능

```
CTDE Critic: V( [o_1, o_2, ..., o_N] )   ← N개 교차로 global obs concat
CTDE Actor:  π_i( a_i | o_i )            ← 자기 obs만 (실행 시 동일)
```

```
┌─────────────────────────────────────────────────────────────┐
│               Simple MAPPO  vs  CTDE-MAPPO                  │
│                                                             │
│  Simple MAPPO                CTDE-MAPPO (--ctde)            │
│  ─────────────               ────────────────────           │
│  o_i ──► Actor (π_i)         o_i ──────────► Actor (π_i)    │
│       └► Critic (V_i)        [o_1‥o_N] ──► Critic (V)       │
│                                                             │
│  Critic 입력: 자기 obs만       Critic 입력: 전체 교차로 obs        │
│  교차로 간 인과 학습 어려움       Green Wave 협조 학습 가능           │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

```bash
brew install sumo

export SUMO_HOME=$(brew --prefix sumo)/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

### 2. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Train & Evaluate

```bash
# 학습 (2x2-brt 시나리오, MAPPO baseline)
python train_mappo.py --map 2x2-brt --num-iters 200 --num-workers 4 --seed 42

# 학습 (CTDE 모드)
python train_mappo.py --ctde --map 2x2-brt --num-iters 200 --num-workers 4 --seed 42

# 평가 — 3-way 비교 (FixedTime vs MAPPO vs CTDE)
python evaluate_mappo.py \
  --model      models/MAPPO_sumo_N \
  --model-ctde models/MAPPO_CTDE_sumo_M \
  --map 2x2-brt --episodes 10
```

---

## Scenarios

| `--map` | 교차로 수 | 특징 | obs_dim |
|---------|----------|------|---------|
| `single` | 1 | 단일 4-way 교차로 | 13 |
| `2x2` | 4 | 2×2 격자, 표준 2-lane | 21 |
| `2x2-brt` | 4 | 2×2 + 좌측 열 BRT corridor (lane 2 = bus 전용) | 25 |
| `3x2` | 6 | 3×2 격자, 모든 edge 2-lane | 21 |
| `3x2-brt` | 6 | 3×2 + 좌측 열 BRT corridor | 25 |

`--traffic high` : 정체 시나리오.
- `2x2` (~1.4 대/초, sumo-rl dense)
- `2x2-brt` (~1.9 대/초, 행복청 세종시 AM peak 실측 직진 100% + cross 보정 ~6,810 veh/h · LOS D)
- `3x2-brt` (~2.4 대/초, **세종시 현실 시뮬레이션 최종 버전** · 6 교차로 실측/추정 + cross 15% · ~8,530 veh/h · LOS D)

`3x2-brt --traffic high` 는 세종시 어진동 일원의 실제 도로망(한누리대로 BRT 회랑 × 갈매로)을 모델링한 학술용 최종 시나리오. 6개 교차로는 3행×2열로 배치됩니다:

| | 한누리대로 (BRT 좌열) | 갈매로 (우열) |
|--|--|--|
| 도움8로 | tl_0 성금 | tl_1 청사 |
| 도움4로 | tl_2 도움4로 | tl_3 도움4로 |
| 가름로 | tl_4 어진 | tl_5 가름로 |

`evaluate_mappo.py --baseline sejong` 와 함께 사용 시 세종 실측 신호 타이밍 (per-TLS 비대칭) 대비 평가 가능.

---

## Environment Design

### Action Space

모든 시나리오 `Discrete(4)`. Yellow phase는 정책이 직접 선택 불가 — phase 전환 요청 시 env가 자동 삽입.

| 시나리오 | Action 0 | Action 1 | Action 2 | Action 3 |
|---------|---------|---------|---------|---------|
| `single` | 방향1 직진+좌회전 | 방향2 직진+좌회전 | 방향3 직진+좌회전 | 방향4 직진+좌회전 |
| `2x2` / `3x2` | NS 직진+우회전 (33s) | NS 좌회전 protected (6s) | EW 직진+우회전 (33s) | EW 좌회전 protected (6s) |
| `*-brt` | NS+BRT 직진+우회전 (33s) | NS 좌회전 protected (6s) | EW 직진+우회전 (33s) | EW 좌회전 protected (6s) |

> NS와 EW는 절대 동시 green 없음 (ring & barrier 표준). 충돌 위험 없음.

### Observation Space

```
[phase_one_hot(4), min_green_flag(1), density_per_lane(L), queue_per_lane(L)]
```

BRT처럼 lane 수가 혼합된 토폴로지에서는 작은 agent의 obs에 0 패딩 적용.

### Reward

```
reward = (이전 step 누적대기시간 − 현재) / 10.0  −  switch_penalty (기본 0.45)
```

`switch_penalty = 0.45` ≈ yellow 3초 × 1대/초 손실 (`yellow_time=3` 기준).
yellow_time=2 로 학습된 이전 체크포인트 재사용 시 `--switch-penalty 0.3` 명시.

`--brt-weight w` (BRT 시나리오 전용): 보상함수에서 BRT(vClass=bus) 대기시간에 부여하는 중요도 계수. `w=1.0` = 일반 차량과 동등 취급 (baseline). `w>1` → BRT 대기시간 감소를 더 크게 보상해 정책이 BRT 우선 신호를 학습하도록 유도. 실제 대기시간을 늘리는 게 아닌 보상 신호의 비중 조정 (권장: `2.0~3.0`).

---

## Usage

### Train

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--map` | `single` | 시나리오 선택 (`single`/`2x2`/`2x2-brt`/`3x2`/`3x2-brt`) |
| `--ctde` | off | Centralized critic 활성화 → `MAPPO_CTDE_sumo_N/` 에 저장 |
| `--traffic` | `default` | `high` = 정체 (세종 실측 / dense routes) |
| `--num-iters` | 200 | 학습 반복 횟수 (1 iter = 8,000 steps) |
| `--num-workers` | 1 | 병렬 env runner 수 (`0` = 디버그) |
| `--brt-weight` | 1.0 | BRT(vClass=bus) 차량 waiting time 가중치. 권장 `2.0~3.0` (BRT 시나리오 전용) |
| `--switch-penalty` | 0.45 | phase switch 마다 reward 페널티. yellow 3초 손실 기준 |
| `--yellow-time` | 3 | yellow phase 길이(초). 세종 실제 신호 3초와 일치 |
| `--min-green` | 13 | 최소 green 유지 시간(초). oscillation 억제 |
| `--seed` | 42 | 재현성용 시드 |

체크포인트는 `models/MAPPO_sumo_N/` (CTDE: `models/MAPPO_CTDE_sumo_N/`)에 자동 버전 저장.
함께 `train_metadata.json` 도 저장되어 평가 시 hyperparameter 가 CSV prefix 컬럼으로 자동 복제됨.

### Evaluate

평가는 **3가지 모드**를 지원합니다. CLI 플래그 조합으로 자동 선택됩니다.

| 모드 | 비교 대상 | 베이스라인 | 권장 시나리오 | 트리거 |
|---|---|---|---|---|
| **2-way** (기본) | 단일 모델 vs FixedTime | symmetric | 빠른 단일 모델 검증 | `--model` **또는** `--model-ctde` 중 하나 |
| **3-way** | MAPPO + CTDE vs FixedTime | symmetric | 알고리즘 간 비교 (학술) | `--model` + `--model-ctde` 동시 |
| **세종** | MAPPO + CTDE vs FixedTime(세종 실측 신호) | sejong | **최종 시뮬레이션** (3x2-brt high) | 3-way + `--baseline sejong` |

```bash
# 2-way: 단일 모델 빠른 검증
python evaluate_mappo.py --model models/MAPPO_sumo_N --map 2x2-brt --episodes 5

# 3-way: 알고리즘 비교 (균등 사이클 베이스라인)
python evaluate_mappo.py \
  --model      models/MAPPO_sumo_N \
  --model-ctde models/MAPPO_CTDE_sumo_M \
  --map 2x2-brt --traffic high --episodes 10

# 세종: 세종 실측 신호 베이스라인 (최종 평가)
python evaluate_mappo.py \
  --model      models/MAPPO_sumo_N \
  --model-ctde models/MAPPO_CTDE_sumo_M \
  --map 3x2-brt --traffic high --baseline sejong --episodes 10
```

**`--baseline sejong`** : 공공데이터포털(2023)·세담터(2026.04) 실측 자료를 기반으로 실제 세종시 교통환경을 구현한 베이스라인. 6개 교차로 각각의 4-phase 비대칭 신호 타이밍을 `SEJONG_PER_TLS_PHASE_SECONDS` dict로 적용 — 실측 데이터로 시뮬레이션 신뢰도를 높였습니다.

결과는 `results/eval_metrics_mappo_N.csv` 에 저장 (에피소드별 raw + 요약 통계 + 모델 메타데이터 prefix 컬럼).

### Record Video

```bash
python record_video_mappo.py --model models/MAPPO_sumo_N --map 2x2-brt
# → videos/mappo_policy_rollout_N.mp4
```

`--mode {continuous, short}`: 기본 `continuous` 는 매 sim sec 당 1 frame (~3,600 frame, 10fps 기본 → **약 2분** 영상, yellow 전환 자연스러움). `short` 는 env.step() 당 1 frame (~720 frame, 5fps 권장 → **약 40초** 영상).

---

## Results

### 실측 예시 — MAPPO_sumo_1 (2x2-brt default, 80 iter, 5 ep 평균)

| 지표 | FixedTime | MAPPO | 개선폭 |
|------|-----------|-------|--------|
| 평균 대기시간 (s) | 59.21 | **37.48** | −36.7% |
| 평균 통행시간 (s) | 234.47 | **103.84** | −55.7% |
| Throughput (대/h) | 2,733 | **3,822** | +39.8% |
| 평균 속도 (m/s) | 1.83 | **4.24** | +132% |
| 차량당 정지 횟수 | 3.59 | **2.09** | −41.8% |
| 차량당 CO₂ (mg) | 610,608 | **280,655** | −54.0% |
| Vehicles lost insert | 1,066 | **51** | −95.2% |
| BRT avg_wait (s) | 32.80 | **15.74** | **−52.0%** |
| Car avg_wait (s) | 56.85 | **31.07** | −45.3% |

> 단일 MAPPO (decentralized critic) 학습만으로도 BRT corridor 우대 효과가 자연 발생 (BRT wait 가 일반차 wait 의 ½). CTDE 추가 시 추가 개선 기대.

### 일반 비교 패턴 (3-way · 정체 시나리오)

| 지표 | Fixed-time | MAPPO | CTDE-MAPPO |
|------|-----------|-------|------------|
| 평균 대기시간 (s) | 높음 | 중간 | **낮음** |
| 차량당 정지 횟수 | ~2–3 | ~1.5 | **~0.5–1** |
| 차량당 CO₂ (mg) | 높음 | 중간 | **낮음** |
| 평균 속도 (m/s) | 낮음 | 중간 | **높음** |

BRT 시나리오의 핵심 지표: `avg_wait_brt` vs `avg_wait_car` (BRT 우선처리 효과 직접 측정).

---

## TensorBoard

```bash
tensorboard --logdir results/tb_mappo
```

| 지표 | 수렴 신호 |
|------|---------|
| `reward/mean` | 초기 음수 → 점진 상승 후 plateau |
| `loss/entropy` | 천천히 ↓ (0.5 ~ 1.5 → 0.1 ~ 0.5). 급격히 0이면 mode collapse |
| `policy/kl_mean` | < 0.02 정상. > 0.05이면 lr / clip_param 조정 필요 |
| `policy/action_dist/action_{0..3}` | 4개 모두 5% 이상. 한쪽 >90%이면 mode collapse |

---

## Expected Effects

Green Wave 제어가 현실에 적용될 경우 기대되는 정량적 효과입니다.

### 탄소 배출 절감

| 항목 | 근거 |
|------|------|
| 정지·출발 반복(stop-and-go) 구간의 연료 소모 | 정속 주행 대비 **약 20~40% 추가 소모** |
| Green Wave 도입 시 차량당 정지 횟수 감소 | 시뮬레이션 기준 ~2–3회 → **~0.5–1회** |
| CO₂ 환산 절감 잠재량 | 도심 간선 기준 연간 차량당 수십 kg CO₂ 절감 기대 |

정지 횟수가 줄면 공회전·급가속이 감소하고, 이는 CO₂·NOx 등 오염물질 배출 감소로 직결됩니다.
교차로 수와 교통량에 비례해 도시 전체 탄소 저감 효과가 누적됩니다.

### 운전자 유류비 절감

| 시나리오 | 효과 |
|---------|------|
| 정속 주행 유도 | 급가속·공회전 감소 → 연료 효율 향상 |
| BRT 우선 신호 | 버스 정시성 향상 → 버스 운행 비용 절감 및 대중교통 이용 유도 |
| 대기시간 단축 | 공회전 시간 감소 → 차량당 유류비 절감 |

### 세종시 BRT 노선 특화 효과

세종시 간선급행버스(BRT)는 전용 차로를 운영하지만 교차로 신호에서 일반 차량과 동일하게 대기합니다.
본 프레임워크의 **BRT 가중 보상함수**(`--brt-weight`)를 적용하면:

- BRT 차량의 교차로 대기시간 우선 최소화 → **버스 정시율 향상**
- 일반 차량 대기시간과의 trade-off를 `brt_weight` 하나로 조정 가능
- CTDE Critic이 BRT 노선 전체의 신호 패턴을 학습 → **노선 전체 Green Wave** 형성 가능

---

## Project Structure

```
GreenWave/
├── env_sumo_pz.py          # PettingZoo ParallelEnv (핵심 환경)
├── ctde_module.py          # CTDE RLModule (centralized critic)
├── map_presets.py          # --map 시나리오 preset 해상
├── train_mappo.py          # 학습
├── evaluate_mappo.py       # 평가 (2-way / 3-way / 세종 모드, → CSV)
├── record_video_mappo.py   # 정책 롤아웃 영상
├── sumo_renderer.py        # matplotlib 신호 시각화
├── sumo_data/
│   ├── single/             # 단일교차로 XML
│   ├── 2x2/                # 2×2 표준 / dense sumocfg
│   ├── 2x2_brt/            # BRT corridor 포함
│   ├── 3x2/                # 3×2 격자
│   └── 3x2_brt/            # 3×2 + BRT
├── models/                 # 체크포인트 (gitignored)
├── results/                # CSV + TensorBoard 로그 (gitignored)
└── samples/                # 커밋된 참조 샘플
```

---

## Citation

```bibtex
@misc{kafa46,
    author = {Prof. Giseop Noh},
    title  = {{S(Grad) Reinforce Learning}},
    url    = {https://www.deepshark.org/courses/grad_reinforce_learning/}
}

@misc{sumorl,
    author = {Lucas N. Alegre},
    title = {{SUMO-RL}},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/LucasAlegre/sumo-rl}},
}
```
