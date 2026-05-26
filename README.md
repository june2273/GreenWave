# GreenWave

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
| **3-way 자동 비교** | Fixed-time vs MAPPO vs CTDE 동일 시드 쌍대 평가 → CSV 자동 저장 |
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
│  o_i ──► Actor (π_i)         o_i ──────────► Actor (π_i)   │
│       └► Critic (V_i)        [o_1‥o_N] ──► Critic (V)      │
│                                                             │
│  Critic 입력: 자기 obs만      Critic 입력: 전체 교차로 obs   │
│  교차로 간 인과 학습 어려움   Green Wave 협조 학습 가능      │
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
# 학습 (2×2 정체 시나리오, CTDE 모드)
python train_mappo.py --ctde --map 2x2 --traffic high --num-iters 200 --num-workers 4 --seed 42

# 3-way 비교 평가 (FixedTime vs MAPPO vs CTDE)
python evaluate_mappo.py \
  --model      models/MAPPO_sumo_N \
  --model-ctde models/MAPPO_CTDE_sumo_M \
  --map 2x2 --traffic high --episodes 10
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

`--traffic high` : 정체 시나리오 (~1.4 대/초, 2x2 계열에서만 지원)

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
reward = (이전 step 누적대기시간 − 현재) / 10.0  −  switch_penalty (기본 0.3)
```

`--brt-weight w` (BRT 시나리오 전용): vClass=bus 차량 대기시간을 w배 가중. `w=1.0` = baseline.

---

## Usage

### Train

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--map` | `single` | 시나리오 선택 |
| `--ctde` | off | Centralized critic 활성화 |
| `--traffic` | `default` | `high` = 정체 시나리오 |
| `--num-iters` | 200 | 학습 반복 횟수 |
| `--num-workers` | 1 | 병렬 env runner 수 (`0` = 디버그) |
| `--brt-weight` | 1.0 | BRT 차량 가중치 (BRT 시나리오 전용) |
| `--seed` | 42 | 재현성용 시드 |

체크포인트는 `models/MAPPO_sumo_N/` (CTDE: `models/MAPPO_CTDE_sumo_N/`)에 자동 버전 저장.

### Evaluate

```bash
python evaluate_mappo.py --model models/MAPPO_sumo_N --map 2x2 --traffic high --episodes 10
```

결과는 `results/eval_metrics_mappo_N.csv` 에 저장 (에피소드별 raw + 요약 통계).

### Record Video

```bash
python record_video_mappo.py --model models/MAPPO_sumo_N --map 2x2 --traffic high
# → videos/mappo_policy_rollout_N.mp4
```

---

## Results

기대 비교 패턴 (2×2 정체 시나리오, 10 에피소드 평균):

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
├── evaluate_mappo.py       # 평가 (3-way 비교 → CSV)
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
