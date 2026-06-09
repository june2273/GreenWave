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
| **이중 보상 모드** | `--reward-mode` 로 `diff-waiting-time`(차분) / `pressure`(max-pressure, 하류 포화 억제) 전환 — 동일 하이퍼파라미터 ablation |
| **동적 시나리오 전환** | `--map` 하나로 단일·격자·BRT 시나리오 전환, TLS 위상 자동 감지 |
| **다양한 평가 모드** | 2-way (단일 모델 vs Fixed) / 3-way (MAPPO vs CTDE vs Fixed) / 세종 (세종 실측 신호 베이스라인) — 동일 시드 쌍대 평가 → CSV 자동 저장 |
| **통계적 신뢰성 평가** | `stats_eval.py` — Friedman + 쌍별 Wilcoxon(Holm) + A₁₂ 효과크기 + 부트스트랩 CI로 비교 우위의 통계적 유의성 검증 |
| **정책 롤아웃 영상** | matplotlib 기반 실시간 신호 시각화, mp4 출력 (3-way 비교 합성 지원) |

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

#### RL-native Green Wave 확장 (선택, 회랑 진행파 학습)

기본 CTDE는 보상에 *진행(progression)* 항이, 관측에 상류 platoon *도착 타이밍*이 없어 진행파가 창발하지 않습니다. 한누리대로 BRT 회랑(좌열, N/S축)에서 **무정차 진행파를 학습으로 유도**하는 3 플래그를 추가합니다 (설계 상세: `CLAUDE.md` 의 "Green Wave Progression" 섹션):

- **`--neighbor-obs` (#1+#3)** — actor obs에 **N/S 이웃**의 상류 혼잡 요약 `[mean_density, mean_queue]×2` 추가(anticipation). `--ctde` 동반 시 critic 입력 = `[own | agent_id one-hot | N/E/S/W 이웃 obs]` → **actor-lean(N/S)/critic-rich(4방향)**.
- **`--upstream-phase` (②)** — N/S 이웃의 위상 시계 `[phase_one_hot, 경과시간]×2` 추가 → 상류 신호 타이밍을 관측해 green wave 오프셋을 학습. (`--neighbor-obs` 필요.)
- **`--progression-coeff β` / `--brt-prog-weight` (①)** — 회랑 교차로(BRT 전용차로 보유, 자동식별)의 직진 lane **가중 평균 속도비**를 β배 보상 → 무정차 통과. `brt-prog-weight>1`이면 BRT 우선(Transit Signal Priority).

회랑·인접·방향은 네트워크에서 **자동 도출**됩니다 (BRT 차로=`lane.getAllowed`, 인접=`out_lane∩in_lane`+상대좌표, 하드코딩 0). obs 차원이 바뀌므로(BRT 시나리오 per-agent 25→29→39, CTDE critic 151→**201**) 미지정 체크포인트와 호환되지 않으며, `train_metadata.json`의 `neighbor_obs`/`upstream_phase`로 추적돼 평가·녹화 시 모델별 자동 로드됩니다. 진행파 측정 지표: `corridor_brt_speed_ratio`.

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

#### 교통 수요 데이터 출처 

`--traffic high` 의 교통량은 **세종시 실측 공공 데이터 2종**을 역할 분담해 구성합니다 (`sumo_data/3x2_brt/routes_3x2_brt_dense.rou.xml` 주석에 전체 유도 과정 기록):

| 자료 | 가져온 부분 | 용도 |
|--|--|--|
| **행복청 제23차 행복도시 교통량 조사** (2025.04.24 AM peak) | 교차로별 **총 교통량(pcu/h)·LOS·지체** | 절대 교통량의 *크기* |
| **공공데이터포털 — 세종시 교차로 방향별 통행량** (2023-03) | 교차로별 **방향 분할 비율**(동/서/남/북 %) | NS/EW *분배 비율* |

- **결합식**: `PDF 총량(pcu/h) × CSV 방향비율 ÷ pcu→veh(~1.155) = 방향별 veh/h`. 예) 성금 `4,245 pcu/h × 58.3%(NS) ÷ 1.155 ≈ 2,142 veh/h` → 회랑 직진 980 veh/h/방향.
- **실측 검증**: 성금 4,245·어진 3,959·청사 2,593·갈매가름 2,782 pcu/h (모두 LOS D) , 방향 비율도 CSV 2023-03 적용
- **추정 부분** : 6교차로 중 `tl_2/tl_3(도움4로)` 2개와 일부 row 는 두 자료에 없어 **인접 교차로 평균으로 추정**. CSV(월 누계)는 절대량이 아닌 *방향 비율*로만 사용. PDF(2025)·CSV(2023) 연도 혼합, pcu→veh 표준 환산계수 가정 포함.

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

`--neighbor-obs` 활성화 시 obs 끝에 **N/S 이웃**의 상류 혼잡 요약 `[mean_density, mean_queue]×2 = +4 dim`이, `--upstream-phase` 까지 켜면 위상 시계 `[phase_one_hot, 경과시간]×2 = +10 dim`이 추가됩니다 (BRT 시나리오 기준 25 → 29 → 39). 위 *RL-native Green Wave 확장* 절 참조.

### Reward

`--reward-mode` 로 두 보상 함수를 선택합니다 (둘 다 phase switch마다 `switch_penalty`를 차감).

**`diff-waiting-time` (기본)** — 차분 기반, 자기 진입로 대기시간만 봅니다.

```
reward = (이전 step 누적대기시간 − 현재) / 10.0  −  switch_penalty (기본 0.45)
```

**`pressure`** — max-pressure / backpressure (Varaiya 2013, PressLight 2019). 상태 기반으로 **하류 포화를 직접 반영**해 과포화(LOS D+) 스필백·grid lock을 억제합니다.

```
reward = Σ#(out_lanes 차량) − Σ#(in_lanes 차량)  −  switch_penalty
```

> `pressure`는 스케일이 작아(±수십, diff-waiting의 ~수천 대비) `switch_penalty` 0.45가 상대적으로 과해집니다 → 학습 시 `--switch-penalty 0.1` 전후 권장. 차량 *수* 기반이라 `brt_weight` 미적용. 동일 하이퍼파라미터로 두 보상의 ablation 비교가 가능합니다.

`switch_penalty = 0.45` ≈ yellow 3초 × 1대/초 손실 (`yellow_time=3` 기준).
yellow_time=2 로 학습된 이전 체크포인트 재사용 시 `--switch-penalty 0.3` 명시.

`--brt-weight w` (BRT 시나리오 전용, `diff-waiting-time`에서만 reward 반영): BRT(vClass=bus) 대기시간 가중치. `w=1.0` = 일반 차량과 동등 취급 (baseline). `w>1` → BRT 대기시간 감소를 더 크게 보상해 정책이 BRT 우선 신호를 학습하도록 유도. 실제 대기시간을 늘리는 게 아닌 보상 신호의 비중 조정 (권장: `2.0~3.0`).

---

## Usage

### Train

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--map` | `single` | 시나리오 선택 (`single`/`2x2`/`2x2-brt`/`3x2`/`3x2-brt`) |
| `--ctde` | off | Centralized critic 활성화 → `MAPPO_CTDE_sumo_N/` 에 저장 |
| `--neighbor-obs` | off | Green Wave 확장: actor에 N/S 이웃 상류 요약(+4dim) + (`--ctde` 동반 시) critic 4-slot. obs 차원 변경(25→29) → 기존 체크포인트와 비호환, resume 시 값 일치 강제 |
| `--upstream-phase` | off | ② 상류 위상 관측: N/S 이웃 phase+경과시간(+10dim, 29→39) → green wave 오프셋 학습. `--neighbor-obs` 필요 |
| `--progression-coeff` | 0.0 | ① 회랑 progression 보상 β: BRT 회랑 직진 무정차 통과 가산. diff-waiting 1~5 / pressure 5~20 |
| `--brt-prog-weight` | 1.0 | ① progression의 BRT 가중치. >1(3~5)=BRT 우선 진행파(TSP) |
| `--traffic` | `default` | `high` = 정체 (세종 실측 / dense routes) |
| `--reward-mode` | `diff-waiting-time` | 보상 함수. `pressure` = max-pressure (하류 포화 억제). pressure 시 `--switch-penalty 0.1` 권장 |
| `--num-iters` | 200 | 학습 반복 횟수 (1 iter = 8,000 steps) |
| `--num-workers` | 1 | 병렬 env runner 수 (`0` = 디버그) |
| `--brt-weight` | 1.0 | BRT(vClass=bus) 차량 waiting time 가중치. 권장 `2.0~3.0` (BRT 시나리오 전용) |
| `--switch-penalty` | 0.45 | phase switch 마다 reward 페널티. yellow 3초 손실 기준 |
| `--yellow-time` | 3 | yellow phase 길이(초). 세종 실제 신호 3초와 일치 |
| `--min-green` | 13 | 최소 green 유지 시간(초). oscillation 억제 |
| `--time-to-teleport` | 300 | SUMO 텔레포트 임계(초). 정체에 갇힌 차량을 정체 너머로 이동시켜 grid lock 자가 회복(deadlock 안전밸브). `-1` = 비활성(과포화 시 보상신호 소멸 → 학습 불가). 과포화 학습엔 유한값 필수 |
| `--entropy-coeff` | 0.03 | 엔트로피 계수(고정). 좌회전 phase mode collapse 방지 |
| `--entropy-schedule` | off | 엔트로피 선형 스케줄 `START END` (예: `0.03 0.005`). 초반 탐색→후반 sharpening, 학습 75% 지점에서 END 도달 후 유지. `--entropy-coeff`보다 우선 |
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

#### 신호 타이밍 데이터 출처

per-TLS 신호 현시는 **실측 신호 데이터 2종**을 교차로별로 매핑해 구성합니다:

| 자료 | 가져온 부분 | 적용 TLS |
|--|--|--|
| **공공데이터포털 — 세종시 교차로 신호 표준데이터** (2023-05-10) | 교차로별 `신호등화시간`(현시 초) | tl_0 성금 · tl_1 청사 · tl_4 어진 |
| **세담터 — 세종시 신호등** (2026-04) | 노선별 녹색 현시 시간 (최신) | tl_5 가름로 |

- 원본 현시열 → 환경의 4-phase ring&barrier `[NS직진, NS좌, EW직진, EW좌]` 로 재매핑:
  - **청사** 원본 `46+20+31+23(+20)` → tl_1 `[46,20,31,23]` (앞 4현시 정확 일치 ✓)
  - **어진** 원본 `18+20+47+47+28` → tl_4 `[47,20,47,28]`
  - **성금** 원본 `45+45+25+45` → tl_0 `[45,25,45,25]`
  - **가름로** 세담터 녹색 `45 / 35`초 → tl_5 `[45,20,35,20]` (직진 45·35 실측, 좌회전 20 추정)
- **추정 부분** : `tl_2 / tl_3 (도움4로)` 는 두 자료 어디에도 없어 도로 폭을 고려하여 **성금·청사 신호를 복제**해 적용. 모든 현시는 4-phase ring&barrier 구조로 재해석된 값(원본 5현시 → 앞 4개 사용 등)이며 yellow 는 환경이 3초로 별도 처리.

결과는 `results/eval_metrics_mappo_N.csv` 에 저장 (에피소드별 raw + 요약 통계 + 모델 메타데이터 prefix 컬럼).

### Record Video

```bash
python record_video_mappo.py --model models/MAPPO_sumo_N --map 2x2-brt
# → videos/mappo_policy_rollout_N.mp4
```

`--mode {continuous, short}`: 기본 `continuous` 는 매 sim sec 당 1 frame (~3,600 frame, 10fps 기본 → **약 2분** 영상, yellow 전환 자연스러움). `short` 는 env.step() 당 1 frame (~720 frame, 5fps 권장 → **약 40초** 영상).

### 3-way 비교 영상 (FixedTime | MAPPO | CTDE)

`build_3way_compare.py` 는 세 정책 영상을 가로로 합성하고 실시간 지표(대기·CO₂·throughput)를 오버레이합니다. 먼저 세 영상을 **동일 seed·동일 시나리오**로 녹화해야 프레임이 정렬됩니다 (`record_video_fixed.py` 는 Fixed-Time 전용, Ray 비의존).

```bash
B=videos/3way_sejong
record_video_fixed.py --baseline sejong --map 3x2-brt --traffic high --seed 777 \
  --output $B/cmp_fixed.mp4 --dump-metrics $B/cmp_fixed.json
record_video_mappo.py --model models/MAPPO_sejong_dense --map 3x2-brt --traffic high --seed 777 \
  --output $B/cmp_mappo.mp4 --dump-metrics $B/cmp_mappo.json
record_video_mappo.py --model models/CTDE_sejong_dense  --map 3x2-brt --traffic high --seed 777 \
  --output $B/cmp_ctde.mp4  --dump-metrics $B/cmp_ctde.json
build_3way_compare.py --base-dir $B --iter 200 --out $B/compare_3way_1080p.mp4
```

---

## Statistical Reliability

성능 지표가 좋다는 것(performance)과 그 우위가 우연이 아니라는 것(reliability)은 다릅니다.
`stats_eval.py` 는 `evaluate_mappo.py` 가 만든 **paired CSV**(세 알고리즘이 매 episode 동일 seed로 실행되는 blocked 설계)를 받아 통계적 신뢰성 증거를 생성합니다 — 재학습·SUMO 불필요, 수 초 소요.

```bash
# 1) 검정력 확보를 위해 충분한 episode 로 평가 (권장 30)
python evaluate_mappo.py --model models/MAPPO_sejong_dense --model-ctde models/CTDE_sejong_dense \
  --map 3x2-brt --traffic high --baseline sejong --episodes 30 --csv-out results/eval_sejong_30.csv

# 2) 통계 분석
python stats_eval.py --csv results/eval_sejong_30.csv
```

| 단계 | 방법 | 의미 |
|------|------|------|
| **Omnibus** | Friedman test | "세 방법 중 어딘가 차이가 있나?" (알고리즘 ≥3 게이트) |
| **사후(post-hoc)** | 쌍별 Wilcoxon signed-rank + Holm 보정 | "어느 쌍이 유의하게 다른가?" (2-way면 단일 Wilcoxon) |
| **효과크기** | Vargha–Delaney A₁₂ | "좋은 쪽이 이길 확률" (지표 방향맵 반영) |
| **신뢰구간** | 부트스트랩 95% CI (mean + IQM) | IQM = 이상치 강건 (Agarwal 2021) |

산출물(`results/stats_<csv명>/`): `stats_tests.csv` · `stats_ci.csv` · `stats_summary.txt`(발표 헤드라인 자동 생성) · `figs/{box,forest}_*.png`.

> **검정력 주의**: paired n이 작으면(예 5) Wilcoxon 양측 최소 p = 2/2ⁿ = 0.0625라 Friedman이 유의해도 쌍별 검정이 유의에 도달할 수 없습니다. 쌍별 유의에는 **n ≥ 7**(Holm 보정 시 더), 안정적 결론엔 **n ≈ 30** 권장. A₁₂ 효과크기는 n이 작아도 해석 가능합니다.

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

### 세종시 실측 3-way 최종 평가 — 3x2-brt `--traffic high` (LOS D, 30 ep paired)

세종시 6교차로 실측 수요(~8,530 veh/h) + 공공데이터포털·세담터 기반 per-TLS 실측 Fixed-Time을 베이스라인으로, MAPPO·CTDE를 **매 episode 동일 seed로 30회** 쌍대 평가한 결과. (`eval_sejong_dense_30.csv`, 모두 `time_to_teleport=300`, `teleported≈0~3`로 결과가 텔레포트 인공물 아님.)

| 지표 | FixedTime(Sejong) | MAPPO | CTDE | RL 개선폭 (vs Fixed) |
|------|------:|------:|------:|:--:|
| 평균 대기시간 (s) | 63.65 | **46.45** | 46.57 | **−27.0%** |
| 평균 통행시간 (s) | 284.56 | **166.62** | 171.00 | −41.4% |
| Throughput (대/h) | 3,812 | **5,958** | 5,869 | **+56.3%** |
| 총 대기열 길이 | 902,611 | **569,051** | 583,127 | −37.0% |
| 평균 속도 (m/s) | 1.66 | **2.87** | 2.80 | +73.2% |
| 차량당 CO₂ (mg) | 756,055 | **445,824** | 457,189 | −41.0% |
| 텔레포트 (대) | 10.97 | **1.40** | 3.33 | −87.2% |
| 차량당 정지 횟수 † | **2.42** | 3.39 | 3.44 | +40% (열위) |
| **BRT 평균대기 (s)** | 44.34 | **24.56** | 26.04 | **−44.6%** |
| 일반차 평균대기 (s) | 68.08 | **43.20** | 44.13 | −36.5% |
| **BRT 평균속도 (m/s)** | 4.71 | **5.85** | 5.76 | **+24.2%** |
| 방향별 대기 NS (s) | 254.8 | **57.1** | 58.7 | −77.6% |
| 방향별 대기 EW (s) | 359.6 | **66.1** | 72.8 | −81.6% |

> † **정지 횟수만 Fixed가 우위** — RL이 Fixed보다 **56% 더 많은 차량을 실제로 통과**시키기 때문(Fixed는 과포화에서 grid lock → 진입 자체가 막혀 통과 차량의 stop은 적게 집계). throughput·대기·통행시간을 모두 종합하면 RL 우위가 명확하며, BRT corridor(NS축)에서도 BRT 대기 −44.6%·속도 +24.2%로 우대 효과가 자연 발생.

**통계적 유의성** (`stats_eval.py`, n=30 paired, Friedman → Wilcoxon+Holm → A₁₂):

- ★ **MAPPO·CTDE 모두 FixedTime 대비 30/30 시나리오 전부 우월** — 평균대기 IQM **−27.0%** [95% CI 26.5~27.5%], Wilcoxon+Holm **p = 5.6e-09**, **A₁₂ = 1.00**(large). 통행시간·throughput·대기열·CO₂·속도·텔레포트 8개 지표 전부 동일하게 유의 (모든 p < 1e-3).
- **MAPPO vs CTDE**: 평균대기는 **통계적 동률**(p_holm = 0.78, A₁₂ = 0.538 negligible). 그 외 통행시간·throughput·대기열·CO₂·속도·텔레포트에서는 **MAPPO가 유의하게 우세**(A₁₂ 0.75~0.84, p < 0.05) — 현 CTDE는 평균만 맞먹고 분산이 큼(throughput/travel CI 폭이 MAPPO의 1.8~2.9배). → topology-aware CTDE 재학습(`--neighbor-obs`)의 게이트 = 이 분산 하락.

산출 그림: `results/stats_eval_sejong_dense_30/figs/{box,forest}_*.png` (분포 박스플롯 / IQM±95% CI forest plot).

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
├── train_mappo.py          # 학습 (diff-waiting-time / pressure 보상)
├── evaluate_mappo.py       # 평가 (2-way / 3-way / 세종 모드, → paired CSV)
├── stats_eval.py           # 통계적 신뢰성 분석 (Friedman/Wilcoxon/A12/부트스트랩 CI)
├── record_video_mappo.py   # 정책 롤아웃 영상
├── record_video_fixed.py   # Fixed-Time 베이스라인 영상 (Ray 비의존)
├── build_3way_compare.py   # 3-way 비교 영상 합성 (지표 오버레이)
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

## Citation & Attribution

> **출처 범위** — 본 프로젝트는 **환경(Environment) 골격을 [SUMO-RL](https://github.com/LucasAlegre/sumo-rl) v1.4.5 (Lucas N. Alegre)에서 채택**했습니다. 구체적으로 ⓐ 액션 스킴(green phase 자동 감지 `Discrete(num_green)` + yellow 자동 삽입), ⓑ 관측 설계 `[phase_one_hot, min_green, density, queue]`, ⓒ `diff-waiting-time` 보상, ⓓ 2x2grid 네트워크(`2x2.net.xml`, 직좌 4-phase ring & barrier)가 이에 해당합니다.
>
> 그 위의 **MAPPO·CTDE 알고리즘(RLlib 기반 자체 구현), BRT 가중 보상(`brt-weight`), switch penalty, BRT 회랑 14-link TLS·관측 패딩, 텔레포트 안전밸브, 세종시 실측 기반 시나리오(2x2-brt / 3x2-brt)는 본 프로젝트의 독자 확장**입니다. (SUMO-RL 라이브러리 자체에는 MAPPO 구현이 없습니다.) 보상 스케일도 SUMO-RL(`/100`)과 달리 `/10`으로 조정했습니다.

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
