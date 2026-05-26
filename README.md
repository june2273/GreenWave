# GreenWave — SUMO 교차로 신호제어 강화학습 (MAPPO / CTDE)

RLlib MAPPO(Multi-Agent PPO)로 SUMO 단일·다중 교차로 신호를 제어하는 실습 프레임워크.
`--map` 사전셋으로 시나리오를 선택합니다: `single` / `2x2` / `2x2-brt` / `3x2` / `3x2-brt`.
각 preset 이 `sumo_cfg` 와 default `tls_ids` 를 자동 해상하며, `--sumo-cfg` / `--tls-ids` 를 명시하면 preset 보다 우선합니다.

**두 가지 학습 모드**:
- **Simple MAPPO** (기본): Actor/Critic 둘 다 local obs만 사용 (decentralized critic)
- **CTDE-MAPPO** (`--ctde`): Actor는 local obs, **Critic은 전체 교차로 global obs**를 봄 → Green Wave 형 교차로 간 협조 학습 유도

**핵심 설계**:
- SUMO TLS phase 자동 감지 → action space `Discrete(num_green)` 동적 결정 (단일/2x2grid/임의 네트워크 자동 호환)
- SUMO program 자동 cycle 차단으로 agent action 만이 phase 전환을 결정
- 양방향 동시 green / 좌회전 / yellow 신호를 정확히 시각화 (matplotlib)
- 학습/평가 CSV에 메타데이터·진단 지표 자동 기록 (mode collapse, memory leak, grid lock 진단)
- Green Wave 효과 직접 측정용 지표 (stops/vehicle, CO2/vehicle, avg_speed, speed_cv, per-direction wait)

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
├── ctde_module.py             # CTDE 커스텀 RLModule (centralized critic, 분리 encoder)
├── map_presets.py             # --map 사전셋 해상 (sumo_cfg + default tls_ids)
├── train_mappo.py             # RLlib MAPPO/CTDE 학습 (--ctde 로 모드 전환)
├── evaluate_mappo.py          # 3-way 비교 (Fixed-time / MAPPO / CTDE) → CSV
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
│   ├── routes_2x2_dense.rou.xml  # 양방향 + 좌회전 cross flow
│   ├── 2x2_brt/               # 2x2 + 좌측 col BRT corridor (3-lane, lane 2 = bus)
│   ├── 3x2/                   # 2 cols × 3 rows, 모든 edge 2-lane
│   └── 3x2_brt/               # 3x2 + 좌측 col BRT corridor
│   # single_intersection.net.xml — 첫 실행 시 netconvert로 자동 생성 (gitignored)
├── models/                    # 학습 체크포인트 (gitignored)
│   ├── MAPPO_sumo_N/                # Simple MAPPO 체크포인트
│   ├── MAPPO_CTDE_sumo_N/           # CTDE-MAPPO 체크포인트 (--ctde 학습 시 자동 분리)
│   └── */train_metadata.json        # 학습 메타데이터 (자동 저장, ctde_mode 플래그 포함)
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
| 단일교차로 (`single`) | `[0, 1, 2, 3]` (모든 phase가 green) | `Discrete(4)` | action 0~3 → phase 0~3 |
| 2x2grid (`2x2`, sumo-rl) | `[0, 2, 4, 6]` (8 phase 중 4개) | `Discrete(4)` | action 0~3 → phase 0/2/4/6 |
| 2x2 BRT (`2x2-brt`) | green 4개 | `Discrete(4)` | BRT TLS: 14-link state, 표준 TLS: 12-link |
| 3x2 (`3x2`) | green 4개 | `Discrete(4)` | 12-link state, 6 TLS |
| 3x2 BRT (`3x2-brt`) | green 4개 | `Discrete(4)` | 좌측 col TLS 14-link, 우측 12-link |

phase 전환 시 SUMO connection 의 `dir` 속성 (s/l/r/t) 기반으로 자동 yellow 삽입.
BRT 시나리오는 NS 직진 phase 에 BRT lane 도 같이 green (별도 BRT phase 없음).

### Observation Space — SUMO-RL 표준 호환

```
[phase_one_hot(num_green), min_green_flag(1), density_per_lane(L), queue_per_lane(L)]
shape = num_green + 1 + 2 * L
```

`L = max(per-agent lanes)` — BRT 시나리오처럼 lane 수가 mixed 인 토폴로지에서는
작은 agent 의 obs 가 density/queue 부분 0 패딩을 받아 같은 shape 로 정렬됩니다.

| 네트워크 (`--map`) | num_green | L (max lanes) | obs_dim |
|----------|-----------|----------------------|-------|
| `single` | 4 | 4 (또는 8) | 13 |
| `2x2` | 4 | 8 | 21 |
| `2x2-brt` | 4 | 10 (3-lane vertical 포함) | 25 |
| `3x2` | 4 | 8 | 21 |
| `3x2-brt` | 4 | 10 | 25 |

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

**공통 switch penalty**: 모든 reward mode 에 `--switch-penalty 0.3` 이 추가 적용됩니다.
agent 가 phase 를 전환한 step 에서 reward 에서 0.3 을 차감 → yellow_seconds 동안 throughput=0 으로
인한 학습 노이즈 / phase oscillation 을 억제. 끄려면 `--switch-penalty 0`.

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

### CTDE 모드 (`--ctde`) — Green Wave 협조 학습

기본 모드는 Actor/Critic 모두 자기 교차로 obs 만 사용합니다 (RLlib PPO 기본). 이는 진정한 MAPPO 가 아니며 교차로 간 협조(Green Wave)를 학습하기 어렵습니다.

**`--ctde` 활성화 시 변경**:

```
─────────────── Simple MAPPO (기본) ───────────────
obs_local ──> [shared encoder] ──> π (actor)
                              └──> V (critic)   ← 자기 교차로만 봄
reward = per-agent local

─────────────── CTDE-MAPPO (--ctde) ───────────────
obs = Box(shape=(D + D × N_agents,))   ← flat 평탄화
       └──────────┬───────────┘
        [local | global concat]

obs[:, :D]  ──> [pi_encoder]  ──> π          ← 실행 시 actor 만 slice 해서 사용
obs[:, D:]  ──> [vf_encoder]  ──> V (critic) ← 학습 시 critic 이 나머지 slice
reward = mean(all agents' local rewards)         ← 모두 동일 (shared)
```

| 항목 | Simple MAPPO | CTDE-MAPPO |
|---|---|---|
| Actor 입력 | local obs | local obs (동일) |
| **Critic 입력** | **local obs** | **전체 agent obs concat (global)** |
| 보상 | per-agent local | `mean(all local rewards)` (shared) 또는 local |
| Decentralized 실행? | ✓ | ✓ (actor는 global을 절대 안 봄) |
| Green Wave 협조 학습 | △ (어려움) | ○ (critic이 교차로 간 인과 학습) |
| 체크포인트 경로 | `models/MAPPO_sumo_N/` | `models/MAPPO_CTDE_sumo_N/` |

**구현**:
- `ctde_module.py:CentralizedCriticPPOModule` — `DefaultPPOTorchRLModule` 상속, `pi_encoder`/`vf_encoder` 분리. `_forward_train`이 `Columns.EMBEDDINGS`를 emit 하지 않아 PPO learner가 `compute_values(batch, None)` 호출 → critic이 global slice 재인코딩
- `env_sumo_pz.py:ctde_mode=True` → obs 가 **flat `Box(D + D×N,)`** 로 평탄화 (`[local | global concat]`). global concat 은 `possible_agents` 순서로 stable.
- `train_mappo.py` 가 `RLModuleSpec.model_config["local_dim"]` 로 slice 경계 (`D`) 를 모듈에 전달. 모듈은 `obs[..., :local_dim]` (actor) / `obs[..., local_dim:]` (critic) 로 slice
- `--ctde-reward shared` (기본): 모든 agent 가 `mean()` 받음. `local`: per-agent reward 유지 (critic만 centralized)
- **단일 교차로 + `--ctde`**: degenerate (global == local 영역 복제). 경고 출력 후 정상 동작 (backward compat)

> **왜 Dict 가 아니라 flat Box 인가**: Dict observation (`{"local", "global"}`) 으로 먼저 구현했더니 `--num-workers ≥ 1` (별도 Ray actor) 에서 RLlib connector pipeline 이 Dict subspace 를 silent drop → `total_steps=0`, NaN reward, weight 업데이트 0회. Driver mode (`--num-workers 0`) 에서는 정상이라 진단 후 flat Box 로 전환. 표준 단일 텐서 라 worker process 호환성 확보, multi-worker 학습 정상 동작 확인.

---

## 실행 순서

### 1. 학습

```bash
# 빠른 검증 (10 iter, ~10분, 단일교차로)
python train_mappo.py --map single --num-iters 10 --reward-mode diff-waiting-time --seed 42

# 본 학습 — 단일교차로 (150 iter)
python train_mappo.py --map single --num-iters 150 --num-workers 1 --seed 42

# 다중 교차로 — 2x2grid 기본 트래픽
python train_mappo.py --map 2x2 \
  --reward-mode diff-waiting-time \
  --num-iters 150 --num-workers 4 --seed 42

# 다중 교차로 — 2x2grid 정체 시나리오 (--traffic high)
python train_mappo.py --map 2x2 --traffic high \
  --reward-mode diff-waiting-time \
  --num-iters 150 --num-workers 4 --seed 42

# BRT 시나리오 — 2x2 + 좌측 col BRT corridor (3-lane, lane 2 = bus)
python train_mappo.py --map 2x2-brt \
  --reward-mode diff-waiting-time \
  --num-iters 200 --num-workers 4 --seed 42

# 3x2 (6 TLS, 2 cols × 3 rows)
python train_mappo.py --map 3x2 --num-iters 200 --num-workers 4 --seed 42

# 3x2 + BRT corridor (6 TLS, 좌측 col 3-lane)
python train_mappo.py --map 3x2-brt --num-iters 200 --num-workers 4 --seed 42

# CTDE-MAPPO — centralized critic + shared reward (Green Wave 협조 학습)
# 체크포인트는 models/MAPPO_CTDE_sumo_N/ 에 자동 분리 저장
python train_mappo.py --ctde --map 2x2 --traffic high \
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
| `--map` | `single` | 시나리오 사전셋 (`single` / `2x2` / `2x2-brt` / `3x2` / `3x2-brt`) |
| `--tls-ids` | preset | SUMO TLS id 목록. 미지정 시 `--map` preset 값 사용 |
| `--sumo-cfg` | preset | SUMO 설정 파일. 명시 시 `--map`/`--traffic` preset 보다 우선 |
| `--traffic` | `default` | `default`: 원본 routes / `high`: `2x2 + high` 조합에서 `2x2grid_dense.sumocfg` 자동 사용 |
| `--seed` | 42 | 전역 랜덤 시드 (random/numpy/torch 일괄) |
| `--reward-mode` | `queue` | `queue` / `diff-waiting-time` / `pressure` |
| `--switch-penalty` | 0.3 | phase 전환 시 reward 차감량 (oscillation 억제). 0 으로 끌 수 있음 |
| `--max-steps` | 3600 | 에피소드 최대 sim 초 |
| `--delta-time` | 5 | env step당 진행 sim 초 |
| `--min-green` | 13 | 최소 green 유지 시간 (초). 10→13: oscillation 억제, delta_time=5 기준 3 env step floor |
| `--yellow-time` | 2 | green→green 전환 시 yellow 시간 (초) |
| `--ctde` | off | centralized critic (Green Wave 협조 학습) 활성화. 체크포인트는 `MAPPO_CTDE_sumo_N` |
| `--ctde-reward` | `shared` | `--ctde` 와 함께: `shared`=mean(all rewards), `local`=per-agent reward 유지 |

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
| `entropy_coeff` | **`0.03`** | 0.02 → 0.005 (B4) → 0.03 (B5: 좌회전 phase mode collapse 방지. 초기 negative reward에 노출된 action이 확률 0으로 collapse되는 현상을 entropy bonus 6× 강화로 억제) |
| `vf_clip_param` | **`1000.0`** | 500 → 10 → 1000 (A3: VF signal 복원. 10으로 낮췄더니 `vf_loss_unclipped=175189` 대비 `vf_loss=9.89`로 VF 학습신호 99% 소실 → 1000으로 복원) |

### 2. 평가

```bash
# 2-way 쌍대 비교 (FixedTime vs MAPPO) — 단일교차로
# 출력: results/eval_metrics_mappo_N.csv  (N = 모델 버전 번호 자동)
python evaluate_mappo.py --model models/MAPPO_sumo_1 --map single --episodes 5

# 학습 시와 동일한 --map / --reward-mode / --traffic 전달
python evaluate_mappo.py --model models/MAPPO_sumo_11 \
  --map 2x2 --traffic high \
  --reward-mode diff-waiting-time --episodes 5

# BRT 시나리오 평가
python evaluate_mappo.py --model models/MAPPO_sumo_20 \
  --map 2x2-brt \
  --reward-mode diff-waiting-time --episodes 5

# 3-way 비교 (FixedTime vs MAPPO vs CTDE-MAPPO) — Green Wave 검증
# --model-ctde 지정 시 CTDE 정책이 자동 추가됨
python evaluate_mappo.py \
  --model      models/MAPPO_sumo_11 \
  --model-ctde models/MAPPO_CTDE_sumo_1 \
  --map 2x2 --traffic high \
  --reward-mode diff-waiting-time --episodes 10
```

> **쌍대 비교**: 동일 에피소드 인덱스에 동일 SUMO seed 를 사용해 교통 수요 차이가 아닌 정책 성능 차이만 측정.

#### CSV 출력 형식

평가 CSV 는 3가지 섹션을 포함:

1. **메타데이터 prefix 컬럼** (모든 행에 동일값):
   `model_path`, `model_path_ctde`, `train_iter`, `train_iter_ctde`, `train_total_steps`, `reward_mode`, `ctde_reward`, `lr`, `entropy_coeff`, `vf_clip_param`, `train_seed`, `tls_ids`, `num_workers`, `traffic`
2. **raw 행** (에피소드별): `algorithm` (MAPPO/CTDE/FixedTime), `episode`, `seed`, 메트릭들
3. **요약 행**: `MAPPO_mean`, `MAPPO_std`, `CTDE_mean`, `CTDE_std`, `FixedTime_mean`, `FixedTime_std` (CTDE는 `--model-ctde` 지정 시에만)

#### 평가 메트릭

**기존 지표 (간접 측정)**:

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

**Oscillation / Lane Saturation 진단 지표**:

| 컬럼 | 설명 |
|------|------|
| `yellow_seconds` | 에피소드 내 yellow phase 로 진행된 총 sim 초 |
| `yellow_ratio` | `yellow_seconds / total_sim_sec` — 높을수록 불필요한 switching 의심 |
| `vehicles_loaded` | SUMO에 로드된 누적 차량 수 (route 수요 총량) |
| `vehicles_departed` | 실제 출발한 누적 차량 수 |
| `vehicles_lost_insert` | `loaded - departed` — lane 포화로 시뮬레이션에 진입 못 한 차량 수 (silent drop) |
| `pending_insert_peak` | 에피소드 내 동시 삽입 대기 차량 최대치 |

**Green Wave 직접 측정 지표 (Tier 1)**:

| 컬럼 | 의미 | 좋은 값 |
|------|------|--------|
| `avg_stops_per_vehicle` | 차량당 평균 정지 횟수 (속도 0.1 m/s 이하 transition) | **↓ 낮을수록 좋음** (Green Wave의 정의 자체) |
| `avg_co2_per_vehicle` | 차량당 CO2 배출량 (mg) | ↓ stop-and-go 가 줄면 즉시 감소 |
| `avg_speed` | 평균 차량 속도 (m/s) | ↑ 부드러운 흐름 = 높음 |

**Green Wave 코리도어 분석 (Tier 2)**:

| 컬럼 | 의미 | 좋은 값 |
|------|------|--------|
| `speed_cv` | 속도 변동계수 `std(v)/mean(v)` | ↓ Green Wave면 차량들이 균일 속도 유지 |
| `per_direction_wait_ew` | 동-서 축 평균 lane 대기시간 | ↓ 한 축에 Green Wave 형성 시 그쪽이 우수 |
| `per_direction_wait_ns` | 남-북 축 평균 lane 대기시간 | ↓ 동일 |

**기대 결과 패턴** (`--ctde` 효과 검증):

| 지표 | FixedTime | MAPPO | CTDE-MAPPO |
|------|-----------|-------|------------|
| `avg_waiting_time` | 높음 | 중간 | **낮음** |
| `avg_stops_per_vehicle` | 높음 (~2-3) | 중간 (~1.5) | **낮음 (~0.5-1)** |
| `avg_co2_per_vehicle` | 높음 | 중간 | **낮음** |
| `avg_speed` | 낮음 | 중간 | **높음** |
| `speed_cv` | 높음 | 중간 | **낮음** |

### 3. 롤아웃 영상 저장

```bash
# 기본: continuous 모드 (sim sec 당 1 frame, 자연스러운 흐름)
# 출력: videos/mappo_policy_rollout_N.mp4
python record_video_mappo.py --model models/MAPPO_sumo_11 \
  --map 2x2 --traffic high \
  --reward-mode diff-waiting-time

# BRT 시나리오 영상
python record_video_mappo.py --model models/MAPPO_sumo_20 --map 2x2-brt

# 짧은 요약 영상 (short 모드 — env step 당 1 frame)
python record_video_mappo.py --model models/MAPPO_sumo_11 \
  --map 2x2 --mode short --fps 5
```

> **CTDE 체크포인트 영상 녹화**: CTDE obs 가 flat Box (Simple MAPPO 와 같은 단일 ndarray) 로 평탄화되어 있어 호출 형태는 호환되지만, `record_video_mappo.py` 가 현재 env 인스턴스화 시 `ctde_mode=True` 를 전달하지 않아 obs 차원이 달라집니다 (`Box(D,)` vs `Box(D+D×N,)`). CTDE 모델 녹화가 필요하면 env_kwargs 에 `ctde_mode=True` 만 추가하면 됩니다 — 추론 호출 코드는 그대로 사용 가능.

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
| **policy/phase_switches** | 에피소드 누적 phase 전환 (4 agent 2x2 정상 범위: 900~1200) |
| **policy/max_queue** | 단일 lane 최대 큐 |
| **policy/teleported** | grid lock 텔레포트 (0 이상이면 정체 한계 초과) |
| **policy/yellow_ratio** | 전체 sim시간 중 yellow 비율 (oscillation 진단; 값이 높으면 불필요한 switching 의심) |
| **policy/vehicles_lost_insert** | lane 포화로 시뮬레이션에 진입 못 한 차량 수 (dense traffic 좌회전 lane saturation 진단) |
| **system/process_rss_mb** | 프로세스 메모리 (누수 감지) |
| **system/sumo_process_count** | SUMO 프로세스 수 (좀비 감지) |
| **system/iter_time_sec** | iter 소요 시간 |

---

## 학습 진단 가이드

**Regression 통과 기준** (10 iter 학습 후):

1. ✅ `reward/mean` 상승 추세 (초기 음수 → 점진 증가)
2. ✅ `policy/action_dist/action_{0,1,2,3}` 모두 **5% 이상** (한쪽 0% 이면 mode collapse)
3. ✅ `loss/entropy` 점진 감소 (0 근처 붕괴 X)
4. ✅ `loss/value` 지속 감소 (수백 → 수십 범위. 정체하면 `vf_clip_param` 확인)
5. ✅ `policy/teleported` 0~소수 (grid lock 없음)
6. ✅ `system/process_rss_mb` 안정 (지속 증가 X)
7. ✅ `system/sumo_process_count` 일정 (좀비 없음)

**문제 진단**:

| 증상 | 원인 후보 | 대응 |
|------|----------|------|
| action 한쪽 0% / 99% | mode collapse (초기 negative reward → collapse) | `entropy_coeff` ↑ (현재 0.03), traffic 다양화 |
| `loss/value` 수렴 안 하고 정체 | `vf_clip_param` 너무 작음 (학습신호 clip 소실) | `vf_clip_param` ↑ (현재 1000) |
| `loss/value` 거의 0 | reward magnitude 부족 | `--traffic high`, reward `/10` 정규화 확인 |
| `phase_switches` > 1200 (4 agent 기준) | 비정상적 과도 switching | `--switch-penalty` ↑, `--min-green` ↑ |
| `policy/yellow_ratio` > 0.15 | phase oscillation (매 step switching) | `--switch-penalty` ↑, `entropy_coeff` ↓ |
| `teleported` > 0 | 정체 한계 초과 (grid lock) | `--traffic` 낮추기, network 수정 |
| `vehicles_lost_insert` 급증 | dense traffic에서 lane 포화 (silent drop) | 차량 수요 완화, 좌회전 phase 학습 확인 |
| `process_rss_mb` 우상향 | 메모리 누수 | env close() 누락 확인 |
| `train_total_steps=0` (CTDE) | obs space 가 Dict 면 worker connector 가 silent drop | flat Box 평탄화 확인 (`obs_space.shape=(D+D×N,)`). 본 repo는 이미 적용됨 |
| `train_total_steps=0` (일반) | Ray worker timeout | `sample_timeout_s` 확인 (현재 600s) |

> **참고**: 4 agent 2x2 grid 에서 `phase_switches` 900~1200 은 정상 범위입니다 (agent 당 225~300회, min_green=13/delta_time=5 기준 최소 2.6 step 간격).

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
#  "vf_clip_param", "train_iter", "train_total_steps", "traffic",
#  "ctde_mode": False, "ctde_reward": None, ...}
# CTDE 체크포인트는 ctde_mode=True 와 ctde_reward="shared"|"local" 가 추가됨
```

**CTDE 체크포인트 복원**:

```python
import torch
from ray.rllib.algorithms.ppo import PPO
from ctde_module import CentralizedCriticPPOModule  # 명시적 import 필요 (FQN 해석)

algo = PPO.from_checkpoint("models/MAPPO_CTDE_sumo_1")
module = algo.get_module("shared_policy")

# CTDE 환경의 obs 는 flat Box(D + D×N,) — Simple MAPPO 와 동일한 단일 ndarray
obs, _ = env.reset()  # env 는 ctde_mode=True 로 생성
action = int(torch.argmax(
    module.forward_inference(
        {"obs": torch.tensor(obs["tl_0"][None], dtype=torch.float32)}
    )["action_dist_inputs"],
    dim=-1,
).item())
# 모듈 내부에서 obs[:, :local_dim] (local) 만 actor 에 사용. 호출자는 평탄화/슬라이싱
# 신경 쓸 필요 없음. Simple MAPPO 와 동일한 호출 패턴.
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

### 학습/평가/영상 호출 일관성
- `--reward-mode`, `--sumo-cfg`/`--traffic`, `--tls-ids` 는 학습·평가·영상에서 **동일하게 지정** 필요
- 평가 CSV 의 메타 prefix 컬럼이 학습 메타데이터 (`train_metadata.json`) 를 자동 로드해 출처 추적 가능

### CTDE 비교 실험 워크플로우
1. **MAPPO 베이스라인 학습**: `python train_mappo.py --map 2x2 --traffic high --num-iters 200`  → `models/MAPPO_sumo_N/`
2. **CTDE 학습** (같은 hyperparam 으로): `python train_mappo.py --ctde --map 2x2 --traffic high --num-iters 200`  → `models/MAPPO_CTDE_sumo_M/`
3. **3-way 비교 평가**: `python evaluate_mappo.py --model models/MAPPO_sumo_N --model-ctde models/MAPPO_CTDE_sumo_M --map 2x2 --traffic high --episodes 10`
4. CSV (`results/eval_metrics_mappo_N.csv`) 의 `avg_stops_per_vehicle`, `avg_co2_per_vehicle`, `speed_cv` 비교 → CTDE 가 Green Wave 효과를 학습했는지 검증
- **CTDE 는 다중 교차로에서만 의미**: 단일 교차로 (`--map single`) 에 `--ctde` 를 주면 global obs = local obs 가 되어 효과 없음 (경고 출력)
- **공정 비교**: 두 모델은 동일한 `--seed`, `--num-iters`, `--reward-mode`, `--map`, `--traffic` 으로 학습해야 critic 구조 차이만 비교 가능

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

Cite:
@misc{sumorl,
    author = {Lucas N. Alegre},
    title = {{SUMO-RL}},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/LucasAlegre/sumo-rl}},
}
