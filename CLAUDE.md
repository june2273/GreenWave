# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GreenWave is a reinforcement learning project for traffic signal control using MAPPO (Multi-Agent PPO). It uses SUMO (Simulation of Urban MObility) as the traffic simulator, PettingZoo as the multi-agent environment interface, and RLlib for the MAPPO algorithm. Scenarios are selected via `--map` (single / 2x2 / 2x2-brt / 3x2 / 3x2-brt); each preset auto-resolves `sumo_cfg` and default `tls_ids`.

**출처/Attribution**: 환경(Environment) 골격은 [SUMO-RL](https://github.com/LucasAlegre/sumo-rl) v1.4.5 (Lucas N. Alegre)에서 채택 — 액션 스킴(`Discrete(num_green)` + yellow 자동삽입), 관측 `[phase_one_hot, min_green, density, queue]`, `diff-waiting-time` 보상, 2x2grid net(`2x2.net.xml`, 직좌 4-phase ring & barrier). 그 위의 **MAPPO·CTDE(RLlib 자체 구현), BRT 가중 보상, switch penalty, BRT 14-link TLS·obs 패딩, 텔레포트 안전밸브, 세종 실측 시나리오는 독자 확장** (SUMO-RL 에는 MAPPO 구현 없음; 보상 스케일도 `/100`→`/10` 수정). 후보였던 thisisjonchen/mschrader15/maxbrenner-ai/Navtegh 레포는 코드·문서·설정 어디에도 흔적 없음 (참조처 아님).

## Prerequisites

**System binaries** (macOS via Homebrew):
```bash
brew install sumo
```

**Environment variables** (required for TraCI/sumolib):
```bash
export SUMO_HOME=$(brew --prefix sumo)/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```
  
**Python dependencies** (in `.venv`):
- `ray[rllib]`, `torch`, `pettingzoo`, `numpy`, `pandas`
- `traci==1.20.0`, `sumolib==1.20.0` (pinned to match Homebrew SUMO 1.20.0)
- `imageio`, `imageio-ffmpeg`, `matplotlib`

## Commands

시나리오 역할 이분화:
- `2x2` / `2x2-brt` — 모델 개발·빠른 실험 (FIXED / MAPPO / CTDE 비교)
- `3x2-brt --traffic high` — **세종시 현실 시뮬레이션 최종 버전** (BRT + CTDE + 실측 Fixed-Time)

```bash
# ── 개발·실험용 (2x2) ────────────────────────────────────────────────────────
python train_mappo.py --map 2x2 --num-iters 200 --num-workers 1
python train_mappo.py --map 2x2-brt --num-iters 200          # BRT corridor on left col
python train_mappo.py --map 2x2-brt --ctde                   # CTDE-MAPPO (per-agent local reward), saves to MAPPO_CTDE_sumo_N/
python train_mappo.py --map 2x2-brt --ctde --neighbor-obs        # topology-aware: actor 에 N/S 이웃 혼잡 요약(+4dim) + critic 4-slot
python train_mappo.py --map 3x2-brt --traffic high --ctde --neighbor-obs --upstream-phase \
  --progression-coeff 3 --brt-prog-weight 4                       # RL-native Green Wave: ②상류 위상 관측 + ①BRT 가중 회랑 progression (설계: "Green Wave Progression" 섹션)

# ── 세종시 현실 시뮬레이션 최종 버전 (3x2-brt) ────────────────────────────────
python train_mappo.py --map 3x2-brt --traffic high --num-iters 300 --num-workers 1
python train_mappo.py --map 3x2-brt --traffic high --ctde --num-iters 300  # CTDE 세종 최종

# Topology-aware CTDE (--neighbor-obs) 재학습 — old CTDE 절차 복제(공정 게이트):
#   old CTDE_sejong_dense 는 dense-direct 가 아니라 light warmup→dense weights-only resume.
#   --neighbor-obs(+--upstream-phase) 는 양 phase 필수(obs 차원 일관 → resume 호환). --out bare name 은 자동 models/ 저장.
python train_mappo.py --map 3x2-brt --ctde --neighbor-obs \
  --num-iters 135 --num-workers 5 --seed 42 --out models/CTDE_sejong_nb            # Phase1 warmup
python train_mappo.py --map 3x2-brt --traffic high --ctde --neighbor-obs \
  --resume-from models/CTDE_sejong_nb --resume-weights-only \
  --num-iters 65 --num-workers 5 --seed 42 --out models/CTDE_sejong_dense_nb        # Phase2 dense (+65→200)
# 게이트 판정: new CTDE vs old CTDE(eval_sejong_dense_30.csv) — 성공 = throughput/travel 분산이
#   MAPPO 수준으로 하락(고분산이 old CTDE 의 진짜 증상; avg_waiting 동률에 가려졌던 것).
#   확실히 개선되면 그때 MAPPO 도 --neighbor-obs 재학습해야 new-CTDE-vs-MAPPO 가 공정해짐.
# ⚠️ eval_sejong_dense_30.csv 는 2026-06-11 지표 수정(waiting-time-memory 포화 해제 + sejong
#   baseline sim-time 화) *이전* 산출물 — 새 eval 과 직접 비교 불가. 게이트를 쓰려면 old
#   체크포인트들(MAPPO_sejong_dense·CTDE_sejong_dense)을 수정된 evaluate 로 30-episode
#   재평가해 게이트 CSV 를 재생성할 것 (체크포인트는 유효 — 재학습 불필요).

# ── 기타 ─────────────────────────────────────────────────────────────────────
python train_mappo.py --map 3x2 --num-iters 200              # 2 cols x 3 rows, all 2-lane

# Monitor training
tensorboard --logdir results/tb_mappo

# Evaluate (use same --map as training)
python evaluate_mappo.py --model models/MAPPO_sumo_1 --map 2x2-brt --episodes 5
python evaluate_mappo.py --model models/MAPPO_sumo_1 --model-ctde models/MAPPO_CTDE_sumo_1 --map 2x2-brt

# 세종시 실측 Fixed-Time 비교 (3x2-brt 최종 평가)
python evaluate_mappo.py \
  --model models/MAPPO_sumo_N \
  --model-ctde models/MAPPO_CTDE_sumo_N \
  --map 3x2-brt --traffic high \
  --baseline sejong
# → 공공데이터포털(2023)·세담터(2026.04) 실측 자료 기반 per-TLS 신호 자동 적용 (SEJONG_PER_TLS_PHASE_SECONDS, 6 TLS 각각 다른 cycle):
#   tl_0 성금=[45,25,45,25] / tl_1 청사=[46,20,31,23] / tl_2 도움4로=[45,25,45,25]
#   tl_3 도움4로=[46,20,31,23] / tl_4 어진=[47,20,47,28] / tl_5 가름로=[45,20,35,20]
# FixedTime(Sejong) vs MAPPO vs CTDE 3-way 비교

# Record policy rollout video
python record_video_mappo.py --model models/MAPPO_sumo_1 --map 2x2-brt
python record_video_mappo.py --model models/MAPPO_sumo_1 --map 2x2-brt --mode continuous  # default, ~3600 frames, 10fps → 약 6분 영상
python record_video_mappo.py --model models/MAPPO_sumo_1 --map 2x2-brt --mode short       # env.step() 당 1 frame (~720 frames), 5fps 권장 → 약 2.4분 영상

# Record Fixed-Time baseline video (RL 모델 없음, 비교용)
# 3-way 비교용 (균등 4×15s 사이클, 모든 agent 동기):
python record_video_fixed.py --baseline symmetric --map 2x2-brt
# → videos/fixed_symmetric_2x2_brt.mp4

# 세종시 실측 per-TLS 비대칭 신호 (3x2-brt 최종 시뮬레이션):
python record_video_fixed.py --baseline sejong --map 3x2-brt --traffic high
# → videos/fixed_sejong_3x2_brt.mp4 (SEJONG_PER_TLS_PHASE_SECONDS, 6 TLS 각각 다른 cycle)

# ── 통계적 신뢰성 분석 (stats_eval.py) ───────────────────────────────────────
# evaluate_mappo.py 의 paired CSV → 통계 검정·효과크기·신뢰구간으로 "신뢰성" 증거 생성.
# (성능 지표가 좋다 ≠ 우연이 아니다. RL 평가 표준: 부트스트랩 CI + 비모수 검정.)
# 1) 검정력 확보를 위해 평가를 충분한 episode 로 (5 → 권장 30):
python evaluate_mappo.py --model models/MAPPO_sumo_N --model-ctde models/MAPPO_CTDE_sumo_N \
  --map 3x2-brt --traffic high --baseline sejong --episodes 30
# 2) 생성된 paired CSV 를 통계 분석 (재학습·SUMO 불필요, 수 초):
python stats_eval.py --csv results/eval_metrics_mappo_N.csv
# → results/stats_eval_metrics_mappo_N/ 아래:
#   stats_tests.csv   (Friedman + 쌍별 Wilcoxon+Holm + A12 효과크기)
#   stats_ci.csv      (알고리즘별 mean·IQM 의 부트스트랩 95% CI)
#   stats_summary.txt (사람이 읽는 요약 + 발표용 헤드라인 문장 자동 생성)
#   figs/box_*.png, figs/forest_*.png (분포 박스플롯 / IQM±CI forest plot)
# 주의: paired n 이 작으면(예 5) Wilcoxon 의 양측 최소 p = 2/2ⁿ = 0.0625 라
#   Friedman 이 유의해도 쌍별 검정이 절대 유의 불가 → 쌍별 유의에는 n≥7 (Holm 3쌍이면
#   더 필요), 안정적 결론엔 n≈30 권장. A12 효과크기는 n 작아도 해석 가능.

# Run environment smoke test (default: single intersection)
python env_sumo_pz.py
```

`train_mappo.py`, `evaluate_mappo.py`, `record_video_mappo.py`, `record_video_fixed.py` 4개 상위 스크립트는 완료 후 `os._exit()` 로 종료해 Ray worker·SUMO 좀비 프로세스를 정리한다 (인터프리터 hang 방지). `stats_eval.py` 는 SUMO/Ray 비의존 순수 사후분석이라 일반 종료한다.

## Architecture

```
RLlib PPO (shared policy)  →({agent: action})→  SumoParallelEnv  →(TraCI)→  SUMO Simulator
                                                       ↑                            |
                                                 sumo_data/ XML              lane states
                                                                                    ↓
RLlib PPO (shared policy)  ←({agent: obs/rew})←  SumoParallelEnv  ←───────────────┘
```

**`env_sumo_pz.py`** — The core PettingZoo `ParallelEnv` (`SumoParallelEnv`):
- Agent IDs: `tl_0`, `tl_1`, ... mapped to SUMO TLS IDs (`"C"`, `"D"`, ...).
- Each `reset()` spawns a new SUMO process via `traci.start()` with a unique UUID label, so parallel envs never conflict.
- If `sumo_data/single/single_intersection.net.xml` is missing, `_maybe_build_network()` auto-generates it via `netconvert`.
- `step()` applies yellow-phase transitions and enforces minimum green time per agent before allowing phase changes.
- All agents share one SUMO connection — phase switching is coordinated in a single `simulationStep()` loop.
- `__init__` runs a one-shot SUMO probe (`_probe_network_spec()`) to extract controlled lanes, green phase indices, and lane capacities; this determines obs/act space shapes before the first `reset()`.
- per-vehicle/lane 데이터 수집은 **TraCI subscription** 기반 (`_subscribe_lanes` + 신규 출발 차량 구독 → `getAllSubscriptionResults` 로컬 조회). 과거 per-vehicle 폴링(LOS D 에서 초당 ~7,000 IPC)을 대체해 rollout **7.5배 가속** (226→30 ms/env-step). 전환 시 20-step bit-identical 게이트로 수치 동일성 검증 완료 — 수집 로직 수정 시 같은 게이트 절차 권장.
- `--neighbor-obs`/`--upstream-phase` 경로: probe 가 `_derive_neighbor_dirs()`(연결성 `out∩in`+상대좌표로 N/E/S/W 분류)로 `neighbor_dir`, `_is_brt_lane()`(`getAllowed`)로 `corridor_agents`/`corridor_ns_lanes` 자동 도출. `_compute_obs` 가 `_agent_dq_summary`(혼잡)+`_agent_phase_summary`(위상 시계) 캐시 → `_enrich_obs()` 가 2-pass 로 **N/S 이웃**(`_OBS_NBR_DIRS`)만 방출(+4 / upstream 면 +14) → `_wrap_obs()` 가 CTDE 면 `[own | agent_id | N/E/S/W 이웃 obs]` flat Box(critic 4-slot). ① progression 은 reward 루프에서 `_corridor_step()` 가 corridor agent 의 ns-through 가중 평균 속도비를 가산(+ `corridor_brt_speed_ratio` 누적).

**Action / Observation / Reward (per agent):**
- Actions: `Discrete(num_green)` — green phase 수는 네트워크에서 자동 감지 (모든 시나리오 4). yellow phase는 정책이 직접 선택 불가, green 전환 시 자동 삽입. **의미 매핑**: `single` 은 4 방향 round-robin (한 번에 한 방향 직진+좌회전), `2x2`/`3x2` 표준은 ring & barrier 비대칭 (action 0/2 = NS/EW 양방향 직진+우회전 33s, action 1/3 = NS/EW 양방향 좌회전 protected 6s). NS 와 EW 절대 동시 green 안 됨.
- Observation: `[phase_one_hot(num_green), min_green_flag(1), density_per_lane(L), queue_per_lane(L)]` shape = `num_green + 1 + 2*L`. **`L=max(per-agent lanes)`**; 작은 agent의 obs는 density/queue 부분을 0 패딩 (BRT 시나리오의 mixed-lane topology 지원). 시나리오별 `obs_dim`: single=13, 2x2/3x2=21, 2x2-brt/3x2-brt=25.
- **RL-native Green Wave 확장 (default off, 설계: "Green Wave Progression" 섹션)** — 회랑(BRT corridor, N/S축)에서 무정차 진행파를 *학습*으로 유도하는 3 플래그:
  - `--neighbor-obs` (#3): 각 agent actor obs 끝에 **N/S 이웃**의 상류 혼잡 요약 `[mean_density, mean_queue]×2 = +4 dim`. (#1) `--ctde` 동반 시 critic 입력 = `[own | agent_id one-hot(N) | N/E/S/W 이웃 obs]` — **actor-lean(N/S)/critic-rich(4방향)**. 인접·방향은 `_derive_neighbor_dirs`(net 연결성+상대좌표)로 자동 도출.
  - `--upstream-phase` (②, `--neighbor-obs` 필요): N/S 이웃의 위상 시계 `[phase_one_hot(num_green), elapsed_norm] × 2 = +10 dim` 추가 → 상류 platoon 도착 타이밍 관측, green wave 오프셋 학습.
  - `--progression-coeff β`, `--brt-prog-weight` (①, reward): corridor agent(=BRT 전용차로 보유, `getAllowed` 자동식별)의 ns-through lane **가중 평균 속도비**를 β배 가산 → 무정차 통과 보상. `brt_prog_weight>1`=BRT 우선(TSP). β는 reward_mode 의존(diff-waiting 1~5 / pressure 5~20, smoke 실측).
  - **차원**(3x2-brt): obs 25(off) → 29(neighbor) → 39(+upstream); CTDE critic 151 → **201**. obs 차원 변경이라 미지정 ckpt 비호환 — `train_metadata.json`의 `neighbor_obs`/`upstream_phase` 추적, evaluate/record 모델별 자동 로드, resume 값 불일치 시 거부(progression은 reward-only→경고). 진행파 측정: info `corridor_brt_speed_ratio`.
- Reward: `--reward-mode` 로 2개 모드 (`REWARD_MODES = ["diff-waiting-time", "pressure"]`). **`diff-waiting-time` (기본)** — `(prev − current) / 10`, 자기 진입로 차분 대기시간. **`pressure`** — `Σ#(out_lanes 차량) − Σ#(in_lanes 차량)` (max-pressure/backpressure, Varaiya 2013·PressLight 2019; 하류 포화·스필백 억제). pressure 는 스케일이 작아(±수십 vs diff-waiting ~수천) `switch_penalty=0.45` 가 과해짐 → `--switch-penalty 0.1` 전후 권장, 차량 *수* 기반이라 `brt_weight` 미적용. 두 모드 공통 `switch_penalty=0.45` (phase switching 시 reward -= 0.45, oscillation 억제, yellow_time=3s 기준).
- `--brt-weight w` (default 1.0, BRT 시나리오 전용): 보상함수에서 BRT(vClass=bus) 대기시간에 부여하는 중요도 계수. 두 경로 모두 **연속 정지시간**(`vehicle.getWaitingTime`; 움직이면 리셋) 기준 — `w=1.0` = 일반 차량과 동등 취급 (baseline, `lane.getWaitingTime` 합과 수학적으로 동일; 과거 w>1 경로가 `getAccumulatedWaitingTime` 을 써서 가중치 토글이 측정치 정의까지 바꾸던 버그는 수정됨). `w>1` → BRT 대기시간 감소를 더 크게 보상해 정책이 BRT 우선 신호를 학습하도록 유도. 실제 대기시간을 늘리는 게 아닌 보상 신호 비중 조정 (`current_wait = Σ_veh w_veh × wait / 10`). 권장: `2.0~3.0`. info dict 의 `avg_wait_brt/car`, `brt_seen`, `car_seen` 는 w 와 무관하게 항상 기록.

**`train_mappo.py`** — Builds a `PPOConfig` with shared policy across all agents, runs the training loop, writes TensorBoard scalars to `results/tb_mappo/<run_name>/`, and saves RLlib checkpoints (directory format) to `models/MAPPO_sumo_N/`. `--ctde` flag enables centralized critic (CTDE-MAPPO), saving to `models/MAPPO_CTDE_sumo_N/`. A `train_metadata.json` is saved alongside each checkpoint with all hyperparameters and map settings. 학습 루프는 `value/explained_var`(크리틱 fit 품질: 1.0=완벽, ~0=평균만, <0=평균보다 나쁨; <0.7 면 크리틱 undertrained → centralized critic 강화 ROI 신호) 스칼라를 TensorBoard 에 기록하고, iter 1 에서 learner 가 노출하는 실제 키 목록을 1회 진단 덤프한다 (RLlib 버전별 키명 차이 대응 — `vf_explained_var`/`vf_explained_variance`/`explained_variance` 순회). **MAPPO·CTDE 하이퍼파라미터 동등성**: `--ctde` 는 `.rl_module(...)` 로 RLModule 만 `CentralizedCriticPPOModule` 로 교체할 뿐 `.training(**hparams)`(lr·entropy_coeff·train_batch_size·clip·vf_clip 등)는 두 경로가 동일하게 적용 → 같은 CLI flag(+`--ctde`)로 학습하면 공정 비교 보장 (차이는 centralized critic 뿐).

**`evaluate_mappo.py`** — 3-way comparison: MAPPO vs CTDE-MAPPO (optional) vs Fixed-time. Results saved to `results/eval_metrics_mappo_N.csv` with metadata prefix columns for model traceability. `--baseline` 선택: `symmetric` (균등 step-기반 사이클 = green 15s + yellow 3s 실 18s/phase; sim-time 화하면 green 12s < min_green 13 으로 전환 묵살되므로 의도적으로 step-기반 유지) / `sejong` (공공데이터포털·세담터 실측 자료 기반 per-TLS 비대칭 신호 — `SEJONG_PER_TLS_PHASE_SECONDS` dict 의 6 TLS 매핑. **sim-time 기반**: `env.sim_step % cycle` 로 phase 결정 → yellow 가 다음 phase green 을 잠식해 실측 사이클 길이(예 tl_0 140s)를 그대로 재현. 과거 step-기반은 사이클이 +12s/cycle(~9%) 늘어났었음 — 2026-06-11 이전 eval CSV 의 FixedTime(Sejong) 행동과 단절). **`neighbor_obs` 는 env_kwargs 공유가 아니라 모델별로 개별 로드** (`train_meta`/`train_meta_ctde` 에서 `nb_mappo`/`nb_ctde`): 옛 모델(False)과 topology-aware 재학습 모델(True)이 섞여도 각 정책이 학습 때와 같은 obs 를 보게 함 (SUMO 동역학·routes·reward 동일 → 공정). FixedTime 은 obs 미사용이라 무관. `upstream_phase` 도 동일하게 모델별 로드(`up_mappo`/`up_ctde`); progression(①)은 reward-only 라 추론 미적용. CSV 에 `neighbor_obs`/`upstream_phase`/`progression_coeff`/`brt_prog_weight`(+ `_ctde`) + `corridor_brt_speed_ratio` 컬럼 기록. `record_video_mappo.py` 도 동일하게 모델 metadata 에서 읽어 추론 env 에 전달.

**`record_video_fixed.py`** — Fixed-Time 베이스라인 영상 녹화 (RL 모델 불필요, Ray 의존 없음). `record_video_mappo.py` 와 동일한 continuous/short 모드 + frame hook 구조. `--baseline symmetric` 은 3-way 비교용 균등 4×15s 사이클; `--baseline sejong` 은 세종시 실측 per-TLS 비대칭 신호 (`SEJONG_PER_TLS_PHASE_SECONDS` 와 `_phase_from_secs` 는 `evaluate_mappo.py` 1:1 복제 → 비디오와 평가 CSV 가 동일 행동 묘사). 기본 출력 경로: `videos/fixed_<baseline>_<map>.mp4`.

**`stats_eval.py`** — `evaluate_mappo.py` 의 paired CSV(`results/eval_metrics_mappo_N.csv`)를 읽어 통계적 신뢰성 분석을 수행하는 독립 사후분석기 (SUMO/Ray 비의존). evaluate 평가 루프가 세 알고리즘을 매 episode 동일 seed 로 실행하는 **paired/blocked 설계**임을 활용 → 지표마다 **Friedman**(omnibus 게이트, 알고리즘 ≥3) → **쌍별 Wilcoxon signed-rank + Holm 보정**(사후, 2-way 면 단일 Wilcoxon 으로 graceful) → **Vargha–Delaney A12** 효과크기(`HIGHER_BETTER` 방향맵 반영) → **부트스트랩 95% CI**(mean + IQM, IQM=양끝 25% 절사평균으로 Agarwal 2021식 이상치 강건). 상수 지표(예 `teleported=0` 전역)는 검정 생략·CI만 기록 (텔레포트 인공물 아님 검증). 산출: `stats_tests.csv`/`stats_ci.csv`/`stats_summary.txt`(발표용 헤드라인 자동 생성)/`figs/{box,forest}_*.png`. 헤드라인 지표는 `avg_waiting_time`. 그림 텍스트는 ASCII(기본 폰트에 한글 글리프 없음). raw row 는 `algorithm` 이 `_mean`/`_std` 가 아닌 행으로 식별.

**`ctde_module.py`** — `CentralizedCriticPPOModule` extends `DefaultPPOTorchRLModule` with separate actor (local obs slice, first `D` dims) and critic (global obs slice, remaining dims) encoders. Both are 2-layer MLPs with Tanh. Bypasses `PPOCatalog`. Registered via `MultiRLModuleSpec` in `train_mappo.py` when `--ctde` is set. The actor uses only `obs[:local_dim]` at execution time; critic is stripped in inference-only workers via `get_non_inference_attributes()`. **차원은 obs space 에서 자동 인식** (하드코딩 없음) → 플래그 토글 시 모듈 코드 무변경. legacy CTDE critic concat = `_obs_dim(25) + _obs_dim×N` (3x2-brt N=6 → 175); `--neighbor-obs`(+`--upstream-phase`) 면 `own(_obs_dim) | agent_id_onehot(6) | N/E/S/W 이웃 obs(_obs_dim×4)` — 3x2-brt: neighbor-obs 29 → critic **151**, +upstream-phase 39 → critic **201**. _obs_dim 39 = phase(4)+min_green(1)+density(10)+queue(10)+N/S 이웃블록(14: 혼잡4+위상10). actor 는 N/S(lean), critic 은 4방향 이웃 obs(rich). critic 은 "위치"가 아니라 공유 함수가 학습 중 agent 별 입력으로 N 번 호출되는 것 (agent-specific per-agent critic; global 단일 critic 은 credit assignment 악화).

**`sumo_renderer.py`** — `SumoRenderer` provides matplotlib-based headless rendering (`render_modes=["rgb_array"]`). Parses `net.xml` directly (via sumolib + ElementTree) to draw road geometry, TLS signal states (green/yellow/red with glow), queue bars per approach direction, and BRT lane overlays (blue). Returns `np.ndarray shape=(H, W, 3) uint8`. Falls back to a blank error frame if net loading fails.

**`map_presets.py`** — `--map` 옵션을 sumo_cfg 경로 + default tls_ids로 해상. 사용자 명시 `--sumo-cfg` 가 preset 보다 우선. `--traffic high` 분기: `2x2` → `2x2grid_dense.sumocfg` (legacy), `2x2-brt` → `2x2_brt_dense.sumocfg` (세종시 실측 ~6,810 veh/h), `3x2-brt` → `3x2_brt_dense.sumocfg` (세종시 6교차로 실측, 최종 버전). `--tls-ids` CLI 인자는 cleanup 으로 제거됨 (모든 시나리오 preset 사용).

**`sumo_data/`** — SUMO 시나리오 파일들 (시나리오별 하위 폴더 구조):
- `single/` — 단일교차로: `nodes.nod.xml`, `edges.edg.xml`, `connections.con.xml`, `tls.tll.xml` (source) → `single_intersection.net.xml` (compiled, gitignored, auto-generated via `netconvert`); `routes.rou.xml`, `single.sumocfg`.
- `2x2/` — `2x2grid.sumocfg` / `2x2grid_dense.sumocfg` (sumo-rl 원본 net 사용, demand만 다름).
- `2x2_brt/` — 4 TLS, 좌측 col vertical edges 3-lane (lane 2 = `allow="bus"`). BRT corridor TLS는 14-link state, 표준 TLS는 12-link.
  - `routes_2x2_brt.rou.xml` (default, ~4,000 veh/h): 세종시 직진 실측의 0.59x 스케일 + cross flow 보정. 빠른 학습 iteration & 좌회전 phase 학습용.
  - `routes_2x2_brt_dense.rou.xml` (`--traffic high`, ~6,810 veh/h): **행복청 제23차 행복도시 교통량 조사(2025.04) AM peak 실측치 직접 적용** (성금/어진/청사/갈매로-가름로 4교차로). 직진은 실측 100%, v1(BRT corridor)=980, v2(우측열)=470 veh/h/방향. BRT 6분 배차. LOS D 조건.
  - **Cross flow 보정**: 두 routes 모두 cross/total ≈ 13% (도시 좌회전 평균 ~15%). 이전 4-5% 는 argmax inference 에서 좌회전 prob (~0.13) < 직진 prob (~0.37) 라서 좌회전 phase 가 영구 미선택 (argmax trap) 됨. cross 증액으로 좌회전 phase prob > 직진 prob 가 되어 argmax 시에도 좌회전 학습 결과 반영.
- `3x2/` — 6 TLS (2 cols × 3 rows), 모든 edge 2-lane, 12-link state.
- `3x2_brt/` — 6 TLS, 좌측 col vertical 3-lane BRT corridor (한누리대로). 좌측 TLS (1,3,5) 14-link, 우측 TLS (2,4,6) 12-link. **세종시 어진동 실제 도로망 기반 최종 시나리오.**
  - `routes_3x2_brt.rou.xml` (default, ~6,120 veh/h, probability 기반): BRT 6분 배차 (10대/h/방향).
  - `routes_3x2_brt_dense.rou.xml` (`--traffic high`, ~8,530 veh/h): **행복청 제23차 행복도시 교통량 조사(2025.04) 1차 + 공공데이터포털 2023 2차** 기반. TLS 매핑 (3행×2열, 좌=한누리대로/우=갈매로) — tl_0 성금·tl_1 청사 (상단), tl_2·tl_3 도움4로 (중단), tl_4 어진·tl_5 가름로 (하단). v1 BRT corridor=980 / v2=470 / h1=h2=h3=750 veh/h/방향 (각 ±10% 추정 보정). cross 도시 좌회전 15% 비율 (375/240/480/240). BRT 6분 배차 (period=360).
  - `3x2_brt_dense.sumocfg`: dense routes 참조용 config.
- BRT phase 구조: NS 직진 phase에 BRT lane도 같이 green (별도 BRT phase 없음). BRT 차량 가중치 reward는 후속 과제로 분리 (memory/project_brt_priority_reward 참조).

## Green Wave Progression — 보상 리스크 & 평가 지표 위계

`--progression-coeff`/`--brt-prog-weight`(①) + `--upstream-phase`(②) 운용 지침. 두 리스크(과포화가 진행파를 깎음 / E3 BRT 우대가 cross 굶김)는 **같은 축**(회랑 NS green 예산 vs EW)이라 한 메커니즘으로 같이 다룬다.

- **β가 마스터 안전 노브**: BRT는 NS 직진 phase를 일반차와 공유 → progression이 우대하는 건 "NS 회랑 흐름" 전체, trade-off는 NS↔EW green 분배뿐. base 보상(diff-waiting/pressure)이 이미 EW를 지키므로 **줄다리기**. β가 작으면 보상 늘릴 유일한 길이 *타이밍 개선*(=진짜 green wave)이고, 크면 EW 페널티 감수하고 *green 훔치기*가 이득 → **β 캘리브레이션이 옳은 해 vs 잘못된 해를 가른다**. smoke 실측(base per-step≈−3.5) 기준 **β≈0.5부터 시작**, sweep. (r_prog∈[0,brt_prog_weight].)
- **속도비 자기완화**: gridlock 에선 속도≈0 → r_prog≈0. 막힌 회랑에서 공짜 보상 파밍 불가 (리스크 #1 부분 자기완화). 진행파의 본거지는 비포화 구간.
- **EW 과포화는 예측이 아니라 게이트로 잡는다**: eval 이 이미 로깅. E2(brt_prog_weight=1) vs E3(3~5) ablation 에서 **`avg_waiting_time`(헤드라인)이 회귀하면 그 config 폐기** → 실험 설계 자체가 안전장치. 별도 reward 페널티는 지금 넣지 말 것(over-engineering); 모니터에서 EW creep 보이면 그때 EW soft penalty 추가.
- **평가 지표 위계** (BRT 통과량을 단독 X, 항상 "do no harm" 가드와 한 쌍으로):
  1. **헤드라인(게이트, 불변)**: `avg_waiting_time` 전체 — RL>Fixed 핵심 주장. 회귀 시 실패.
  2. **밸런스 가드**: `per_direction_wait_ew` (vs `_ns`) — EW 안 굶겼다 증명.
  3. **진행파 증거**: `corridor_brt_speed_ratio` (+ 후속 회랑 BRT stops/travel, arrivals-on-green).
  4. **동반이득**: throughput, stops/veh, CO₂.
  - 내러티브: "BRT 우선 진행파(3)를 학습해 BRT 개선 + 네트워크 헤드라인(1)·교차류(2) 무해." 3을 1·2와 같은 화면에.
- **과포화 시나리오(3x2-brt high=LOS D)**: green wave 본거지(비포화)가 거의 없음. base=`pressure`(gridlock 담당) + **작은 β** 로 회랑을 throughput 허용 범위에서 다듬기. 정직한 주장 = "BRT 속도비 X%↑, 무회귀". **깨끗한 진행파(time-space 밴드)는 비포화 regime**(3x2-brt default / 저수요 구간)에서 시연. LOS D 에서 진행파 과장 금지.

### 설계 요약 (차원 / ablation)

목표: MAPPO/CTDE 가 한누리대로 BRT 회랑(좌열 tl_0→tl_2→tl_4)에서 무정차 진행파를 *학습*으로 만든다. 두 레버는 한 몸 — **②(`--upstream-phase`)가 상류 신호 타이밍을 관측, ①(`--progression-coeff`)이 무정차 통과를 보상**. ②없이 ①만 주면 platoon 도착 타이밍을 몰라 학습난 → 구현·실험 순서 ②→①. 회랑 자동식별 = BRT 전용차로(`getAllowed` 에 bus 포함·passenger 제외), 하드코딩 0. **actor lean(N/S만)/critic rich(N/E/S/W 4-slot)**.

obs 레이아웃: base `[phase_onehot4, min_green1, density10, queue10]=25` + 방향블록 `[mean_density, mean_queue (+ upstream 면 phase_onehot4, elapsed_norm1)]` × {N,S}. `elapsed_norm=min(1, _elapsed_phase_time/60)` (`_PHASE_TIME_NORM=60`) → 전부 [0,1]. r_prog = corridor ns-through lane 의 차량 가중평균 속도비 ∈ [0,1] (버스는 `brt_prog_weight` 가중; **docstring 의 상한 brt_prog_weight 는 과표기, 실제 정규화 평균이라 [0,1]**).

| 구성 | actor obs (_obs_dim) | CTDE critic |
|---|---|---|
| base (플래그 없음) | 25 | — |
| `--neighbor-obs` (N/S 혼잡 +4) | 29 | 29+6+29×4 = 151 |
| `+--upstream-phase` (N/S 위상+경과 +10) | **39** | 39+6+39×4 = **201** |

**Ablation 매트릭스** — 각 run 은 old CTDE(`eval_sejong_dense_30.csv`) + 동일조건 MAPPO 와 비교. 공정 위해 MAPPO 도 같은 obs 플래그로 재학습, warmup→dense weights-only resume 복제(`--upstream-phase` 는 양 phase 필수):

| run | neighbor | upstream | β | brt_prog_w | obs/critic | 의미 |
|---|---|---|---|---|---|---|
| E0 | ✓ | ✗ | 0 | – | 29/151 | N/S 혼잡 인지만 |
| E1 | ✓ | ✓ | 0 | – | 39/201 | +상류 타이밍 관측 (② 단독; 깨끗한 ablation = phase 하나 차이) |
| E2 | ✓ | ✓ | β | 1.0 | 39/201 | +회랑 진행파 (BRT 무가중) |
| E3 | ✓ | ✓ | β | 3~5 | 39/201 | +BRT 우선 진행파 (full TSP) |

진행파 측정: info `corridor_brt_speed_ratio` / `corridor_brt_stops`. 비-BRT 맵: corridor=∅ → progression 전역 no-op. ctde_module 은 차원 자동인식이라 플래그 토글 시 무변경. obs 차원 변경 → 미지정 ckpt 비호환: metadata `neighbor_obs`/`upstream_phase` 추적, resume 불일치 거부, eval/record 모델별 로드(progression 은 reward-only → 추론 무관).

## Key Constraints

- Default TLS id is `"C"` (center intersection node). Green/yellow phases are detected dynamically via `_extract_phase_structure()` — do not remove yellow phases from `tls.tll.xml`.
- `delta_time=5` means each `step()` advances the simulation by 5 seconds (plus yellow time on phase change). `max_steps=3600` corresponds to 1 simulated hour.
- SUMO TLS auto-cycle is blocked via `_set_phase_locked()` (`setPhaseDuration(100000s)`) — agent actions are the sole source of phase transitions.
- `min_green=13` (default) — agent must hold a phase for at least 13 sim seconds before switching. `delta_time=5` → minimum 3 env steps between switches. phase 전환 시 삽입되는 yellow 3초는 **전환하지 않은 agent** 의 경과시간에도 산입됨 (min_green 게이팅·`--upstream-phase` 의 `elapsed_norm` 위상 시계 정확화 — 2026-06 수정).
- `--waiting-time-memory 10000` — env 가 SUMO 시작 시 항상 설정 (튜닝 노브 아님). SUMO 기본 100s 는 `getAccumulatedWaitingTime` 기반 지표 체인(`avg_waiting_time` 헤드라인)을 차량당 ~100s 에서 **포화**시켜 LOS D 결과를 왜곡했음 (실측: 동일 시나리오에서 차량 최대 대기 482s 가 ~100s 로 잘려 기록). ⚠️ **2026-06-11 이전에 생성된 모든 eval CSV 의 `avg_waiting_time` 은 이 포화 영향권** — 절대값·격차 비교 불가, 재평가 필요 (보상 기본 경로는 `lane.getWaitingTime`(순간값)이라 체크포인트는 전부 유효, 재학습 불필요). 포화가 대기 긴 Fixed-Time 을 더 크게 잘랐으므로 수정 후 RL 우위는 오히려 커짐 (스모크: Fixed 180.7s vs RL ~91s).
- **RLlib checkpoint loading**: checkpoints are directories. Do NOT use `PPO.from_checkpoint()` — it tries to spin up SUMO workers and fails with `IndexError`. Use `RLModule.from_checkpoint()` pointing to the sub-path `models/MAPPO_sumo_N/learner_group/learner/rl_module/shared_policy/`. See `_load_rl_module()` in `evaluate_mappo.py` and `record_video_mappo.py`.
- `models/`, `results/`, `videos/` are gitignored. Curated reference artifacts go in `samples/` (see `samples/README.md`).
- `entropy_coeff=0.03` (current) — raised from 0.005 to prevent left-turn phase mode collapse in dense traffic scenarios.
- `vf_clip_param=10.0` (value_norm on, current default) — value-target 정규화 후 σ-단위 outlier guard. value_norm off 시 `1000.0` (real-space, legacy). 자세히는 "Value-function 학습 붕괴" 섹션.
- `switch_penalty=0.45` (current) — reward 페널티 per phase switch. `yellow_time=3s × 1대/s 손실 ≈ 0.45`. yellow_seconds → throughput 0 으로 인한 oscillation 학습 차단. 0 으로 끄려면 `--switch-penalty 0`. yellow_time=2s 기준 이전 체크포인트 재사용 시 `--switch-penalty 0.3` 명시.
- `yellow_time=3` (current) — XML TLS 정의(`duration="3"`) 및 세종시 실제 신호(3초)와 일치. Python 환경이 직접 `_simulate_seconds(yellow_time)` 호출 (XML duration은 무시됨).
- `time_to_teleport=300` (current default) — 차량이 정체에 이 시간 이상 갇히면 SUMO가 정체 너머로 순간이동시켜 grid lock 을 자가 회복 (deadlock 안전밸브). 과거 하드코딩 `-1`(비활성)은 과포화(LOS D+) 시 영구 grid lock → diff-waiting-time 의 행동 의존 신호 소멸 → 학습 불가의 직접 원인이었음. 이제 `env_sumo_pz.py` 의 SUMO 시작 명령은 `self.time_to_teleport` 사용, `--time-to-teleport` CLI 로 override (train/evaluate/record 전부). 너무 짧으면(<120) 정상 대기 차량까지 순간이동 → 지표 과대평가. 평가 시 info `teleported`≈0 확인 = 결과가 텔레포트 인공물이 아님을 검증. 공정 비교 위해 MAPPO·CTDE·Fixed-Time 동일 값(env_kwargs 공유 + evaluate 는 `train_metadata.json` 자동 로드). 옛 거동 재현은 `--time-to-teleport -1`.
- BRT 시나리오에서 BRT 차량은 `vClass="bus"`로 lane 2 진입 / 일반 차량은 `passenger`로 lane 2 진입 차단. routes XML에서 `departLane="2"`로 BRT 강제. 일반 차량의 우회전·좌회전은 BRT corridor edge 진입 시 lane 0/1로 들어가 BRT lane과 충돌 없음.
- Diagnostic info dict 키: `vehicles_loaded` / `vehicles_departed` / `vehicles_lost_insert` / `pending_insert_peak` / `pending_insert_final` (insertion-failure 가시화 — `--time-to-teleport -1`(텔레포트 비활성) 환경에서 silent vehicle drop 진단; 기본값 300 에서는 deadlock 자가 회복), `yellow_seconds` / `yellow_ratio` (oscillation 진단), `phase_switches` / `max_queue` / `action_counts` (mode collapse 진단).

## PPO Hyperparameters (Current)

| Parameter | Value | Note |
|-----------|-------|------|
| `lr` | `1e-4` | policy 진동 억제. `--lr` 로 override 가능 |
| `train_batch_size` | `8000` | gradient noise 감소. `--train-batch-size 4000` (Colab/빠른 실험 권장) |
| `num_epochs` | `6` | — |
| `minibatch_size` | `128` | — |
| `gamma` | `0.99` | — |
| `lambda_` | `0.95` | GAE λ |
| `clip_param` | `0.2` | PPO clip ε |
| `vf_loss_coeff` | `0.5` | — |
| `entropy_coeff` | `0.03` | 좌회전 phase mode collapse 방지 (dense 시나리오 초기 음수 reward → collapse 억제). `--entropy-schedule START END` 로 선형 스케줄(예 `0.03 0.005`, 초반 탐색→후반 sharpening, 75% 지점에서 END 도달) 가능 — 새 API 스택은 `entropy_coeff=[[step,val],...]` 형식. `hparams` 공유라 MAPPO·CTDE 동일 적용 |
| `vf_clip_param` | `10.0` (value_norm on) / `1000.0` (off) | **value_norm on(기본): σ-단위 outlier guard** (10=3.16σ, 스케일 독립). off: real-space legacy. `--vf-clip-param` override(미명시 시 metadata→default). 단 **resume 시 value_norm 상태가 metadata 와 다르면 metadata vf_clip 은 무시**(σ-단위 vs real-space 단위 불일치 — legacy 1000 상속 시 σ-공간 사실상 unclip footgun 차단). `hparams` 공유라 MAPPO·CTDE 동일. **튜닝 전 아래 "Value-function 학습 붕괴" 섹션 필독.** |
| `value_norm` | `True` | value-target normalization(MAPPO ValueNorm). 크리틱 정규화 출력→denorm, σ-단위 손실 → value 학습 붕괴 해소. `--no-value-norm` 으로 off. MAPPO·CTDE 공통 |
| `vn_beta` | `0.99999` | ValueNorm running-stat EMA decay. 채택 근거는 **MAPPO 원 구현 정렬뿐** — "느린 β→σ 안정→explained_var 노이즈↓" 가설은 0.999 vs 0.99999 A/B 로 **반증**(차이 없음; 노이즈 주범=window=1 로깅+미수렴 정책). ⚠️ 느린 EMA 는 regime shift(warmup→dense resume)에서 σ-clip 재동결 위험 — dense 진입 직후 `vf_loss_unclipped` vs `loss/value` 갭 확인, 재발 시 normalizer 버퍼 리셋 또는 `--vn-beta 0.9999`. "더 느린 β=더 안정" 일반화 금지 |
| `yellow_time` | `3` | XML TLS 정의 및 세종시 실제 신호(3초)와 일치. 이전 값 2초로 학습된 모델은 `--yellow-time 2` 명시 |
| `switch_penalty` | `0.45` | yellow 3초 × 1대/초 손실. yellow_time=2 기준 이전 모델 재사용 시 0.3 명시 |
| `time_to_teleport` | `300` | SUMO 텔레포트 임계(초). deadlock 안전밸브. `-1`=비활성(과포화 시 영구 grid lock → 학습 불가). `--time-to-teleport` CLI. evaluate 는 metadata 자동 로드 |

## Value-function 학습 붕괴 — value-target 정규화로 **해결됨** (필독)

> 이 절은 후속 작업자(사람/봇)가 **같은 실수를 반복하지 않도록** 못박는다. 한 줄 요약: **`vf_clip_param` 을 손으로 튜닝하지 말 것. value-target 정규화(`--value-norm`, 기본 on)가 적용돼 saga 는 끝났다.**

> **✅ 해결 완료 (구현됨, 기본 on)** — MAPPO ValueNorm 적용. 크리틱이 **정규화 공간** value 를 출력하고 `compute_values` 가 `σ·z+μ` 로 denormalize, 손실은 σ-단위로 계산(`value_norm.py`/`value_norm_learner.py`/`value_norm_module.py`). 효과: ① `vf_clip_param` 이 **σ-단위(스케일 독립) outlier guard** 로 바뀜(default 10 = 3.16σ; return 스케일에 손으로 맞출 값 아님 → saga 종결), ② 크리틱 출력 +μ offset 으로 **cold-start freeze 방지**(unit test: loss-only 면 첫 배치 99.9% freeze, 정규화-출력은 0.15%). `--no-value-norm` 으로 legacy 거동. **MAPPO·CTDE 공통**. 아래 saga 기록은 *왜* 정규화가 정답인지의 근거로 보존.
>
> **만지기 전 주의 2가지**: (1) **활성 노브는 β 가 아니라 `vf_clip_param`(σ-단위)** — Adam 이 손실의 1/σ² gradient 스케일을 흡수하므로(상쇄) "작은 grad=안정" 논리는 거짓이고, 진짜 안정자는 σ-단위 clip 의 일관된 outlier freeze. 튜닝하려면 vf_clip ∈ {5,10,20} sweep. (2) **value clip 과 normalization 을 섞지 말 것**(Zheng 2023)의 의미 = *스케일 맞춤용* clip 금지. 현재 vf_clip 은 σ-단위 outlier 가드라 무해.

> 📜 **이하 §증상~§금지 = 구현 전 saga 기록 (보존; 현재형 서술·"1000" 언급은 당시 시점).** 지금 최신 상태는 위 "✅ 해결 완료" — vf_clip default 는 10(σ-단위), value_norm 기본 on.

**증상**: `value/explained_var` 가 **0.04~0.08** (3x2-brt high 실측: MAPPO 0.044 / CTDE 0.076). 크리틱이 return 분산의 4~8%만 설명 = 사실상 "평균만 찍음" → PPO advantage 가 노이즈 → 정책이 노이즈로 학습. MAPPO·CTDE 공통(hparams 공유).

**근본 원인 (한 줄)**: `vf_clip_param` 은 *per-step reward* 가 아니라 *누적 return(=value 타깃, diff-waiting 은 ~수천)* 스케일에 맞춰야 한다. 작게 두면 value 예측오차가 clip 에 잘려 크리틱이 초기값에 고정됨(과거 "value=10 고정 버그"). 실측: clip 1000 에서 `vf_loss_unclipped≈104,000` vs `vf_loss≈927` (약 100배 갭) → **현재 1000 도 여전히 under-set**.

**금지 — clip 을 또 추측으로 만지지 말 것.** 이 값은 explained_var 검증 없이 5번 추측으로 오갔다: 기본 10(→`value=10 고정 버그`, 769652b) → 100 → 500(0937409) → **10 으로 재하향(95e8b50, reward `/100→/10` 동시변경하며 "스케일 맞춤"이라 오판)** → 학습붕괴 → 1000 복원(cdb6518: clip 10 에서 `vf_loss_unclipped=175189` vs `vf_loss=9.89` 로 신호 소실). **10→100→500→10→1000, 전부 미검증 추측이고 1000 도 정답 아님.** clip 을 더 키우면(예 1e5) explained_var 는 오르지만 raw value loss(O(1e5))로 gradient 불안정 위험 → 진단용일 뿐.

**진단 근거 (구현 전 probe, 보존)**: `value/explained_var`(TB 로깅) 확인. `--vf-clip-param 1e5` 30-iter probe → 0.076→last5 0.61(8배) 상승 = clip 이 throttle 범인 입증. 단 value loss O(1e4)로 spiky·iter217 에 0.018 붕괴 = "clip 키우기는 throttle 풀지만 불안정 산다" → **clip 튜닝이 아니라 정규화가 정답**임을 데이터로 확인.

**적용된 해결 = value-target normalization (MAPPO ValueNorm)**: 크리틱이 *정규화된* return 에 회귀, GAE/inference 때 denormalize (`compute_values`→`σz+μ`). RLlib 새 API 내장 없어 직접 구현 (`value_norm.py` 정규화기 + `value_norm_learner.py` σ-단위 손실 + `value_norm_module.py` MAPPO 모듈 + CTDE 는 `ctde_module.py` 자체 보유). **reward 재스케일로 대체 금지** — reward 를 나누면 entropy_coeff·switch_penalty(절대스케일) 균형까지 깨져 재튜닝 필요. 정규화는 *타깃만* 건드려 부작용 없음. ⚠️ `value_norm_learner.py` 는 ray 2.55.1 `PPOTorchLearner.compute_loss_for_module` 복제(vf 블록만 교체) — RLlib 업글 시 parent diff 확인.

**왜 중요**: 두 정책 모두 (망가진) 노이즈 advantage 로도 Fixed 를 압도(−25% wait, +56% throughput) → 거대한 미실현 여지. 망가진 와중에도 CTDE(0.076)>MAPPO(0.044) 1.7배 → value 학습을 고치면 CTDE 구조적 우위가 드러나 "결정적 CTDE>MAPPO" 기회. 다음 단계: value_norm on 으로 **clean β=0 재학습**(progression 끄고) → CTDE>MAPPO 재측정.

**문헌**: MAPPO [arXiv:2103.01955](https://arxiv.org/abs/2103.01955) (Appendix value-norm ablation) · PopArt [arXiv:1602.07714](https://arxiv.org/pdf/1602.07714) · Secrets of RLHF Part I [arXiv:2307.04964](https://arxiv.org/pdf/2307.04964) · Return-based Scaling [arXiv:2105.05347](https://arxiv.org/pdf/2105.05347).

## Troubleshooting

| 증상 | 원인 | 대응 |
|------|------|------|
| action 한쪽 0% / 99% | mode collapse (초기 negative reward) | `entropy_coeff` ↑ (현재 0.03), traffic 다양화 |
| `loss/value` 수렴 안 함 | value_norm off + `vf_clip_param` 너무 작음 (신호 clip 소실) | `--value-norm`(기본 on) 확인; off 면 `vf_clip_param` ↑ |
| `value/explained_var` < 0.7 | 크리틱 undertrained / value_norm off | `--value-norm` on 확인 ("Value-function 학습 붕괴" 섹션). value_norm 에선 vf_clip ∈{5,10,20} sweep, centralized critic(`--ctde`) ROI 높음 |
| `phase_switches` > 1200 (4 agent 기준) | 과도 switching | `--switch-penalty` ↑, `--min-green` ↑ |
| `policy/yellow_ratio` > 0.15 | phase oscillation (매 step switching) | `--switch-penalty` ↑, `entropy_coeff` ↓ |
| `teleported` ≫ 0 | grid lock (정체 한계 초과, 안전밸브 빈발) | `--traffic` 낮추기, network 수정. 소수(≈0)는 정상 deadlock 회복 |
| `vehicles_lost_insert` 급증 | dense traffic lane 포화 (silent drop) | 차량 수요 완화, 좌회전 phase 학습 확인 |
| `process_rss_mb` 우상향 | 메모리 누수 | env close() 누락 확인 |
| `train_total_steps=0` (CTDE) | Dict obs → worker connector silent drop | flat Box 평탄화 확인 (`obs_space.shape=(D+D×N,)`). 본 repo는 이미 적용됨 |
| `train_total_steps=0` (일반) | Ray worker timeout | `sample_timeout_s` 확인 (현재 1800s) |
| evaluate/record `IndexError` | `PPO.from_checkpoint()` 잘못 사용 | `RLModule.from_checkpoint()` + sub-path 사용 (see `_load_rl_module()`) |

> 4 agent 2x2 grid에서 `phase_switches` 900~1200은 정상 범위 (agent당 225~300회, min_green=13/delta_time=5 기준 최소 2.6 step 간격).

## Smoke-Test 기준치 (코드 변경 후 동작 검증용)

직접 SUMO 600s 실행 시 (`sumo -c <cfg> --end 600`) 예상 차량 로드량:

| 시나리오 | loaded@600s | 환산 veh/h | 비고 |
|---|---|---|---|
| `2x2-brt` default | ~671 | ~4,030 | LOS B-C |
| `2x2-brt` high | ~1,140 (waiting ~205) | ~6,840 loaded | LOS D, pending 18% |
| `3x2-brt` high | ~1,447 (waiting ~276) | ~8,680 loaded | LOS D 최종 버전, pending 19% |

Python env 호출 후 (`env.reset` + 임의 action 20-step):
- `obs_dim`: single=13, 2x2/3x2=21, 2x2-brt/3x2-brt=25
- `yellow_time=3`, `switch_penalty=0.45`, `time_to_teleport=300` (env 기본값, 학습 CLI default 와 일치)

`env_sumo_pz.py` 의 `SumoParallelEnv.__init__` defaults 는 `train_mappo.py` CLI defaults 와 항상 동기화. 직접 객체 생성 시 (`SumoParallelEnv(...)`) 에도 학습 환경과 동일 reward 식 적용됨.

## 미사용/더미 코드 인벤토리 (2026-06-07 스캔, 삭제 X — 참고용)

green-wave 파이프라인(train→eval→record) 기준 AST/grep 스캔 결과. **삭제하지 말 것** — B/C 는 의도된 보존.

**A. 진짜 orphan — 2026-06-07 삭제 완료:**
- ~~`env_sumo_pz.py` `clear_step_hooks()`~~ 삭제. (`add_step_hook`/`_step_hooks`/`live_metrics` 는 record_video 가 사용 → 유지.)
- ~~`sumo_renderer.py` `_cached_lane_movement()` + `_lane_to_movement` 캐시 체인(init·population)~~ 삭제 (유일 reader 였음).
- ~~`train_mappo.py` `from gymnasium import spaces`~~ 삭제 (미사용 import).

**B. 이 파이프라인에선 미실행이나 *의도된 보존* (legacy/조건부):**
- `observation_space` 의 non-neighbor CTDE `else` 브랜치 (`_obs_dim + _obs_dim×N`, 175) — green-wave 는 항상 `--neighbor-obs` → 미실행. legacy CTDE(neighbor 없이) 체크포인트 호환용. 보존.
- `record_video_fixed.py`, `build_3way_compare.py` — RL train/eval 파이프라인 밖(발표용 영상 툴링). 기능적으로 사용됨, 더미 아님.
- `pressure` reward mode (`REWARD_MODES`) — green-wave 의 과포화 base 로 *권장*. 더미 아님.

**C. False positive (orphan 으로 보이나 프레임워크가 호출 — 건드리지 말 것):**
- `ctde_module.py` 의 `get_inference_action_dist_cls`/`get_exploration_action_dist_cls`/`get_train_action_dist_cls`/`get_initial_state`/`_forward_train`/`compute_values`/`get_non_inference_attributes` — RLlib RLModule 인터페이스 오버라이드. 이름 직접 호출 없지만 프레임워크가 호출.
- `from __future__ import annotations` (sumo_renderer.py) — future statement.
