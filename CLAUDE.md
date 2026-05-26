# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GreenWave is a reinforcement learning project for traffic signal control using MAPPO (Multi-Agent PPO). It uses SUMO (Simulation of Urban MObility) as the traffic simulator, PettingZoo as the multi-agent environment interface, and RLlib for the MAPPO algorithm. Scenarios are selected via `--map` (single / 2x2 / 2x2-brt / 3x2 / 3x2-brt); each preset auto-resolves `sumo_cfg` and default `tls_ids`.

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
- `traci`, `sumolib` (SUMO Python bindings)
- `imageio`, `imageio-ffmpeg`

## Commands

```bash
# Train MAPPO — auto-versions checkpoint to models/MAPPO_sumo_N/
python train_mappo.py --map 2x2 --num-iters 200 --num-workers 1
python train_mappo.py --map 2x2-brt --num-iters 200          # BRT corridor on left col
python train_mappo.py --map 3x2 --num-iters 200              # 2 cols x 3 rows, all 2-lane
python train_mappo.py --map 3x2-brt --num-iters 200          # 3x2 + BRT corridor on left col

# Evaluate (use same --map as training)
python evaluate_mappo.py --model models/MAPPO_sumo_1 --map 2x2-brt --episodes 5

# Record policy rollout video
python record_video_mappo.py --model models/MAPPO_sumo_1 --map 2x2-brt

# Run environment smoke test (default: single intersection)
python env_sumo_pz.py
```

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
- If `sumo_data/single_intersection.net.xml` is missing, `_maybe_build_network()` auto-generates it via `netconvert`.
- `step()` applies yellow-phase transitions and enforces minimum green time per agent before allowing phase changes.
- All agents share one SUMO connection — phase switching is coordinated in a single `simulationStep()` loop.

**Action / Observation / Reward (per agent):**
- Actions: `Discrete(num_green)` — green phase 수는 네트워크에서 자동 감지 (모든 시나리오 4). yellow phase는 정책이 직접 선택 불가, green 전환 시 자동 삽입.
- Observation: `[phase_one_hot(num_green), min_green_flag(1), density_per_lane(L), queue_per_lane(L)]` shape = `num_green + 1 + 2*L`. **`L=max(per-agent lanes)`**; 작은 agent의 obs는 density/queue 부분을 0 패딩 (BRT 시나리오의 mixed-lane topology 지원). 시나리오별 `obs_dim`: single=13, 2x2/3x2=21, 2x2-brt/3x2-brt=25.
- Reward (선택): `queue` (−(mean+max)/10 + throughput×0.5) / `diff-waiting-time` (대기시간 변화량/10) / `pressure` (출력−입력 차량). 모든 모드에 공통으로 `switch_penalty=0.3` 적용 (phase switching 시 reward -= 0.3, oscillation 억제).

**`train_mappo.py`** — Builds a `PPOConfig` with shared policy across all agents, runs the training loop, writes TensorBoard scalars to `results/tb_mappo/<run_name>/`, and saves RLlib checkpoints (directory format) to `models/MAPPO_sumo_N/`. `--ctde` flag enables centralized critic (CTDE-MAPPO), saving to `models/MAPPO_CTDE_sumo_N/`.

**`evaluate_mappo.py`** — 3-way comparison: MAPPO vs CTDE-MAPPO (optional) vs Fixed-time. Results saved to `results/eval_metrics_mappo_N.csv` with metadata prefix columns for model traceability.

**`map_presets.py`** — `--map` 옵션을 sumo_cfg 경로 + default tls_ids로 해상. 사용자 명시 `--sumo-cfg`/`--tls-ids` 가 preset보다 우선. `2x2 + --traffic high` 조합은 dense routes 사용 (legacy 호환).

**`sumo_data/`** — SUMO 시나리오 파일들:
- 단일교차로: `nodes.nod.xml`, `edges.edg.xml`, `connections.con.xml`, `tls.tll.xml` (source) → `single_intersection.net.xml` (compiled, gitignored, auto-generated via `netconvert`); `routes.rou.xml`, `single.sumocfg`.
- `2x2grid.sumocfg` / `2x2grid_dense.sumocfg` — sumo-rl 원본 2x2 net 사용, demand만 다름.
- `2x2_brt/` — 4 TLS, 좌측 col vertical edges 3-lane (lane 2 = `allow="bus"`). BRT corridor TLS는 14-link state, 표준 TLS는 12-link.
- `3x2/` — 6 TLS (2 cols × 3 rows), 모든 edge 2-lane, 12-link state.
- `3x2_brt/` — 6 TLS, 좌측 col vertical 3-lane BRT corridor. 좌측 TLS (1,3,5) 14-link, 우측 TLS (2,4,6) 12-link.
- BRT phase 구조: NS 직진 phase에 BRT lane도 같이 green (별도 BRT phase 없음). BRT 차량 가중치 reward는 후속 과제로 분리 (memory/project_brt_priority_reward 참조).

## Key Constraints

- Default TLS id is `"C"` (center intersection node). Green/yellow phases are detected dynamically via `_extract_phase_structure()` — do not remove yellow phases from `tls.tll.xml`.
- `delta_time=5` means each `step()` advances the simulation by 5 seconds (plus yellow time on phase change). `max_steps=3600` corresponds to 1 simulated hour.
- SUMO TLS auto-cycle is blocked via `_set_phase_locked()` (`setPhaseDuration(100000s)`) — agent actions are the sole source of phase transitions.
- `min_green=13` (default) — agent must hold a phase for at least 13 sim seconds before switching. `delta_time=5` → minimum 3 env steps between switches.
- RLlib checkpoints are saved as directories, not `.zip` files. Use `PPO.from_checkpoint(path)` to restore.
- `models/`, `results/`, `videos/` are gitignored. Curated reference artifacts go in `samples/` (see `samples/README.md`).
- `entropy_coeff=0.03` (current) — raised from 0.005 to prevent left-turn phase mode collapse in dense traffic scenarios.
- `vf_clip_param=1000.0` (current) — matches diff-waiting-time reward scale (~thousands); too small causes VF loss signal loss.
- `switch_penalty=0.3` (current) — reward 페널티 per phase switch. yellow_seconds → throughput 0 으로 인한 oscillation 학습 차단. 0 으로 끄려면 `--switch-penalty 0`.
- BRT 시나리오에서 BRT 차량은 `vClass="bus"`로 lane 2 진입 / 일반 차량은 `passenger`로 lane 2 진입 차단. routes XML에서 `departLane="2"`로 BRT 강제. 일반 차량의 우회전·좌회전은 BRT corridor edge 진입 시 lane 0/1로 들어가 BRT lane과 충돌 없음.
- Diagnostic info dict 키: `vehicles_loaded` / `vehicles_departed` / `vehicles_lost_insert` / `pending_insert_peak` / `pending_insert_final` (insertion-failure 가시화 — `--time-to-teleport -1` 환경에서 silent vehicle drop 진단), `yellow_seconds` / `yellow_ratio` (oscillation 진단), `phase_switches` / `max_queue` / `action_counts` (mode collapse 진단).
