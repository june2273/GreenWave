# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GreenWave is a reinforcement learning project for traffic signal control using MAPPO (Multi-Agent PPO). It uses SUMO (Simulation of Urban MObility) as the traffic simulator, PettingZoo as the multi-agent environment interface, and RLlib for the MAPPO algorithm. Supports single intersection (`tls_ids=["C"]`) and scales to multiple intersections via the `--tls-ids` argument.

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
python train_mappo.py --num-iters 200 --num-workers 1

# Evaluate MAPPO vs fixed-time baseline (saves to results/eval_metrics_mappo.csv)
python evaluate_mappo.py --model models/MAPPO_sumo_1 --episodes 5

# Record policy rollout video (saves to videos/mappo_policy_rollout.mp4)
python record_video_mappo.py --model models/MAPPO_sumo_1

# Run environment smoke test
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
- Actions: `Discrete(4)` — 0=North, 1=South, 2=East, 3=West single-direction green
- Observation: `shape=(10,)` — `[queue_N, queue_S, queue_E, queue_W, speed_N, speed_S, speed_E, speed_W, phase_norm, elapsed_norm]`
- Reward: `−total_queue_length` (sum of halting vehicles across all 4 incoming lanes of that agent's intersection)

**`train_mappo.py`** — Builds a `PPOConfig` with shared policy across all agents, runs the training loop, writes TensorBoard scalars to `results/tb_mappo/<run_name>/`, and saves RLlib checkpoints (directory format) to `models/MAPPO_sumo_N/`.

**`evaluate_mappo.py`** — Runs two policies head-to-head: the trained MAPPO checkpoint (`compute_single_action`) and a fixed-time cyclic baseline. Results are aggregated into `results/eval_metrics_mappo.csv`.

**`sumo_data/`** — SUMO network definition: `nodes.nod.xml`, `edges.edg.xml`, `connections.con.xml`, `tls.tll.xml` are the source files; `single_intersection.net.xml` is the compiled output (gitignored, auto-generated at runtime); `routes.rou.xml` defines vehicle demand; `single.sumocfg` is the SUMO config entrypoint.

## Key Constraints

- Default TLS id is `"C"` (center intersection node). The yellow phase is detected dynamically by scanning for a phase whose state contains only `'y'` and `'r'` characters — do not remove this phase from `tls.tll.xml`.
- `delta_time=5` means each `step()` advances the simulation by 5 seconds (plus yellow time on phase change). `max_steps=3600` corresponds to 1 simulated hour.
- RLlib checkpoints are saved as directories, not `.zip` files. Use `PPO.from_checkpoint(path)` to restore.
- `models/`, `results/`, `videos/` are gitignored. Curated reference artifacts go in `samples/` (see `samples/README.md`).
