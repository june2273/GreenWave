# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GreenWave is a reinforcement learning project for traffic signal control at a single intersection. It uses SUMO (Simulation of Urban MObility) as the traffic simulator, Gymnasium as the RL environment interface, and Stable Baselines 3 (SB3) for the DQN algorithm.

## Prerequisites

**System binaries** (must be installed separately):
```bash
sudo apt install -y sumo sumo-tools sumo-doc
```

**Environment variables** (required for TraCI/sumolib):
```bash
export SUMO_HOME=/usr/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

**Python dependencies** (in `.venv`):
- `gymnasium`, `stable-baselines3`, `numpy`, `pandas`
- `traci`, `sumolib` (SUMO Python bindings)
- `imageio`, `imageio-ffmpeg`

## Commands

```bash
# Train DQN (saves to models/dqn_sumo_single.zip)
python train_dqn.py --timesteps 120000

# Evaluate DQN vs fixed-time baseline (saves to results/eval_metrics.csv)
python evaluate.py --model models/dqn_sumo_single.zip --episodes 5

# Record policy rollout video (saves to videos/dqn_policy_rollout.mp4)
python record_video.py --model models/dqn_sumo_single.zip --output videos/dqn_policy_rollout.mp4

# Run environment smoke test
python env_sumo_single.py
```

## Architecture

```
SB3 DQN Agent  →(action: Discrete 4)→  SumoSingleIntersectionEnv  →(TraCI)→  SUMO Simulator
                                               ↑                                      |
                                         sumo_data/ XML files                   lane states
                                                                                      ↓
SB3 DQN Agent  ←(obs 10, reward, done)←  SumoSingleIntersectionEnv  ←──────────────┘
```

**`env_sumo_single.py`** — The core Gymnasium environment (`SumoSingleIntersectionEnv`):
- Each `reset()` spawns a new SUMO process via `traci.start()` with a unique UUID label, so multiple envs can run in parallel without port conflicts.
- If `sumo_data/single_intersection.net.xml` is missing, `_maybe_build_network()` auto-generates it from the XML source files using `netconvert`.
- `step()` enforces a yellow-phase transition (inserting the all-yellow phase from `tls.tll.xml`) and a minimum green time before allowing phase changes.
- `render()` returns a synthetic RGB array (no SUMO GUI required); real SUMO-GUI can be enabled with `use_gui=True`.

**Action / Observation / Reward:**
- Actions: `Discrete(4)` — 0=North, 1=South, 2=East, 3=West single-direction green
- Observation: `shape=(10,)` — `[queue_N, queue_S, queue_E, queue_W, speed_N, speed_S, speed_E, speed_W, phase_norm, elapsed_norm]`
- Reward: `−total_queue_length` (sum of halting vehicles across all 4 incoming lanes)

**`train_dqn.py`** — Wraps the env in SB3's `Monitor`, configures DQN hyperparameters, and saves the model as a `.zip` archive. TensorBoard logs go to `results/tb_dqn/`.

**`evaluate.py`** — Runs two policies head-to-head: the trained DQN (deterministic predict) and a fixed-time cyclic baseline (round-robin phases every `--baseline-phase-steps` steps). Results are aggregated into `results/eval_metrics.csv`.

**`sumo_data/`** — SUMO network definition: `nodes.nod.xml`, `edges.edg.xml`, `connections.con.xml`, `tls.tll.xml` are the source files; `single_intersection.net.xml` is the compiled output; `routes.rou.xml` defines vehicle demand; `single.sumocfg` is the SUMO config entrypoint.

## Key Constraints

- The `tls_id` is always `"C"` (the center intersection node). The yellow phase is detected dynamically by scanning for a phase whose state contains only `'y'` and `'r'` characters — do not remove this phase from `tls.tll.xml`.
- `delta_time=5` means each `step()` advances the simulation by 5 seconds (plus yellow time on phase change). `max_steps=3600` corresponds to 1 simulated hour.
- SB3 model files use `.zip` format regardless of the extension used when saving.