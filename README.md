# GreenWave

Multi-agent reinforcement learning (MAPPO) for traffic signals at six intersections on a SUMO model of a BRT corridor in Sejong, Korea, compared against the city's measured fixed-time signal plans. **Simulation only** — nothing here has been tested on real signals.

![Average waiting time, IQM ± 95% CI over 30 paired episodes](samples/results/sejong%203-way/figs/forest_avg_waiting_time.png)

Side-by-side video (Fixed-Time | MAPPO | CTDE): [samples/videos/sejong 3-way/3x2compare_3way_1080p.mp4](samples/videos/sejong%203-way/3x2compare_3way_1080p.mp4)

## Problem

Fixed-time signals repeat the same cycle regardless of the queues in front of them. On the modeled corridor (Hannuri-daero BRT × Galmae-ro, 6 intersections, ~8,530 veh/h, LOS D) that means long waits and vehicles that cannot enter the network at all. Each intersection only sees its own approaches, while its decisions change the traffic arriving at its neighbors.

## Approach

- **Environment.** A PettingZoo `ParallelEnv` over SUMO (TraCI). Each intersection is an agent with `Discrete(4)` green phases; yellow (3 s) is inserted by the environment, minimum green is 13 s, and SUMO's own cycling is locked out so only the agents switch phases. The action scheme, observation layout and `diff-waiting-time` reward are adopted from SUMO-RL v1.4.5.
- **Learning.** MAPPO with one shared policy on Ray RLlib. A CTDE variant gives the critic the neighbors' observations during training while each actor still uses only its own intersection at execution.
- **Own extensions:** MAPPO/CTDE on RLlib, BRT-weighted reward (`--brt-weight`), switch penalty (0.45 per switch), 14-link BRT signals with observation padding, a teleport safety valve (`time_to_teleport=300`) so gridlock does not erase the learning signal, and value-target normalization for the critic.
- **Scenario from public data.** Demand combines the 2025-04 AM-peak intersection counts (total volume, LOS) with 2023-03 direction splits from the public data portal. The fixed-time baseline maps measured signal timings (public data portal 2023, Sejong open data 2026-04) per intersection. Two of the six intersections (`tl_2`, `tl_3`) appear in neither source and are estimated from their neighbors.
- **Evaluation.** All three controllers run on the same seed in each episode (paired, 30 episodes). `stats_eval.py` runs Friedman → pairwise Wilcoxon with Holm correction → Vargha–Delaney A₁₂ → bootstrap 95% CI on mean and IQM.

## Results

### Sejong 3-way, `3x2-brt --traffic high` (LOS D, 30 paired episodes)

Re-evaluated on 2026-06-11 after fixing a metric bug: SUMO's default `--waiting-time-memory` (100 s) capped waiting time per vehicle at ~100 s under LOS D. With the fix, the RL improvement in average waiting time grew from 27% to 50%. Older tables are not comparable in absolute terms.

| Metric | FixedTime (Sejong) | MAPPO | CTDE | RL vs Fixed |
|------|------:|------:|------:|:--:|
| Avg waiting time (s) | 180.14 | **90.54** | 97.74 | **−49.7%** |
| Avg travel time (s) | 249.59 | **174.79** | 184.20 | −30.0% |
| Throughput (veh/h) | 4,030 | **5,772** | 5,500 | **+43.2%** |
| Total queue length | 859,901 | **544,052** | 565,612 | −36.7% |
| Avg speed (m/s) | 1.80 | **2.73** | 2.57 | +51.8% |
| CO₂ per vehicle (mg) | 708,194 | **462,913** | 488,514 | −34.6% |
| Teleported vehicles | 24.27 | **2.30** | 3.03 | −90.5% |
| Stops per vehicle † | **2.57** | 4.40 | 4.66 | +71% (worse) |
| **BRT avg wait (s)** | 48.51 | **18.02** | 29.02 | **−62.9%** |
| Car avg wait (s) | 188.69 | **65.79** | 71.10 | −65.1% |
| **BRT avg speed (m/s)** | 5.14 | **6.60** | 6.36 | **+28.3%** |
| Corridor BRT speed ratio | 0.299 | **0.432** | 0.415 | +44.5% |
| Wait by direction NS (s) | 226.1 | **37.7** | 43.0 | −83.3% |
| Wait by direction EW (s) | 320.7 | **52.0** | 58.3 | −83.8% |

† Fixed-time has fewer stops per vehicle because it lets fewer vehicles in: under gridlock about 50% of loaded vehicles fail to enter, versus about 30% for MAPPO, so the vehicles that do get counted stop less.

The BRT gains appear with `brt_weight = 1.0`, i.e. without any extra reward for buses.

**Statistics** (`stats_eval.py`, n = 30 paired):
- MAPPO and CTDE each beat FixedTime in 30 of 30 episodes. MAPPO average waiting time: IQM −49.4% [95% CI 48.7–50.2%], Wilcoxon + Holm p = 5.6e-09, A₁₂ = 1.00. CTDE: IQM −45.8% [45.0–46.7%].
- MAPPO vs CTDE: MAPPO is significantly better on almost every metric, including average waiting time (90.5 vs 97.7 s, p = 5.6e-09, A₁₂ = 1.00); only teleports are not significant. CTDE also has larger variance (std ratio CTDE/MAPPO: throughput 6.4×, travel time 2.8×, waiting time 3.6×).

Raw data and statistics: [`samples/results/sejong 3-way/`](samples/results/sejong%203-way/).

### Development run, `2x2-brt` default demand (MAPPO_sumo_1, 80 iter, 5 ep mean)

| Metric | FixedTime | MAPPO | Change |
|------|-----------|-------|--------|
| Avg waiting time (s) | 59.21 | **37.48** | −36.7% |
| Avg travel time (s) | 234.47 | **103.84** | −55.7% |
| Throughput (veh/h) | 2,733 | **3,822** | +39.8% |
| Avg speed (m/s) | 1.83 | **4.24** | +132% |
| Stops per vehicle | 3.59 | **2.09** | −41.8% |
| CO₂ per vehicle (mg) | 610,608 | **280,655** | −54.0% |
| Vehicles lost at insertion | 1,066 | **51** | −95.2% |
| BRT avg wait (s) | 32.80 | **15.74** | **−52.0%** |
| Car avg wait (s) | 56.85 | **31.07** | −45.3% |

This table predates the 2026-06-11 metric fix. The scenario is not saturated (LOS B–C), so the distortion is small, but do not compare its absolute values with the Sejong table.

## Tech stack

- Python, PyTorch
- Ray RLlib (MAPPO / PPO, custom CTDE RLModule and value normalization)
- SUMO 1.20 via TraCI / sumolib, PettingZoo, Gymnasium
- numpy, pandas; statistics in `stats_eval.py`
- matplotlib + imageio for rollout videos, TensorBoard for training curves

## Run

```bash
brew install sumo
export SUMO_HOME=$(brew --prefix sumo)/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# train (Sejong scenario)
python train_mappo.py --map 3x2-brt --traffic high --num-iters 300
python train_mappo.py --map 3x2-brt --traffic high --ctde --num-iters 300

# evaluate against the measured Sejong signal plans, then run the statistics
python evaluate_mappo.py --model models/MAPPO_sumo_N --model-ctde models/MAPPO_CTDE_sumo_M \
  --map 3x2-brt --traffic high --baseline sejong --episodes 30 --csv-out results/eval_sejong_30.csv
python stats_eval.py --csv results/eval_sejong_30.csv
```

Scenarios (`single`, `2x2`, `2x2-brt`, `3x2`, `3x2-brt`), every CLI flag, the reward options and video recording are documented in [README.ko.md](README.ko.md).

## Lessons learned

- **A metric bug hid the size of the effect.** SUMO's default waiting-time memory capped waits at ~100 s, which cut Fixed-Time's long waits the most. Re-evaluating the same checkpoints changed the headline from −27% to −50% and overturned an earlier "MAPPO and CTDE tie on waiting time" conclusion. Results produced before the fix are kept only for reference.
- **CTDE did not beat MAPPO here.** Its average is worse and its variance is larger. A critic fix (value normalization, explained variance 0.076 → 0.5–0.7) did not recover the headline, so improving CTDE is left as future work and MAPPO is the proposed controller.
- **Throughput depends on who gets in.** Even with MAPPO about 30% of loaded vehicles fail to enter the network in this LOS D scenario. Stops per vehicle get worse with RL because more vehicles are let through.
- **Limits.** Simulation only. Demand mixes 2025 counts with 2023 direction splits, assumes a standard pcu→veh factor, and estimates two intersections. One corridor, one seed set, one demand level.

## References

- Environment skeleton adopted from [SUMO-RL](https://github.com/LucasAlegre/sumo-rl) v1.4.5 (Lucas N. Alegre, 2019): action scheme, observation layout, `diff-waiting-time` reward, `2x2.net.xml`. SUMO-RL has no MAPPO implementation; reward scale changed from `/100` to `/10`.
- Prof. Giseop Noh, *S(Grad) Reinforce Learning* course. https://www.deepshark.org/courses/grad_reinforce_learning/
- C. Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games," *NeurIPS Datasets and Benchmarks*, 2022. (MAPPO)
- P. A. Lopez et al., "Microscopic Traffic Simulation using SUMO," *IEEE ITSC*, 2018.
- E. Liang et al., "RLlib: Abstractions for Distributed Reinforcement Learning," *ICML*, 2018.
- J. K. Terry et al., "PettingZoo: Gym for Multi-Agent Reinforcement Learning," *NeurIPS*, 2021.

---
한국어 문서: [README.ko.md](README.ko.md)
