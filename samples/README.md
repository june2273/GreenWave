# Sample Artifacts

This directory holds curated reference artifacts for reproducibility and demonstration.
`models/`, `results/`, and `videos/` at the repo root are **gitignored** — only promoted samples live here.

## Policy

| Type | Root (gitignored) | Here (tracked) |
|------|-------------------|----------------|
| Trained models | `models/MAPPO_sumo_N/` | `samples/models/<tag>/` |
| Eval metrics | `results/eval_metrics_mappo_N.csv` | `samples/results/<tag>_eval_metrics.csv` |
| TensorBoard logs | `results/tb_mappo/MAPPO_sumo_N/` | — (too large; omit) |
| Rollout videos | `videos/mappo_policy_rollout_N.mp4` | `samples/videos/<tag>_rollout.mp4` |

## How to promote a result

After a notable training run, copy the artifact here with a descriptive tag:

```bash
# Example: promote v0.2.0 2x2grid dense-traffic result
cp -r models/MAPPO_sumo_11                samples/models/mappo_v0.2.0_2x2grid_dense
cp    results/eval_metrics_mappo_11.csv    samples/results/mappo_v0.2.0_2x2grid_dense_eval_metrics.csv
cp    videos/mappo_policy_rollout_11.mp4   samples/videos/mappo_v0.2.0_2x2grid_dense_rollout.mp4
git add samples/
git commit -m "samples: promote v0.2.0 2x2grid dense-traffic artifacts"
```

## Naming convention

`<algorithm>_v<semver>[_<note>]` — e.g. `mappo_v0.2.0_2x2grid_dense`, `mappo_v0.1.0_single`
