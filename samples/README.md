# Sample Artifacts

This directory holds curated reference artifacts for reproducibility and demonstration.  
`models/`, `results/`, and `videos/` at the repo root are **gitignored** — only promoted samples live here.

## Policy

| Type | Root (gitignored) | Here (tracked) |
|------|-------------------|----------------|
| Trained models | `models/*.zip` | `samples/models/<tag>.zip` |
| Eval metrics | `results/eval_metrics.csv` | `samples/results/<tag>_eval_metrics.csv` |
| TensorBoard logs | `results/tb_*/` | — (too large; omit) |
| Rollout videos | `videos/*.mp4` | `samples/videos/<tag>_rollout.mp4` |

## How to promote a result

After a notable training run, copy the artifact here with a descriptive tag:

```bash
# Example: promote v0.1.2 baseline results
cp models/dqn_sumo_single.zip        samples/models/dqn_v0.1.2.zip
cp results/eval_metrics.csv          samples/results/dqn_v0.1.2_eval_metrics.csv
cp videos/dqn_policy_rollout.mp4     samples/videos/dqn_v0.1.2_rollout.mp4
git add samples/
git commit -m "samples: promote v0.1.2 baseline artifacts"
```

## Naming convention

`<algorithm>_v<semver>[_<note>]`  — e.g. `mappo_v0.2.0_4agent`, `dqn_v0.1.2`
