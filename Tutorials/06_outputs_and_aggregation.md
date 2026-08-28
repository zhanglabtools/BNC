# Outputs and aggregation

Training output is deliberately separated from versioned `paper_data/`.
Never point a fresh run at `paper_data/` or `figures/paper_data/`.

## Run directory contract

A run directory is identified by architecture, modulus, seed, and any
protocol-specific grid value. It contains some or all of:

| File | Role | Safe to aggregate? |
|---|---|---:|
| `config_resolved.yaml` | Exact configuration recorded for the run | Inspect first |
| `status.json` | Run state and completion metadata | Only complete is final |
| `metrics.csv` | Checkpoint-level measurements | Yes, after completion |
| model/checkpoint file | Final or prerequisite weights | No |

File names can vary for protocol-specific final checkpoints. Treat
`status.json` and the resolved config as the authoritative identity checks.

## Inspect statuses before aggregation

```bash
python -c "from pathlib import Path; import json; files=list(Path('outputs').rglob('status.json')); print('\n'.join(f'{p}: {json.loads(p.read_text()).get(\"status\")}' for p in files))"
```

Do not aggregate a mixture of `running`, `failed`, and `complete` runs. The core
CLI has no resume support; preserve interrupted directories for diagnosis and
restart in a clearly named fresh output root.

The aggregator does not enforce this rule: it recursively reads every
`metrics.csv` below `--runs-root` without checking `status.json`. Use a clean
root that contains only compatible, complete runs.

## Aggregate alignment runs

```bash
python -m bnc_repro.cli aggregate \
  --figure fig_s3 \
  --runs-root outputs/fig_s3 \
  --output outputs/aggregates/fig_s3_summary.csv
```

The alignment aggregator groups by architecture, K, and epoch; it reports mean,
sample standard deviation, and the number of unique seeds for matched and
shuffled metrics. For centered feature/classifier alignment, use Figure S4.

## Aggregate Figure 2 runs

```bash
python -m bnc_repro.cli aggregate \
  --figure fig2 \
  --runs-root outputs/fig2 \
  --output outputs/aggregates/fig2
```

Figure 2 produces raw, aggregate, and onset tables in the output directory.
Onset is the first of two consecutive checkpoints at or above the threshold;
right-censored runs remain explicit.

## Plot an aggregate

```bash
python -m bnc_repro.cli plot \
  --figure fig_s3 \
  --data outputs/aggregates/fig_s3_summary.csv \
  --output figures/custom/fig_s3
```

Keep generated custom plots outside `figures/paper_data/`, which is reserved
for audited, versioned supplied-data figures.

## Aggregation checklist

- every included run is complete;
- resolved configurations match the intended profile;
- expected architectures, K values, seeds, and epochs are present;
- no duplicate run identity or duplicate epoch exists;
- matched and negative-control columns are both present;
- plot labels say whether the data are supplied, pilot, or formal fresh runs.
