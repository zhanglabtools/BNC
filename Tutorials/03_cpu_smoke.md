# Four-architecture CPU smoke

The smoke path answers one narrow question: can the current package train,
record, aggregate, and plot one tiny run for every supported model family?

## 1. Review the smoke configuration

Open `configs/smoke/cpu_all_architectures.yaml`. Its defining properties are:

- K=17;
- seed 1;
- MLP, Transformer, LSTM, and RNN;
- two epochs on CPU;
- four fixed shuffled controls.

The configuration has `formal: false`. This is intentional.

## 2. Run the wrapper

```bash
python scripts/run_smoke.py --output outputs/smoke
```

The wrapper runs training, aggregates matched and shuffled alignment metrics,
and creates plots. A typical output tree is:

```text
outputs/smoke/
|-- smoke_alignment/
|   |-- mlp/K17/seed_1/
|   |-- transformer/K17/seed_1/
|   |-- lstm/K17/seed_1/
|   `-- rnn/K17/seed_1/
|-- token_geometry_summary.csv
|-- centered_feature_summary.csv
|-- token_geometry.{png,pdf,svg}
`-- centered_feature.{png,pdf,svg}
```

Each run directory should contain `config_resolved.yaml`, `status.json`, and a
non-empty `metrics.csv`.

## 3. Run the same config through the unified CLI

Use this path when you want to exercise the same configuration through the
public CLI:

```bash
python -m bnc_repro.cli train \
  --config configs/smoke/cpu_all_architectures.yaml \
  --output outputs/tutorial_smoke
```

Use a fresh output directory when repeating a smoke run. The current core CLI
does not resume interrupted runs.

## 4. Inspect the negative control

```bash
python -c "import pandas as pd; p='outputs/smoke/token_geometry_summary.csv'; d=pd.read_csv(p); print(d[['architecture','matched_mean','shuffled_mean']].to_string(index=False))"
```

The command is an operational check, not a pass/fail research criterion. Two
epochs on K=17 are too small for a paper claim. Record only that the metric and
control paths executed and produced finite values.

## 5. Decide whether to continue

- **Go to a bounded pilot** only if all four run statuses are complete, the
  summaries are readable, and the plots are visually valid.
- **Stop and diagnose** if any run is still `running`, contains a traceback, or
  is missing metrics.

Next: [configured training](04_training.md).
