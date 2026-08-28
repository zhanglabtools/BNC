# Figure-by-figure recipes

There are two distinct reproduction modes:

- **Data-only plot:** validate and plot supplied tables. This is fast.
- **Fresh training:** generate new run directories, aggregate them, then plot.
  This can require many GPU-days and missing prerequisites.

Never describe a data-only replot as a fresh experimental replication.

## Figure 1: mod-97 PCA

The supplied package contains a reference PNG and metrics JSON but no PCA
coordinates or checkpoint. A numerical data-only replot is therefore blocked.

Fresh training configuration:

```bash
python -m bnc_repro.cli train \
  --config configs/fig1/manuscript_seed111.yaml \
  --output outputs
```

The bundled reference uses a different seed; read
[`../docs/known_discrepancies.md`](../docs/known_discrepancies.md).

## Figure 2: classifier-first cyclic geometry

Replot the supplied tuned-MLP aggregate:

```bash
python scripts/reproduce_fig2.py \
  --data paper_data/fig2/mlp_aggregate_metrics.csv \
  --output figures/tutorial/fig2_mlp
```

Fresh four-architecture training:

```bash
python -m bnc_repro.cli train \
  --config configs/fig2/tuned_mlp_and_descriptive_architectures.yaml \
  --output outputs
```

The supplied paper data cannot create the four-architecture plot because only
the MLP aggregate was included.

## Figure S1: rank homotopy

Data-only plot:

```bash
python scripts/reproduce_fig_s1.py \
  --data paper_data/fig_s1/mean_std_trajectory.csv \
  --output figures/tutorial/fig_s1
```

Fresh training requires the dense checkpoint grid first:

```bash
python -m bnc_repro.cli train \
  --config configs/dense/s1_s2_dense_checkpoints.yaml \
  --output outputs

python -m bnc_repro.cli train \
  --config configs/fig_s1/lambda_ablation.yaml \
  --output outputs
```

## Figure S2: explicit rank-2 effective dimension

Data-only plot:

```bash
python scripts/reproduce_fig_s2.py \
  --data paper_data/fig_s2/mean_std_trajectory.csv \
  --output figures/tutorial/fig_s2
```

After the same dense prerequisite:

```bash
python -m bnc_repro.cli train \
  --config configs/fig_s2/effective_dimension.yaml \
  --output outputs
```

## Figure S3: token/classifier geometry

```bash
python scripts/reproduce_fig_s3.py \
  --data paper_data/fig_s3/alignment_summary.csv \
  --output figures/tutorial/fig_s3

python -m bnc_repro.cli train \
  --config configs/fig_s3/token_geometry_20k.yaml \
  --output outputs
```

## Figure S4: centered feature/classifier alignment

```bash
python scripts/reproduce_fig_s4.py \
  --data paper_data/fig_s4/centered_alignment_summary.csv \
  --output figures/tutorial/fig_s4

python -m bnc_repro.cli train \
  --config configs/fig_s4/centered_feature_classifier_20k.yaml \
  --output outputs
```

## Before a full grid

1. Run tests, paper-data validation, and CPU smoke.
2. Run one non-formal pilot on the intended GPU environment.
3. Confirm run directories, heartbeat, checkpoints, and aggregation.
4. Confirm the relevant shuffled or negative control.
5. Only then submit the complete formal grid.
