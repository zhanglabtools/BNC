# Source audit and canonicalization decisions

## Selection rule

For each manuscript target, the newest complete figure package is authoritative for data and formal metadata; the newest server source was used to verify whether later code existed. Older packages remain provenance/reference material and are not imported by `src/bnc_repro`.

| Target | Canonical supplied source | Core implementation |
|---|---|---|
| Figure 1 | `mod97_modular_addition_pca_code_20260717.zip` plus revised manuscript seed specification | `protocols/fig1.py`, `plotting/fig1.py` |
| Figure 2 MLP/BCS/tuning | `mlp_classifier_first_latest_complete_no_checkpoints_20260721(1).zip` | `metrics/bcs.py`, `protocols/fig2.py`, `aggregation/fig2.py`, `plotting/fig2.py` |
| Four-architecture definitions | Latest server native rerun and July multiarchitecture bundles | `models/`, common model interface, figure protocols |
| Dense prerequisite | Multiarchitecture dense/spectral sources | `protocols/dense.py` |
| Figure S1 | `classifier_rank_lambda_figure_ymax10_code_and_explanation_20260731.zip` | `protocols/rank_homotopy.py`, `aggregation/rank.py`, `plotting/fig_s1.py` |
| Figure S2 | `rank2_effective_dimension_figure_package_20260731.zip` | `protocols/rank2_finetune.py`, `plotting/fig_s2.py` |
| Figure S3 | `embedding_classifier_alignment_20k_figure_code_data_notes_20260731(1).zip` | `protocols/alignment.py`, `metrics/token_geometry.py`, `plotting/alignment.py` |
| Figure S4 | `feature_classifier_centered_alignment_multiarchitecture_20k_code_results_no_checkpoints_20260731.zip` | `metrics/feature_classifier.py`, shared alignment protocol/plotter |

## Scientific checks

- The BCS code was rebuilt from the exact July implementation: row-centering, two-dimensional SVD PCA coordinates, RMS normalization, pointwise unit-circle projection, all invertible multipliers modulo K, and both orientations.
- The common interface makes classifier orientation explicit. `classifier()` returns the feature-by-class matrix used in logits; `classifier_matrix()` returns class rows for geometry metrics.
- The four architectures have independent classifier heads. MLP is role-specific and bias-free. Transformer uses a K+1 vocabulary, causal single block, no layer norm/bias, and the named Q/K/V/O and MLP matrices. LSTM/RNN use a shared K+1 embedding with independent heads.
- S1 uses a cosine tail gate from epoch 1 through 6,000, CE for full and rank-2 heads, and the balanced A/B tail penalty.
- S2 uses the formal metadata profile rather than inconsistent historical argparse defaults.
- S3/S4 use `unique(rint(geomspace(1, 20000, 321)))`, yielding 240 positive checkpoints, and shuffle seed `10000 + seed` with 16 controls.

## Server comparison

The newest restricted project-server tree had no post-July experiment implementation. A source-only snapshot was downloaded and audited. Exact SHA-256 matches were found for the Figure 2 runner/aggregator, S1 trainer/aggregator, S3 trainer/plotter, S4 trainer/validator, and spectral runner. The S2 server file differed from the packaged file only in self-contained default-path handling; the formal algorithm/configuration was unchanged.

Server snapshots kept under `experiments/server_latest_audit/` are audit evidence only. Core commands run exclusively through `bnc_repro`.

## Rejected alternatives

- Earlier mod-97 exploratory packages are historical precursors and do not override later multiarchitecture or 20k packages.
- Raw, uncentered feature/classifier alignment is retained as an auxiliary server snapshot but does not replace the centered S4 metric.
- Spectral-rank dynamics remains under `experiments/auxiliary_spectral_rank/`; it is not one of the six requested manuscript figure targets.
- Missing checkpoints and missing Figure 2 aggregates were not inferred from plots.
