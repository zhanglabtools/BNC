# Python source migration map

This file accounts for every `.py` entry in the 15 supplied experiment-code ZIPs. “Ported” means the behavior was reimplemented behind the unified package interface; “preserved” means a script is retained outside the core package for audit; “superseded” means a later supplied package is canonical; “duplicate” means a byte/role duplicate was intentionally not copied again.

## `classifier_rank_lambda_ablation_K97_lambda0p01_to_3_20260726_no_checkpoints.zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `spectral_rank_lambda_ablation_K97_20260726/aggregate_and_plot_lambda.py` | Superseded | July 31 S1 figure package; `aggregation/rank.py`, `plotting/fig_s1.py` |
| `spectral_rank_lambda_ablation_K97_20260726/run_lambda_ablation.py` | Superseded | July 31 formal S1 profile; `protocols/rank_homotopy.py` |

## `classifier_rank_lambda_figure_ymax10_code_and_explanation_20260731.zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `.../code/aggregate_and_plot_lambda.py` | Ported | `aggregation/rank.py`, `plotting/fig_s1.py` |
| `.../code/plot_classifier_rank_lambda_ymax10.py` | Ported | `plotting/fig_s1.py` |
| `.../code/run_classifier_first_multiarchitecture.py` | Ported | shared `models/` and `protocols/dense.py` |
| `.../code/run_lambda_ablation.py` | Ported | `protocols/rank_homotopy.py` |

## `embedding_classifier_alignment_20k_figure_code_data_notes_20260731(1).zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `.../code/plot_alignment_architectures.py` | Ported | `plotting/alignment.py` |
| `.../code/summarize_alignment.py` | Ported | `aggregation/alignment.py` |
| `.../code/train_alignment_architecture.py` | Ported | `protocols/alignment.py`, `metrics/token_geometry.py` |

## `embedding_classifier_alignment_multiarchitecture_20260721_no_checkpoints(1).zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `.../scripts/plot_alignment_architectures.py` | Superseded | July 31 S3 package |
| `.../scripts/summarize_alignment.py` | Superseded | July 31 S3 package |
| `.../scripts/train_alignment_architecture.py` | Superseded | July 31 S3 package |
| `.../source_reference/original_alignment_trainer.py` | Historical reference | Defining metric audited into `metrics/token_geometry.py` |
| `.../source_reference/supplied_architecture_search.py` | Ported | canonical four-architecture implementations under `models/` |

## `feature_classifier_centered_alignment_multiarchitecture_20k_code_results_no_checkpoints_20260731.zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `.../scripts/plot_feature_classifier_centered_alignment.py` | Ported | `plotting/alignment.py` |
| `.../scripts/summarize_feature_classifier_centered_alignment.py` | Ported | `aggregation/alignment.py` |
| `.../scripts/train_feature_classifier_centered_alignment.py` | Ported | `protocols/alignment.py`, `metrics/feature_classifier.py` |
| `.../scripts/validate_centered_results.py` | Ported | `validation.py`, `scripts/validate_reference_data.py` |
| `.../tests/test_centered_alignment_metric.py` | Ported and extended | `tests/test_alignment_metrics.py` |

## `mlp_classifier_first_latest_complete_no_checkpoints_20260721(1).zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `.../aggregate_selected_mlp.py` | Preserved + ported | `experiments/fig2_tuning/aggregate_selected_mlp.py`, `aggregation/fig2.py` |
| `.../plot_tuned_mlp_results.py` | Preserved + ported | `experiments/fig2_tuning/plot_tuned_mlp_results.py`, `plotting/fig2.py` |
| `.../run_classifier_first_multiarchitecture.py` | Preserved + ported | `experiments/fig2_tuning/`, `protocols/fig2.py` |
| `.../search_mlp_classifier_first.py` | Preserved | `experiments/fig2_tuning/search_mlp_classifier_first.py` |
| `.../search_mlp_round2.py` | Preserved | `experiments/fig2_tuning/search_mlp_round2.py` |
| `.../search_mlp_round3.py` | Preserved | `experiments/fig2_tuning/search_mlp_round3.py` |
| `.../final_artifacts/aggregate_selected_mlp.py` | Duplicate | canonical copy already retained above |
| `.../final_artifacts/plot_tuned_mlp_results.py` | Duplicate | canonical copy already retained above |
| `.../final_artifacts/run_classifier_first_multiarchitecture.py` | Duplicate | canonical copy already retained above |
| `.../final_artifacts/search_mlp_classifier_first.py` | Duplicate | canonical copy already retained above |
| `.../final_artifacts/search_mlp_round2.py` | Duplicate | canonical copy already retained above |
| `.../final_artifacts/search_mlp_round3.py` | Duplicate | canonical copy already retained above |
| `.../round2/run_classifier_first_multiarchitecture.py` | Search-stage variant | audit parameters retained in round-2 ranking/per-run records |
| `.../round3/run_classifier_first_multiarchitecture.py` | Search-stage variant | audit parameters retained in round-3 ranking/per-run records |

## `mod97_classifier_embedding_rank2_dynamics_20260720.zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `scripts/make_mod97_dual_rank2_report.py` | Historical | superseded by multiarchitecture S1/S2 plotting |
| `scripts/run_mod97_soft_rank_homotopy.py` | Historical precursor | audited into `protocols/rank_homotopy.py` |

## `mod97_embedding_classifier_alignment_code_20260720(1).zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `scripts/make_mod97_embedding_classifier_alignment_report.py` | Historical | superseded by July 31 S3 plotter |
| `scripts/run_mod97_embedding_classifier_alignment.py` | Historical precursor | superseded by multiarchitecture 20k S3 trainer |

## `mod97_modular_addition_pca_code_20260717.zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `run_mod_add_pca_probe.py` | Ported | `protocols/fig1.py`, `plotting/fig1.py`; reference artifact in `paper_data/fig1` |

## `mod97_rank2_timescales_bundle_20260720(1).zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `.../scripts/make_mod97_rank2_timescales.py` | Historical | spectral/rank auxiliary, outside requested figure set |
| `.../scripts/run_mod97_progressive_rank_compression.py` | Historical precursor | later S1/S2 protocols are canonical |

## `mod97_weight_decay_rank2_speed_20260720.zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `scripts/make_mod97_rank2_wd_dynamics_report.py` | Historical | later multiarchitecture rank figures are canonical |
| `scripts/run_mod97_progressive_rank_compression.py` | Historical precursor | later S1/S2 protocols are canonical |

## `rank2_collapsed_multiarchitecture_bundle_no_checkpoints_20260721(1).zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `.../aggregate_and_plot.py` | Superseded + ported | July 31 S2 package; `aggregation/rank.py`, `plotting/fig_s2.py` |
| `.../run_rank2_representation_collapse.py` | Superseded + ported | July 31 S2 formal metadata; `protocols/rank2_finetune.py` |

## `rank2_effective_dimension_figure_package_20260731.zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `.../code/aggregate_and_plot.py` | Ported | `aggregation/rank.py`, `plotting/fig_s2.py` |
| `.../code/plot_figure.py` | Ported | `plotting/fig_s2.py` |
| `.../code/run_classifier_first_multiarchitecture.py` | Ported | common `models/`, `protocols/dense.py` |
| `.../code/run_rank2_representation_collapse.py` | Ported with formal metadata correction | `protocols/rank2_finetune.py` |

## `spectral_rank_dynamics_multiarchitecture_20260721_no_checkpoints(1).zip`

| Original path | Status | Destination / reason |
|---|---|---|
| `.../aggregate_and_plot.py` | Preserved auxiliary | `experiments/auxiliary_spectral_rank/aggregate_and_plot.py` |
| `.../run_spectral_rank_multiarchitecture.py` | Preserved auxiliary | `experiments/auxiliary_spectral_rank/run_spectral_rank_multiarchitecture.py` |

## Archives with no `.py` entries

`mod97_classifier_first_20260720.zip` contains shell/config/reference material but no Python file. All Python entries in the other 14 experiment archives are listed above. The revision ZIP is manuscript/reviewer material, not an executable-code archive.
