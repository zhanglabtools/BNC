# Claims-to-code map

| Reproduction claim | Configuration | Implementation | Test/data evidence |
|---|---|---|---|
| Figure 1 shows modular cyclic PCA geometry | `configs/fig1/*.yaml` | `protocols/fig1.py`, `plotting/fig1.py` | Bundled PNG/metrics in `paper_data/fig1`; rerun required for coordinates |
| Classifier BCS can precede embedding BCS | `configs/fig2/tuned_mlp_and_descriptive_architectures.yaml` | `metrics/bcs.py`, `protocols/fig2.py`, `aggregation/fig2.py` | 3,015 raw MLP rows; censor-aware onset table; `test_bcs.py` |
| Rank pressure drives classifier/effective dimension toward two | `configs/fig_s1/lambda_ablation.yaml` | `protocols/rank_homotopy.py`, `metrics/participation.py` | 39,520 raw rows; schedule and rank tests |
| Explicit rank-2 heads support low-dimensional representation | `configs/fig_s2/effective_dimension.yaml` | `protocols/rank2_finetune.py`, `protocols/rank_common.py` | 33,660 raw rows; numerical-rank invariant; formal-config tests |
| Token and classifier geometries align during training | `configs/fig_s3/token_geometry_20k.yaml` | `metrics/token_geometry.py`, `protocols/alignment.py` | 2,880 aggregate rows; orthogonal-invariance test |
| Centered class features align with classifier directions | `configs/fig_s4/centered_feature_classifier_20k.yaml` | `metrics/feature_classifier.py`, `protocols/alignment.py` | 14,400 raw/2,880 aggregate rows; offset/sign/control tests |
| Result handling is reproducible | all configs | `training/checkpoints.py`, `aggregation/`, `validation.py` | resolved configs, runtime metadata, row-count tests, CPU smoke |

This map links software to experimental evidence. Interpretation beyond these measurable claims remains the responsibility of the associated paper.
