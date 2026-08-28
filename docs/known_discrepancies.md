# Known discrepancies and missing artifacts

## Figure 1 seed

The revised manuscript's Table S2 and reviewer response specify seed 111. The supplied July 17 reference artifact records seed 1. Both are preserved explicitly:

- `configs/fig1/manuscript_seed111.yaml` is the formal manuscript profile.
- `configs/fig1/bundled_reference_seed1.yaml` explains the included PNG/metrics artifact.

The reference directory has no PCA coordinate table and no checkpoint. Copying the PNG is auditable; claiming it was regenerated from data is not. `scripts/reproduce_fig1.py` therefore supports training/output generation, while the data-only plotting status remains blocked.

## Figure 2 architecture coverage

The MLP tuning package contains complete MLP raw and aggregate metrics for K=79/97/113 and seeds 1–5. No aggregate results for Transformer, LSTM, or RNN were supplied. The four-architecture runner and aggregator are retained, but the data-only command emits a structured blocked state for the full panel and still creates the MLP-only figure.

The architecture comparison is descriptive under specified per-architecture settings; it is not presented as a causal architecture-only ablation.

## Figure S2 command-line defaults

The final S2 package's `formal_experiment_config.json` specifies overall coefficient 1, participation weight 5, tail weight 0, balance weight 0, and a 200-epoch ramp. Defaults in an accompanying historical script used participation 1, tail 5, and balance 0.5. The formal metadata is authoritative. Strict configuration validation rejects the historical defaults for a formal run.

## Checkpoints

All named experiment packages were explicitly supplied without checkpoints. S1/S2 require dense initialization; their formal fine-tuning cannot start until the dense grid has been generated. No checkpoint is fabricated or downloaded.

## License

The supplied source archives contained no license. `LICENSE_PENDING.md` is therefore used instead of assigning rights that the authors have not approved.
