# Reproduction status

Status as audited on 2026-08-28.

| Item | Status | Evidence / next action |
|---|---|---|
| Core package installation | Ready | Editable install verified in independent Windows and Linux environments. |
| Unit tests | Ready | 23 tests pass. |
| CPU smoke | Ready | MLP, Transformer, LSTM, and RNN train, aggregate, plot, and write completion status. |
| Figure 1 numerical replot | Blocked by missing data | The July 17 bundle has only a reference PNG and metrics JSON; rerun `manuscript_seed111.yaml` or supply coordinates/checkpoint. |
| Figure 1 formal training | Ready, expensive | `manuscript_seed111.yaml` follows Table S2/reviewer seed 111. The bundled artifact instead uses seed 1. |
| Figure 2 tuned MLP replot | Ready | 3,015 raw rows and 603 aggregate rows validated; MLP-only PNG/PDF/SVG generated. |
| Figure 2 four-architecture replot | Blocked by missing data | Transformer/LSTM/RNN aggregate tables were not present; runner and aggregator are implemented. |
| Figure S1 replot | Ready | 39,520 raw rows and 7,904 aggregate rows validated. |
| Figure S2 replot | Ready | 33,660 raw rows and 6,732 aggregate rows validated against formal metadata. |
| Figure S3 replot | Ready | 2,880 aggregate rows validated. |
| Figure S4 replot | Ready | 14,400 raw rows reconstructed from 60 runs and 2,880 aggregate rows validated. |
| Formal S1/S2 fresh run | Requires prerequisite | Dense checkpoints were excluded from the supplied packages. Generate them first with the dense config. |
| Full formal GPU sweep | Not launched | Intentionally outside the static/test/smoke validation requested by the task prompt. |

Machine-readable plot and validation states are written to `figures/paper_data/plot_status.json` and `paper_data/validation_report.json`.
