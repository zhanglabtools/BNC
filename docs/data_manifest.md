# Paper-data manifest

Each figure directory contains machine-readable `metadata.json` with its source ZIP, source-relative path, ZIP SHA-256, row counts, columns, figure mapping, and raw/aggregate classification.

| Directory | Primary files | Rows | Role |
|---|---|---:|---|
| `paper_data/fig1` | reference PNG, metrics JSON | n/a | Bundled reference artifact only |
| `paper_data/fig2` | raw, aggregate, onset CSVs | 3,015 / 603 / 15 | MLP training and data-only plot |
| `paper_data/fig_s1` | `all_runs_metrics.csv.gz`, trajectory CSV | 39,520 / 7,904 | Raw compressed and aggregate |
| `paper_data/fig_s2` | `all_runs_metrics.csv.gz`, trajectory CSV | 33,660 / 6,732 | Raw compressed and aggregate |
| `paper_data/fig_s3` | alignment and final CSVs | 2,880 aggregate | Aggregate only |
| `paper_data/fig_s4` | `all_runs_metrics.csv.gz`, centered summary | 14,400 / 2,880 | Raw reconstructed and aggregate |

`scripts/validate_reference_data.py` checks the defining row counts, schedules, seed counts, K values, architecture values, and formal S2 metadata. It writes `paper_data/validation_report.json`. `scripts/plot_all_paper_data.py` reads only these repository-relative files and writes `figures/paper_data/plot_status.json` plus PNG/PDF/SVG figures.

The S4 raw gzip was assembled without altering measurements from the 60 supplied per-run `metrics.csv` files. Architecture, K, and seed were parsed from audited run-directory names, and the combined row count is 60 × 240 = 14,400.
