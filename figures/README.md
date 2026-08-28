# Versioned paper figures

These files are release artifacts generated from the versioned data under `paper_data/`, except for the explicitly labeled Figure 1 reference image.

| Figure | Files | Source | Status |
|---|---|---|---|
| Figure 1 | `paper_data/fig1_bundled_reference.png` | Supplied July 17 reference artifact | Reference only; coordinate data/checkpoint absent |
| Figure 2 | `paper_data/fig2_mlp.{png,pdf,svg}` | `paper_data/fig2/mlp_aggregate_metrics.csv` | Reproducible MLP-only panel |
| Figure S1 | `paper_data/fig_s1.{png,pdf,svg}` | `paper_data/fig_s1/mean_std_trajectory.csv` | Reproducible |
| Figure S2 | `paper_data/fig_s2.{png,pdf,svg}` | `paper_data/fig_s2/mean_std_trajectory.csv` | Reproducible |
| Figure S3 | `paper_data/fig_s3.{png,pdf,svg}` | `paper_data/fig_s3/alignment_summary.csv` | Reproducible |
| Figure S4 | `paper_data/fig_s4.{png,pdf,svg}` | `paper_data/fig_s4/centered_alignment_summary.csv` | Reproducible |

Regenerate all self-contained plots with:

```bash
python scripts/plot_all_paper_data.py
```

The command also writes `paper_data/plot_status.json` in this directory. It records the missing-data limitations for Figure 1 numerical replotting and the full four-architecture Figure 2 panel.
