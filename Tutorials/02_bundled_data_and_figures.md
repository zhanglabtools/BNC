# Bundled data and figures

This tutorial recreates every plot supported by the supplied tables. It does
not train a model and does not fill missing data.

## 1. Inspect provenance before plotting

Read these files first:

- [`../docs/data_manifest.md`](../docs/data_manifest.md) for files and row counts.
- [`../docs/known_discrepancies.md`](../docs/known_discrepancies.md) for blocked targets.
- Each `paper_data/<figure>/metadata.json` for source archive hashes and roles.

Raw measurements, aggregates, and reference-only artifacts have different
evidential status. Do not treat an aggregate-only directory as raw data.

## 2. Validate contracts

```bash
python scripts/validate_reference_data.py
```

The command writes `paper_data/validation_report.json` and prints the same
records. It checks row counts and schedule structure, not whether a research
claim is true.

## 3. Plot every available target

```bash
python scripts/plot_all_paper_data.py
```

Expected outputs:

```text
figures/paper_data/
|-- fig1_bundled_reference.png
|-- fig2_mlp.{png,pdf,svg}
|-- fig_s1.{png,pdf,svg}
|-- fig_s2.{png,pdf,svg}
|-- fig_s3.{png,pdf,svg}
|-- fig_s4.{png,pdf,svg}
`-- plot_status.json
```

Figure 1 is a supplied reference image, not a regenerated numerical plot. The
Figure 2 output is MLP-only. `plot_status.json` must continue to mark the
unavailable targets as blocked.

## 4. Plot one target with the CLI

```bash
python -m bnc_repro.cli plot \
  --figure fig_s4 \
  --data paper_data/fig_s4/centered_alignment_summary.csv \
  --output figures/tutorial/fig_s4
```

The output argument is a filename stem, so the plotter creates three formats.

## 5. Verify that outputs are non-empty

```bash
python -c "from pathlib import Path; p=Path('figures/paper_data'); files=list(p.glob('*.*')); assert files and all(f.stat().st_size for f in files); print(len(files), 'files checked')"
```

The existence check does not replace visual inspection. Open each PNG and check
titles, axes, legends, and whether the plotted target matches its label.
