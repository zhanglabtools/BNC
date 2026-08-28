# Command-line reference

Install the package in editable mode before using the module or console entry
point:

```bash
python -m pip install -e ".[dev]"
```

These two forms are equivalent:

```bash
python -m bnc_repro.cli --help
bnc-repro --help
```

## `train`

Run every grid item in one YAML configuration.

```bash
python -m bnc_repro.cli train --config CONFIG [--output OUTPUT_ROOT]
```

| Argument | Required | Meaning |
|---|---:|---|
| `--config` | Yes | YAML profile loaded and validated by `bnc_repro.config` |
| `--output` | No | Root for generated run directories; config/default is used when omitted |

The loader validates selected experiment-specific values for formal profiles,
but it does not bind every protocol name or architecture list. The current
unified CLI does not expose grid overrides or resume. Review the YAML and use a
separate `formal: false` pilot config rather than changing a manuscript profile
in place.

## `aggregate`

Combine run-level `metrics.csv` files below a selected root.

```bash
python -m bnc_repro.cli aggregate \
  --figure FIGURE \
  --runs-root RUNS_ROOT \
  --output OUTPUT
```

`FIGURE` is one of `fig1`, `fig2`, `fig_s1`, `fig_s2`, `fig_s3`, or `fig_s4`.
The command recursively reads every `metrics.csv` below `--runs-root` and does
not inspect `status.json`; point it only at compatible runs whose status is
`complete`.
Figure 1 has no generic aggregate path because its PCA requires a trained
checkpoint. Figure 2 writes a directory of raw, aggregate, and onset tables;
the other supported aggregators write the requested CSV path.

## `plot`

Create PNG, PDF, and SVG from one compatible data table.

```bash
python -m bnc_repro.cli plot \
  --figure FIGURE \
  --data INPUT_TABLE \
  --output OUTPUT_STEM \
  [--font-family Arial]
```

`--output` is a stem, not a directory. For example,
`--output figures/custom/fig_s3` creates `fig_s3.png`, `fig_s3.pdf`, and
`fig_s3.svg`.

## `validate`

Validate supplied paper data for one target or all targets.

```bash
python -m bnc_repro.cli validate \
  --figure {fig1,fig2,fig_s1,fig_s2,fig_s3,fig_s4,all} \
  [--data paper_data]
```

The command prints JSON. A deliberate blocked state is not the same as an
exception: Figure 1 and the full Figure 2 plot have documented missing inputs.

## Wrapper scripts

| Script | Purpose |
|---|---|
| `scripts/run_smoke.py` | Train four tiny CPU runs, aggregate, and plot |
| `scripts/validate_reference_data.py` | Validate all supplied paper data and write a report |
| `scripts/plot_all_paper_data.py` | Plot every available supplied-data target and write statuses |
| `scripts/reproduce_fig*.py` | Preselect one figure for the `plot` subcommand |
| `scripts/check_tutorials.py` | Verify repository-local Markdown links |

Run commands from the repository root so relative config and data paths match
the documented examples.
