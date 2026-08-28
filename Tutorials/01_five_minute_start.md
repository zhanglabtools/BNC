# Five-minute start

This path verifies the package, CLI, supplied paper data, and documentation
without launching training.

## 1. Create an isolated environment

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell (run this instead of the previous line)
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Confirm that the interpreter belongs to the environment:

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
```

## 2. Check the command-line interface

```bash
python -m bnc_repro.cli --help
python -m bnc_repro.cli train --help
```

The first command should list `train`, `aggregate`, `plot`, and `validate`.
The training help should list `--config` and `--output`.

## 3. Compile and test

```bash
python -m compileall -q src scripts tests examples
pytest -q
python scripts/check_tutorials.py
```

Compilation is a syntax check. The tests cover numerical invariants, schedules,
model interfaces, plotting, supplied-data contracts, recovery behavior, and
documentation links; they do not reproduce the full paper.

## 4. Validate supplied data

```bash
python -m bnc_repro.cli validate --figure all
```

Expected high-level states:

- Figure 1: `reference-artifact-only`; numerical replot is blocked because
  coordinates/checkpoint were not supplied.
- Figure 2: MLP data validates; the full four-architecture plot is blocked
  because three aggregate tables were not supplied.
- Figures S1-S4: supplied data validates.

Treat a different blocked reason, row count, or exception as a real failure.

## 5. Optional data-only plot

```bash
python scripts/reproduce_fig_s3.py \
  --data paper_data/fig_s3/alignment_summary.csv \
  --output figures/tutorial/fig_s3
```

Successful output contains `fig_s3.png`, `fig_s3.pdf`, and `fig_s3.svg` under
`figures/tutorial/`.

Next: [recreate all available paper-data figures](02_bundled_data_and_figures.md)
or [run the CPU smoke](03_cpu_smoke.md).
