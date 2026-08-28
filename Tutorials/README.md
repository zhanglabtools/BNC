# Tutorials

Choose the shortest path that answers your question. All commands assume the
repository root as the current directory and an editable installation:

```bash
python -m pip install -e ".[dev]"
```

| Goal | Tutorial | Expected artifact |
|---|---|---|
| Confirm the package and supplied data are usable | [Five-minute start](01_five_minute_start.md) | Validation JSON in the terminal |
| Recreate plots without training | [Bundled data and figures](02_bundled_data_and_figures.md) | PNG/PDF/SVG under `figures/paper_data/` |
| Exercise all four model families | [CPU smoke](03_cpu_smoke.md) | Run metrics, summaries, and plots under `outputs/smoke/` |
| Launch a configured grid safely | [Configured training](04_training.md) | Run directories under `outputs/` |
| Reproduce one named figure | [Figure recipes](05_figure_recipes.md) | Target-specific data, run, or plot outputs |
| Inspect and aggregate a custom sweep | [Outputs and aggregation](06_outputs_and_aggregation.md) | Aggregate CSV and custom plots |
| Reuse metrics in another program | [Python API](07_python_api.md) | Printed metric values from `examples/` |
| Diagnose failures | [Troubleshooting](08_troubleshooting.md) | A bounded diagnosis and next action |

## Recommended order

1. Run the five-minute start.
2. Recreate the supplied-data figures.
3. Run the CPU smoke.
4. Read the configured-training and output-contract tutorials.
5. Launch one bounded GPU run before any full formal grid.

Do not interpret smoke results as experimental evidence. A formal result is
shareable only after its configuration, run status, aggregate, and relevant
negative control have all been checked.
