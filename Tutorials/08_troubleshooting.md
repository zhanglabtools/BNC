# Troubleshooting

Diagnose the smallest failing layer first. Do not relaunch a formal grid until
the corresponding five-minute, smoke, or one-run check passes.

## The CLI cannot import `bnc_repro`

Cause: the editable package is not installed in the active interpreter.

```bash
python -c "import sys; print(sys.executable)"
python -m pip install -e ".[dev]"
python -c "import bnc_repro; print(bnc_repro.__file__)"
```

Use `python -m pip`, not a bare `pip`, so installation and execution use the
same interpreter.

## Python is too old

`pyproject.toml` requires Python 3.10 or newer. Create a new environment with a
supported interpreter; do not edit the requirement to hide the mismatch.

## PyTorch cannot load a DLL on Windows

Common causes include an incompatible CUDA/PyTorch build, missing runtime DLLs,
or insufficient page file. Check the exact exception, Python bitness, PyTorch
build, free memory, and page-file availability. Do not repeatedly rerun the same
failing import.

## CUDA was requested but is unavailable

Formal configs intentionally request CUDA. Verify the selected interpreter and
driver:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

Use the CPU smoke config for local verification. Do not silently change a
formal manuscript config from CUDA to CPU and call it the same experiment.

## A remote UI keeps reconnecting

The repository has no WebSocket or network client. First distinguish a dropped
viewer from a dead training process:

1. inspect the run's `status.json`, logs, and last modified outputs;
2. inspect the training log and process/scheduler state;
3. reconnect without starting a duplicate job;
4. if the process died, preserve the interrupted directory;
5. restart in a clearly named fresh output root after diagnosing the failure.

Use `python -u` and log redirection so emitted output is not buffered. The core
CLI has no resume support and cannot repair VPN, firewall, SSH, or service
connectivity.

## A run remains `running`

An unclean interruption can leave the last status as `running`. Check the
process state. If no process is alive, preserve the directory and restart in a
new output root. Do not mark it complete or merge partial CSVs by hand.

## S1/S2 cannot find dense checkpoints

This is an expected missing prerequisite, not a download retry problem. The
source packages omitted dense checkpoints. Generate them with:

```bash
python -m bnc_repro.cli train \
  --config configs/dense/s1_s2_dense_checkpoints.yaml \
  --output outputs
```

Then confirm that the fine-tuning config resolves the generated paths.

## Figure 1 or full Figure 2 data-only plotting is blocked

These are audited data gaps. Figure 1 lacks numerical coordinates/checkpoint;
full Figure 2 lacks three architecture aggregates. Supply real missing data or
run training. Do not synthesize substitute tables.

## Plot files exist but look wrong

Check the input CSV, figure selector, output stem, axis labels, legend, and
`plot_status.json`. File existence alone is not visual validation.

## Collect a bounded bug report

Include:

- operating system, Python, PyTorch, and CUDA versions;
- exact command and config path;
- traceback or final 100 log lines;
- `config_resolved.yaml` and `status.json`;
- whether five-minute checks and CPU smoke pass;
- no checkpoints, private data, credentials, or full large output trees.
