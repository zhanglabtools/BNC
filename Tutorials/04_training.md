# Configured training

The repository uses YAML profiles rather than long command lines. This keeps
the defining experiment values reviewable. For formal profiles, the loader
checks selected experiment-specific grids and key hyperparameters; it does not
bind every protocol or architecture field, so review the YAML itself.

## Start with the smoke profile

```bash
python -m bnc_repro.cli train \
  --config configs/smoke/cpu_all_architectures.yaml \
  --output outputs/tutorial_smoke
```

This runs four architectures on K=17 for two CPU epochs. Inspect every
`status.json`, `config_resolved.yaml`, and `metrics.csv` before moving on.

## Launch a formal profile

```bash
python -m bnc_repro.cli train \
  --config configs/fig_s3/token_geometry_20k.yaml \
  --output outputs
```

The Figure S3 profile is a 60-run GPU grid with up to 20,000 epochs. Do not use
it as a first environment check. Review `docs/experiment_matrix.md` and
`docs/compute_requirements.md` before launch.

## Create a bounded pilot

To test one architecture/modulus/seed without weakening the manuscript profile:

1. copy the closest YAML to a new file under `configs/pilot/`;
2. give it a new experiment name;
3. set `formal: false`;
4. reduce the grid and epochs explicitly;
5. send outputs to a new pilot directory;
6. label every resulting table and plot as pilot data.

Do not edit a formal YAML in place and keep the same experiment label.

## Remote process lifetime

The core CLI currently has no resume option and prints little during long
training. Run it under the intended scheduler or a terminal multiplexer, and
write stdout/stderr to `logs/` rather than the metrics directory:

```bash
mkdir -p logs
python -u -m bnc_repro.cli train \
  --config configs/fig_s3/token_geometry_20k.yaml \
  --output outputs 2>&1 | tee logs/fig_s3.log
```

`python -u` keeps emitted output unbuffered; it does not add missing progress
messages. A reconnecting viewer does not prove the training process died. Check
the scheduler/process state and run files before starting a duplicate job.

## Current interruption boundary

Run metrics are generally finalized at the end of a run, and the unified core
CLI does not recover optimizer/RNG state after interruption. Preserve an
interrupted run directory for diagnosis, then restart in a clearly named fresh
output root. Historical scripts under `experiments/` may have different resume
logic, but they are audit material and are not the supported package interface.

Next: [outputs and aggregation](06_outputs_and_aggregation.md).
