# Python API walkthrough

The CLI is the supported path for experiment grids. The Python API is useful
for metric checks, theory examples, and custom analysis that does not change a
formal configuration.

## Run the metric example

```bash
python examples/metric_walkthrough.py
```

The script constructs an exact cyclic code, rotates it, and compares it with a
fixed shuffled negative control. Expected qualitative behavior:

- best cyclic score is close to 1 for the exact cyclic code;
- token-geometry correlation is close to 1 after an orthogonal rotation;
- the fixed shuffled control is lower for the chosen example;
- participation rank is close to 2.

Exact floating-point values can vary slightly across PyTorch builds.

## Core imports

```python
import torch

from bnc_repro.metrics.bcs import best_cyclic_score
from bnc_repro.metrics.participation import participation_rank, spectrum_summary
from bnc_repro.metrics.token_geometry import (
    fixed_shuffle_control,
    token_geometry_correlation,
)
```

All public geometry matrices use rows as tokens or classes (`K x d`) unless a
function explicitly says it accepts classifier columns (`d x K`). Shape
conversions are not implicit.

## Load and validate a configuration

```python
from bnc_repro.config import load_config

config = load_config("configs/smoke/cpu_all_architectures.yaml")
print(config["protocol"], config["grid"])
```

Formal profiles reject selected experiment-specific value changes, but they do
not bind every protocol or architecture field. Review the YAML directly. To
create an exploratory pilot, copy a config, set `formal: false`, use a new
experiment/output name, and label its outputs as pilot data.

## Execute a config programmatically

```python
from bnc_repro.config import load_config
from bnc_repro.training.engine import execute_config

config = load_config("configs/smoke/cpu_all_architectures.yaml")
paths = execute_config(
    config,
    "outputs/api_smoke",
)
print(*paths, sep="\n")
```

For long jobs, prefer the CLI so the exact invocation is visible in logs.

## Negative controls are part of the API contract

S3/S4 use fixed permutations derived from `10000 + training_seed`. Reusing the
same permutations across checkpoints makes trajectories comparable. Do not
resample a new permutation at every checkpoint.
