# Compute requirements

## Safe local checks

Compilation, the 22-test suite, paper-data validation, and data-only plotting are CPU tasks. The K=17 smoke configuration trains one seed for two epochs on each of MLP, Transformer, LSTM, and RNN and is intended for ordinary development machines.

## Formal runs

Formal configurations default to CUDA because they cover K in {79, 97, 113}, five seeds, four architectures, dense grids, and up to 20,000 epochs. Figure 2 alone records 201 checkpoints per run; S3/S4 use 240 unique geometrically spaced checkpoints. S1 includes 8 lambda values at K=97, and S2 spans three K values. Run each seed/configuration in its own output directory so interrupted work can be diagnosed and rerun without mixing tables. The unified core CLI does not currently resume optimizer/RNG state.

Exact GPU memory and wall time depend on architecture, device, PyTorch build, and scheduling, and were not benchmarked as part of this consolidation. Start with the CPU smoke, then a single formal CUDA run, before submitting the full grid. S1/S2 also require the dense checkpoint grid, which was not supplied.

Recommended preflight:

```bash
python scripts/run_smoke.py
python -m bnc_repro.cli train --config configs/dense/s1_s2_dense_checkpoints.yaml --output outputs
```

The second command is a real formal sweep; only launch it on the intended GPU system after reviewing the configuration.
