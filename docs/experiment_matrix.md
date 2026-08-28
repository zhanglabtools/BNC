# Experiment matrix

All formal grids use a 70/30 seeded split of the complete ordered modular-addition table.

| Target | Architectures | K | Seeds | Epochs | Main configuration |
|---|---|---|---|---:|---|
| Figure 1 | MLP | 97 | 111 formal; 1 bundled reference | 10,000 | 256 embedding, 128 hidden, AdamW lr 1e-3, wd 1, betas (0.9, 0.98) |
| Figure 2 | MLP, Transformer, LSTM, RNN | 79, 97, 113 | 1–5 | 10,000 | MLP 512/512; embedding lr 2e-4/wd 0; other lr 3e-3/wd 0.4; log every 50 |
| Dense prerequisite | All four | 79, 97, 113 | 1–5 | 10,000 | MLP 256/128; embedding lr 2e-4/wd 0; other lr 2e-3/wd 0.8 |
| Figure S1 | All four | 97 | 1–5 | 10,000 | lambda in {0.01, 0.1, 0.5, 0.7, 1, 1.5, 2, 3}; max rank 16 to rank 2 |
| Figure S2 | All four | 79, 97, 113 | 1–5 | 6,000 | explicit balanced rank-2 head; participation target 2 |
| Figure S3 | All four | 79, 97, 113 | 1–5 | 20,000 | token-geometry alignment, 240 unique geometric checkpoints, 16 controls |
| Figure S4 | All four | 79, 97, 113 | 1–5 | 20,000 | centered feature/classifier alignment, same checkpoints/controls |

## Architecture dimensions

The non-Figure-2 MLP uses 256-dimensional role-specific embeddings and a 128-dimensional hidden layer. The tuned Figure-2 MLP uses 512/512 and is intentionally a separate profile. Transformer uses `d_model=128`, 4 heads of width 32, and MLP width 512. LSTM and RNN use 128-dimensional embeddings and hidden states.

## Schedule counts

- Figure 2: epochs 0, 50, ..., 10,000 = 201 rows per run; 15 MLP runs = 3,015 raw rows.
- S1: its piecewise logging schedule yields 247 rows per run; 4 architectures × 5 seeds × 8 lambdas = 160 runs and 39,520 rows.
- S2: epochs 0–200 each epoch, 205–1,000 every 5, then every 25 through 6,000 = 561 rows per run; 60 runs and 33,660 rows.
- S3/S4: `unique(rint(geomspace(1, 20000, 321)))` yields 240 checkpoints; 60 runs and 14,400 raw rows when raw tables are available, and 4 × 3 × 240 = 2,880 aggregate rows.

Formal YAML files live under `configs/`; the loader checks these defining values before training.
