# Metric definitions

## Best cyclic score (BCS)

For a K-by-d embedding or classifier-row matrix, subtract the row mean, take the first two SVD PCA coordinates `U[:, :2] S[:2]`, and divide by the root-mean-square radius. Project each nonzero 2D point to the complex unit circle. Compare it with cyclic targets for every multiplier coprime to K and for both orientations; the largest absolute mean complex correlation is the BCS in [0, 1]. Near-zero/degenerate inputs raise an error.

## Participation rank and spectral summaries

For singular-value energy `e_i = sigma_i^2`, participation rank is `(sum e_i)^2 / sum e_i^2`. The reported effective representation dimension is the average participation rank of the two role-conditioned codes `R_x` and `R_y`. Entropy rank uses `exp(-sum p_i log p_i)` with `p_i = sigma_i / sum sigma_i`. Top-2 tail is `sum_{i>2} e_i / sum_i e_i`.

Role-conditioned codes average penultimate features over all complete-grid examples sharing the first or second input symbol and subtract the complete-grid mean.

## S1 rank homotopy

The dense classifier is balanced-factorized into `A @ B` with maximum rank 16. Components after the first two are multiplied by a cosine gate that decreases from 1 at epoch 1 to 0 at epoch 6,000. The loss is full-head cross entropy + rank-2-head cross entropy + `lambda * 0.5 * (mean(A_tail^2) + mean(B_tail^2))`. AdamW learning rate decays by cosine from 1e-3 to 1e-5.

## S2 explicit rank-2 fine-tuning

The dense classifier is balanced-factorized directly as rank 2. Training uses train-split cross entropy and a full-grid role-conditioned participation penalty `0.5 * sum_role log(PR_role / 2)^2`, weighted by 5 and linearly ramped for 200 epochs. Tail and balance weights are zero in the authoritative formal profile. A numerical-rank invariant rejects a classifier rank greater than 2.

## Token-geometry alignment (S3)

Center and row-normalize each token/class representation. Form each matrix's pairwise cosine Gram matrix, retain the strict upper triangle, and compute the Pearson correlation between the two vectors. Fixed label permutations give a shuffled control.

## Centered feature/classifier alignment (S4)

Compute the K class means of penultimate features on the complete grid. Center class means and classifier rows separately across classes, row-normalize, then average corresponding-class cosine similarity. The shuffled control permutes classifier rows before the corresponding-class average. Degenerate centered rows raise an error.

For S3/S4, each run uses 16 fixed permutations generated from seed `10000 + training_seed`; the same permutations are reused across checkpoints.
