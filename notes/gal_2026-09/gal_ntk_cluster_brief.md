# Gal's NTK question: one-step diagnostic

Goal: separate a wrong merged-coordinate objective, ambiguity in a free-coefficient span fit, and failure of image reconstruction. Use self-trained models with known data. Do not assume a first-layer adapter recovers images or a head adapter necessarily returns blends.

Use the existing small MLP and chart setup, N = 2 and 8, one full-batch SGD step, B0 = 0, known A0, unit LoRA scaling, float64, fixed preprocessing, and no momentum, regularization, dropout, or changing batch-normalization statistics. Compare adapter placement on the first layer and on the head. Record all layer dimensions, chart dimensions, and ranks; placement comparisons are descriptive if parameter budgets differ. Require q = rank(H) < r and check excitation rather than assuming it.

## Objectives on each identical release

1. **Naive merged control.** Reproduce the earlier raw-gradient mixture fit to B1 A1, omitting the sketch. This is a diagnostic of that objective, not the strongest attack on merged weights.
2. **Adapter-coordinate NTK fit.** Use model-output gradients with respect to the actually trained factors at (A0, B0). For multiclass output, allow one coefficient per example and output coordinate, as in Loo et al. Appendix F. Keep the downstream Jacobian's dependence on the candidate image. Record that the initial A-gradient is zero.
3. **Free-vector factor fit.** Fit B1 = U(A1 H_candidate)^T with U unrestricted, eliminating U by least squares for each candidate set. This tests only a subspace condition. At a linear head and T = 1 it can coincide with the unconstrained multiclass NTK fit; record that equality rather than claiming the two are always distinct.
4. **Exact one-step replay.** Compute loss derivatives from the candidate images and training labels, then use B1 = -eta D0(A0 H_candidate)^T and A1 = A0. Use known labels for this diagnostic. Unknown labels are a separate experiment.

## Evaluate before optimizing images

- Evaluate at the true images, fitting only permitted free coefficients. Compare normalized residuals to each objective's own numerical floor.
- Evaluate mixtures H_candidate = H K for invertible K with columns summing to one. Distinguish freely mixed feature vectors from features attained by admissible chart images. Do not identify the two after a nonlinear backbone.
- For realizable candidate blends, evaluate exact replay as well as certificate and factor-fit residuals. A certificate zero is not an endpoint alias.
- Then run a small, fixed budget of matched image starts. Record initialization, actual compute, residuals, and recovery under optimal permutation. A high optimized residual with a low truth residual is a search failure, not evidence of non-identifiability.

Deliver one figure with rows for adapter location: residual at truth, residual on realizable blends, and reconstruction success across matched starts for the four objectives. Include q, r, dimensions, and parameter counts in the caption. Report “no aliases found in these runs”; finite restarts cannot prove uniqueness.

## Sanity counterexample

Let W0 = 0, H = [e1,e2] in R^4, A0 be a full-row-rank 3-by-4 matrix injective on span(H), and use a 3-class softmax head with labels e1 and e2. At initialization D = (1/3)11^T - [e1,e2] is fixed. Put U = -eta D and B1 = U(A0 H)^T.

For K = [[0.75,0.25],[0.25,0.75]], set H' = H K and U' = U K^(-T). Then U'(A0 H')^T = B1 exactly. But actual replay retains D, giving B1' = U K^T(A0 H)^T, which differs from B1. Both U and A0 H have rank two, so equality would force K = I. Thus the free-coefficient symmetry is not automatically a symmetry of the training release, even at a head.

For a numerical check with NumPy default_rng(713), A0 drawn standard normal, and eta = 0.1, the free-fit relative error was 2.65e-16; the replay relative error was 0.2998. These are a new toy sanity check, not a reproduction of the CIFAR results.
