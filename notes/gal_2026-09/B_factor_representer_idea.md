# Separate research idea: fitting the released B factor

Recorded 8 September 2026 at Yoad's request. Keep this as a comparison idea, separate from the proof note for Gal and from the original NTK experiment.

## The method

Given released factors A_T and B_T, search jointly for candidate images x_i with features h_i = phi(x_i), using

\[
B_T = \sum_{i=1}^{N} u_i (A_T h_i)^\top,
\qquad u_i \in \mathbb R^m.
\]

The coefficients are free vectors. This is a factor representation, not the earlier attempt to fit the merged update B_T A_T by a mixture of fixed-anchor example gradients. If implemented, eliminate the coefficients by linear least squares for each candidate batch rather than optimizing them as extra variables.

## What it adds to the certificate

Let C = proj_{row(B_T)^perp} A_T. Suppose the private features are independent, q = N, and rank(B_T) = rank(A_T H) = N. An exact candidate fit exists if and only if every candidate satisfies C h_i = 0 and the N readings A_T h_i are independent. Indeed, the candidate readings must span the N-dimensional row space of B_T.

Thus this formulation adds a batch independence requirement to the individual certificate tests. It cannot resolve the remaining feature-space ambiguity: an independent set of blends can also span the same space. It gives no general guarantee of training-image membership or complete recovery.

The rank(A_T H) = N condition is additional to the certificate theorem. Without it, the true batch need not admit this representation even when C H = 0. The stated equivalence also must not be carried over unchanged to N > q: spare candidates can then remain unconstrained by an exact joint fit.

## Working judgment

Keep C as the main per-image search objective. It separates candidate searches and avoids mixture coefficients. The joint factor fit couples the whole batch, so there is no general reason it should optimize more easily. Its loss may have different numerical conditioning despite the same exact solutions under the stated assumptions.

Use the factor representation as a possible coverage and selection step: collect low-residual candidates with C, select readings spanning row(B_T), and consider a joint refinement if directions are missing. Whether refinement improves recovery needs testing. Spanning the space alone does not establish that the selected images are the private images.

Do not call this the NTK method we already ran, and do not put it back into Gal's proof PDF without a further request.

## Yoad's proposed combination: partial candidates plus open slots

Use several good candidates found by C as an initial partial batch, add fresh candidate-image slots, and fit the released B factor to complete the batch. This is an untested strategy to retain, not a claim of demonstrated improvement.

1. Run independent certificate searches. Retain candidates with small residuals and nondegenerate, linearly independent readings A_T h_i; visual difference alone is insufficient.
2. Initially hold p selected candidates fixed. In the q = N regime, add N - p slots parameterized by actual images or public-chart coordinates, with diverse random initializations. Do not use unconstrained feature vectors, which can fill a span without corresponding to images.
3. Fit B_T jointly, eliminating the free coefficients by least squares. With the selected candidates fixed, only (N - p)k image coordinates remain to optimize instead of Nk. Continue checking C on candidates and watch for rank collapse; a small joint residual can hide a poorly constrained individual candidate numerically.
4. If the remaining search stalls, try other selected sets or restarts, or optionally allow the initial candidates to refine. Passing C is compatibility, so initially freezing an alias should not be treated as a verified identification.

There is an exact geometric motivation. If the p selected readings lie in S = row(B_T) and are independent, let Q_g project onto their span. Then

\[
B_{\mathrm{res}} = B_T(I-Q_g),
\qquad \operatorname{rank} B_{\mathrm{res}} = q-p.
\]

This follows because projecting the q-dimensional row space of B_T removes exactly those p directions. The remaining candidate readings, after projection by I-Q_g, must cover the missing q-p directions. For approximate certificate hits this rank statement need not hold exactly.

The method can therefore target missing directions instead of repeatedly rediscovering the same solution. It does not guarantee that those directions correspond to previously missing private images: independent blends or other compatible aliases can still complete the fit. If N > q, q-p counts missing feature directions, not the number of missing training images. The advantage to test is easier search and better coverage, not additional information or guaranteed identifiability.
