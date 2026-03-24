---
description: Help write, edit, or review LaTeX content for the thesis — sections, equations, arguments
disable-model-invocation: true
---

## Existing notes
!`ls notes/*.tex`

## Task
$ARGUMENTS

## Process
1. **Understand the section** — what argument are we making? What's the reader's context?
2. **Check existing content** — read relevant `.tex` files in `notes/` and papers in `papers/`.
3. **Write or edit** — produce clear, precise academic prose with proper LaTeX formatting.
4. **Equations** — use `align` environments, define notation consistently with existing files.
5. **References** — cite papers from `papers/` directory using proper academic citation format.
6. **Save** — write LaTeX source to `notes/`. Compilation happens on Overleaf, not locally.

## Key notation conventions
- $\theta$ = network parameters, $W$ = weight matrix
- $\Delta W = BA$ = LoRA update (B is d×r, A is r×k)
- $\lambda_i$ = Lagrange multipliers (KKT)
- $\Phi(\theta; x)$ = network output
- $\nabla_W \mathcal{L}$ = gradient w.r.t. weights

## Thesis structure
1. LoRA Reconstruction via the "Gradient Bridge" (primary)
2. LoRA in the NTK Regime (supporting theory)
3. Generative Priors / SDS (extension)
