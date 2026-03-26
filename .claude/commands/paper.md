---
description: Write or edit thesis content — LaTeX sections, abstracts, summaries, emails, presentation notes
disable-model-invocation: true
---

## Existing notes
!`ls notes/*.tex`

## Task
$ARGUMENTS

## Process
1. **Identify audience** — supervisor email? Thesis chapter? Conference abstract? Adapt tone and detail.
2. **Check existing content** — read relevant `.tex` in `notes/`, papers in `papers/`, STATUS.md.
3. **Draft** — clear, precise academic prose. Lead with the contribution, not methodology.
4. **Verify claims** — every claim needs either a citation or experimental evidence from `results/`.
5. **Save** — LaTeX → `notes/` (compiled on Overleaf). Emails/notes → inline response.

## Key notation
- $\theta$ = parameters, $W$ = weight matrix, $\Delta W = BA$ = LoRA update
- $\lambda_i$ = Lagrange multipliers (KKT), $\Phi(\theta; x)$ = network output
- $\nabla_W \mathcal{L}$ = gradient w.r.t. weights

## Style rules
- Read STYLE_GUIDE.md before generating any formatted document
- Quantify: "SSIM improved from 0.79 to 0.83" not "results improved"
- Active voice preferred
- Never fabricate citations — verify every reference exists in `papers/`

## Thesis structure
1. LoRA Reconstruction via "Gradient Bridge" (primary)
2. LoRA in the NTK Regime (supporting theory)
3. Generative Priors / SDS (extension)
