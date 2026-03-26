---
description: Design, run, and analyze reconstruction experiments for WEXAC GPU cluster
disable-model-invocation: false
---

## Current infrastructure
!`ls experiments/*.py | head -20`
!`ls scripts/*.sh | head -20`
!`ls results/*.csv | tail -10`

## Task
$ARGUMENTS

## Process

### Designing experiments
1. **Clarify hypothesis** — what are we testing? What's the independent variable?
2. **Check prior results** — scan `results/` for related sweeps. Don't repeat what's done.
3. **Design config** — parameter grid, expected runtime, output format.
4. **Write code** — under `experiments/`. Follow existing patterns.
5. **Write WEXAC script** — under `scripts/`. Must use CUDA, never MPS.
6. **Update STATUS.md** — add to "What's Pending".

### Analyzing results
1. **Inventory artifacts** — find all CSVs and .pth files for the experiment.
2. **Lock the comparison** — which configs vs which? Primary metric?
3. **Compute stats** — mean +/- std across seeds, best/worst per config.
4. **Generate figures** — comparison grids from .pth tensors → `figures/`.
5. **QA gate** — before finishing, verify:
   - [ ] Sample size / seed count stated
   - [ ] Both best AND worst examples shown
   - [ ] Ground truth + control included in every comparison
   - [ ] No cherry-picked numbers
6. **Update docs** — key numbers → STATUS.md, insights → LESSONS_LEARNED.md.

## Rules
- ALL experiments on WEXAC (NVIDIA L40S, CUDA 12.6), never Mac/MPS
- PyTorch 1.11 compatible (no `weights_only=`, no `torch.compile`)
- Always save image tensors (.pth) AND metrics (.csv), not just numbers
- Never fabricate statistics — if seeds are too few, say so
- Remind to rsync: `rsync -avz --exclude='__pycache__' experiments/ wexac:~/experiments/`
