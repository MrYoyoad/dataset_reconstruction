---
description: Design, configure, and prepare reconstruction experiments for WEXAC GPU cluster
disable-model-invocation: true
---

## Current experiment infrastructure
!`ls experiments/*.py | head -20`
!`ls scripts/*.sh | head -20`

## Task
$ARGUMENTS

## Process
1. **Clarify the experiment** — what hypothesis are we testing? What's the independent variable?
2. **Check prior results** — scan `results/` for related sweeps. Don't repeat what's been done.
3. **Design the config** — define parameter grid (rank, T, lr, etc.), expected runtime, output format.
4. **Write the code** — create or modify experiment scripts under `experiments/`. Follow existing patterns.
5. **Write the WEXAC script** — create a job submission script under `scripts/`. Must use CUDA, never MPS.
6. **Save everything** — experiments must save image tensors (.pth) AND metrics (.csv), not just numbers.
7. **Update docs** — add experiment to STATUS.md "What's Pending" section.

## Rules
- ALL experiments run on WEXAC (NVIDIA L40S, CUDA 12.6), never on Mac/MPS
- Code must be compatible with PyTorch 1.11 (no `weights_only=` in torch.load, no torch.compile)
- Always save visual examples (ground truth + reconstruction grids), not just SSIM numbers
- Include both best AND worst results in any output
- Remind user to rsync before running: `rsync -avz --exclude='__pycache__' experiments/ wexac:~/experiments/`
