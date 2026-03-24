---
description: Debug experiment failures, code errors, or unexpected results
---

## Recent changes
!`git log --oneline -5`
!`git diff --stat HEAD~1`

## Task
$ARGUMENTS

## Process
1. **Reproduce** — understand exactly what fails and how. Read error messages carefully.
2. **Check the obvious** — wrong paths, missing files, device mismatch (CUDA vs MPS vs CPU), PyTorch version issues.
3. **Read the code** — don't guess. Read the actual function that's failing.
4. **Trace the data flow** — for reconstruction issues, check: model loading → weight shapes → KKT loss computation → gradient flow.
5. **Test incrementally** — fix one thing at a time, verify each fix.
6. **Document the fix** — log the root cause and fix in LESSONS_LEARNED.md if it's a non-obvious pitfall.

## Common pitfalls in this project
- PyTorch 1.11 (WEXAC) vs 2.2 (Mac) incompatibilities
- `weights_only=True` crashes on WEXAC (use `torch.load(path, map_location=device)`)
- LoRA rank mismatch between config and saved weights
- Device mismatches (tensors on different devices)
- WEXAC conda env not activated (`conda activate rec`)
