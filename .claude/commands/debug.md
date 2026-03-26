---
description: Debug experiment failures, code errors, or unexpected results — systematic root cause analysis
---

## Recent changes
!`git log --oneline -5`
!`git diff --stat HEAD~1`

## Task
$ARGUMENTS

## Process
1. **Reproduce** — understand exactly what fails. Read error messages word by word.
2. **Check the obvious** — wrong paths, missing files, device mismatch, PyTorch version.
3. **Read the code** — don't guess. Read the actual function that's failing.
4. **Form hypotheses** — list 2-3 possible causes ranked by likelihood.
5. **Verify the top hypothesis** — add prints, check shapes, inspect tensors. One change at a time.
6. **Apply the fix** — minimal change. Don't refactor surrounding code.

## Solution tiers (from wshobson/smart-debug)
- **Quick fix** — minimal patch to unblock, with risk assessment
- **Proper fix** — correct long-term solution with tests
- **Prevention** — what pattern/check would have caught this earlier?

## Common pitfalls in this project
- PyTorch 1.11 (WEXAC) vs 2.2 (Mac) — `weights_only=True` crashes on WEXAC
- dtype mismatch — extraction needs float64, not float32
- Pre-trained vs random init confusion — always use pre-trained for NTK experiments
- NaN from ReLU at high T — use LeakyReLU for multi-step
- Device mismatches — tensors on different devices after model.to()
- LoRA rank mismatch between config and saved weights

## After fixing
- Log non-obvious bugs in LESSONS_LEARNED.md
