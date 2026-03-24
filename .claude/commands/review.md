---
description: Review code changes for correctness, style, and compatibility before committing
allowed-tools: Read, Grep, Glob, Bash(git *)
---

## Changes to review
!`git diff --stat`
!`git diff`

## Task
$ARGUMENTS

## Process
1. **Read every changed file** — don't skim.
2. **Check correctness** — does the logic do what it claims? Edge cases?
3. **Check compatibility** — PyTorch 1.11 safe? No `weights_only=`, no `torch.compile`, no 2.x features?
4. **Check for regressions** — does this break existing experiments or configs?
5. **Check for leaks** — no hardcoded paths, no secrets, no accidental data files in git.
6. **Check docs** — if behavior changed, are STATUS.md / CLAUDE.md updated?
7. **Summarize** — list issues found (if any) with file:line references, then give a go/no-go recommendation.

## Severity levels
- **BLOCK**: Must fix before commit (bugs, compatibility breaks, security issues)
- **WARN**: Should fix, but won't break anything (style, missing docs)
- **NOTE**: Optional improvements for later
