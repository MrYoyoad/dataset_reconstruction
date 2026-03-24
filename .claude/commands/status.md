---
description: Review and update STATUS.md with current project state — what's done, pending, blocked
---

## Current status
!`cat STATUS.md`

## Recent git activity
!`git log --oneline -15`

## Task
$ARGUMENTS

## Process
1. **Compare** STATUS.md against actual git history and file state — find anything that's out of date.
2. **Update "What's Done"** — add completed work, with dates and key metrics.
3. **Update "What's Pending"** — reprioritize based on current state. Remove stale items.
4. **Flag blockers** — anything that's stuck or needs external input.
5. **Commit** the updated STATUS.md.

## Rules
- Be factual — only claim something is done if the code/results exist
- Include specific metrics where available (SSIM scores, experiment counts)
- Prioritize by thesis impact: Gradient Bridge > NTK > Generative Priors
