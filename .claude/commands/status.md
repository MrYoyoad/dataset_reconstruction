---
description: Update project state — progress tracking, lesson logging, planning next steps
---

## Current status
!`cat STATUS.md`

## Recent activity
!`git log --oneline -15`

## Task
$ARGUMENTS

If no specific task, do a full status review.

## Process

### Status update (default)
1. **Compare** STATUS.md against git history and file state — find anything out of date.
2. **Update "What's Done"** — add completed work with dates and key metrics.
3. **Update "What's Pending"** — reprioritize. Remove stale items.
4. **Flag blockers** — anything stuck or needing external input.
5. **Commit** the updated STATUS.md.

### Logging a lesson ($ARGUMENTS mentions "lesson" or "insight" or "learned")
1. **Categorize** — Theory / Implementation / Experiment / Pitfall / Design Decision.
2. **Append to LESSONS_LEARNED.md** with date, category, finding, context, and impact on future work.
3. **Cross-reference** — update STATUS.md or CLAUDE.md if affected.

### Planning ($ARGUMENTS mentions "plan" or "next" or "priorities")
1. **Read current state** — STATUS.md, LESSONS_LEARNED.md, recent git history.
2. **Identify gaps** — what's done, stuck, missing, overdue?
3. **Propose concrete next steps** — prioritized by thesis impact.

## Rules
- Be factual — only claim done if code/results exist
- Include specific metrics (SSIM scores, experiment counts)
- Priorities: Gradient Bridge > NTK > Generative Priors
