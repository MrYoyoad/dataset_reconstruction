---
description: Log an insight, pitfall, or design decision to LESSONS_LEARNED.md
---

## Current lessons
!`tail -30 LESSONS_LEARNED.md`

## Task
Log the following insight:

$ARGUMENTS

## Process
1. **Categorize** — is this a Theory / Implementation / Experiment / Pitfall / Design Decision?
2. **Write the entry** — append to LESSONS_LEARNED.md with today's date, category, and a clear description.
3. **Include context** — what were you doing when you learned this? Why does it matter?
4. **Note the impact** — how should this change future work?
5. **Cross-reference** — if this affects STATUS.md or CLAUDE.md, update those too.

## Format
```
### [DATE] Category: Short Title
**Finding:** What you learned.
**Context:** What you were doing.
**Impact:** How this changes future work.
```
