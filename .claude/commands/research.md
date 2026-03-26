---
description: Deep research on a topic — literature review, gap analysis, idea validation, novelty checking
context: fork
agent: Explore
---

## Local knowledge
!`ls papers/`
!`ls notes/`

## Task
$ARGUMENTS

## Process
1. **Check local first** — search `papers/`, `notes/`, `LESSONS_LEARNED.md`, and `experiments/`.
2. **Search the web** — arxiv, GitHub, Google Scholar. Primary sources over blog posts.
3. **Go deep** — read actual papers and code, not abstracts. Follow citations.
4. **Devil's advocate** — after forming a view, actively search for evidence against it.
5. **Synthesize** — structured write-up:
   - Key findings (bullets)
   - Relevant equations/methods
   - Connection to thesis (LoRA → gradient → reconstruction)
   - Open questions / next steps
   - References with links or file paths
6. **Update docs** — insights → LESSONS_LEARNED.md, status changes → STATUS.md.

## For idea brainstorming ($ARGUMENTS mentions "brainstorm" or "idea")
- Generate 3-5 concrete approaches, each with: hypothesis, required compute, expected signal, risk
- Check novelty on arxiv — has this been done?
- Rank by: thesis impact > compute cost > implementation complexity
- Design a minimal pilot for the top idea

## Rules
- Never fabricate citations — verify every reference exists before including it
- Flag weak or conflicting evidence explicitly
- Connect everything to: can LoRA adapters leak training data?
