# docs/sessions/

Cross-session **handover notes**, managed by the `/handover` skill
(`.claude/skills/handover/SKILL.md`).

- `handover-latest.md` — the most recent note. A `SessionStart` hook in
  `.claude/settings.json` prints this ("📋 Handover note: …") at the top of every new
  session. Created on the first `/handover save`; absent until then (so the hook stays
  quiet on a clean repo).
- `handover-log.md` — append-only history of every saved note, newest appended last.

Workflow:

- `/handover save` — write where work stands before ending a session that leaves
  something in flight.
- `/handover resume` — read the latest note, verify it against the live repo, and
  continue.

This is the short-horizon baton. The durable project record is STATUS.md /
LESSONS_LEARNED.md.
