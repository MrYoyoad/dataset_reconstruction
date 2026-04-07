# Documentation Guide

How to maintain and update documentation for this thesis project.

---

## Philosophy

- **Document "why" not "what"** — code shows what; docs explain motivation and decisions
- **Keep READMEs scannable** — details go in `docs/`, top-level stays short
- **Include negative results** — this is a thesis; failures are informative
- **Single source of truth** — canonical numbers live in [GET_UP_TO_SPEED.md](GET_UP_TO_SPEED.md) "NUMBERS TO REMEMBER"
- **Per-directory READMEs are self-contained** — readable without navigating up

---

## Documentation Inventory

| File | Purpose | Update when |
|------|---------|-------------|
| [README.md](README.md) | GitHub landing page | New major result or structural change |
| [STATUS.md](STATUS.md) | Sprint-by-sprint progress | After every experiment run |
| [LESSONS_LEARNED.md](LESSONS_LEARNED.md) | Insights and pitfalls log | After debugging or unexpected results |
| [GET_UP_TO_SPEED.md](GET_UP_TO_SPEED.md) | 30-minute onboarding | When major milestones change |
| [STYLE_GUIDE.md](STYLE_GUIDE.md) | Formatting rules | When creating new document types |
| [CLAUDE.md](CLAUDE.md) | AI assistant context | New workflows, tools, or architecture |
| [docs/architecture.md](docs/architecture.md) | Code architecture | When adding new modules |
| [docs/pipeline.md](docs/pipeline.md) | End-to-end code walkthrough | When changing core algorithm |
| [docs/experiment_guide.md](docs/experiment_guide.md) | How to run experiments | When adding new experiment types |
| [experiments/README.md](experiments/README.md) | Experiment file index | When adding/removing Python files |
| [scripts/README.md](scripts/README.md) | WEXAC script index | When adding new job scripts |
| [results/README.md](results/README.md) | Result file guide | When adding new result categories |
| [figures/README.md](figures/README.md) | Figure index | When generating new figures |
| [experiments/tests/README.md](experiments/tests/README.md) | Test suite overview | When adding tests |

---

## Update Checklist (Before Every Git Push)

- [ ] If new experiment file: update [experiments/README.md](experiments/README.md) file index
- [ ] If new WEXAC script: update [scripts/README.md](scripts/README.md) script index
- [ ] If new result files: update [results/README.md](results/README.md) categories
- [ ] If new figure: update [figures/README.md](figures/README.md) index + regeneration command
- [ ] If new canonical number: `grep -rn "OLD_VALUE" *.md docs/*.md` to find stale references
- [ ] If structural change: update README.md directory tree
- [ ] Always: update STATUS.md and LESSONS_LEARNED.md (per CLAUDE.md rules)

---

## How to Add a New Experiment

1. **Write the code:** create `experiments/run_<name>.py` with argparse + `main()`
2. **Add docstring:** module-level docstring explaining purpose, modes, usage (follow existing pattern)
3. **Update experiments/README.md:** add entry to file index table
4. **Create WEXAC script:** `scripts/run_<name>_wexac.sh` with standard preamble
5. **Update scripts/README.md:** add entry to script index
6. **Run the experiment:** submit to WEXAC
7. **After completion:**
   - Save results to `results/`, update [results/README.md](results/README.md)
   - Generate figures, update [figures/README.md](figures/README.md)
   - Update [STATUS.md](STATUS.md) with findings
   - Log insights in [LESSONS_LEARNED.md](LESSONS_LEARNED.md)
8. **If new module/architecture change:** update [docs/architecture.md](docs/architecture.md)

---

## Cross-Reference Rules

- **Canonical numbers** live in [GET_UP_TO_SPEED.md](GET_UP_TO_SPEED.md) "NUMBERS TO REMEMBER" table
- **README.md** links to `docs/` for details — never duplicates full content
- **CLAUDE.md** is for AI assistant context — not linked from README.md
- **Per-directory READMEs** should be self-contained (no assumed knowledge from parent)
- **DOCUMENTATION_GUIDE.md** (this file) is the only place listing ALL documentation files

---

## Staleness Prevention

- **After each sprint:** review GET_UP_TO_SPEED.md status table
- **After number changes:** `grep -rn "SSIM" *.md docs/*.md` to audit consistency
- **Monthly:** scan per-directory READMEs against actual file listings
- **Archive outdated guides:** rename with date suffix (e.g., `_archived_april2026.md`)
