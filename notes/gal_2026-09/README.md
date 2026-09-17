# `notes/gal_2026-09/` — the September 2026 proof notes and plans, as delivered

Imported 2026-09-17 from `Thesis_files_2026-09-17.zip` (the zip stays at the repo root, untracked). These are the
documents as they were written and shown, **not** re-derived here. Where one of them disagrees with a repo file the
conflict is recorded in `notes/corrections_from_archive_2026-09-17.md` §3 and left open for the approver lane.

| file | date | what it is |
|---|---|---|
| `The_LoRA_certificate.{tex,pdf}` | 2026-09-10 | The Gal-facing certificate note. Assumptions A1–A5, Claims 1–3, the Theorem, and the remarks on NTK coordinates, the free-coefficient factor fit and equation counts. The `.tex` is the editable source. |
| `multilayer_lora_theory{_source.tex,.pdf}` | 2026-09-07 | 56 pp. Exact closure over the training span, the truncated-certificate bound, short-time theorems for trained two-layer MLPs (softplus and ReLU), the quadratic residual family, cross-layer rank, local reconstruction. No experiments. |
| `chart_inversion_theory.{tex,pdf}` | 2026-09-08 | 65 pp on charts. Local vs global identifiability, the strict `k < r − q` theorem, twelve counterexamples, the decoder-backed atlas, and Appendix A's corrections to our own earlier premises. No experiments. (The PDF was named `charts.pdf` in the archive.) |
| `Text_Certificate_Chart_Research_Plan.md` | 2026-09-15 | The text extension: the endpoint certificate in text, finite-codebook separation, the known-initialisation sparse route, the rank-saturation obstruction, and a staged programme. One computation only. |
| `LoRA_meeting_audit_2026-09-14.md` | 2026-09-14 | The pre-meeting audit: corrections to claims that were circulating, the primary-source literature pass, figure roles and what remains unverified. |
| `B_factor_representer_idea.md` | 2026-09-08 | The joint factor fit `B_T = Σ u_i (A_T h_i)^T`, kept separate from the proof note, with its batch-independence requirement and its limits. |
| `gal_ntk_cluster_brief.md` | 2026-09-10 | A four-objective one-step diagnostic, and the sanity counterexample showing the free-coefficient symmetry is not a symmetry of training replay. |

**Reading order** for someone new: the certificate note, then the charts note's §1–§2, then the meeting record
(`notes/meeting_summary_2026-09-15.md`), then `notes/technical_record_2026-09-17.md` for how these line up with
what has been measured here.

**Do not edit these documents.** Disagreements go to `notes/corrections_from_archive_2026-09-17.md` with evidence,
following the same rule the exact-inversion track uses for its theory documents.
