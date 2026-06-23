# Supervisor Meeting Deck — Changelog

Audit trail of every change made to `notes/supervisor_meeting_2026_05_14.pptx` from first build (overnight) to final (this morning).

---

**Initial deck → final: 12 → 22 slides** (after changelog moved out of the PPTX into this file)

## Structure

• 4 section dividers (Part I NTK / Part II ViT / Part III KKT failure / Part IV theory)
• Added R2F unlearning ↔ reconstruction bridge slide (early, slide 3)
• Added NEW slide: NTK reconstruction at T=10 steps (slide 7)
• Added NEW slide: Batches and epochs — NTK still applies in expectation (slide 9)
• Reordered "Multiple images: no collapse" → after face slide (Part II)
• Replaced "TV dominance" → "How well does the NTK assumption hold?"
• Split single theory-question slide → Q-A (noise) + Q-B (distribution overlap)
• Replaced "Running on WEXAC" → "Built + landed" (work-shown narrative)
## Visuals

• Slide 7: rebuilt with 3-face grid (face1+2+3) at honest multi-seed SSIM
• Slide 12 (ViT gate crossed): now uses rose-recon image (fig_d2_top_recon)
• Slide 14 (N>1): added actual N=3 joint-reconstruction visual
• Slide 18 (Q-A): replaced bar chart with 28-config scatter (fig_loss_vs_ssim)
• Slide 8 (NTK stability): refreshed with activation comparison + LoRA-vs-Full panels
## Math + style

• All math typeset as proper LaTeX images (was ugly ASCII)
• KKT equation re-rendered cleaner (2-line)
• "Phase 0" renamed to "ViT gradient inversion" throughout
• PEFT acronym expanded on slide 1
• Removed all oracle-vs-free-c framing
• Prominent theory-question callouts on every content slide
## Speaker notes

• Rich speaker notes per slide: WHAT WE DID / WHY / THEORY CONNECTION
• 5-vs-6 separation idea + diversity_penalty pointer noted on slide 5

---

## Files

- `notes/supervisor_meeting_2026_05_14.pptx` — the deck (22 slides after changelog moved out)
- `notes/supervisor_meeting_2026_05_14.BACKUP.pptx` — pre-fix backup
- `notes/meeting_qa_cheatsheet.md` — speaker reference + Q&A killer lines
- `notes/tier1_paper_notes.md` — SVS 2025 + Gronich-Vardi 2026 digest
- `figures/phase0/deck_assets/` — 13 LaTeX equation PNGs
- `figures/sprint1/ntk_stability_v2.png` — activation comparison + LoRA-vs-Full panels
- `figures/sprint1/ntk_T10_grid.png` — T=10 reconstruction grid
- `figures/phase0/n3_three_faces.png` — N=3 joint reconstruction visual
- `figures/phase0/n3_cross_matrix.png` — cross-SSIM matrix (backup)