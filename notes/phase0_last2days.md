# Phase 0 — Last 2 Days (2026-04-27 → 2026-04-29)

What was run, what we learned, and where the artifacts are.

## TL;DR

1. **D3 priors (frequency + LPIPS) on Flowers102 don't help.** The TV-only D2 winner (SSIM=0.548) is matched but not beaten by any of 7 prior configs. Best D3 result is freq=1e-3 at SSIM=0.558 — within seed/restart noise. Strong priors actively hurt.
2. **Real face reconstruction works at the D2 winner config.** face1.jpg reaches **SSIM=0.522, PSNR=13.8dB, cos_sim=0.974** — visibly identifiable person. The privacy attack crosses the gate on a real photo, not just Flowers.
3. **Gaps**: only face1 has been re-run with the new winner; face2/face3 still sit at the old SSIM 0.21–0.24. No N>1 anywhere in Phase 0. Single seed=42 only.

## What we ran (chronological)

### 1. D3v2 — Image-prior ablation on Flowers102
- 7 configs, all using the D2 winner backbone (signAdam, tv=1e-1, lr=0.05, 30K iters, n_restarts=2 for speed)
- Variables: `freq_weight ∈ {0, 1e-3, 1e-2, 1e-1}` × `lpips_weight ∈ {0, 1e-3, 1e-2}` (subset of 7 cells)
- Submission: [scripts/run_phase0_d3_v2.sh](../scripts/run_phase0_d3_v2.sh)
- Logs: `wexac_logs/phase0_d3v2_*_*.out` (7 jobs, ~5500s each)
- Tensors: `results/phase0_full_r8_n1_s42_20260428_12071*_d3v2_idx{0..6}_*.pth`

| idx | config | SSIM | PSNR | cos_sim |
|---|---|---|---|---|
| 0 | freq=1e-3 | **0.558** | 11.9 | 0.975 |
| 4 | lpips=1e-2 | 0.548 | 12.7 | 0.974 |
| 1 | freq=1e-2 | 0.495 | 12.6 | 0.951 |
| 6 | freq=1e-2 + lpips=1e-2 | 0.478 | 12.6 | 0.944 |
| 2 | freq=1e-1 | 0.423 | 12.9 | 0.933 |
| 3 | lpips=1e-3 | 0.416 | 12.4 | 0.946 |
| 5 | freq=1e-2 + lpips=1e-3 | 0.411 | 12.4 | 0.949 |

### 2. face_d3winner — Real face with the D3 winner config
- One face (face1.jpg), single seed (42)
- Config: signAdam, tv=1e-1, lr=0.05, freq=1e-3, lpips=0, 30K iters, **n_restarts=8**
- Auto-submitted by [scripts/analyze_d3v2_and_submit_face.sh](../scripts/analyze_d3v2_and_submit_face.sh)
- Log: `wexac_logs/phase0_face_d3winner_freq1e-3_771537.out`
- Tensor: [results/phase0_full_r8_n1_s42_20260428_134922_face_d3winner_freq1e-3.pth](../results/phase0_full_r8_n1_s42_20260428_134922_face_d3winner_freq1e-3.pth)
- Result: **SSIM=0.522, PSNR=13.8dB, cos_sim=0.974**

## What we learned

1. **TV is doing all the prior work.** The 100× jump in TV from D1→D2 (1e-2 → 1e-1) bought 3.8× SSIM. Adding frequency/LPIPS on top of that gives nothing measurable. The intuition: TV at 1e-1 is already aggressive enough that imposing a *second* smoothness-like prior either duplicates work or pushes the reconstruction off-manifold (visible color artifacts in the strong-freq config).
2. **Strong priors actively degrade SSIM.** freq=1e-1 and the combined LPIPS+freq configs sit at SSIM 0.41–0.43 — well below the TV-only baseline. This is the classic over-regularization failure: cos_sim stays high (~0.93–0.95) but pixel structure is wrong.
3. **The D2 winner generalizes from flowers to faces.** Same hyperparameters, no re-tuning, transferred from a Flowers102 image to a real human portrait and crossed the SSIM=0.3 gate. This is the more meaningful privacy result — flowers were the gate, faces are the payload.
4. **Iterations 0 → 30K shows a clean coarse-to-fine trajectory.** From noise at iter 0, recognizable face shape emerges by iter 5K, and detail (skin tone, collar, eye placement) sharpens between iter 10K and 30K (see iteration figure). No collapse / drift in late iterations at this config.
5. **Conclusion: stop spending compute on freq/LPIPS at this stage.** The next levers worth testing are (a) latent-space recon, (b) SDS, (c) LoRA-only mode at the winning config.

## Figures

All in [figures/phase0_report/last2days/](../figures/phase0_report/last2days/):

| File | What it shows |
|---|---|
| [fig_face1_recon.png](../figures/phase0_report/last2days/fig_face1_recon.png) | face1: GT vs reconstruction vs |GT − recon|, with metrics |
| [fig_face1_iters.png](../figures/phase0_report/last2days/fig_face1_iters.png) | face1 reconstruction at iter 0 / 2K / 5K / 10K / 20K / 30K |
| [fig_d3_grid.png](../figures/phase0_report/last2days/fig_d3_grid.png) | Flowers102 GT + 7 D3 prior reconstructions side-by-side |
| [fig_d3_ssim_bar.png](../figures/phase0_report/last2days/fig_d3_ssim_bar.png) | SSIM bar chart vs the D2 baseline + the SSIM=0.3 gate |

PDF versions of each figure live alongside the PNGs. Per-iteration snapshots (every 1K iters × 8 restarts = 248 PNGs) are in [figures/phase0/snapshots/full_r8_n1_20260428_134922_face_d3winner_freq1e-3/](../figures/phase0/snapshots/full_r8_n1_20260428_134922_face_d3winner_freq1e-3/).

## What's NOT here (gaps in the experiment, not in the doc)

These are things you asked about that don't have data yet — flagging so we don't pretend they exist:

- **face2.jpg, face3.jpg with the new winner config**: only face1 was re-run on 2026-04-28. The face2/face3 numbers in the older logs (`wexac_logs/phase0_face{2,3}_584*.out`, SSIM 0.21–0.24) were from the March sweep with weak TV (tv=1e-2) — they're stale and have no saved tensors, so we can't even render them.
- **N > 1 reconstruction**: every Phase 0 .pth on disk is `n1`. We have never run multi-image inversion in Phase 0 — that's still in the Sprint 3 backlog (S3.5 + the superposition section in CLAUDE.md).
- **Multi-seed faces**: the face1 SSIM=0.522 number is *one seed*. No mean±std yet. Same for the D3 priors (n_restarts=2, but seed=42 only).
- **LoRA-only at the winner**: still pending. Every last-2-days run is `--mode full` (86M-param gradient).

## Suggested next runs (cheapest first)

1. **face2.jpg + face3.jpg with the D3 winner config** (2 jobs, ~5h each) — gives "different faces" as a proper figure.
2. **face1 at 5 seeds** (5 jobs, ~5h each) — gives canonical SSIM mean±std on the headline face number.
3. **N=2 on Flowers** at the D2 winner — first ever Phase 0 N>1; will probably show heavy superposition (per CLAUDE.md), but that itself is a paper-worthy negative result.
4. Once those land, the LoRA-only sweep at rank ∈ {8, 16, 32, 64} on the winning config (4 jobs).

## D4 face-structure prior (2026-04-29) — infra in, sweep pending

The TL;DR above said freq+LPIPS-as-priors don't help because they only re-impose smoothness on top of TV. **The thing TV cannot fix is structural correctness** — eyes near mouth, multiple disconnected face fragments. So the next prior to try is *semantic*: a frozen face detector + landmark-layout penalty.

What was added this turn (no WEXAC compute yet, just plumbing):

- [experiments/face_prior.py](../experiments/face_prior.py) — three losses on top of kornia YuNet:
  - **Presence**: maximize top-1 detection confidence; suppress 2nd-best (multi-face fragments).
  - **Landmark layout**: ReLU-hinge penalties on the 5 keypoints — `y_eye < y_nose < y_mouth`, mouth ≥ 0.15 below eyes, eye-spacing in [0.2, 0.5] of face width, nose midline aligned with eye centers. Coordinates normalized to bbox so penalties are scale-invariant.
  - **Bbox symmetry**: L1 between face crop and its horizontal flip.
- [experiments/phase0_vit_inversion.py](../experiments/phase0_vit_inversion.py) — new flags `--cos_weight`, `--face_weight`, `--face_layout_weight`, `--face_sym_weight`, `--face_warmup_iters`, `--face_ramp_iters`, `--face_model`. With `face_weight=0` the path is byte-equivalent to the D3-era pipeline.
- [scripts/run_phase0_face_prior_sweep.sh](../scripts/run_phase0_face_prior_sweep.sh) — 9-arm bsub grid (A control, B face-only, C/D TV+face low/high, E1/E4 face_weight strength sweep, F1/F3/F4 cos_weight sweep). Each arm: 30K iters × 8 restarts × signAdam at lr=0.05 + freq=1e-3 (D3 backbone).
- [experiments/analyze_face_prior_sweep.py](../experiments/analyze_face_prior_sweep.py) — produces, after the sweep finishes: per-arm reconstruction grid with predicted-landmark overlays, face_weight strength curve, cos_weight sweep, per-arm loss panels, winner landmark evolution, and a metrics CSV. Outputs to [figures/phase0/face_prior/](../figures/phase0/face_prior/).
- [experiments/tests/test_face_prior.py](../experiments/tests/test_face_prior.py) — 9 pytest tests, all pass on CPU (~20 s).

Two findings worth documenting now (rest pending sweep results):

1. **kornia FaceDetector backward returns NaN** unless you patch `postprocess` to add `+1e-12` inside `(cls * iou.clamp(0,1)).sqrt()` — `sqrt(0)` backward is `0.5/0=inf`, and `0 * inf = NaN` poisons the input gradient even for filtered-out anchors. See [LESSONS_LEARNED.md](../LESSONS_LEARNED.md#bug-kornia-facedetector-backward-returns-nan-gradient-via-sqrt0-2026-04-29).
2. **The face term needs a warm-up.** Default `--face_warmup_iters=5000`: the detector cannot fire on Gaussian-noise iterates, so adding the prior at iter 0 just stalls. Let TV form coarse structure first, then ramp the face term to full strength over 2000 iters.

Once the sweep runs on WEXAC, the headline result + winner config will be appended here and the analyzer-generated grid added to the figures table above.
