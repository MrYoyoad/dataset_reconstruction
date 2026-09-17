# Handover — 2026-09-18 02:04

## State
Branch `step1-activation-rescore-retrieval`. Coordinator session of the **18 Sept package** (WP0–WP5, plan + audit in
`notes/plan_2026-09-18_cnn_ranklaw_newclass_charts.md`): CNN rank law, class composition × chart × T, pretrained-decoder
chart, bootstrap chart, MNIST landing gate, all built by sibling sessions after a plan audit. WP0, WP1, WP5-letters,
WP5-digits, WP4-MNIST are finished, written up, committed (STATUS top section; ledger §Q rows Q0–Q9). Still running:
WP2 (58 cells 355926–355983 + an ae fill-in), WP3 (18 decoder-chart cells 356034–356098), WP4-CIFAR (355988),
WP5 `d15_letter_a` (356075–356088), a second-reader audit of Q1–Q4, and the unrelated A100 gap check 355778.

## Done this session
- Base-training gate measured on every checkpoint (job 355833); three fully-trained twins made, none substituted.
- CNN rank law (355907): one conv layer pins any chart at zero drift (VACUOUS for the two-law test); T arm plateau.
- MNIST letters gate (355845–355858): cliff between chart error 0.0069 and 0.0139; no all-8 end; PCA 0.31.
- d15 digits release records nothing (355883–355896): degenerate certificate, wrong-release floors too.
- Decoder chart interim: SD VAE CEILING-BOUND on all three sets; letters local latent chart first to beat pixel PCA.
- Bootstrap chart MNIST (355987): both variants STALL; per-slot local-chart fidelity 0.27–0.31 but slot collapse.
- Composition interim: certificate composition-blind at k=16, flat in T; NTK route non-monotone.
- Flagged: two_walls.py and ladder_cell.py select different eight images under the same seed (ledger Q7).

## Next step(s)
- Read the finished rows as they land and finalize: WP2 §7 in `experiments/cifar/RESULT.md` (per-class vs pooled,
  AE, CNN cells), WP3 `experiments/decoder_chart/RESULT.md`, WP4 CIFAR rows in `experiments/bootstrap_chart/RESULT.md`,
  WP5 third MNIST section (`d15_letter_a`) in `experiments/oracle_ladder/RESULT.md`; promote ledger Q rows from
  interim, add second-read PASSes.
- Recompute the C7 shortfall ratios on the ladder's image set (lane 6e's `experiments/e1b/two_walls.py`).
- Then a plain-language summary for Yoad in the "what did we try / what did it mean" form he asked for.

## Open threads / gotchas
- Builders never commit; the coordinator commits per package. Every builder is told not to touch STATUS/LESSONS/ledger.
- `diffusers` lives in `.conda/extra_pkgs` via PYTHONPATH (rec env untouched); WP3 jobs need `gmem=20G` (OOM on shared A40s).
- Raw-private cells must reference the oracle-start chart optimum, not the pixel projection (ledger Q6).
- d15 fails the base gate and is used unchanged where the depth window was measured; `_full` twins exist for a later re-run.
- Other lanes are active in the same tree (real_encoder_ranklaw.py changed under me; "6e" lane) — check `git status` before editing shared files.

## Pointers
- Plan + audit: `notes/plan_2026-09-18_cnn_ranklaw_newclass_charts.md`; base gate: `experiments/exact_inversion/BASE_TRAINING_GATE.md`
- Runners: `scripts/run_conv_ranklaw_wexac.sh`, `scripts/run_decoder_chart_wexac.sh`, `scripts/run_bootstrap_chart_wexac.sh`,
  `experiments/cifar/submit_ntk_vs_cert.sh <cell> [cnn|mlp_overtrained]`, `experiments/oracle_ladder/submit_ladder.sh <examples>`
- Rows: `results/multilayer_cert/conv_ranklaw_355907.jsonl`, `results/oracle_ladder/rows.jsonl`, `results/bootstrap_chart/`,
  `results/decoder_chart/`, `results/ntk_vs_cert/sweep_*_3559*.jsonl`, `results/base_training_gate.jsonl`
- Ledger: `results/CLAIMS_LEDGER.md` §Q; STATUS top section "18 Sept package"; LESSONS 2026-09-18 entries (four).
