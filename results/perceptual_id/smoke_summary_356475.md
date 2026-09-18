# Perceptual identification sweep -- job 356475, 2026-09-18 02:46

Tier 1 = exact landing (relative pixel error < 0.01). Tier 2 = line-up identification: the truth among 1+99 public images of the same class (train split, fixed decoy seed 0); rank of the truth by SSIM / pixel L2 / base-model feature L2 to the recovery. 'ctrl' = SSIM(recovery, nearest public image to the truth). Rows with attacker_output=False are references (chart projections, oracle selections), not attacks.

## Sources seen

- **oracle_ladder**: 4 file(s); producer jobs still running at sweep time: ol_d15_letter_a_e0.03, ol_d15_letter_a_e0.05, ol_d15_letter_a_e0.10, ol_d15_letter_a_e0.20, ol_d15_letter_a_e0.30, ol_d15_letter_a_e0.40, ol_d15_letter_a_e0.60

## Per cell (attacker outputs first)

| source | cell | arm | target | attack? | n | exact landed | SSIM top-1 | SSIM top-5 | L2 top-1 | feat top-1 | feat top-5 | med SSIM truth | med SSIM ctrl | chart err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle_ladder | mlp_letter_a_eps0 | cert_best | raw | yes | 8 | 7 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.748 | 0.0000 |
| oracle_ladder | mlp_letter_a_eps0.03 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.928 | 0.694 | 0.0208 |
| oracle_ladder | mlp_letter_a_eps0.1 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.706 | 0.492 | 0.0677 |
| oracle_ladder | mlp_letter_a_pca | cert_best | raw | yes | 8 | 0 | 7 | 7 | 7 | 7 | 7 | 0.619 | 0.536 | 0.3123 |
| oracle_ladder | mlp_letter_a_eps0 | chart_projection | raw | ref | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.770 | 0.0000 |
| oracle_ladder | mlp_letter_a_eps0.03 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.936 | 0.697 | 0.0208 |
| oracle_ladder | mlp_letter_a_eps0.1 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.728 | 0.501 | 0.0677 |
| oracle_ladder | mlp_letter_a_pca | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | 0.3123 |

## Oracle ladder: identification as the chart error grows (cert_best vs raw truth)

eps is the ladder's perturbation; 'chart err' is the MEASURED mean projection error of the true images. Exact landings die first; the question is where SSIM top-1 dies.

### mlp_letter_a

| cell | chart | eps | chart err (mean) | exact landed | SSIM top-1 | SSIM top-5 | L2 top-1 | feat top-1 | med SSIM truth | med SSIM ctrl | med SSIM truth-vs-ctrl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_letter_a_eps0 | oracle | 0.000 | 0.0000 | 7 | 8 | 8 | 8 | 8 | 1.000 | 0.748 | 0.770 |
| mlp_letter_a_eps0.03 | oracle | 0.030 | 0.0208 | 0 | 8 | 8 | 8 | 8 | 0.928 | 0.694 | 0.770 |
| mlp_letter_a_eps0.1 | oracle | 0.100 | 0.0677 | 0 | 8 | 8 | 8 | 8 | 0.706 | 0.492 | 0.770 |
| mlp_letter_a_pca | pca | - | 0.3123 | 0 | 7 | 7 | 7 | 7 | 0.619 | 0.536 | 0.770 |

- first oracle cell with 0 exact landings: eps 0.030 (chart err 0.0208); SSIM top-1 there 8/8
- 0 exact landings but ALL 8 SSIM-identified top-1 among 100: eps up to 0.100 (chart err 0.0677)
- 0 exact landings but at least one SSIM top-1: eps up to 0.100 (chart err 0.0677, 8/8)


_8 arms scored in 17s; rows in results/perceptual_id/smoke_<source>_356475.jsonl_
