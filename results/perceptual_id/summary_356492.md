# Perceptual identification sweep -- job 356492, 2026-09-18 03:51

Tier 1 = exact landing (relative pixel error < 0.01). Tier 2 = line-up identification: the truth among 1+99 public images of the same class (train split, fixed decoy seed 0); rank of the truth by SSIM / pixel L2 / base-model feature L2 to the recovery. 'ctrl' = SSIM(recovery, nearest public image to the truth). Rows with attacker_output=False are references (chart projections, oracle selections), not attacks.

## Sources seen

- **oracle_ladder**: 63 file(s); producer jobs still running at sweep time: ol_d15_letter_a_e0.03, ol_d15_letter_a_e0.05, ol_d15_letter_a_e0.10, ol_d15_letter_a_e0.20, ol_d15_letter_a_e0.30, ol_d15_letter_a_e0.40, ol_d15_letter_a_e0.60
- **ntk_vs_cert**: 40 file(s); producer jobs still running at sweep time: nvc_cifar_bottle_cnn_T100_ae, nvc_cifar_bottle_cnn_T100_pca, nvc_cifar_bottle_cnn_T1_ae, nvc_cifar_bottle_cnn_T1_pca, nvc_cifar_bottle_cnn_T20_ae, nvc_cifar_bottle_cnn_T20_pca, nvc_cifar_bottle_cnn_T400_ae, nvc_cifar_bottle_cnn_T400_pca, nvc_cifar_bottle_cnn_T5_ae, nvc_cifar_bottle_cnn_T5_pca, nvc_cifar_bottle_mlp_overtrained, nvc_cifar_mixed_mb_cnn_T100_ae, nvc_cifar_mixed_mb_cnn_T100_pca, nvc_cifar_mixed_mb_cnn_T100_pca_perclass, nvc_cifar_mixed_mb_cnn_T1_ae, nvc_cifar_mixed_mb_cnn_T1_pca, nvc_cifar_mixed_mb_cnn_T1_pca_perclass, nvc_cifar_mixed_mb_cnn_T20_ae, nvc_cifar_mixed_mb_cnn_T20_pca, nvc_cifar_mixed_mb_cnn_T20_pca_perclass, nvc_cifar_mixed_mb_cnn_T400_ae, nvc_cifar_mixed_mb_cnn_T400_pca, nvc_cifar_mixed_mb_cnn_T400_pca_perclass, nvc_cifar_mixed_mb_cnn_T5_ae, nvc_cifar_mixed_mb_cnn_T5_pca, nvc_cifar_mixed_mb_cnn_T5_pca_perclass, nvc_cifar_mixed_mb_mlp_overtrained, nvc_cifar_mixed_mb_samerow_cnn_T100_ae, nvc_cifar_mixed_mb_samerow_cnn_T100_pca, nvc_cifar_mixed_mb_samerow_cnn_T100_pca_perclass, nvc_cifar_mixed_mb_samerow_cnn_T1_ae, nvc_cifar_mixed_mb_samerow_cnn_T1_pca, nvc_cifar_mixed_mb_samerow_cnn_T1_pca_perclass, nvc_cifar_mixed_mb_samerow_cnn_T20_ae, nvc_cifar_mixed_mb_samerow_cnn_T20_pca, nvc_cifar_mixed_mb_samerow_cnn_T20_pca_perclass, nvc_cifar_mixed_mb_samerow_cnn_T400_ae, nvc_cifar_mixed_mb_samerow_cnn_T400_pca, nvc_cifar_mixed_mb_samerow_cnn_T400_pca_perclass, nvc_cifar_mixed_mb_samerow_cnn_T5_ae, nvc_cifar_mixed_mb_samerow_cnn_T5_pca, nvc_cifar_mixed_mb_samerow_cnn_T5_pca_perclass, nvc_cifar_mixed_mb_samerow_mlp_overtrained, nvc_cifar_motorcycle_cnn_T100_ae, nvc_cifar_motorcycle_cnn_T100_pca, nvc_cifar_motorcycle_cnn_T1_ae, nvc_cifar_motorcycle_cnn_T1_pca, nvc_cifar_motorcycle_cnn_T20_ae, nvc_cifar_motorcycle_cnn_T20_pca, nvc_cifar_motorcycle_cnn_T400_ae, nvc_cifar_motorcycle_cnn_T400_pca, nvc_cifar_motorcycle_cnn_T5_ae, nvc_cifar_motorcycle_cnn_T5_pca, nvc_cifar_motorcycle_mlp_overtrained, nvc_mnist_a, nvc_mnist_mixed, nvc_mnist_mixed_samerow, nvc_mnist_t
- **bootstrap_chart**: 23 file(s); producer jobs still running at sweep time: bsc_cifar
- **decoder_chart**: 17 file(s); producer jobs still running at sweep time: dc_full_cnn_keyboard_truth_nn_K64, dc_full_mlp_motorcycle_proxy_nn_K64, dc_full_mnist_letter_a_truth_nn_K64

## Per cell (attacker outputs first)

| source | cell | arm | target | attack? | n | exact landed | SSIM top-1 | SSIM top-5 | L2 top-1 | feat top-1 | feat top-5 | med SSIM truth | med SSIM ctrl | chart err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bootstrap_chart | mnist_A_round1_oracle_class_355986 | matched_candidate | raw | yes | 8 | 0 | 4 | 4 | 3 | 3 | 4 | 0.421 | 0.419 | 0.2452 |
| bootstrap_chart | mnist_A_round1_oracle_class_355987 | matched_candidate | raw | yes | 8 | 0 | 3 | 3 | 3 | 3 | 3 | 0.423 | 0.448 | 0.2452 |
| bootstrap_chart | mnist_A_round1_recognised_355986 | matched_candidate | raw | yes | 8 | 0 | 4 | 4 | 3 | 3 | 4 | 0.421 | 0.419 | 0.2452 |
| bootstrap_chart | mnist_A_round1_recognised_355987 | matched_candidate | raw | yes | 8 | 0 | 3 | 3 | 3 | 3 | 3 | 0.423 | 0.448 | 0.2452 |
| bootstrap_chart | mnist_A_round1_wrong_class_355986 | matched_candidate | raw | yes | 8 | 0 | 0 | 1 | 1 | 1 | 4 | 0.315 | 0.308 | 0.3423 |
| bootstrap_chart | mnist_A_round1_wrong_class_355987 | matched_candidate | raw | yes | 8 | 0 | 1 | 2 | 1 | 1 | 1 | 0.423 | 0.422 | 0.3064 |
| bootstrap_chart | mnist_B_round1_oracle_anchor_355986 | matched_slot | raw | yes | 8 | 0 | 3 | 4 | 3 | 3 | 3 | 0.553 | 0.532 | 0.2491 |
| bootstrap_chart | mnist_B_round1_oracle_anchor_355987 | matched_slot | raw | yes | 8 | 0 | 3 | 4 | 3 | 3 | 3 | 0.553 | 0.532 | 0.2491 |
| bootstrap_chart | mnist_B_round1_random_anchor_355986 | matched_slot | raw | yes | 8 | 0 | 2 | 2 | 2 | 2 | 3 | 0.497 | 0.519 | 0.4221 |
| bootstrap_chart | mnist_B_round1_random_anchor_355987 | matched_slot | raw | yes | 8 | 0 | 4 | 5 | 3 | 4 | 5 | 0.531 | 0.525 | 0.4545 |
| bootstrap_chart | mnist_B_round1_recovery_355986 | matched_slot | raw | yes | 8 | 0 | 2 | 3 | 2 | 3 | 3 | 0.490 | 0.490 | 0.2753 |
| bootstrap_chart | mnist_B_round1_recovery_355987 | matched_slot | raw | yes | 8 | 0 | 2 | 3 | 2 | 2 | 2 | 0.460 | 0.480 | 0.2811 |
| bootstrap_chart | mnist_B_round2_oracle_anchor_355987 | matched_slot | raw | yes | 8 | 0 | 3 | 4 | 3 | 3 | 3 | 0.553 | 0.532 | 0.2491 |
| bootstrap_chart | mnist_B_round2_random_anchor_355987 | matched_slot | raw | yes | 8 | 0 | 4 | 5 | 4 | 5 | 5 | 0.575 | 0.527 | 0.3393 |
| bootstrap_chart | mnist_B_round2_recovery_355987 | matched_slot | raw | yes | 8 | 0 | 2 | 3 | 2 | 2 | 2 | 0.467 | 0.458 | 0.2988 |
| bootstrap_chart | mnist_B_round3_oracle_anchor_355987 | matched_slot | raw | yes | 8 | 0 | 3 | 4 | 3 | 3 | 3 | 0.553 | 0.532 | 0.2491 |
| bootstrap_chart | mnist_B_round3_random_anchor_355987 | matched_slot | raw | yes | 8 | 0 | 4 | 5 | 4 | 4 | 5 | 0.615 | 0.599 | 0.3288 |
| bootstrap_chart | mnist_B_round3_recovery_355987 | matched_slot | raw | yes | 8 | 0 | 2 | 3 | 2 | 2 | 2 | 0.481 | 0.468 | 0.2907 |
| bootstrap_chart | mnist_B_round4_oracle_anchor_355987 | matched_slot | raw | yes | 8 | 0 | 3 | 4 | 3 | 3 | 3 | 0.553 | 0.532 | 0.2491 |
| bootstrap_chart | mnist_B_round4_random_anchor_355987 | matched_slot | raw | yes | 8 | 0 | 5 | 6 | 5 | 5 | 5 | 0.578 | 0.554 | 0.3139 |
| bootstrap_chart | mnist_B_round4_recovery_355987 | matched_slot | raw | yes | 8 | 0 | 2 | 2 | 2 | 2 | 2 | 0.474 | 0.473 | 0.3152 |
| bootstrap_chart | mnist_round0_round0_generic_355986 | matched_candidate | raw | yes | 8 | 0 | 1 | 3 | 2 | 3 | 4 | 0.441 | 0.422 | 0.2865 |
| bootstrap_chart | mnist_round0_round0_generic_355987 | matched_candidate | raw | yes | 8 | 0 | 4 | 5 | 4 | 4 | 4 | 0.501 | 0.449 | 0.2865 |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | glob_s128_k128 | raw | yes | 8 | 0 | 4 | 5 | 6 | 0 | 1 | 0.394 | 0.356 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | glob_s128_k16 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 0 | 0.303 | 0.403 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | glob_s128_k32 | raw | yes | 8 | 0 | 0 | 1 | 5 | 0 | 0 | 0.306 | 0.398 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | glob_s128_k66 | raw | yes | 8 | 0 | 1 | 2 | 5 | 0 | 1 | 0.344 | 0.379 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | glob_s256_k128 | raw | yes | 8 | 0 | 4 | 5 | 6 | 0 | 0 | 0.439 | 0.353 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | glob_s256_k16 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 0 | 0.291 | 0.413 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | glob_s256_k32 | raw | yes | 8 | 0 | 0 | 2 | 5 | 0 | 0 | 0.328 | 0.381 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | glob_s256_k66 | raw | yes | 8 | 0 | 3 | 4 | 5 | 0 | 0 | 0.398 | 0.362 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | glob_s64_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 3 | 5 | 0.440 | 0.240 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | glob_s64_k16 | raw | yes | 8 | 0 | 0 | 1 | 3 | 0 | 1 | 0.282 | 0.433 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | glob_s64_k32 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 1 | 0.305 | 0.412 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | glob_s64_k66 | raw | yes | 8 | 0 | 2 | 4 | 6 | 1 | 1 | 0.353 | 0.315 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | local_K256_k128_proxy_nn | raw | yes | 8 | 0 | 8 | 8 | 8 | 2 | 6 | 0.509 | 0.217 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | local_K256_k16_proxy_nn | raw | yes | 8 | 0 | 2 | 2 | 5 | 0 | 0 | 0.316 | 0.386 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | local_K256_k32_proxy_nn | raw | yes | 8 | 0 | 4 | 4 | 6 | 0 | 2 | 0.366 | 0.279 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | local_K256_k66_proxy_nn | raw | yes | 8 | 0 | 5 | 6 | 7 | 1 | 3 | 0.413 | 0.259 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 1 | 4 | 0.499 | 0.187 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | pixpca_k16 | raw | yes | 8 | 0 | 1 | 2 | 5 | 0 | 1 | 0.285 | 0.362 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | pixpca_k32 | raw | yes | 8 | 0 | 3 | 5 | 6 | 0 | 1 | 0.349 | 0.287 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | pixpca_k66 | raw | yes | 8 | 0 | 5 | 5 | 6 | 0 | 2 | 0.419 | 0.260 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | glob_s128_k128 | raw | yes | 8 | 0 | 6 | 8 | 7 | 8 | 8 | 0.415 | 0.337 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | glob_s128_k16 | raw | yes | 8 | 0 | 0 | 3 | 4 | 1 | 6 | 0.307 | 0.365 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | glob_s128_k32 | raw | yes | 8 | 0 | 1 | 4 | 6 | 3 | 8 | 0.341 | 0.360 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | glob_s128_k66 | raw | yes | 8 | 0 | 3 | 4 | 7 | 7 | 8 | 0.361 | 0.346 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | glob_s256_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 8 | 0.455 | 0.341 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | glob_s256_k16 | raw | yes | 8 | 0 | 1 | 2 | 3 | 1 | 5 | 0.311 | 0.353 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | glob_s256_k32 | raw | yes | 8 | 0 | 3 | 5 | 6 | 5 | 8 | 0.352 | 0.360 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | glob_s256_k66 | raw | yes | 8 | 0 | 6 | 8 | 7 | 6 | 8 | 0.411 | 0.352 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | glob_s64_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.433 | 0.252 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | glob_s64_k16 | raw | yes | 8 | 0 | 0 | 3 | 3 | 1 | 2 | 0.293 | 0.376 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | glob_s64_k32 | raw | yes | 8 | 0 | 2 | 4 | 4 | 2 | 6 | 0.311 | 0.367 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | glob_s64_k66 | raw | yes | 8 | 0 | 4 | 8 | 8 | 8 | 8 | 0.357 | 0.304 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | local_K256_k128_proxy_nn | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.491 | 0.211 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | local_K256_k16_proxy_nn | raw | yes | 8 | 0 | 2 | 5 | 6 | 7 | 8 | 0.302 | 0.291 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | local_K256_k32_proxy_nn | raw | yes | 8 | 0 | 6 | 7 | 8 | 8 | 8 | 0.372 | 0.303 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | local_K256_k66_proxy_nn | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.420 | 0.254 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.490 | 0.199 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | pixpca_k16 | raw | yes | 8 | 0 | 3 | 5 | 7 | 5 | 6 | 0.302 | 0.299 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | pixpca_k32 | raw | yes | 8 | 0 | 5 | 8 | 8 | 8 | 8 | 0.339 | 0.280 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | pixpca_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.398 | 0.247 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | glob_s112_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.782 | 0.583 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | glob_s112_k16 | raw | yes | 8 | 0 | 6 | 7 | 8 | 3 | 3 | 0.324 | 0.278 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | glob_s112_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 5 | 0.383 | 0.329 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | glob_s112_k66 | raw | yes | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.508 | 0.407 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | glob_s224_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.431 | 0.305 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | glob_s224_k16 | raw | yes | 8 | 0 | 6 | 7 | 7 | 1 | 3 | 0.313 | 0.262 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | glob_s224_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 4 | 0.343 | 0.286 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | glob_s224_k66 | raw | yes | 8 | 0 | 7 | 8 | 8 | 5 | 6 | 0.412 | 0.296 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | glob_s56_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.700 | 0.525 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | glob_s56_k16 | raw | yes | 8 | 0 | 7 | 8 | 7 | 4 | 6 | 0.407 | 0.385 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | glob_s56_k32 | raw | yes | 8 | 0 | 7 | 8 | 7 | 4 | 7 | 0.568 | 0.508 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | glob_s56_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 8 | 0.725 | 0.583 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | local_K256_k128_proxy_nn | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.844 | 0.624 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | local_K256_k16_proxy_nn | raw | yes | 8 | 0 | 8 | 8 | 8 | 6 | 6 | 0.395 | 0.335 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | local_K256_k32_proxy_nn | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 8 | 0.530 | 0.395 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | local_K256_k66_proxy_nn | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.661 | 0.486 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.838 | 0.631 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | pixpca_k16 | raw | yes | 8 | 0 | 7 | 7 | 6 | 6 | 6 | 0.663 | 0.625 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | pixpca_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | pixpca_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.811 | 0.640 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | glob_s128_k128 | raw | yes | 8 | 0 | 4 | 5 | 6 | 0 | 1 | 0.394 | 0.356 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | glob_s128_k16 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 0 | 0.303 | 0.403 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | glob_s128_k32 | raw | yes | 8 | 0 | 0 | 1 | 5 | 0 | 0 | 0.306 | 0.398 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | glob_s128_k66 | raw | yes | 8 | 0 | 1 | 2 | 5 | 0 | 1 | 0.344 | 0.379 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | glob_s256_k128 | raw | yes | 8 | 0 | 4 | 5 | 6 | 0 | 0 | 0.439 | 0.353 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | glob_s256_k16 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 0 | 0.291 | 0.413 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | glob_s256_k32 | raw | yes | 8 | 0 | 0 | 2 | 5 | 0 | 0 | 0.328 | 0.381 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | glob_s256_k66 | raw | yes | 8 | 0 | 3 | 4 | 5 | 0 | 0 | 0.398 | 0.362 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | glob_s64_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 3 | 5 | 0.440 | 0.240 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | glob_s64_k16 | raw | yes | 8 | 0 | 0 | 1 | 3 | 0 | 1 | 0.282 | 0.433 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | glob_s64_k32 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 1 | 0.305 | 0.412 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | glob_s64_k66 | raw | yes | 8 | 0 | 2 | 4 | 6 | 1 | 1 | 0.353 | 0.315 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | local_K64_k128_proxy_nn | raw | yes | 8 | 0 | 7 | 8 | 7 | 2 | 2 | 0.386 | 0.308 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | local_K64_k16_proxy_nn | raw | yes | 8 | 0 | 2 | 2 | 5 | 0 | 1 | 0.338 | 0.403 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | local_K64_k32_proxy_nn | raw | yes | 8 | 0 | 3 | 5 | 5 | 1 | 2 | 0.358 | 0.348 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | local_K64_k66_proxy_nn | raw | yes | 8 | 0 | 7 | 8 | 7 | 2 | 2 | 0.382 | 0.315 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 1 | 4 | 0.499 | 0.187 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | pixpca_k16 | raw | yes | 8 | 0 | 1 | 2 | 5 | 0 | 1 | 0.285 | 0.362 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | pixpca_k32 | raw | yes | 8 | 0 | 3 | 5 | 6 | 0 | 1 | 0.349 | 0.287 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | pixpca_k66 | raw | yes | 8 | 0 | 5 | 5 | 6 | 0 | 2 | 0.419 | 0.260 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | glob_s112_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.782 | 0.583 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | glob_s112_k16 | raw | yes | 8 | 0 | 6 | 7 | 8 | 3 | 3 | 0.324 | 0.278 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | glob_s112_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 5 | 0.383 | 0.329 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | glob_s112_k66 | raw | yes | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.508 | 0.407 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | glob_s224_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.431 | 0.305 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | glob_s224_k16 | raw | yes | 8 | 0 | 6 | 7 | 7 | 1 | 3 | 0.313 | 0.262 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | glob_s224_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 4 | 0.343 | 0.286 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | glob_s224_k66 | raw | yes | 8 | 0 | 7 | 8 | 8 | 5 | 6 | 0.412 | 0.296 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | glob_s56_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.700 | 0.525 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | glob_s56_k16 | raw | yes | 8 | 0 | 7 | 8 | 7 | 4 | 6 | 0.407 | 0.385 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | glob_s56_k32 | raw | yes | 8 | 0 | 7 | 8 | 7 | 4 | 7 | 0.568 | 0.508 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | glob_s56_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 8 | 0.725 | 0.583 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | local_K64_k128_proxy_nn | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.687 | 0.538 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | local_K64_k16_proxy_nn | raw | yes | 8 | 0 | 8 | 8 | 8 | 6 | 7 | 0.406 | 0.360 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | local_K64_k32_proxy_nn | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.519 | 0.404 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | local_K64_k66_proxy_nn | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.723 | 0.570 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.838 | 0.631 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | pixpca_k16 | raw | yes | 8 | 0 | 7 | 7 | 6 | 6 | 6 | 0.663 | 0.625 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | pixpca_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | pixpca_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.811 | 0.640 | - |
| decoder_chart | fidelity_smoke_mlp_motorcycle_355910 | glob_s128_k16 | raw | yes | 8 | 0 | 0 | 3 | 4 | 1 | 6 | 0.307 | 0.365 | - |
| decoder_chart | fidelity_smoke_mlp_motorcycle_355910 | glob_s256_k16 | raw | yes | 8 | 0 | 1 | 2 | 3 | 1 | 5 | 0.311 | 0.353 | - |
| decoder_chart | fidelity_smoke_mlp_motorcycle_355910 | glob_s64_k16 | raw | yes | 8 | 0 | 0 | 3 | 3 | 1 | 2 | 0.293 | 0.376 | - |
| decoder_chart | fidelity_smoke_mlp_motorcycle_355910 | local_K64_k16_proxy_nn | raw | yes | 8 | 0 | 1 | 5 | 4 | 4 | 7 | 0.278 | 0.340 | - |
| decoder_chart | fidelity_smoke_mlp_motorcycle_355910 | pixpca_k16 | raw | yes | 8 | 0 | 3 | 5 | 7 | 5 | 6 | 0.302 | 0.299 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | glob_s128_k128 | raw | yes | 8 | 0 | 4 | 5 | 6 | 0 | 1 | 0.394 | 0.356 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | glob_s128_k16 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 0 | 0.303 | 0.403 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | glob_s128_k32 | raw | yes | 8 | 0 | 0 | 1 | 5 | 0 | 0 | 0.306 | 0.398 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | glob_s128_k66 | raw | yes | 8 | 0 | 1 | 2 | 5 | 0 | 1 | 0.344 | 0.379 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | glob_s256_k128 | raw | yes | 8 | 0 | 4 | 5 | 6 | 0 | 0 | 0.439 | 0.353 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | glob_s256_k16 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 0 | 0.291 | 0.413 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | glob_s256_k32 | raw | yes | 8 | 0 | 0 | 2 | 5 | 0 | 0 | 0.328 | 0.381 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | glob_s256_k66 | raw | yes | 8 | 0 | 3 | 4 | 5 | 0 | 0 | 0.398 | 0.362 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | glob_s64_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 3 | 5 | 0.440 | 0.240 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | glob_s64_k16 | raw | yes | 8 | 0 | 0 | 1 | 3 | 0 | 1 | 0.282 | 0.433 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | glob_s64_k32 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 1 | 0.305 | 0.412 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | glob_s64_k66 | raw | yes | 8 | 0 | 2 | 4 | 6 | 1 | 1 | 0.353 | 0.315 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 1 | 4 | 0.499 | 0.187 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | pixpca_k16 | raw | yes | 8 | 0 | 1 | 2 | 5 | 0 | 1 | 0.285 | 0.362 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | pixpca_k32 | raw | yes | 8 | 0 | 3 | 5 | 6 | 0 | 1 | 0.349 | 0.287 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | pixpca_k66 | raw | yes | 8 | 0 | 5 | 5 | 6 | 0 | 2 | 0.419 | 0.260 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | glob_s128_k128 | raw | yes | 8 | 0 | 6 | 8 | 7 | 8 | 8 | 0.415 | 0.337 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | glob_s128_k16 | raw | yes | 8 | 0 | 0 | 3 | 4 | 1 | 6 | 0.307 | 0.365 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | glob_s128_k32 | raw | yes | 8 | 0 | 1 | 4 | 6 | 3 | 8 | 0.341 | 0.360 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | glob_s128_k66 | raw | yes | 8 | 0 | 3 | 4 | 7 | 7 | 8 | 0.361 | 0.346 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | glob_s256_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 8 | 0.455 | 0.341 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | glob_s256_k16 | raw | yes | 8 | 0 | 1 | 2 | 3 | 1 | 5 | 0.311 | 0.353 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | glob_s256_k32 | raw | yes | 8 | 0 | 3 | 5 | 6 | 5 | 8 | 0.352 | 0.360 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | glob_s256_k66 | raw | yes | 8 | 0 | 6 | 8 | 7 | 6 | 8 | 0.411 | 0.352 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | glob_s64_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.433 | 0.252 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | glob_s64_k16 | raw | yes | 8 | 0 | 0 | 3 | 3 | 1 | 2 | 0.293 | 0.376 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | glob_s64_k32 | raw | yes | 8 | 0 | 2 | 4 | 4 | 2 | 6 | 0.311 | 0.367 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | glob_s64_k66 | raw | yes | 8 | 0 | 4 | 8 | 8 | 8 | 8 | 0.357 | 0.304 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.490 | 0.199 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | pixpca_k16 | raw | yes | 8 | 0 | 3 | 5 | 7 | 5 | 6 | 0.302 | 0.299 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | pixpca_k32 | raw | yes | 8 | 0 | 5 | 8 | 8 | 8 | 8 | 0.339 | 0.280 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | pixpca_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.398 | 0.247 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | glob_s112_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.782 | 0.583 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | glob_s112_k16 | raw | yes | 8 | 0 | 6 | 7 | 8 | 3 | 3 | 0.324 | 0.278 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | glob_s112_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 5 | 0.383 | 0.329 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | glob_s112_k66 | raw | yes | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.508 | 0.407 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | glob_s224_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.431 | 0.305 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | glob_s224_k16 | raw | yes | 8 | 0 | 6 | 7 | 7 | 1 | 3 | 0.313 | 0.262 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | glob_s224_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 4 | 0.343 | 0.286 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | glob_s224_k66 | raw | yes | 8 | 0 | 7 | 8 | 8 | 5 | 6 | 0.412 | 0.296 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | glob_s56_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.700 | 0.525 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | glob_s56_k16 | raw | yes | 8 | 0 | 7 | 8 | 7 | 4 | 6 | 0.407 | 0.385 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | glob_s56_k32 | raw | yes | 8 | 0 | 7 | 8 | 7 | 4 | 7 | 0.568 | 0.508 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | glob_s56_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 8 | 0.725 | 0.583 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.838 | 0.631 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | pixpca_k16 | raw | yes | 8 | 0 | 7 | 7 | 6 | 6 | 6 | 0.663 | 0.625 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | pixpca_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | pixpca_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.811 | 0.640 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | glob_s128_k128 | raw | yes | 8 | 0 | 4 | 5 | 6 | 0 | 1 | 0.394 | 0.356 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | glob_s128_k16 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 0 | 0.303 | 0.403 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | glob_s128_k32 | raw | yes | 8 | 0 | 0 | 1 | 5 | 0 | 0 | 0.306 | 0.398 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | glob_s128_k66 | raw | yes | 8 | 0 | 1 | 2 | 5 | 0 | 1 | 0.344 | 0.379 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | glob_s256_k128 | raw | yes | 8 | 0 | 4 | 5 | 6 | 0 | 0 | 0.439 | 0.353 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | glob_s256_k16 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 0 | 0.291 | 0.413 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | glob_s256_k32 | raw | yes | 8 | 0 | 0 | 2 | 5 | 0 | 0 | 0.328 | 0.381 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | glob_s256_k66 | raw | yes | 8 | 0 | 3 | 4 | 5 | 0 | 0 | 0.398 | 0.362 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | glob_s64_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 3 | 5 | 0.440 | 0.240 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | glob_s64_k16 | raw | yes | 8 | 0 | 0 | 1 | 3 | 0 | 1 | 0.282 | 0.433 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | glob_s64_k32 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 1 | 0.305 | 0.412 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | glob_s64_k66 | raw | yes | 8 | 0 | 2 | 4 | 6 | 1 | 1 | 0.353 | 0.315 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 1 | 4 | 0.499 | 0.187 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | pixpca_k16 | raw | yes | 8 | 0 | 1 | 2 | 5 | 0 | 1 | 0.285 | 0.362 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | pixpca_k32 | raw | yes | 8 | 0 | 3 | 5 | 6 | 0 | 1 | 0.349 | 0.287 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | pixpca_k66 | raw | yes | 8 | 0 | 5 | 5 | 6 | 0 | 2 | 0.419 | 0.260 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | glob_s128_k128 | raw | yes | 8 | 0 | 6 | 8 | 7 | 8 | 8 | 0.415 | 0.337 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | glob_s128_k16 | raw | yes | 8 | 0 | 0 | 3 | 4 | 1 | 6 | 0.307 | 0.365 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | glob_s128_k32 | raw | yes | 8 | 0 | 1 | 4 | 6 | 3 | 8 | 0.341 | 0.360 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | glob_s128_k66 | raw | yes | 8 | 0 | 3 | 4 | 7 | 7 | 8 | 0.361 | 0.346 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | glob_s256_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 8 | 0.455 | 0.341 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | glob_s256_k16 | raw | yes | 8 | 0 | 1 | 2 | 3 | 1 | 5 | 0.311 | 0.353 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | glob_s256_k32 | raw | yes | 8 | 0 | 3 | 5 | 6 | 5 | 8 | 0.352 | 0.360 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | glob_s256_k66 | raw | yes | 8 | 0 | 6 | 8 | 7 | 6 | 8 | 0.411 | 0.352 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | glob_s64_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.433 | 0.252 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | glob_s64_k16 | raw | yes | 8 | 0 | 0 | 3 | 3 | 1 | 2 | 0.293 | 0.376 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | glob_s64_k32 | raw | yes | 8 | 0 | 2 | 4 | 4 | 2 | 6 | 0.311 | 0.367 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | glob_s64_k66 | raw | yes | 8 | 0 | 4 | 8 | 8 | 8 | 8 | 0.357 | 0.304 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.490 | 0.199 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | pixpca_k16 | raw | yes | 8 | 0 | 3 | 5 | 7 | 5 | 6 | 0.302 | 0.299 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | pixpca_k32 | raw | yes | 8 | 0 | 5 | 8 | 8 | 8 | 8 | 0.339 | 0.280 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | pixpca_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.398 | 0.247 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | glob_s112_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.782 | 0.583 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | glob_s112_k16 | raw | yes | 8 | 0 | 6 | 7 | 8 | 3 | 3 | 0.324 | 0.278 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | glob_s112_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 5 | 0.383 | 0.329 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | glob_s112_k66 | raw | yes | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.508 | 0.407 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | glob_s224_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.431 | 0.305 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | glob_s224_k16 | raw | yes | 8 | 0 | 6 | 7 | 7 | 1 | 3 | 0.313 | 0.262 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | glob_s224_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 4 | 0.343 | 0.286 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | glob_s224_k66 | raw | yes | 8 | 0 | 7 | 8 | 8 | 5 | 6 | 0.412 | 0.296 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | glob_s56_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.700 | 0.525 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | glob_s56_k16 | raw | yes | 8 | 0 | 7 | 8 | 7 | 4 | 6 | 0.407 | 0.385 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | glob_s56_k32 | raw | yes | 8 | 0 | 7 | 8 | 7 | 4 | 7 | 0.568 | 0.508 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | glob_s56_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 8 | 0.725 | 0.583 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.838 | 0.631 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | pixpca_k16 | raw | yes | 8 | 0 | 7 | 7 | 6 | 6 | 6 | 0.663 | 0.625 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | pixpca_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | pixpca_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.811 | 0.640 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | glob_s128_k128 | raw | yes | 8 | 0 | 4 | 5 | 6 | 0 | 1 | 0.394 | 0.356 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | glob_s128_k16 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 0 | 0.303 | 0.403 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | glob_s128_k32 | raw | yes | 8 | 0 | 0 | 1 | 5 | 0 | 0 | 0.306 | 0.398 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | glob_s128_k66 | raw | yes | 8 | 0 | 1 | 2 | 5 | 0 | 1 | 0.344 | 0.379 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | glob_s256_k128 | raw | yes | 8 | 0 | 4 | 5 | 6 | 0 | 0 | 0.439 | 0.353 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | glob_s256_k16 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 0 | 0.291 | 0.413 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | glob_s256_k32 | raw | yes | 8 | 0 | 0 | 2 | 5 | 0 | 0 | 0.328 | 0.381 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | glob_s256_k66 | raw | yes | 8 | 0 | 3 | 4 | 5 | 0 | 0 | 0.398 | 0.362 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | glob_s64_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 3 | 5 | 0.440 | 0.240 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | glob_s64_k16 | raw | yes | 8 | 0 | 0 | 1 | 3 | 0 | 1 | 0.282 | 0.433 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | glob_s64_k32 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 1 | 0.305 | 0.412 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | glob_s64_k66 | raw | yes | 8 | 0 | 2 | 4 | 6 | 1 | 1 | 0.353 | 0.315 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 1 | 4 | 0.499 | 0.187 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | pixpca_k16 | raw | yes | 8 | 0 | 1 | 2 | 5 | 0 | 1 | 0.285 | 0.362 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | pixpca_k32 | raw | yes | 8 | 0 | 3 | 5 | 6 | 0 | 1 | 0.349 | 0.287 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | pixpca_k66 | raw | yes | 8 | 0 | 5 | 5 | 6 | 0 | 2 | 0.419 | 0.260 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | glob_s128_k128 | raw | yes | 8 | 0 | 6 | 8 | 7 | 8 | 8 | 0.415 | 0.337 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | glob_s128_k16 | raw | yes | 8 | 0 | 0 | 3 | 4 | 1 | 6 | 0.307 | 0.365 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | glob_s128_k32 | raw | yes | 8 | 0 | 1 | 4 | 6 | 3 | 8 | 0.341 | 0.360 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | glob_s128_k66 | raw | yes | 8 | 0 | 3 | 4 | 7 | 7 | 8 | 0.361 | 0.346 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | glob_s256_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 8 | 0.455 | 0.341 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | glob_s256_k16 | raw | yes | 8 | 0 | 1 | 2 | 3 | 1 | 5 | 0.311 | 0.353 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | glob_s256_k32 | raw | yes | 8 | 0 | 3 | 5 | 6 | 5 | 8 | 0.352 | 0.360 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | glob_s256_k66 | raw | yes | 8 | 0 | 6 | 8 | 7 | 6 | 8 | 0.411 | 0.352 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | glob_s64_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.433 | 0.252 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | glob_s64_k16 | raw | yes | 8 | 0 | 0 | 3 | 3 | 1 | 2 | 0.293 | 0.376 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | glob_s64_k32 | raw | yes | 8 | 0 | 2 | 4 | 4 | 2 | 6 | 0.311 | 0.367 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | glob_s64_k66 | raw | yes | 8 | 0 | 4 | 8 | 8 | 8 | 8 | 0.357 | 0.304 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.490 | 0.199 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | pixpca_k16 | raw | yes | 8 | 0 | 3 | 5 | 7 | 5 | 6 | 0.302 | 0.299 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | pixpca_k32 | raw | yes | 8 | 0 | 5 | 8 | 8 | 8 | 8 | 0.339 | 0.280 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | pixpca_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.398 | 0.247 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | glob_s112_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.782 | 0.583 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | glob_s112_k16 | raw | yes | 8 | 0 | 6 | 7 | 8 | 3 | 3 | 0.324 | 0.278 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | glob_s112_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 5 | 0.383 | 0.329 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | glob_s112_k66 | raw | yes | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.508 | 0.407 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | glob_s224_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.431 | 0.305 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | glob_s224_k16 | raw | yes | 8 | 0 | 6 | 7 | 7 | 1 | 3 | 0.313 | 0.262 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | glob_s224_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 4 | 0.343 | 0.286 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | glob_s224_k66 | raw | yes | 8 | 0 | 7 | 8 | 8 | 5 | 6 | 0.412 | 0.296 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | glob_s56_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.700 | 0.525 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | glob_s56_k16 | raw | yes | 8 | 0 | 7 | 8 | 7 | 4 | 6 | 0.407 | 0.385 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | glob_s56_k32 | raw | yes | 8 | 0 | 7 | 8 | 7 | 4 | 7 | 0.568 | 0.508 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | glob_s56_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 8 | 0.725 | 0.583 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.838 | 0.631 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | pixpca_k16 | raw | yes | 8 | 0 | 7 | 7 | 6 | 6 | 6 | 0.663 | 0.625 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | pixpca_k32 | raw | yes | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | pixpca_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.811 | 0.640 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | glob_s128_k128 | raw | yes | 8 | 0 | 4 | 5 | 6 | 0 | 1 | 0.394 | 0.356 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | glob_s128_k16 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 0 | 0.303 | 0.403 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | glob_s128_k32 | raw | yes | 8 | 0 | 0 | 1 | 5 | 0 | 0 | 0.306 | 0.398 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | glob_s128_k66 | raw | yes | 8 | 0 | 1 | 2 | 5 | 0 | 1 | 0.344 | 0.379 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | glob_s256_k128 | raw | yes | 8 | 0 | 4 | 5 | 6 | 0 | 0 | 0.439 | 0.353 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | glob_s256_k16 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 0 | 0.291 | 0.413 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | glob_s256_k32 | raw | yes | 8 | 0 | 0 | 2 | 5 | 0 | 0 | 0.328 | 0.381 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | glob_s256_k66 | raw | yes | 8 | 0 | 3 | 4 | 5 | 0 | 0 | 0.398 | 0.362 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | glob_s64_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 3 | 5 | 0.440 | 0.240 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | glob_s64_k16 | raw | yes | 8 | 0 | 0 | 1 | 3 | 0 | 1 | 0.282 | 0.433 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | glob_s64_k32 | raw | yes | 8 | 0 | 0 | 1 | 4 | 0 | 1 | 0.305 | 0.412 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | glob_s64_k66 | raw | yes | 8 | 0 | 2 | 4 | 6 | 1 | 1 | 0.353 | 0.315 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 1 | 4 | 0.499 | 0.187 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | pixpca_k16 | raw | yes | 8 | 0 | 1 | 2 | 5 | 0 | 1 | 0.285 | 0.362 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | pixpca_k32 | raw | yes | 8 | 0 | 3 | 5 | 6 | 0 | 1 | 0.349 | 0.287 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | pixpca_k66 | raw | yes | 8 | 0 | 5 | 5 | 6 | 0 | 2 | 0.419 | 0.260 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | glob_s128_k128 | raw | yes | 8 | 0 | 6 | 8 | 7 | 8 | 8 | 0.415 | 0.337 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | glob_s128_k16 | raw | yes | 8 | 0 | 0 | 3 | 4 | 1 | 6 | 0.307 | 0.365 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | glob_s128_k32 | raw | yes | 8 | 0 | 1 | 4 | 6 | 3 | 8 | 0.341 | 0.360 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | glob_s128_k66 | raw | yes | 8 | 0 | 3 | 4 | 7 | 7 | 8 | 0.361 | 0.346 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | glob_s256_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 8 | 0.455 | 0.341 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | glob_s256_k16 | raw | yes | 8 | 0 | 1 | 2 | 3 | 1 | 5 | 0.311 | 0.353 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | glob_s256_k32 | raw | yes | 8 | 0 | 3 | 5 | 6 | 5 | 8 | 0.352 | 0.360 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | glob_s256_k66 | raw | yes | 8 | 0 | 6 | 8 | 7 | 6 | 8 | 0.411 | 0.352 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | glob_s64_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.433 | 0.252 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | glob_s64_k16 | raw | yes | 8 | 0 | 0 | 3 | 3 | 1 | 2 | 0.293 | 0.376 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | glob_s64_k32 | raw | yes | 8 | 0 | 2 | 4 | 4 | 2 | 6 | 0.311 | 0.367 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | glob_s64_k66 | raw | yes | 8 | 0 | 4 | 8 | 8 | 8 | 8 | 0.357 | 0.304 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | pixpca_k128 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.490 | 0.199 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | pixpca_k16 | raw | yes | 8 | 0 | 3 | 5 | 7 | 5 | 6 | 0.302 | 0.299 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | pixpca_k32 | raw | yes | 8 | 0 | 5 | 8 | 8 | 8 | 8 | 0.339 | 0.280 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | pixpca_k66 | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.398 | 0.247 | - |
| ntk_vs_cert | cifar_keyboard_N1_r64__pca_k32_T1 | cert_best | onchart | yes | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1.000 | 0.329 | 0.2970 |
| ntk_vs_cert | cifar_keyboard_N1_r64__pca_k32_T1 | cert_best | raw | yes | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0.342 | 0.355 | 0.2970 |
| ntk_vs_cert | cifar_keyboard_N1_r64__pca_k32_T1 | ntk_best | onchart | yes | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1.000 | 0.329 | 0.2970 |
| ntk_vs_cert | cifar_keyboard_N1_r64__pca_k32_T1 | ntk_best | raw | yes | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0.342 | 0.355 | 0.2970 |
| ntk_vs_cert | cifar_keyboard_N1_r64__pca_k32_T400 | cert_best | onchart | yes | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1.000 | 0.329 | 0.2970 |
| ntk_vs_cert | cifar_keyboard_N1_r64__pca_k32_T400 | cert_best | raw | yes | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0.342 | 0.355 | 0.2970 |
| ntk_vs_cert | cifar_keyboard_N1_r64__pca_k32_T400 | ntk_best | onchart | yes | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1.000 | 0.329 | 0.2970 |
| ntk_vs_cert | cifar_keyboard_N1_r64__pca_k32_T400 | ntk_best | raw | yes | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0.342 | 0.355 | 0.2970 |
| ntk_vs_cert | cifar_mlp_overtrained_bottle_N8_r64_pca_k16_T1 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.435 | 0.2762 |
| ntk_vs_cert | cifar_mlp_overtrained_bottle_N8_r64_pca_k16_T1 | cert_best | raw | yes | 8 | 0 | 3 | 4 | 7 | 6 | 6 | 0.461 | 0.446 | 0.2762 |
| ntk_vs_cert | cifar_mlp_overtrained_bottle_N8_r64_pca_k16_T1 | ntk_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.435 | 0.2762 |
| ntk_vs_cert | cifar_mlp_overtrained_bottle_N8_r64_pca_k16_T1 | ntk_best | raw | yes | 8 | 0 | 3 | 4 | 7 | 6 | 6 | 0.460 | 0.446 | 0.2762 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_N8_r64_pca_k16_T1 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.299 | 0.3433 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_N8_r64_pca_k16_T1 | cert_best | raw | yes | 8 | 0 | 3 | 5 | 7 | 6 | 6 | 0.302 | 0.299 | 0.3433 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_N8_r64_pca_k16_T1 | ntk_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.299 | 0.3433 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_N8_r64_pca_k16_T1 | ntk_best | raw | yes | 8 | 0 | 3 | 5 | 7 | 6 | 6 | 0.302 | 0.299 | 0.3433 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T1 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.408 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T1 | cert_best | raw | yes | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T1 | ntk_best | onchart | yes | 8 | 5 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.406 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T1 | ntk_best | raw | yes | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.380 | 0.345 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T100 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.408 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T100 | cert_best | raw | yes | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T100 | ntk_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.408 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T100 | ntk_best | raw | yes | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T20 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.408 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T20 | cert_best | raw | yes | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T20 | ntk_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.408 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T20 | ntk_best | raw | yes | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T400 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.408 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T400 | cert_best | raw | yes | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T400 | ntk_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.408 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T400 | ntk_best | raw | yes | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T5 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.408 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T5 | cert_best | raw | yes | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T5 | ntk_best | onchart | yes | 8 | 5 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.408 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T5 | ntk_best | raw | yes | 8 | 0 | 3 | 5 | 5 | 7 | 7 | 0.381 | 0.356 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T1 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.384 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T1 | cert_best | raw | yes | 8 | 0 | 6 | 7 | 8 | 8 | 8 | 0.417 | 0.344 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T1 | ntk_best | onchart | yes | 8 | 1 | 7 | 7 | 8 | 8 | 8 | 0.821 | 0.288 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T1 | ntk_best | raw | yes | 8 | 0 | 3 | 5 | 6 | 5 | 7 | 0.302 | 0.284 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T20 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.384 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T20 | cert_best | raw | yes | 8 | 0 | 6 | 7 | 8 | 8 | 8 | 0.417 | 0.344 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T20 | ntk_best | onchart | yes | 8 | 2 | 8 | 8 | 8 | 7 | 8 | 0.985 | 0.339 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T20 | ntk_best | raw | yes | 8 | 0 | 4 | 5 | 7 | 6 | 7 | 0.404 | 0.295 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T5 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.384 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T5 | cert_best | raw | yes | 8 | 0 | 6 | 7 | 8 | 8 | 8 | 0.417 | 0.344 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T5 | ntk_best | onchart | yes | 8 | 2 | 7 | 8 | 8 | 7 | 8 | 0.989 | 0.339 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T5 | ntk_best | raw | yes | 8 | 0 | 4 | 5 | 7 | 6 | 8 | 0.405 | 0.290 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_samerow_N8_r64_pca_k16_T1 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.408 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_samerow_N8_r64_pca_k16_T1 | cert_best | raw | yes | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_samerow_N8_r64_pca_k16_T1 | ntk_best | onchart | yes | 8 | 2 | 8 | 8 | 8 | 8 | 8 | 0.999 | 0.407 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_samerow_N8_r64_pca_k16_T1 | ntk_best | raw | yes | 8 | 0 | 3 | 5 | 5 | 7 | 7 | 0.383 | 0.363 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_samerow_N8_r64_pca_k16_T5 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.408 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_samerow_N8_r64_pca_k16_T5 | cert_best | raw | yes | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_samerow_N8_r64_pca_k16_T5 | ntk_best | onchart | yes | 8 | 4 | 8 | 8 | 8 | 8 | 8 | 0.999 | 0.407 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_samerow_N8_r64_pca_k16_T5 | ntk_best | raw | yes | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.395 | 0.364 | 0.3378 |
| ntk_vs_cert | mnist_letter_a_N1_r64__pca_k32_T1 | cert_best | onchart | yes | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1.000 | 0.665 | 0.2334 |
| ntk_vs_cert | mnist_letter_a_N1_r64__pca_k32_T1 | cert_best | raw | yes | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0.735 | 0.665 | 0.2334 |
| ntk_vs_cert | mnist_letter_a_N1_r64__pca_k32_T1 | ntk_best | onchart | yes | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1.000 | 0.665 | 0.2334 |
| ntk_vs_cert | mnist_letter_a_N1_r64__pca_k32_T1 | ntk_best | raw | yes | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0.735 | 0.665 | 0.2334 |
| ntk_vs_cert | mnist_letter_a_N1_r64__pca_k32_T400 | cert_best | onchart | yes | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1.000 | 0.665 | 0.2334 |
| ntk_vs_cert | mnist_letter_a_N1_r64__pca_k32_T400 | cert_best | raw | yes | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0.735 | 0.665 | 0.2334 |
| ntk_vs_cert | mnist_letter_a_N1_r64__pca_k32_T400 | ntk_best | onchart | yes | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1.000 | 0.665 | 0.2334 |
| ntk_vs_cert | mnist_letter_a_N1_r64__pca_k32_T400 | ntk_best | raw | yes | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0.735 | 0.665 | 0.2334 |
| ntk_vs_cert | mnist_letter_a_N8_r64__pca_k32_T1 | cert_best | onchart | yes | 8 | 4 | 5 | 6 | 5 | 0 | 0 | 0.812 | 0.561 | 0.2346 |
| ntk_vs_cert | mnist_letter_a_N8_r64__pca_k32_T1 | cert_best | raw | yes | 8 | 0 | 4 | 5 | 4 | 0 | 0 | 0.606 | 0.548 | 0.2346 |
| ntk_vs_cert | mnist_letter_a_N8_r64__pca_k32_T1 | ntk_best | onchart | yes | 8 | 0 | 5 | 7 | 6 | 0 | 0 | 0.557 | 0.494 | 0.2346 |
| ntk_vs_cert | mnist_letter_a_N8_r64__pca_k32_T1 | ntk_best | raw | yes | 8 | 0 | 4 | 6 | 3 | 0 | 0 | 0.508 | 0.494 | 0.2346 |
| ntk_vs_cert | mnist_letter_a_N8_r64__pca_k32_T400 | cert_best | onchart | yes | 8 | 4 | 6 | 7 | 5 | 0 | 0 | 0.812 | 0.560 | 0.2346 |
| ntk_vs_cert | mnist_letter_a_N8_r64__pca_k32_T400 | cert_best | raw | yes | 8 | 0 | 4 | 5 | 4 | 0 | 0 | 0.660 | 0.560 | 0.2346 |
| ntk_vs_cert | mnist_letter_a_N8_r64__pca_k32_T400 | ntk_best | onchart | yes | 8 | 0 | 7 | 8 | 7 | 0 | 0 | 0.524 | 0.495 | 0.2346 |
| ntk_vs_cert | mnist_letter_a_N8_r64__pca_k32_T400 | ntk_best | raw | yes | 8 | 0 | 5 | 7 | 6 | 0 | 0 | 0.496 | 0.495 | 0.2346 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_ae_k16_T1 | cert_best | onchart | yes | 8 | 5 | 7 | 7 | 7 | 8 | 8 | 1.000 | 0.752 | 0.2359 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_ae_k16_T1 | cert_best | raw | yes | 8 | 0 | 7 | 7 | 7 | 7 | 8 | 0.796 | 0.752 | 0.2359 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_ae_k16_T1 | ntk_best | onchart | yes | 8 | 0 | 4 | 8 | 4 | 5 | 8 | 0.631 | 0.606 | 0.2359 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_ae_k16_T1 | ntk_best | raw | yes | 8 | 0 | 4 | 8 | 3 | 4 | 5 | 0.611 | 0.600 | 0.2359 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T100 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.635 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T100 | cert_best | raw | yes | 8 | 0 | 7 | 7 | 6 | 6 | 6 | 0.663 | 0.625 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T100 | ntk_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.636 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T100 | ntk_best | raw | yes | 8 | 0 | 7 | 7 | 6 | 6 | 6 | 0.663 | 0.625 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T20 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.635 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T20 | cert_best | raw | yes | 8 | 0 | 7 | 7 | 6 | 6 | 6 | 0.663 | 0.625 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T20 | ntk_best | onchart | yes | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.955 | 0.602 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T20 | ntk_best | raw | yes | 8 | 0 | 7 | 7 | 6 | 5 | 6 | 0.611 | 0.566 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T5 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.635 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T5 | cert_best | raw | yes | 8 | 0 | 7 | 7 | 6 | 6 | 6 | 0.663 | 0.625 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T5 | ntk_best | onchart | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.958 | 0.623 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T5 | ntk_best | raw | yes | 8 | 0 | 7 | 7 | 7 | 5 | 6 | 0.650 | 0.605 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T100 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.627 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T100 | cert_best | raw | yes | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T100 | ntk_best | onchart | yes | 8 | 0 | 7 | 7 | 7 | 5 | 6 | 0.611 | 0.512 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T100 | ntk_best | raw | yes | 8 | 0 | 5 | 6 | 5 | 4 | 6 | 0.529 | 0.512 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T20 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.627 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T20 | cert_best | raw | yes | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T20 | ntk_best | onchart | yes | 8 | 0 | 8 | 8 | 7 | 6 | 7 | 0.678 | 0.543 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T20 | ntk_best | raw | yes | 8 | 0 | 6 | 7 | 6 | 5 | 6 | 0.592 | 0.543 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T5 | cert_best | onchart | yes | 8 | 7 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.627 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T5 | cert_best | raw | yes | 8 | 0 | 6 | 8 | 8 | 7 | 8 | 0.715 | 0.622 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T5 | ntk_best | onchart | yes | 8 | 0 | 7 | 8 | 8 | 7 | 7 | 0.588 | 0.519 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T5 | ntk_best | raw | yes | 8 | 0 | 6 | 7 | 6 | 5 | 6 | 0.526 | 0.519 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T100 | cert_best | onchart | yes | 8 | 2 | 8 | 8 | 7 | 7 | 7 | 0.541 | 0.447 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T100 | cert_best | raw | yes | 8 | 0 | 6 | 7 | 5 | 7 | 7 | 0.472 | 0.447 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T100 | ntk_best | onchart | yes | 8 | 0 | 8 | 8 | 8 | 4 | 8 | 0.513 | 0.427 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T100 | ntk_best | raw | yes | 8 | 0 | 5 | 7 | 7 | 4 | 7 | 0.478 | 0.427 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T20 | cert_best | onchart | yes | 8 | 2 | 6 | 8 | 4 | 5 | 5 | 0.464 | 0.398 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T20 | cert_best | raw | yes | 8 | 0 | 4 | 5 | 4 | 5 | 5 | 0.417 | 0.398 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T20 | ntk_best | onchart | yes | 8 | 0 | 8 | 8 | 8 | 6 | 7 | 0.538 | 0.480 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T20 | ntk_best | raw | yes | 8 | 0 | 5 | 6 | 4 | 5 | 6 | 0.499 | 0.480 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T5 | cert_best | onchart | yes | 8 | 1 | 3 | 8 | 4 | 5 | 5 | 0.465 | 0.460 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T5 | cert_best | raw | yes | 8 | 0 | 3 | 7 | 3 | 4 | 5 | 0.456 | 0.431 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T5 | ntk_best | onchart | yes | 8 | 0 | 7 | 7 | 8 | 5 | 7 | 0.560 | 0.475 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T5 | ntk_best | raw | yes | 8 | 0 | 6 | 7 | 5 | 6 | 6 | 0.525 | 0.476 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_N8_r64_pca_k16_T20 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.571 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_N8_r64_pca_k16_T20 | cert_best | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 7 | 0.553 | 0.544 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_N8_r64_pca_k16_T20 | ntk_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.572 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_N8_r64_pca_k16_T20 | ntk_best | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 7 | 0.554 | 0.544 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_N8_r64_pca_k16_T5 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.571 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_N8_r64_pca_k16_T5 | cert_best | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 7 | 0.553 | 0.544 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_N8_r64_pca_k16_T5 | ntk_best | onchart | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.997 | 0.573 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_N8_r64_pca_k16_T5 | ntk_best | raw | yes | 8 | 0 | 6 | 8 | 8 | 3 | 6 | 0.559 | 0.542 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_samerow_N8_r64_pca_k16_T1 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.571 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_samerow_N8_r64_pca_k16_T1 | cert_best | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 7 | 0.553 | 0.544 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_samerow_N8_r64_pca_k16_T1 | ntk_best | onchart | yes | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.899 | 0.570 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_samerow_N8_r64_pca_k16_T1 | ntk_best | raw | yes | 8 | 0 | 5 | 7 | 7 | 3 | 5 | 0.552 | 0.549 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_samerow_N8_r64_pca_k16_T5 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.571 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_samerow_N8_r64_pca_k16_T5 | cert_best | raw | yes | 8 | 0 | 7 | 8 | 8 | 3 | 7 | 0.553 | 0.544 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_samerow_N8_r64_pca_k16_T5 | ntk_best | onchart | yes | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.887 | 0.571 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_samerow_N8_r64_pca_k16_T5 | ntk_best | raw | yes | 8 | 0 | 6 | 8 | 6 | 3 | 5 | 0.542 | 0.539 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T1 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.607 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T1 | cert_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.592 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T1 | ntk_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.607 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T1 | ntk_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.592 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T100 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.607 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T100 | cert_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.592 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T100 | ntk_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.607 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T100 | ntk_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.592 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T20 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.607 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T20 | cert_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.592 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T20 | ntk_best | onchart | yes | 8 | 2 | 8 | 8 | 8 | 8 | 8 | 0.989 | 0.605 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T20 | ntk_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.608 | 0.585 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T400 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.607 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T400 | cert_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.592 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T400 | ntk_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.607 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T400 | ntk_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.591 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T5 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.607 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T5 | cert_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.592 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T5 | ntk_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.607 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T5 | ntk_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.592 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k32_T1 | cert_best | onchart | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.617 | 0.3668 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k32_T1 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.628 | 0.608 | 0.3668 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k32_T1 | ntk_best | onchart | yes | 8 | 1 | 8 | 8 | 8 | 5 | 5 | 0.802 | 0.482 | 0.3668 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k32_T1 | ntk_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 5 | 5 | 0.477 | 0.482 | 0.3668 |
| oracle_ladder | cnn_keyboard_eps0 | cert_best | raw | yes | 8 | 7 | 8 | 8 | 7 | 8 | 8 | 1.000 | 0.155 | 0.0000 |
| oracle_ladder | cnn_keyboard_eps0.01 | cert_best | raw | yes | 8 | 1 | 8 | 8 | 7 | 8 | 8 | 0.997 | 0.154 | 0.0045 |
| oracle_ladder | cnn_keyboard_eps0.02 | cert_best | raw | yes | 8 | 0 | 7 | 7 | 6 | 7 | 7 | 0.987 | 0.152 | 0.0090 |
| oracle_ladder | cnn_keyboard_eps0.03 | cert_best | raw | yes | 8 | 0 | 7 | 7 | 6 | 7 | 7 | 0.969 | 0.146 | 0.0135 |
| oracle_ladder | cnn_keyboard_eps0.05 | cert_best | raw | yes | 8 | 0 | 7 | 7 | 6 | 7 | 7 | 0.907 | 0.139 | 0.0224 |
| oracle_ladder | cnn_keyboard_eps0.075 | cert_best | raw | yes | 8 | 0 | 7 | 7 | 6 | 7 | 7 | 0.808 | 0.132 | 0.0334 |
| oracle_ladder | cnn_keyboard_eps0.1 | cert_best | raw | yes | 8 | 0 | 6 | 6 | 5 | 5 | 5 | 0.624 | 0.179 | 0.0441 |
| oracle_ladder | cnn_keyboard_eps0.15 | cert_best | raw | yes | 8 | 0 | 6 | 7 | 4 | 3 | 3 | 0.365 | 0.150 | 0.0646 |
| oracle_ladder | cnn_keyboard_eps0.2 | cert_best | raw | yes | 8 | 0 | 4 | 6 | 3 | 3 | 4 | 0.314 | 0.115 | 0.0834 |
| oracle_ladder | cnn_keyboard_eps0.3 | cert_best | raw | yes | 8 | 0 | 3 | 4 | 2 | 2 | 2 | 0.145 | 0.109 | 0.1158 |
| oracle_ladder | cnn_keyboard_eps0.4 | cert_best | raw | yes | 8 | 0 | 3 | 4 | 2 | 2 | 2 | 0.186 | 0.091 | 0.1417 |
| oracle_ladder | cnn_keyboard_eps0.6 | cert_best | raw | yes | 8 | 0 | 0 | 1 | 1 | 1 | 1 | 0.139 | 0.163 | 0.1781 |
| oracle_ladder | cnn_keyboard_eps0_wrongrelease | cert_best | raw | yes | 8 | 0 | 5 | 6 | 4 | 0 | 3 | 0.328 | 0.192 | 0.0000 |
| oracle_ladder | cnn_keyboard_pca | cert_best | raw | yes | 8 | 0 | 0 | 1 | 0 | 1 | 1 | 0.130 | 0.145 | 0.2432 |
| oracle_ladder | d15_digits_eps0 | cert_best | raw | yes | 8 | 0 | 7 | 7 | 6 | 5 | 5 | 0.586 | 0.563 | 0.0000 |
| oracle_ladder | d15_digits_eps0.01 | cert_best | raw | yes | 8 | 0 | 6 | 7 | 6 | 4 | 4 | 0.554 | 0.553 | 0.0083 |
| oracle_ladder | d15_digits_eps0.02 | cert_best | raw | yes | 8 | 0 | 6 | 7 | 5 | 4 | 4 | 0.549 | 0.554 | 0.0166 |
| oracle_ladder | d15_digits_eps0.03 | cert_best | raw | yes | 8 | 0 | 6 | 6 | 5 | 5 | 5 | 0.542 | 0.545 | 0.0248 |
| oracle_ladder | d15_digits_eps0.05 | cert_best | raw | yes | 8 | 0 | 7 | 7 | 6 | 4 | 4 | 0.527 | 0.521 | 0.0413 |
| oracle_ladder | d15_digits_eps0.075 | cert_best | raw | yes | 8 | 0 | 6 | 6 | 5 | 5 | 5 | 0.502 | 0.486 | 0.0616 |
| oracle_ladder | d15_digits_eps0.1 | cert_best | raw | yes | 8 | 0 | 6 | 6 | 5 | 5 | 5 | 0.466 | 0.450 | 0.0815 |
| oracle_ladder | d15_digits_eps0.15 | cert_best | raw | yes | 8 | 0 | 6 | 6 | 6 | 4 | 4 | 0.404 | 0.373 | 0.1198 |
| oracle_ladder | d15_digits_eps0.2 | cert_best | raw | yes | 8 | 0 | 4 | 6 | 5 | 4 | 4 | 0.357 | 0.333 | 0.1554 |
| oracle_ladder | d15_digits_eps0.3 | cert_best | raw | yes | 8 | 0 | 4 | 6 | 5 | 4 | 4 | 0.263 | 0.240 | 0.2173 |
| oracle_ladder | d15_digits_eps0.4 | cert_best | raw | yes | 8 | 0 | 5 | 5 | 4 | 5 | 5 | 0.244 | 0.248 | 0.2666 |
| oracle_ladder | d15_digits_eps0.6 | cert_best | raw | yes | 8 | 0 | 3 | 4 | 5 | 5 | 5 | 0.195 | 0.181 | 0.3346 |
| oracle_ladder | d15_digits_eps0_wrongrelease | cert_best | raw | yes | 8 | 0 | 6 | 7 | 5 | 0 | 0 | 0.566 | 0.542 | 0.0000 |
| oracle_ladder | d15_digits_pca | cert_best | raw | yes | 8 | 0 | 2 | 4 | 4 | 6 | 6 | 0.535 | 0.531 | 0.4165 |
| oracle_ladder | d15_letter_a_eps0 | cert_best | raw | yes | 8 | 0 | 6 | 7 | 5 | 0 | 1 | 0.457 | 0.429 | 0.0000 |
| oracle_ladder | d15_letter_a_eps0.01 | cert_best | raw | yes | 8 | 0 | 6 | 8 | 7 | 4 | 4 | 0.520 | 0.458 | 0.0069 |
| oracle_ladder | d15_letter_a_eps0.02 | cert_best | raw | yes | 8 | 0 | 6 | 8 | 7 | 4 | 4 | 0.510 | 0.449 | 0.0139 |
| oracle_ladder | d15_letter_a_eps0.075 | cert_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 4 | 4 | 0.438 | 0.372 | 0.0513 |
| oracle_ladder | d15_letter_a_eps0.15 | cert_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 4 | 4 | 0.355 | 0.287 | 0.0987 |
| oracle_ladder | d15_letter_a_eps0_wrongrelease | cert_best | raw | yes | 8 | 0 | 7 | 8 | 7 | 1 | 1 | 0.471 | 0.401 | 0.0000 |
| oracle_ladder | d15_letter_a_pca | cert_best | raw | yes | 8 | 0 | 4 | 7 | 4 | 3 | 3 | 0.483 | 0.457 | 0.3123 |
| oracle_ladder | mlp_letter_a_eps0 | cert_best | raw | yes | 8 | 7 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.748 | 0.0000 |
| oracle_ladder | mlp_letter_a_eps0.01 | cert_best | raw | yes | 8 | 7 | 8 | 8 | 8 | 8 | 8 | 0.990 | 0.757 | 0.0069 |
| oracle_ladder | mlp_letter_a_eps0.02 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.963 | 0.724 | 0.0139 |
| oracle_ladder | mlp_letter_a_eps0.03 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.928 | 0.694 | 0.0208 |
| oracle_ladder | mlp_letter_a_eps0.05 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.853 | 0.629 | 0.0345 |
| oracle_ladder | mlp_letter_a_eps0.075 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.771 | 0.552 | 0.0513 |
| oracle_ladder | mlp_letter_a_eps0.1 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.706 | 0.492 | 0.0677 |
| oracle_ladder | mlp_letter_a_eps0.15 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.613 | 0.414 | 0.0987 |
| oracle_ladder | mlp_letter_a_eps0.2 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.551 | 0.369 | 0.1271 |
| oracle_ladder | mlp_letter_a_eps0.3 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.482 | 0.323 | 0.1748 |
| oracle_ladder | mlp_letter_a_eps0.4 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.435 | 0.295 | 0.2114 |
| oracle_ladder | mlp_letter_a_eps0.6 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.366 | 0.257 | 0.2596 |
| oracle_ladder | mlp_letter_a_eps0_wrongrelease | cert_best | raw | yes | 8 | 0 | 4 | 7 | 4 | 1 | 4 | 0.462 | 0.466 | 0.0000 |
| oracle_ladder | mlp_letter_a_pca | cert_best | raw | yes | 8 | 0 | 7 | 7 | 7 | 7 | 7 | 0.619 | 0.536 | 0.3123 |
| oracle_ladder | mlp_motorcycle_eps0 | cert_best | raw | yes | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.198 | 0.0000 |
| oracle_ladder | mlp_motorcycle_eps0.01 | cert_best | raw | yes | 8 | 7 | 8 | 8 | 8 | 8 | 8 | 0.998 | 0.197 | 0.0062 |
| oracle_ladder | mlp_motorcycle_eps0.02 | cert_best | raw | yes | 8 | 3 | 8 | 8 | 8 | 8 | 8 | 0.994 | 0.195 | 0.0124 |
| oracle_ladder | mlp_motorcycle_eps0.03 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.987 | 0.191 | 0.0186 |
| oracle_ladder | mlp_motorcycle_eps0.05 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.965 | 0.182 | 0.0309 |
| oracle_ladder | mlp_motorcycle_eps0.075 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.930 | 0.172 | 0.0460 |
| oracle_ladder | mlp_motorcycle_eps0.1 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.889 | 0.160 | 0.0608 |
| oracle_ladder | mlp_motorcycle_eps0.15 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.810 | 0.139 | 0.0891 |
| oracle_ladder | mlp_motorcycle_eps0.2 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.740 | 0.122 | 0.1151 |
| oracle_ladder | mlp_motorcycle_eps0.3 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.620 | 0.096 | 0.1598 |
| oracle_ladder | mlp_motorcycle_eps0.4 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.527 | 0.079 | 0.1949 |
| oracle_ladder | mlp_motorcycle_eps0.6 | cert_best | raw | yes | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.382 | 0.061 | 0.2428 |
| oracle_ladder | mlp_motorcycle_eps0_wrongrelease | cert_best | raw | yes | 8 | 0 | 2 | 3 | 3 | 1 | 2 | 0.107 | 0.148 | 0.0000 |
| oracle_ladder | mlp_motorcycle_pca | cert_best | raw | yes | 8 | 0 | 2 | 3 | 5 | 6 | 6 | 0.151 | 0.208 | 0.3176 |
| bootstrap_chart | mnist_A_round1_oracle_class_355986 | chart_optimum_oracle_start | raw | ref | 8 | 0 | 7 | 8 | 8 | 7 | 7 | 0.632 | 0.558 | 0.2452 |
| bootstrap_chart | mnist_A_round1_oracle_class_355986 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | 0.2452 |
| bootstrap_chart | mnist_A_round1_oracle_class_355987 | chart_optimum_oracle_start | raw | ref | 8 | 0 | 7 | 8 | 8 | 7 | 7 | 0.632 | 0.558 | 0.2452 |
| bootstrap_chart | mnist_A_round1_oracle_class_355987 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | 0.2452 |
| bootstrap_chart | mnist_A_round1_recognised_355986 | chart_optimum_oracle_start | raw | ref | 8 | 0 | 7 | 8 | 8 | 7 | 7 | 0.632 | 0.558 | 0.2452 |
| bootstrap_chart | mnist_A_round1_recognised_355986 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | 0.2452 |
| bootstrap_chart | mnist_A_round1_recognised_355987 | chart_optimum_oracle_start | raw | ref | 8 | 0 | 7 | 8 | 8 | 7 | 7 | 0.632 | 0.558 | 0.2452 |
| bootstrap_chart | mnist_A_round1_recognised_355987 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | 0.2452 |
| bootstrap_chart | mnist_A_round1_wrong_class_355986 | chart_optimum_oracle_start | raw | ref | 8 | 0 | 4 | 7 | 7 | 7 | 7 | 0.467 | 0.455 | 0.3423 |
| bootstrap_chart | mnist_A_round1_wrong_class_355986 | chart_projection | raw | ref | 8 | 0 | 7 | 7 | 8 | 3 | 4 | 0.603 | 0.555 | 0.3423 |
| bootstrap_chart | mnist_A_round1_wrong_class_355987 | chart_optimum_oracle_start | raw | ref | 8 | 0 | 6 | 7 | 8 | 8 | 8 | 0.558 | 0.537 | 0.3064 |
| bootstrap_chart | mnist_A_round1_wrong_class_355987 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 6 | 8 | 0.642 | 0.599 | 0.3064 |
| bootstrap_chart | mnist_round0_round0_generic_355986 | best_any_start | raw | ref | 8 | 0 | 1 | 3 | 2 | 3 | 4 | 0.441 | 0.422 | 0.2865 |
| bootstrap_chart | mnist_round0_round0_generic_355986 | chart_optimum_oracle_start | raw | ref | 8 | 0 | 7 | 8 | 8 | 8 | 8 | 0.581 | 0.553 | 0.2865 |
| bootstrap_chart | mnist_round0_round0_generic_355986 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 8 | 8 | 0.674 | 0.617 | 0.2865 |
| bootstrap_chart | mnist_round0_round0_generic_355987 | best_any_start | raw | ref | 8 | 0 | 5 | 8 | 8 | 8 | 8 | 0.551 | 0.511 | 0.2865 |
| bootstrap_chart | mnist_round0_round0_generic_355987 | chart_optimum_oracle_start | raw | ref | 8 | 0 | 7 | 8 | 8 | 8 | 8 | 0.581 | 0.553 | 0.2865 |
| bootstrap_chart | mnist_round0_round0_generic_355987 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 8 | 8 | 0.674 | 0.617 | 0.2865 |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | ae_128 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.948 | 0.171 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | ae_256 | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.990 | 0.166 | - |
| decoder_chart | fidelity_proxy_nn_K256_cnn_keyboard_356041 | ae_64 | raw | ref | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.714 | 0.160 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | ae_128 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.914 | 0.212 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | ae_256 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.975 | 0.214 | - |
| decoder_chart | fidelity_proxy_nn_K256_mlp_motorcycle_356102 | ae_64 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.660 | 0.198 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | ae_112 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.920 | 0.711 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | ae_224 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.862 | 0.623 | - |
| decoder_chart | fidelity_proxy_nn_K256_mnist_letter_a_356096 | ae_56 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.739 | 0.557 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | ae_128 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.948 | 0.171 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | ae_256 | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.990 | 0.166 | - |
| decoder_chart | fidelity_proxy_nn_K64_cnn_keyboard_356113 | ae_64 | raw | ref | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.714 | 0.160 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | ae_112 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.920 | 0.711 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | ae_224 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.862 | 0.623 | - |
| decoder_chart | fidelity_proxy_nn_K64_mnist_letter_a_356046 | ae_56 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.739 | 0.557 | - |
| decoder_chart | fidelity_smoke_mlp_motorcycle_355910 | ae_128 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.914 | 0.212 | - |
| decoder_chart | fidelity_smoke_mlp_motorcycle_355910 | ae_256 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.975 | 0.214 | - |
| decoder_chart | fidelity_smoke_mlp_motorcycle_355910 | ae_64 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.660 | 0.198 | - |
| decoder_chart | fidelity_smoke_mlp_motorcycle_355910 | local_K64_k16_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.977 | 0.208 | - |
| decoder_chart | fidelity_smoke_mlp_motorcycle_355910 | local_K64_k16_truth_nn | raw | ref | 8 | 0 | 1 | 4 | 5 | 3 | 5 | 0.290 | 0.324 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | ae_128 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.948 | 0.171 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | ae_256 | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.990 | 0.166 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | ae_64 | raw | ref | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.714 | 0.160 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | local_K256_k128_truth_latent | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.991 | 0.160 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | local_K256_k16_truth_latent | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.990 | 0.165 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | local_K256_k32_truth_latent | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.990 | 0.166 | - |
| decoder_chart | fidelity_truth_latent_K256_cnn_keyboard_356103 | local_K256_k66_truth_latent | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.991 | 0.161 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | ae_128 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.914 | 0.212 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | ae_256 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.975 | 0.214 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | ae_64 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.660 | 0.198 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | local_K256_k128_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.980 | 0.199 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | local_K256_k16_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.977 | 0.205 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | local_K256_k32_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.978 | 0.205 | - |
| decoder_chart | fidelity_truth_latent_K256_mlp_motorcycle_356091 | local_K256_k66_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.978 | 0.203 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | ae_112 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.920 | 0.711 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | ae_224 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.862 | 0.623 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | ae_56 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.739 | 0.557 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | local_K256_k128_truth_latent | raw | ref | 8 | 4 | 8 | 8 | 8 | 8 | 8 | 0.975 | 0.726 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | local_K256_k16_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.881 | 0.631 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | local_K256_k32_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.893 | 0.642 | - |
| decoder_chart | fidelity_truth_latent_K256_mnist_letter_a_356098 | local_K256_k66_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.930 | 0.687 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | ae_128 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.948 | 0.171 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | ae_256 | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.990 | 0.166 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | ae_64 | raw | ref | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.714 | 0.160 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | local_K64_k128_truth_latent | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.991 | 0.164 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | local_K64_k16_truth_latent | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.990 | 0.165 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | local_K64_k32_truth_latent | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.990 | 0.165 | - |
| decoder_chart | fidelity_truth_latent_K64_cnn_keyboard_356095 | local_K64_k66_truth_latent | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.991 | 0.164 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | ae_128 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.914 | 0.212 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | ae_256 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.975 | 0.214 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | ae_64 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.660 | 0.198 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | local_K64_k128_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.978 | 0.203 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | local_K64_k16_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.977 | 0.208 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | local_K64_k32_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.977 | 0.206 | - |
| decoder_chart | fidelity_truth_latent_K64_mlp_motorcycle_356038 | local_K64_k66_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.978 | 0.203 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | ae_112 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.920 | 0.711 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | ae_224 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.862 | 0.623 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | ae_56 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.739 | 0.557 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | local_K64_k128_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.912 | 0.677 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | local_K64_k16_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.881 | 0.628 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | local_K64_k32_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.894 | 0.645 | - |
| decoder_chart | fidelity_truth_latent_K64_mnist_letter_a_356097 | local_K64_k66_truth_latent | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.912 | 0.677 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | ae_128 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.948 | 0.171 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | ae_256 | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.990 | 0.166 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | ae_64 | raw | ref | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.714 | 0.160 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | local_K256_k128_truth_nn | raw | ref | 8 | 0 | 8 | 8 | 8 | 4 | 6 | 0.505 | 0.209 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | local_K256_k16_truth_nn | raw | ref | 8 | 0 | 1 | 2 | 5 | 0 | 0 | 0.321 | 0.386 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | local_K256_k32_truth_nn | raw | ref | 8 | 0 | 4 | 4 | 6 | 0 | 1 | 0.373 | 0.276 | - |
| decoder_chart | fidelity_truth_nn_K256_cnn_keyboard_356116 | local_K256_k66_truth_nn | raw | ref | 8 | 0 | 6 | 7 | 6 | 1 | 3 | 0.422 | 0.262 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | ae_128 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.914 | 0.212 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | ae_256 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.975 | 0.214 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | ae_64 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.660 | 0.198 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | local_K256_k128_truth_nn | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.498 | 0.218 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | local_K256_k16_truth_nn | raw | ref | 8 | 0 | 2 | 4 | 7 | 7 | 8 | 0.294 | 0.308 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | local_K256_k32_truth_nn | raw | ref | 8 | 0 | 6 | 7 | 8 | 8 | 8 | 0.374 | 0.304 | - |
| decoder_chart | fidelity_truth_nn_K256_mlp_motorcycle_356090 | local_K256_k66_truth_nn | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.424 | 0.240 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | ae_112 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.920 | 0.711 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | ae_224 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.862 | 0.623 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | ae_56 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.739 | 0.557 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | local_K256_k128_truth_nn | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.870 | 0.655 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | local_K256_k16_truth_nn | raw | ref | 8 | 0 | 8 | 8 | 8 | 3 | 5 | 0.362 | 0.316 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | local_K256_k32_truth_nn | raw | ref | 8 | 0 | 8 | 8 | 8 | 7 | 8 | 0.504 | 0.380 | - |
| decoder_chart | fidelity_truth_nn_K256_mnist_letter_a_356049 | local_K256_k66_truth_nn | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.672 | 0.462 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | ae_128 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.948 | 0.171 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | ae_256 | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.990 | 0.166 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | ae_64 | raw | ref | 8 | 0 | 8 | 8 | 8 | 7 | 7 | 0.714 | 0.160 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | local_K64_k128_truth_nn | raw | ref | 8 | 0 | 7 | 8 | 7 | 2 | 2 | 0.392 | 0.312 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | local_K64_k16_truth_nn | raw | ref | 8 | 0 | 3 | 3 | 5 | 1 | 2 | 0.343 | 0.392 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | local_K64_k32_truth_nn | raw | ref | 8 | 0 | 4 | 5 | 6 | 1 | 3 | 0.359 | 0.344 | - |
| decoder_chart | fidelity_truth_nn_K64_cnn_keyboard_356115 | local_K64_k66_truth_nn | raw | ref | 8 | 0 | 7 | 8 | 7 | 2 | 2 | 0.392 | 0.312 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | ae_128 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.914 | 0.212 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | ae_256 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.975 | 0.214 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | ae_64 | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.660 | 0.198 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | local_K64_k128_truth_nn | raw | ref | 8 | 0 | 6 | 8 | 8 | 8 | 8 | 0.361 | 0.304 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | local_K64_k16_truth_nn | raw | ref | 8 | 0 | 1 | 4 | 5 | 3 | 5 | 0.290 | 0.324 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | local_K64_k32_truth_nn | raw | ref | 8 | 0 | 4 | 6 | 6 | 8 | 8 | 0.313 | 0.322 | - |
| decoder_chart | fidelity_truth_nn_K64_mlp_motorcycle_356036 | local_K64_k66_truth_nn | raw | ref | 8 | 0 | 6 | 8 | 8 | 8 | 8 | 0.361 | 0.304 | - |
| ntk_vs_cert | cifar_keyboard_N1_r64__pca_k32_T1 | chart_projection | raw | ref | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0.342 | 0.355 | 0.2970 |
| ntk_vs_cert | cifar_keyboard_N1_r64__pca_k32_T400 | chart_projection | raw | ref | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0.342 | 0.355 | 0.2970 |
| ntk_vs_cert | cifar_mlp_overtrained_bottle_N8_r64_pca_k16_T1 | chart_projection | raw | ref | 8 | 0 | 3 | 4 | 7 | 6 | 6 | 0.461 | 0.446 | 0.2762 |
| ntk_vs_cert | cifar_mlp_overtrained_bottle_N8_r64_pca_k16_T1 | control_public_nn | raw | ref | 8 | 0 | 1 | 3 | 0 | 1 | 4 | 0.433 | 0.499 | 0.2762 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_N8_r64_pca_k16_T1 | chart_projection | raw | ref | 8 | 0 | 3 | 5 | 7 | 6 | 6 | 0.302 | 0.299 | 0.3433 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_N8_r64_pca_k16_T1 | control_public_nn | raw | ref | 8 | 0 | 0 | 1 | 1 | 1 | 2 | 0.264 | 0.354 | 0.3433 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T1 | chart_projection | raw | ref | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T1 | control_public_nn | raw | ref | 8 | 0 | 1 | 3 | 3 | 3 | 5 | 0.333 | 0.416 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T100 | chart_projection | raw | ref | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T100 | control_public_nn | raw | ref | 8 | 0 | 1 | 3 | 3 | 3 | 5 | 0.333 | 0.416 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T20 | chart_projection | raw | ref | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T20 | control_public_nn | raw | ref | 8 | 0 | 1 | 3 | 3 | 3 | 5 | 0.333 | 0.416 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T400 | chart_projection | raw | ref | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T400 | control_public_nn | raw | ref | 8 | 0 | 1 | 3 | 3 | 3 | 5 | 0.333 | 0.416 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T5 | chart_projection | raw | ref | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k16_T5 | control_public_nn | raw | ref | 8 | 0 | 1 | 3 | 3 | 3 | 5 | 0.333 | 0.416 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T1 | chart_projection | raw | ref | 8 | 0 | 6 | 7 | 8 | 8 | 8 | 0.417 | 0.344 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T1 | control_public_nn | raw | ref | 8 | 0 | 3 | 3 | 4 | 2 | 4 | 0.321 | 0.458 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T20 | chart_projection | raw | ref | 8 | 0 | 6 | 7 | 8 | 8 | 8 | 0.417 | 0.344 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T20 | control_public_nn | raw | ref | 8 | 0 | 3 | 3 | 4 | 2 | 4 | 0.321 | 0.458 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T5 | chart_projection | raw | ref | 8 | 0 | 6 | 7 | 8 | 8 | 8 | 0.417 | 0.344 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_N8_r64_pca_k32_T5 | control_public_nn | raw | ref | 8 | 0 | 3 | 3 | 4 | 2 | 4 | 0.321 | 0.458 | 0.2907 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_samerow_N8_r64_pca_k16_T1 | chart_projection | raw | ref | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_samerow_N8_r64_pca_k16_T1 | control_public_nn | raw | ref | 8 | 0 | 1 | 3 | 3 | 3 | 5 | 0.333 | 0.416 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_samerow_N8_r64_pca_k16_T5 | chart_projection | raw | ref | 8 | 0 | 3 | 5 | 6 | 7 | 7 | 0.394 | 0.360 | 0.3378 |
| ntk_vs_cert | cifar_mlp_overtrained_motorcycle_and_bottle_samerow_N8_r64_pca_k16_T5 | control_public_nn | raw | ref | 8 | 0 | 1 | 3 | 3 | 3 | 5 | 0.333 | 0.416 | 0.3378 |
| ntk_vs_cert | mnist_letter_a_N1_r64__pca_k32_T1 | chart_projection | raw | ref | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0.735 | 0.665 | 0.2334 |
| ntk_vs_cert | mnist_letter_a_N1_r64__pca_k32_T400 | chart_projection | raw | ref | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0.735 | 0.665 | 0.2334 |
| ntk_vs_cert | mnist_letter_a_N8_r64__pca_k32_T1 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 0 | 0 | 0.733 | 0.625 | 0.2346 |
| ntk_vs_cert | mnist_letter_a_N8_r64__pca_k32_T400 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 0 | 0 | 0.733 | 0.625 | 0.2346 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_ae_k16_T1 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.858 | 0.770 | 0.2359 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_ae_k16_T1 | control_public_nn | raw | ref | 8 | 0 | 7 | 8 | 7 | 6 | 7 | 0.762 | 0.927 | 0.2359 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T100 | chart_projection | raw | ref | 8 | 0 | 7 | 7 | 6 | 6 | 6 | 0.663 | 0.625 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T100 | control_public_nn | raw | ref | 8 | 0 | 6 | 7 | 6 | 3 | 5 | 0.618 | 0.591 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T20 | chart_projection | raw | ref | 8 | 0 | 7 | 7 | 6 | 6 | 6 | 0.663 | 0.625 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T20 | control_public_nn | raw | ref | 8 | 0 | 6 | 7 | 6 | 3 | 5 | 0.618 | 0.591 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T5 | chart_projection | raw | ref | 8 | 0 | 7 | 7 | 6 | 6 | 6 | 0.663 | 0.625 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k16_T5 | control_public_nn | raw | ref | 8 | 0 | 6 | 7 | 6 | 3 | 5 | 0.618 | 0.591 | 0.3203 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T100 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T100 | control_public_nn | raw | ref | 8 | 0 | 6 | 7 | 7 | 6 | 7 | 0.629 | 0.702 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T20 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T20 | control_public_nn | raw | ref | 8 | 0 | 6 | 7 | 7 | 6 | 7 | 0.629 | 0.702 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T5 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k32_T5 | control_public_nn | raw | ref | 8 | 0 | 6 | 7 | 7 | 6 | 7 | 0.629 | 0.702 | 0.2452 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T100 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 8 | 8 | 0.793 | 0.664 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T100 | control_public_nn | raw | ref | 8 | 0 | 6 | 7 | 7 | 7 | 7 | 0.634 | 0.753 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T20 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 8 | 8 | 0.793 | 0.664 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T20 | control_public_nn | raw | ref | 8 | 0 | 6 | 7 | 7 | 7 | 7 | 0.634 | 0.753 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T5 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 8 | 8 | 0.793 | 0.664 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_N8_r64_pca_k48_T5 | control_public_nn | raw | ref | 8 | 0 | 6 | 7 | 7 | 7 | 7 | 0.634 | 0.753 | 0.1957 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_N8_r64_pca_k16_T20 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 3 | 7 | 0.553 | 0.544 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_N8_r64_pca_k16_T20 | control_public_nn | raw | ref | 8 | 0 | 6 | 7 | 7 | 2 | 4 | 0.544 | 0.566 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_N8_r64_pca_k16_T5 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 3 | 7 | 0.553 | 0.544 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_N8_r64_pca_k16_T5 | control_public_nn | raw | ref | 8 | 0 | 6 | 7 | 7 | 2 | 4 | 0.544 | 0.566 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_samerow_N8_r64_pca_k16_T1 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 3 | 7 | 0.553 | 0.544 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_samerow_N8_r64_pca_k16_T1 | control_public_nn | raw | ref | 8 | 0 | 6 | 7 | 7 | 2 | 4 | 0.544 | 0.566 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_samerow_N8_r64_pca_k16_T5 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 3 | 7 | 0.553 | 0.544 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_a_and_letter_t_samerow_N8_r64_pca_k16_T5 | control_public_nn | raw | ref | 8 | 0 | 6 | 7 | 7 | 2 | 4 | 0.544 | 0.566 | 0.4385 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T1 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.592 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T1 | control_public_nn | raw | ref | 8 | 0 | 3 | 8 | 5 | 3 | 6 | 0.615 | 0.603 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T100 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.592 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T100 | control_public_nn | raw | ref | 8 | 0 | 3 | 8 | 5 | 3 | 6 | 0.615 | 0.603 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T20 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.592 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T20 | control_public_nn | raw | ref | 8 | 0 | 3 | 8 | 5 | 3 | 6 | 0.615 | 0.603 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T400 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.592 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T400 | control_public_nn | raw | ref | 8 | 0 | 3 | 8 | 5 | 3 | 6 | 0.615 | 0.603 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T5 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 7 | 5 | 6 | 0.606 | 0.592 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k16_T5 | control_public_nn | raw | ref | 8 | 0 | 3 | 8 | 5 | 3 | 6 | 0.615 | 0.603 | 0.4438 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k32_T1 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.628 | 0.608 | 0.3668 |
| ntk_vs_cert | mnist_mnist_mlp_strong_letter_t_N8_r64_pca_k32_T1 | control_public_nn | raw | ref | 8 | 0 | 5 | 8 | 7 | 6 | 8 | 0.618 | 0.641 | 0.3668 |
| oracle_ladder | cnn_keyboard_eps0 | chart_projection | raw | ref | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.155 | 0.0000 |
| oracle_ladder | cnn_keyboard_eps0.01 | chart_projection | raw | ref | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 0.999 | 0.155 | 0.0045 |
| oracle_ladder | cnn_keyboard_eps0.02 | chart_projection | raw | ref | 8 | 6 | 8 | 8 | 8 | 8 | 8 | 0.997 | 0.155 | 0.0090 |
| oracle_ladder | cnn_keyboard_eps0.03 | chart_projection | raw | ref | 8 | 2 | 8 | 8 | 8 | 8 | 8 | 0.993 | 0.154 | 0.0135 |
| oracle_ladder | cnn_keyboard_eps0.05 | chart_projection | raw | ref | 8 | 1 | 8 | 8 | 8 | 8 | 8 | 0.981 | 0.153 | 0.0224 |
| oracle_ladder | cnn_keyboard_eps0.075 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.961 | 0.150 | 0.0334 |
| oracle_ladder | cnn_keyboard_eps0.1 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.937 | 0.147 | 0.0441 |
| oracle_ladder | cnn_keyboard_eps0.15 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 7 | 8 | 0.884 | 0.141 | 0.0646 |
| oracle_ladder | cnn_keyboard_eps0.2 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 4 | 7 | 0.830 | 0.135 | 0.0834 |
| oracle_ladder | cnn_keyboard_eps0.3 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 3 | 4 | 0.732 | 0.126 | 0.1158 |
| oracle_ladder | cnn_keyboard_eps0.4 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 3 | 3 | 0.651 | 0.121 | 0.1417 |
| oracle_ladder | cnn_keyboard_eps0.6 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 1 | 2 | 0.539 | 0.131 | 0.1781 |
| oracle_ladder | cnn_keyboard_eps0_wrongrelease | chart_projection | raw | ref | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.155 | 0.0000 |
| oracle_ladder | cnn_keyboard_pca | chart_projection | raw | ref | 8 | 0 | 3 | 5 | 6 | 0 | 1 | 0.349 | 0.287 | 0.2432 |
| oracle_ladder | d15_digits_eps0 | chart_projection | raw | ref | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.850 | 0.0000 |
| oracle_ladder | d15_digits_eps0.01 | chart_projection | raw | ref | 8 | 6 | 8 | 8 | 8 | 6 | 8 | 0.990 | 0.841 | 0.0083 |
| oracle_ladder | d15_digits_eps0.02 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 6 | 8 | 0.963 | 0.814 | 0.0166 |
| oracle_ladder | d15_digits_eps0.03 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 5 | 8 | 0.925 | 0.772 | 0.0248 |
| oracle_ladder | d15_digits_eps0.05 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 4 | 8 | 0.836 | 0.679 | 0.0413 |
| oracle_ladder | d15_digits_eps0.075 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 3 | 7 | 0.730 | 0.580 | 0.0616 |
| oracle_ladder | d15_digits_eps0.1 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 3 | 7 | 0.652 | 0.508 | 0.0815 |
| oracle_ladder | d15_digits_eps0.15 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 0 | 5 | 0.550 | 0.415 | 0.1198 |
| oracle_ladder | d15_digits_eps0.2 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 0 | 5 | 0.488 | 0.364 | 0.1554 |
| oracle_ladder | d15_digits_eps0.3 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 0 | 3 | 0.427 | 0.319 | 0.2173 |
| oracle_ladder | d15_digits_eps0.4 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 1 | 3 | 0.395 | 0.297 | 0.2666 |
| oracle_ladder | d15_digits_eps0.6 | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 0 | 3 | 0.366 | 0.290 | 0.3346 |
| oracle_ladder | d15_digits_eps0_wrongrelease | chart_projection | raw | ref | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.850 | 0.0000 |
| oracle_ladder | d15_digits_pca | chart_projection | raw | ref | 8 | 0 | 7 | 7 | 8 | 0 | 2 | 0.682 | 0.652 | 0.4165 |
| oracle_ladder | d15_letter_a_eps0 | chart_projection | raw | ref | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.770 | 0.0000 |
| oracle_ladder | d15_letter_a_eps0.01 | chart_projection | raw | ref | 8 | 7 | 8 | 8 | 8 | 6 | 8 | 0.991 | 0.760 | 0.0069 |
| oracle_ladder | d15_letter_a_eps0.02 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 6 | 8 | 0.968 | 0.731 | 0.0139 |
| oracle_ladder | d15_letter_a_eps0.075 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 2 | 7 | 0.790 | 0.557 | 0.0513 |
| oracle_ladder | d15_letter_a_eps0.15 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 1 | 6 | 0.639 | 0.431 | 0.0987 |
| oracle_ladder | d15_letter_a_eps0_wrongrelease | chart_projection | raw | ref | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.770 | 0.0000 |
| oracle_ladder | d15_letter_a_pca | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 1 | 4 | 0.733 | 0.625 | 0.3123 |
| oracle_ladder | mlp_letter_a_eps0 | chart_projection | raw | ref | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.770 | 0.0000 |
| oracle_ladder | mlp_letter_a_eps0.01 | chart_projection | raw | ref | 8 | 7 | 8 | 8 | 8 | 8 | 8 | 0.991 | 0.760 | 0.0069 |
| oracle_ladder | mlp_letter_a_eps0.02 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.968 | 0.731 | 0.0139 |
| oracle_ladder | mlp_letter_a_eps0.03 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.936 | 0.697 | 0.0208 |
| oracle_ladder | mlp_letter_a_eps0.05 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.867 | 0.632 | 0.0345 |
| oracle_ladder | mlp_letter_a_eps0.075 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.790 | 0.557 | 0.0513 |
| oracle_ladder | mlp_letter_a_eps0.1 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.728 | 0.501 | 0.0677 |
| oracle_ladder | mlp_letter_a_eps0.15 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.639 | 0.431 | 0.0987 |
| oracle_ladder | mlp_letter_a_eps0.2 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.582 | 0.396 | 0.1271 |
| oracle_ladder | mlp_letter_a_eps0.3 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.527 | 0.366 | 0.1748 |
| oracle_ladder | mlp_letter_a_eps0.4 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.499 | 0.356 | 0.2114 |
| oracle_ladder | mlp_letter_a_eps0.6 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.477 | 0.358 | 0.2596 |
| oracle_ladder | mlp_letter_a_eps0_wrongrelease | chart_projection | raw | ref | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.770 | 0.0000 |
| oracle_ladder | mlp_letter_a_pca | chart_projection | raw | ref | 8 | 0 | 7 | 8 | 8 | 7 | 8 | 0.733 | 0.625 | 0.3123 |
| oracle_ladder | mlp_motorcycle_eps0 | chart_projection | raw | ref | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.198 | 0.0000 |
| oracle_ladder | mlp_motorcycle_eps0.01 | chart_projection | raw | ref | 8 | 7 | 8 | 8 | 8 | 8 | 8 | 0.999 | 0.197 | 0.0062 |
| oracle_ladder | mlp_motorcycle_eps0.02 | chart_projection | raw | ref | 8 | 3 | 8 | 8 | 8 | 8 | 8 | 0.994 | 0.195 | 0.0124 |
| oracle_ladder | mlp_motorcycle_eps0.03 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.987 | 0.192 | 0.0186 |
| oracle_ladder | mlp_motorcycle_eps0.05 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.966 | 0.183 | 0.0309 |
| oracle_ladder | mlp_motorcycle_eps0.075 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.933 | 0.175 | 0.0460 |
| oracle_ladder | mlp_motorcycle_eps0.1 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.897 | 0.167 | 0.0608 |
| oracle_ladder | mlp_motorcycle_eps0.15 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.827 | 0.153 | 0.0891 |
| oracle_ladder | mlp_motorcycle_eps0.2 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.759 | 0.143 | 0.1151 |
| oracle_ladder | mlp_motorcycle_eps0.3 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.650 | 0.134 | 0.1598 |
| oracle_ladder | mlp_motorcycle_eps0.4 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.571 | 0.133 | 0.1949 |
| oracle_ladder | mlp_motorcycle_eps0.6 | chart_projection | raw | ref | 8 | 0 | 8 | 8 | 8 | 8 | 8 | 0.466 | 0.140 | 0.2428 |
| oracle_ladder | mlp_motorcycle_eps0_wrongrelease | chart_projection | raw | ref | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.198 | 0.0000 |
| oracle_ladder | mlp_motorcycle_pca | chart_projection | raw | ref | 8 | 0 | 5 | 8 | 8 | 8 | 8 | 0.339 | 0.280 | 0.3176 |

## Oracle ladder: identification as the chart error grows (cert_best vs raw truth)

eps is the ladder's perturbation; 'chart err' is the MEASURED mean projection error of the true images. Exact landings die first; the question is where SSIM top-1 dies.

### cnn_keyboard

| cell | chart | eps | chart err (mean) | exact landed | SSIM top-1 | SSIM top-5 | L2 top-1 | feat top-1 | med SSIM truth | med SSIM ctrl | med SSIM truth-vs-ctrl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn_keyboard_eps0 | oracle | 0.000 | 0.0000 | 7 | 8 | 8 | 7 | 8 | 1.000 | 0.155 | 0.155 |
| cnn_keyboard_eps0.01 | oracle | 0.010 | 0.0045 | 1 | 8 | 8 | 7 | 8 | 0.997 | 0.154 | 0.155 |
| cnn_keyboard_eps0.02 | oracle | 0.020 | 0.0090 | 0 | 7 | 7 | 6 | 7 | 0.987 | 0.152 | 0.155 |
| cnn_keyboard_eps0.03 | oracle | 0.030 | 0.0135 | 0 | 7 | 7 | 6 | 7 | 0.969 | 0.146 | 0.155 |
| cnn_keyboard_eps0.05 | oracle | 0.050 | 0.0224 | 0 | 7 | 7 | 6 | 7 | 0.907 | 0.139 | 0.155 |
| cnn_keyboard_eps0.075 | oracle | 0.075 | 0.0334 | 0 | 7 | 7 | 6 | 7 | 0.808 | 0.132 | 0.155 |
| cnn_keyboard_eps0.1 | oracle | 0.100 | 0.0441 | 0 | 6 | 6 | 5 | 5 | 0.624 | 0.179 | 0.155 |
| cnn_keyboard_eps0.15 | oracle | 0.150 | 0.0646 | 0 | 6 | 7 | 4 | 3 | 0.365 | 0.150 | 0.155 |
| cnn_keyboard_eps0.2 | oracle | 0.200 | 0.0834 | 0 | 4 | 6 | 3 | 3 | 0.314 | 0.115 | 0.155 |
| cnn_keyboard_eps0.3 | oracle | 0.300 | 0.1158 | 0 | 3 | 4 | 2 | 2 | 0.145 | 0.109 | 0.155 |
| cnn_keyboard_eps0.4 | oracle | 0.400 | 0.1417 | 0 | 3 | 4 | 2 | 2 | 0.186 | 0.091 | 0.155 |
| cnn_keyboard_eps0.6 | oracle | 0.600 | 0.1781 | 0 | 0 | 1 | 1 | 1 | 0.139 | 0.163 | 0.155 |
| cnn_keyboard_pca | pca | - | 0.2432 | 0 | 0 | 1 | 0 | 1 | 0.130 | 0.145 | 0.155 |
| cnn_keyboard_eps0_wrongrelease | oracle WRONG-RELEASE | 0.000 | 0.0000 | 0 | 5 | 6 | 4 | 0 | 0.328 | 0.192 | 0.155 |

- first oracle cell with 0 exact landings: eps 0.020 (chart err 0.0090); SSIM top-1 there 7/8
- 0 exact landings but at least one SSIM top-1: eps up to 0.400 (chart err 0.1417, 3/8)

### d15_digits

| cell | chart | eps | chart err (mean) | exact landed | SSIM top-1 | SSIM top-5 | L2 top-1 | feat top-1 | med SSIM truth | med SSIM ctrl | med SSIM truth-vs-ctrl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d15_digits_eps0 | oracle | 0.000 | 0.0000 | 0 | 7 | 7 | 6 | 5 | 0.586 | 0.563 | 0.850 |
| d15_digits_eps0.01 | oracle | 0.010 | 0.0083 | 0 | 6 | 7 | 6 | 4 | 0.554 | 0.553 | 0.850 |
| d15_digits_eps0.02 | oracle | 0.020 | 0.0166 | 0 | 6 | 7 | 5 | 4 | 0.549 | 0.554 | 0.850 |
| d15_digits_eps0.03 | oracle | 0.030 | 0.0248 | 0 | 6 | 6 | 5 | 5 | 0.542 | 0.545 | 0.850 |
| d15_digits_eps0.05 | oracle | 0.050 | 0.0413 | 0 | 7 | 7 | 6 | 4 | 0.527 | 0.521 | 0.850 |
| d15_digits_eps0.075 | oracle | 0.075 | 0.0616 | 0 | 6 | 6 | 5 | 5 | 0.502 | 0.486 | 0.850 |
| d15_digits_eps0.1 | oracle | 0.100 | 0.0815 | 0 | 6 | 6 | 5 | 5 | 0.466 | 0.450 | 0.850 |
| d15_digits_eps0.15 | oracle | 0.150 | 0.1198 | 0 | 6 | 6 | 6 | 4 | 0.404 | 0.373 | 0.850 |
| d15_digits_eps0.2 | oracle | 0.200 | 0.1554 | 0 | 4 | 6 | 5 | 4 | 0.357 | 0.333 | 0.850 |
| d15_digits_eps0.3 | oracle | 0.300 | 0.2173 | 0 | 4 | 6 | 5 | 4 | 0.263 | 0.240 | 0.850 |
| d15_digits_eps0.4 | oracle | 0.400 | 0.2666 | 0 | 5 | 5 | 4 | 5 | 0.244 | 0.248 | 0.850 |
| d15_digits_eps0.6 | oracle | 0.600 | 0.3346 | 0 | 3 | 4 | 5 | 5 | 0.195 | 0.181 | 0.850 |
| d15_digits_pca | pca | - | 0.4165 | 0 | 2 | 4 | 4 | 6 | 0.535 | 0.531 | 0.850 |
| d15_digits_eps0_wrongrelease | oracle WRONG-RELEASE | 0.000 | 0.0000 | 0 | 6 | 7 | 5 | 0 | 0.566 | 0.542 | 0.850 |

- first oracle cell with 0 exact landings: eps 0.000 (chart err 0.0000); SSIM top-1 there 7/8
- 0 exact landings but at least one SSIM top-1: eps up to 0.600 (chart err 0.3346, 3/8)

### d15_letter_a

| cell | chart | eps | chart err (mean) | exact landed | SSIM top-1 | SSIM top-5 | L2 top-1 | feat top-1 | med SSIM truth | med SSIM ctrl | med SSIM truth-vs-ctrl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d15_letter_a_eps0 | oracle | 0.000 | 0.0000 | 0 | 6 | 7 | 5 | 0 | 0.457 | 0.429 | 0.770 |
| d15_letter_a_eps0.01 | oracle | 0.010 | 0.0069 | 0 | 6 | 8 | 7 | 4 | 0.520 | 0.458 | 0.770 |
| d15_letter_a_eps0.02 | oracle | 0.020 | 0.0139 | 0 | 6 | 8 | 7 | 4 | 0.510 | 0.449 | 0.770 |
| d15_letter_a_eps0.075 | oracle | 0.075 | 0.0513 | 0 | 7 | 8 | 7 | 4 | 0.438 | 0.372 | 0.770 |
| d15_letter_a_eps0.15 | oracle | 0.150 | 0.0987 | 0 | 7 | 8 | 7 | 4 | 0.355 | 0.287 | 0.770 |
| d15_letter_a_pca | pca | - | 0.3123 | 0 | 4 | 7 | 4 | 3 | 0.483 | 0.457 | 0.770 |
| d15_letter_a_eps0_wrongrelease | oracle WRONG-RELEASE | 0.000 | 0.0000 | 0 | 7 | 8 | 7 | 1 | 0.471 | 0.401 | 0.770 |

- first oracle cell with 0 exact landings: eps 0.000 (chart err 0.0000); SSIM top-1 there 6/8
- 0 exact landings but at least one SSIM top-1: eps up to 0.150 (chart err 0.0987, 7/8)

### mlp_letter_a

| cell | chart | eps | chart err (mean) | exact landed | SSIM top-1 | SSIM top-5 | L2 top-1 | feat top-1 | med SSIM truth | med SSIM ctrl | med SSIM truth-vs-ctrl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_letter_a_eps0 | oracle | 0.000 | 0.0000 | 7 | 8 | 8 | 8 | 8 | 1.000 | 0.748 | 0.770 |
| mlp_letter_a_eps0.01 | oracle | 0.010 | 0.0069 | 7 | 8 | 8 | 8 | 8 | 0.990 | 0.757 | 0.770 |
| mlp_letter_a_eps0.02 | oracle | 0.020 | 0.0139 | 0 | 8 | 8 | 8 | 8 | 0.963 | 0.724 | 0.770 |
| mlp_letter_a_eps0.03 | oracle | 0.030 | 0.0208 | 0 | 8 | 8 | 8 | 8 | 0.928 | 0.694 | 0.770 |
| mlp_letter_a_eps0.05 | oracle | 0.050 | 0.0345 | 0 | 8 | 8 | 8 | 8 | 0.853 | 0.629 | 0.770 |
| mlp_letter_a_eps0.075 | oracle | 0.075 | 0.0513 | 0 | 8 | 8 | 8 | 8 | 0.771 | 0.552 | 0.770 |
| mlp_letter_a_eps0.1 | oracle | 0.100 | 0.0677 | 0 | 8 | 8 | 8 | 8 | 0.706 | 0.492 | 0.770 |
| mlp_letter_a_eps0.15 | oracle | 0.150 | 0.0987 | 0 | 8 | 8 | 8 | 8 | 0.613 | 0.414 | 0.770 |
| mlp_letter_a_eps0.2 | oracle | 0.200 | 0.1271 | 0 | 8 | 8 | 8 | 8 | 0.551 | 0.369 | 0.770 |
| mlp_letter_a_eps0.3 | oracle | 0.300 | 0.1748 | 0 | 8 | 8 | 8 | 8 | 0.482 | 0.323 | 0.770 |
| mlp_letter_a_eps0.4 | oracle | 0.400 | 0.2114 | 0 | 8 | 8 | 8 | 7 | 0.435 | 0.295 | 0.770 |
| mlp_letter_a_eps0.6 | oracle | 0.600 | 0.2596 | 0 | 8 | 8 | 8 | 8 | 0.366 | 0.257 | 0.770 |
| mlp_letter_a_pca | pca | - | 0.3123 | 0 | 7 | 7 | 7 | 7 | 0.619 | 0.536 | 0.770 |
| mlp_letter_a_eps0_wrongrelease | oracle WRONG-RELEASE | 0.000 | 0.0000 | 0 | 4 | 7 | 4 | 1 | 0.462 | 0.466 | 0.770 |

- first oracle cell with 0 exact landings: eps 0.020 (chart err 0.0139); SSIM top-1 there 8/8
- 0 exact landings but ALL 8 SSIM-identified top-1 among 100: eps up to 0.600 (chart err 0.2596)
- 0 exact landings but at least one SSIM top-1: eps up to 0.600 (chart err 0.2596, 8/8)

### mlp_motorcycle

| cell | chart | eps | chart err (mean) | exact landed | SSIM top-1 | SSIM top-5 | L2 top-1 | feat top-1 | med SSIM truth | med SSIM ctrl | med SSIM truth-vs-ctrl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_motorcycle_eps0 | oracle | 0.000 | 0.0000 | 8 | 8 | 8 | 8 | 8 | 1.000 | 0.198 | 0.198 |
| mlp_motorcycle_eps0.01 | oracle | 0.010 | 0.0062 | 7 | 8 | 8 | 8 | 8 | 0.998 | 0.197 | 0.198 |
| mlp_motorcycle_eps0.02 | oracle | 0.020 | 0.0124 | 3 | 8 | 8 | 8 | 8 | 0.994 | 0.195 | 0.198 |
| mlp_motorcycle_eps0.03 | oracle | 0.030 | 0.0186 | 0 | 8 | 8 | 8 | 8 | 0.987 | 0.191 | 0.198 |
| mlp_motorcycle_eps0.05 | oracle | 0.050 | 0.0309 | 0 | 8 | 8 | 8 | 8 | 0.965 | 0.182 | 0.198 |
| mlp_motorcycle_eps0.075 | oracle | 0.075 | 0.0460 | 0 | 8 | 8 | 8 | 8 | 0.930 | 0.172 | 0.198 |
| mlp_motorcycle_eps0.1 | oracle | 0.100 | 0.0608 | 0 | 8 | 8 | 8 | 8 | 0.889 | 0.160 | 0.198 |
| mlp_motorcycle_eps0.15 | oracle | 0.150 | 0.0891 | 0 | 8 | 8 | 8 | 8 | 0.810 | 0.139 | 0.198 |
| mlp_motorcycle_eps0.2 | oracle | 0.200 | 0.1151 | 0 | 8 | 8 | 8 | 8 | 0.740 | 0.122 | 0.198 |
| mlp_motorcycle_eps0.3 | oracle | 0.300 | 0.1598 | 0 | 8 | 8 | 8 | 8 | 0.620 | 0.096 | 0.198 |
| mlp_motorcycle_eps0.4 | oracle | 0.400 | 0.1949 | 0 | 8 | 8 | 8 | 8 | 0.527 | 0.079 | 0.198 |
| mlp_motorcycle_eps0.6 | oracle | 0.600 | 0.2428 | 0 | 8 | 8 | 8 | 8 | 0.382 | 0.061 | 0.198 |
| mlp_motorcycle_pca | pca | - | 0.3176 | 0 | 2 | 3 | 5 | 6 | 0.151 | 0.208 | 0.198 |
| mlp_motorcycle_eps0_wrongrelease | oracle WRONG-RELEASE | 0.000 | 0.0000 | 0 | 2 | 3 | 3 | 1 | 0.107 | 0.148 | 0.198 |

- first oracle cell with 0 exact landings: eps 0.030 (chart err 0.0186); SSIM top-1 there 8/8
- 0 exact landings but ALL 8 SSIM-identified top-1 among 100: eps up to 0.600 (chart err 0.2428)
- 0 exact landings but at least one SSIM top-1: eps up to 0.600 (chart err 0.2428, 8/8)


_767 arms scored in 2761s; rows in results/perceptual_id/<source>_356492.jsonl_
