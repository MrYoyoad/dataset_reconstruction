#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/fullft_valley_stage0_%J.out
#BSUB -e scripts/wexac_logs/fullft_valley_stage0_%J.err
#BSUB -J fullft_valley_stage0
# FULL-FT VALLEY COMPARISON — STAGE-0 PLUMBING GATE (plan v1.2).
# Exercises every --arm end to end on tiny config (K=10, 1 target) WITHOUT calibration
# (each dial/removal arm uses resolve_config's stage0 PROVISIONAL lr/eps — plumbing only,
# NOT science). Checks: metric self-test -> full-FT training converges (max_bce printed)
# -> per-layer/concat metric integrates -> d=0 identity rung ~0 (artifact-kill) -> the
# leave-one-out contrast runs both regimes -> the B1 coordinate-rescore reads arm-D's
# saved stage-0 stacks. Arm D MUST run before B1 (B1 rescores D_stage0_*.pth).
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
FV="python -u -m experiments.dataset_sensitivity.fullft_valley"

echo "########## GATE 0: whitened-metric acceptance self-test (CPU) ##########"
python -u -m experiments.dataset_sensitivity.whitened_metric || { echo FATAL metric; exit 1; }

echo "########## GATE 0.5: calib (lr-search into max_bce band + ε) — WRITES calibration.json (arms need it) ##########"
$FV --arm calib --stage0 --device cuda || { echo FATAL calib; exit 1; }

echo "########## GATE 1: arm C (full-rank single layer L0, ε-noise) ##########"
$FV --arm C --stage0 --device cuda || { echo FATAL armC; exit 1; }

echo "########## GATE 2: arm D (full FT all layers, per-layer readout) — SAVES stacks ##########"
$FV --arm D --stage0 --device cuda || { echo FATAL armD; exit 1; }

echo "########## GATE 3: arm E_b0 (LoRA r=8, B0-reseed noise) ##########"
$FV --arm E_b0 --stage0 --device cuda || { echo FATAL armE_b0; exit 1; }

echo "########## GATE 4: arm E_eps (LoRA r=8, fixed B0 + θ0 ε-perturb) ##########"
$FV --arm E_eps --stage0 --device cuda || { echo FATAL armE_eps; exit 1; }

echo "########## GATE 5: arm B2 (full FT under SGD minibatch-order noise) ##########"
$FV --arm B2 --stage0 --device cuda || { echo FATAL B2; exit 1; }

echo "########## GATE 6: arm F (leave-one-out, full vs LoRA + g0 piggyback) ##########"
$FV --arm F --stage0 --device cuda || { echo FATAL F; exit 1; }

echo "########## GATE 7: arm B1 (dimension-invariance rescore of arm-D stage-0 stacks, CPU) ##########"
$FV --arm B1 --stage0 --arm_d_tag _stage0 --device cpu || { echo FATAL B1; exit 1; }

echo "=== ALL STAGE-0 GATES PASSED $(date) ==="
echo "NOTE: stage-0 uses PROVISIONAL lr/eps (no calibration.json) — headline numbers come"
echo "only from the calibrated wave (run_fullft_valley_wexac.sh + _part2)."
