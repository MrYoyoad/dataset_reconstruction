#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0 && hname!='lgn28' && hname!='hgn46' && hname!='hgn45' && hname!='lgn13']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/fullft_valley_core1_%J.out
#BSUB -e scripts/wexac_logs/fullft_valley_core1_%J.err
#BSUB -J fullft_valley_core1

# =====================================================================
# FULL-FT VALLEY COMPARISON — core wave, PART 1 of 2.
# Plan: notes/fullft_valley_comparison_plan.md (v1.2 FINAL, unanimous audit pass).
# Sequencing per plan §6: (1) metric self-test gate -> (2) stage-0 smoke (arm C
# tiny) -> (3) eps-calibration + 3-point bracket + §4.1.3 null gate + S6c arm-A
# pre-check -> (4) arm C full dial -> (5) {K,2K} adequacy on arm C (reduced rungs).
# PART 2 (run_fullft_valley_part2_wexac.sh) runs arms D, E_b0, E_eps, B2 and
# REQUIRES this job's calibration.json — submit it with -w 'done(<this job>)'.
# Split choice: the whole wave is ~3k trainings (~3x job 268959); two sequenced
# jobs bound the walltime risk.  rsync experiments/ before submit (house rule).
# NOTE (S6c): after step (3), check the printed mid-rung recommendation; if it is
# not p3_rot15, pass --mid_rung <rec> to the PART-2 dial arms (and rerun arm C
# with it if the recommendation differs — the arm-A pre-check governs, plan D3).
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## GATE 1: whitened-metric self-test (CPU, synthetic) ##########"; date
python -u -m experiments.dataset_sensitivity.whitened_metric
if [ $? -ne 0 ]; then echo "FATAL: metric self-test FAILED. Aborting wave."; exit 1; fi
echo "metric self-test PASSED."

echo ""; echo "########## STEP 2: calibration FIRST (lr search + eps fixed-point + bracket + null gate + S6c pre-check) — arms need calibration.json ##########"; date
python -u -m experiments.dataset_sensitivity.fullft_valley --arm calib --device cuda
if [ $? -ne 0 ]; then echo "FATAL: calibration FAILED. Aborting wave."; exit 1; fi

echo ""; echo "########## GATE 3: stage-0 smoke (arm C tiny, now with calibrated lr) ##########"; date
python -u -m experiments.dataset_sensitivity.fullft_valley --arm C --stage0 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: stage-0 smoke FAILED. Aborting wave."; exit 1; fi
echo "stage-0 PASSED."

if [ ! -f results/fullft_valley/calibration.json ]; then
    echo "FATAL: calibration.json missing after calib step. Aborting."; exit 1
fi
if [ ! -f results/fullft_valley/calibration.json ]; then
    echo "FATAL: calibration.json missing after calib step. Aborting."; exit 1
fi
# S5 BAND ENFORCEMENT: freeze lr from calibration.json (arms read it via resolve_config)
# and ABORT if arm C's tuned lr did not land max_bce in [1e-4,1e-3) — never run off-band.
CBAND=$(python -c "import json;print(json.load(open('results/fullft_valley/calibration.json'))['lr']['C']['in_band'])")
if [ "$CBAND" != "True" ]; then
    echo "FATAL (S5): arm-C lr NOT in the memorization band [1e-4,1e-3). Widen/retune LR_GRID"
    echo "and rerun calib before quoting any dial number. Aborting."; exit 1
fi
echo "calibration DONE (arm-C lr in-band; check the printed eps-bracket alpha gate + null gate + mid-rung rec)."

echo ""; echo "########## STEP 4: ARM C — full-rank single layer (L0), eps-noise ##########"; date
python -u -m experiments.dataset_sensitivity.fullft_valley --arm C \
    --K 50 --n_targets 2 --T 1000 --rank 8 --N 16 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: arm C FAILED."; exit 1; fi

echo ""; echo "########## STEP 5: {K,2K} adequacy on arm C (§4.1.6; reduced rungs, 2K=100) ##########"; date
python -u -m experiments.dataset_sensitivity.fullft_valley --arm C \
    --K 100 --n_targets 2 --T 1000 --rank 8 --N 16 --tag _2K \
    --rungs p00_identity,p0_noise,r_cross --device cuda
if [ $? -ne 0 ]; then echo "WARNING: {K,2K} check failed (non-fatal for the dial)."; fi

echo ""; echo "=== DONE $(date) ==="
echo "READ: (1) eps-bracket alpha must be -2±0.3 with no curvature, else shrink eps"
echo "and RERUN calib before quoting anything; (2) null gate frac(p<.05)<=0.2, qeff~0;"
echo "(3) arm-C p00_identity ~0/p~1 (artifact-kill) and p(r_cross)<0.05 (normalizer);"
echo "(4) compare mini-d* between C and C_2K (K-adequacy on the reported ratio, S6a);"
echo "(5) submit part 2 with: bsub -w 'done(<this jobid>)' < scripts/run_fullft_valley_part2_wexac.sh"
