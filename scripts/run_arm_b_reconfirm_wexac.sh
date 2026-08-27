#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/arm_b_reconfirm_%J.out
#BSUB -e scripts/wexac_logs/arm_b_reconfirm_%J.err
#BSUB -J arm_b_reconfirm

# =====================================================================
# ARM B — BELT-AND-SUSPENDERS RE-CONFIRM on the FIXED 3-way whitened metric.
# The retracted "sharpens with N" was a winner's-curse denominator artifact; the
# 3-way disjoint cross-fit (whitened_metric.py, 2026-08-27) removes it.
# TEST: sensitivity(N) must now (1) CONVERGE K=50 vs K=100 (old 2-way blew 63->161),
#       and (2) read FLAT in N (detection real + magnitude stable). p-value stays robust.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## GATE 0a: 3-way metric self-test (on-node) ##########"; date
python -u -m experiments.dataset_sensitivity.whitened_metric
if [ $? -ne 0 ]; then echo "FATAL: whitened_metric self-test failed on node. Aborting."; exit 1; fi
echo "GATE 0a PASSED (4/4 incl K-convergence)."

echo ""; echo "########## GATE 0b: arm-b sanity (N=8, K=12, 2 targets) ##########"; date
python -u -m experiments.dataset_sensitivity.arm_b_dilution --stage0 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: arm-b Stage-0 failed. Aborting."; exit 1; fi
echo "GATE 0b PASSED."

echo ""; echo "########## K=50 : N in {4,8,16,32} ##########"; date
python -u -m experiments.dataset_sensitivity.arm_b_dilution \
    --N_list 4 8 16 32 --K 50 --n_targets 4 --lr 0.5 --T 1000 --rank 8 --device cuda

echo ""; echo "########## K=100 : N in {4,8,16,32} (convergence check vs K=50) ##########"; date
python -u -m experiments.dataset_sensitivity.arm_b_dilution \
    --N_list 4 8 16 32 --K 100 --n_targets 4 --lr 0.5 --T 1000 --rank 8 --device cuda

echo ""; echo "=== DONE $(date) ==="
echo "READ: (1) sensitivity(N) at K=50 vs K=100 must AGREE (converged) -- old 2-way inflated ~2.5x;"
echo "      (2) sensitivity(N) FLAT across N with robust p; magnitude now trustworthy (3-way)."
