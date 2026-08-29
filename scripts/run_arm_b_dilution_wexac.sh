#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/arm_b_dilution_%J.out
#BSUB -e scripts/wexac_logs/arm_b_dilution_%J.err
#BSUB -J arm_b_dilution

# =====================================================================
# ARM B — per-image adapter sensitivity vs N (the 1/N dilution law).
# notes/dataset_sensitivity_program_plan.md, first spine. Audit-clean (yoado-34);
# floor-decomposition pre-audited (yoado-a2). Metric ρ = swap / seed-noise-floor,
# ΔW=BA only, σ_repeat vs σ_reseed decomposed (+NaN-drop), raw swap_abs(N) &
# reseed(N) reported un-normalized (S2), {K,2K} stability. mnist/gelu/binary, float64.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## STAGE 0: arm-b sanity (N=8, K=3, 3 targets) ##########"; date
python -u -m experiments.dataset_sensitivity.arm_b_dilution --stage0 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: arm-b Stage-0 sanity failed. Aborting."; exit 1; fi
echo "Stage-0 PASSED."

echo ""; echo "########## FULL RUN: N in {2,4,8,16,32,64} ##########"; date
python -u -m experiments.dataset_sensitivity.arm_b_dilution \
    --N_list 2 4 8 16 32 64 --K 50 --n_targets 4 --lr 0.5 --T 1000 --rank 8 --device cuda

echo ""; echo "=== DONE $(date) ==="
echo "READ: rho(N)=swap/reseed; check repeat/reseed ratio small (else uninterpretable);"
echo "fit swap_abs(N) AND reseed(N) exponents SEPARATELY before any 1/N claim (S2)."
