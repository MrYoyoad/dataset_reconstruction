#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/fullft_jacobian_%J.out
#BSUB -e scripts/wexac_logs/fullft_jacobian_%J.err
#BSUB -J fullft_jac

# =====================================================================
# Arm G (full-FT valley comparison §2.2) — the T-sweep + comparison.
# J_full = ∂vec(Δθ_full)/∂a vs J_LoRA = ∂vec(BA)/∂a on the SAME tangents/θ₀/D.
#
#   Stage 0  GATE      : jacobian_spectrum gate + arm-G FD gate (abort on fail).
#   Stage 1  PRIMARY   : layer-0 J_full T-sweep T={1,5,20} (the like-for-like
#                        weight block vs J_LoRA). RAW spectra + r_J (P6, T=5-cond
#                        consistency) + ‖J·a_nn‖/‖J·a_far‖ (P7, LEADS) + max_bce.
#   Stage 2  STRETCH   : all-layer J_full T-sweep (heavier — dimΘ≈1.79M; the
#                        double-backward graph over all weight matrices is the
#                        memory-bound case; a full-batch full-param unroll). Runs
#                        AFTER the primary so a memory abort never sinks stage 1.
#
# RAW / noise-free readouts are PRIMARY (P7). SNR-whitened q_eff is P4-conditional
# secondary and NOT in this job (a downstream rescore once P4 clears).
# Every readout = early-training Jacobian (T∈{1,5,20}), NOT the converged valley.
# GELU-only, float64. python -u (LSF buffers otherwise).
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## STAGE 0: gates (abort on fail) ##########"; date
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: jacobian_spectrum gate failed. Aborting."; exit 1; fi
python -u -m experiments.dataset_sensitivity.fullft_jacobian --stage0 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: arm-G stage-0 gate failed. Aborting."; exit 1; fi

echo ""; echo "########## STAGE 1: PRIMARY layer-0 J_full T-sweep T={1,5,20} ##########"; date
python -u -m experiments.dataset_sensitivity.fullft_jacobian --tsweep \
    --N 4 --k 8 --Ts 1 5 20 --rank 8 --activation gelu \
    --dataset mnist --seed 42 --save --device cuda

echo ""; echo "########## STAGE 2: STRETCH all-layer J_full T-sweep (heavy) ##########"; date
python -u -m experiments.dataset_sensitivity.fullft_jacobian --tsweep --all_layers \
    --N 4 --k 8 --Ts 1 5 20 --rank 8 --activation gelu \
    --dataset mnist --seed 42 --save --device cuda

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
