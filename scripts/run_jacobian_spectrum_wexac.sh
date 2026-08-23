#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_spectrum_%J.out
#BSUB -e scripts/wexac_logs/jacobian_spectrum_%J.err
#BSUB -J jacobian_spectrum

# =====================================================================
# Phase J0 — data-latent Jacobian of LoRA fine-tuning.
# notes/jacobian_leakage_experiment_plan.md
#
# J = d vec(A_T,B_T) / d a  where x_i(a_i) = x_i^0 + U_i a_i.
# Recover the private coordinates a from the released adapter; test whether
# sigma_i(J) predicts which coordinates survive. GELU only, float64.
#
#   Stage 0  GATE  : toy-AD finite-difference check (<1e-6) + jvp-vs-reverse
#                    (<1e-8) + LSQ recovery. ABORTS the job on failure —
#                    nothing downstream is trusted without it.
#   Stage 1  smoke : real MNIST single-module J (dimY=r*(784+1000)), FD <1e-4.
#   Stage 2  J0    : coordinate-recovery-vs-eps sweeps, save .pth + spectrum
#                    figures. 'qr' (clean rank k) AND 'svd' (injected decay:
#                    the falsification test that sigma_i(J) tracks a known
#                    rank deficiency).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "=== START $(date) on $(hostname) ==="

# ---------------------------------------------------------------------
# Stage 0 + 1 — the AD gate and the real single-module smoke.
# --smoke runs toy_ad_gate() (sys.exit(1) on fail) then real_smoke().
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 0+1: AD gate + MNIST smoke ##########"; date
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then
    echo "FATAL: toy-AD gate or MNIST smoke failed. Aborting — J is not trustworthy."
    exit 1
fi
echo "Stage 0+1 PASSED — J validated against finite differences."

# ---------------------------------------------------------------------
# Stage 2 — Phase J0 coordinate-recovery sweeps.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 2: Phase J0 sweeps ##########"; date
for TANGENT in qr svd; do
  for N in 2 4; do
    for K in 4 8 16; do
      echo ""; echo "---- J0 N=$N k=$K tangent=$TANGENT ----"; date
      python -u -m experiments.jacobian_spectrum --j0 \
          --N $N --k $K --T 5 --rank 8 --activation gelu \
          --tangent $TANGENT \
          --eps_list 0.001 0.01 0.1 0.3 1.0 \
          --save --device cuda
    done
  done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
