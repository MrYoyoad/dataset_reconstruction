#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_robust_%J.out
#BSUB -e scripts/wexac_logs/jacobian_robust_%J.err
#BSUB -J jacobian_robust

# =====================================================================
# Robustness of the on-manifold leakage finding (qr random vs pca principal:
# init masks little for random dirs, ~half for principal dirs). Vary the WORK
# POINT to confirm it is not specific to one setting:
#   - datasets:      mnist / fashion / flowers  (all 784-dim, reuse MNIST θ₀)
#   - private draw:  seed 42 / 1                 (different private images)
#   - anchor α:      0.0 / 0.9                   (θ_anchor=(1−α)θ₀+αθ_T work pt)
#   - tangents:      qr / pca                    (the contrast under test)
# Metric: J1 col(J) iso_ratio + q_eff|col(J) + Σ_J/μ mode count.
#
# Plus a LOCALITY block: J0 with the linearization residual lin_res reported at
# ε=1e-4..1e-1 (the linear regime only holds where lin_res ≪ 1; pca needs small ε).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## STAGE 0: AD gate ##########"; date
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: AD gate failed. Aborting."; exit 1; fi

# ---------------------------------------------------------------------
# Locality block: is the regime valid? (lin_res at small ε, qr vs pca)
# ---------------------------------------------------------------------
echo ""; echo "########## LOCALITY: J0 lin_res vs ε ##########"; date
for DS in mnist fashion; do
  for TAN in qr pca; do
    echo ""; echo "-- locality $DS $TAN --"
    python -u -m experiments.jacobian_spectrum --j0 \
        --dataset $DS --N 2 --k 8 --T 5 --rank 8 --tangent $TAN \
        --eps_list 0.0001 0.001 0.01 0.1 --save --device cuda
  done
done

# ---------------------------------------------------------------------
# Robustness grid: J1 col(J) across datasets × seeds × anchors × tangents.
# ---------------------------------------------------------------------
echo ""; echo "########## ROBUSTNESS: J1 col(J) grid ##########"; date
for DS in mnist fashion flowers; do
  for TAN in qr pca; do
    for SEED in 42 1; do
      for A in 0.0 0.9; do
        echo ""; echo "-- J1 $DS $TAN seed=$SEED anchor=$A --"; date
        python -u -m experiments.jacobian_spectrum --j1 \
            --dataset $DS --N 2 --k 8 --T 5 --rank 8 --tangent $TAN \
            --seed $SEED --anchor_alpha $A \
            --S_list 64 128 --shrink_list 0.01 \
            --eps_list 0.1 0.3 1.0 3.0 10.0 --save --device cuda
      done
    done
  done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
