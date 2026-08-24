#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_h2_%J.out
#BSUB -e scripts/wexac_logs/jacobian_h2_%J.err
#BSUB -J jacobian_h2

# =====================================================================
# H2 — nonlinear recovery beyond first order (plan rev 2/3).
# Does a nonlinear inverse recover the near-null coordinate v_min (the collinear
# direction J flattens) past the linear q_eff ceiling? Target a_true = eps*v_min;
# recover with the real nonlinear forward_Y (Adam on a-hat, box_weight=0, grad-clip,
# many restarts). Three-way verdict via the null component + loss gap:
#   loss>>0 -> optimizer-failure; loss~0 & null-match -> NONLINEAR-WIN;
#   loss~0 & null-mismatch -> collision (Y-match, wrong a). Local vs global init.
# svd = graceful ill-conditioning; pca = near-hard null (the real substrate).
# Known-init UPPER bound (distinct from the SGD-noise phase). eps capped below NaN.
# long-gpu; outer_iters kept low (only Nk coords; the unroll dominates per-iter).
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

echo ""; echo "########## STAGE 1: H2 smoke (pca N=2, 1 restart, 50 iters) ##########"; date
python -u -m experiments.jacobian_spectrum --h2 --tangent pca --N 2 --k 8 \
    --T 5 --rank 8 --seed 42 --eps_list 0.1 1.0 --n_restarts 1 --outer_iters 50 \
    --device cuda
if [ $? -ne 0 ]; then echo "FATAL: H2 smoke failed. Aborting."; exit 1; fi

echo ""; echo "########## STAGE 2: H2 sweep (tangent x seed) ##########"; date
for TAN in pca svd; do
  for SEED in 42 1; do
    echo ""; echo "-- H2 $TAN seed=$SEED --"; date
    python -u -m experiments.jacobian_spectrum --h2 \
        --tangent $TAN --N 2 --k 8 --T 5 --rank 8 --seed $SEED \
        --eps_list 0.01 0.1 0.3 1.0 3.0 --n_restarts 8 --outer_iters 400 \
        --save --device cuda
  done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
