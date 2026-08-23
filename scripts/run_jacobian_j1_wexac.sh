#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_j1_%J.out
#BSUB -e scripts/wexac_logs/jacobian_j1_%J.err
#BSUB -J jacobian_j1

# =====================================================================
# Phase J1 — seed-whitened Jacobian, q_eff (first privacy-meaningful number)
#            + T-sweep de-confound of the J0 eff_rank readout.
# notes/jacobian_leakage_experiment_plan.md
#
# Σ_seed = Cov over LoRA-B0 init draws (unknown-init attacker; full-batch ⇒ B0
# is the only training randomness). J_SNR = Σ_seed^{-1/2} J; q_eff(ε)=#{εσ_i>1}.
# BRACKET: deterministic eff_rank = known-init upper bound; q_eff = unknown-init
# conservative bound.
#
#   Stage 0  GATE     : toy-AD FD check (<1e-6) + MNIST smoke. Aborts on fail.
#   Stage 1  T-sweep  : eff_rank(J) at T=5/20/50 — underfitting vs structural.
#   Stage 2  J1       : q_eff over S×ρ×ε for the bracket configs (N=2 vs N=4).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "=== START $(date) on $(hostname) ==="

# ---------------------------------------------------------------------
# Stage 0 — AD gate (abort on fail).
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 0: AD gate + MNIST smoke ##########"; date
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then
    echo "FATAL: AD gate failed. Aborting."
    exit 1
fi

# ---------------------------------------------------------------------
# Stage 1 — T-sweep: is the N=4 eff_rank deficiency structural or underfitting?
# Small Nk configs (fewer JVP columns) keep the deep T=50 unroll graph in budget.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 1: T-sweep (eff_rank vs T) ##########"; date
for NK in "2 8" "4 4"; do
  set -- $NK
  echo ""; echo "---- T-sweep N=$1 k=$2 ----"
  python -u -m experiments.jacobian_spectrum --T_sweep \
      --N $1 --k $2 --rank 8 --Ts 5 20 50 --tangent qr --device cuda
done

# ---------------------------------------------------------------------
# Stage 2 — J1 whitening / q_eff on the bracket configs.
# ---------------------------------------------------------------------
echo ""; echo "########## STAGE 2: J1 whitening / q_eff ##########"; date
for NK in "2 4" "2 8" "4 4" "4 8"; do
  set -- $NK
  echo ""; echo "---- J1 N=$1 k=$2 ----"; date
  python -u -m experiments.jacobian_spectrum --j1 \
      --N $1 --k $2 --T 5 --rank 8 --tangent qr \
      --S_list 16 32 64 --shrink_list 0.0001 0.01 0.1 \
      --eps_list 0.01 0.1 0.3 1.0 3.0 10.0 \
      --save --device cuda
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
