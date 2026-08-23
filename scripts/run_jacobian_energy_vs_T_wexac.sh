#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_energyT_%J.out
#BSUB -e scripts/wexac_logs/jacobian_energyT_%J.err
#BSUB -J jacobian_energyT

# =====================================================================
# Mechanism test (yoado-29): is J ⊥ Σ_seed a structural A₀=0 effect, or a
# small-T artifact? A₀=0 ⇒ data signal enters the A-block, init noise the
# B-block; they mix only as A≠0 grows with T. PREDICTION: the "J-energy in
# measured noise subspace" fraction should GROW with T.
# Reuses run_j1 (its per-S line prints the energy fraction) at T=5/20/50.
# Minimal S/ρ/ε sweep — we only need the energy-fraction + moments line.
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

echo ""; echo "########## energy-vs-T: J-energy in noise subspace vs T ##########"; date
for NK in "2 8" "4 8"; do
  set -- $NK
  for T in 5 20 50; do
    echo ""; echo "---- N=$1 k=$2 T=$T ----"
    python -u -m experiments.jacobian_spectrum --j1 \
        --N $1 --k $2 --T $T --rank 8 --tangent qr \
        --S_list 64 --shrink_list 0.01 --eps_list 1.0 \
        --save --device cuda
  done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
