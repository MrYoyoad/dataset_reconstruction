#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gate_matrix_test_%J.out
#BSUB -e scripts/wexac_logs/gate_matrix_test_%J.err
#BSUB -J gate_matrix_test

# The Section-8 test from notes/linearization_leakage_theory.tex: per-activation gate-matrix rank,
# per-neuron gate variance/range, |sigma''|, and pairwise cosine of the per-sample gradient features,
# at theta_0 (alpha=0 anchor), for N=2 and N=10. Forward + one backward pass, no extraction -> fast.
# Confirms or refutes the feature-ceiling half of the theory against the measured LoRA leakage margins.

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
python -u -m experiments.gate_matrix_test --device cuda
echo "=== DONE $(date) ==="
