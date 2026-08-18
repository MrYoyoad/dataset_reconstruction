#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gb_phase2_prior_%J.out
#BSUB -e scripts/wexac_logs/gb_phase2_prior_%J.err
#BSUB -J gb_phase2_prior
# Fix after TV failed: compare priors MATCHED to digit statistics (L1-sparsity, non-negativity) vs TV
# vs raw-SVD, sweeping the weight. Decides prior-limited vs decoder-cosine-limited. gelu FIRST (its
# two-sided decode is only ~0.65 -- the requested activation), then softplus. Efficient (ep70).
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
python -u -m experiments.gradient_bridge.phase2_prior --activations gelu softplus --device cuda
echo "=== DONE $(date) ==="
