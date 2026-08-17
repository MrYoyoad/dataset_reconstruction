#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/theory_closure_%J.out
#BSUB -e scripts/wexac_logs/theory_closure_%J.err
#BSUB -J theory_closure

# Exp A: quantitatively close the corrected Lemma B (lin-error ~ sigma''/||grad Phi||) and the high-k
# item (eff_rank(X) MNIST vs flowers32 vs flowers64). Forward + one backward pass, no extraction.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
python -u -m experiments.theory_closure_test
echo "=== DONE $(date) ==="
