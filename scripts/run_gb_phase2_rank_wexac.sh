#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gb_phase2_rank_%J.out
#BSUB -e scripts/wexac_logs/gb_phase2_rank_%J.err
#BSUB -J gb_phase2_rank
# Learning chain: TV failed -> L1/non-neg failed -> NOT prior-limited, it's the decoder cosine.
# Mechanistic next lever: x is the ROW factor of dL/dW0; two-sided rank-r observes an r-dim slice of
# the 784-dim row space, so raising rank should DIRECTLY capture more of x (unlike single-sided rank,
# which did nothing). Two-sided x rank {8,32,64,128}, input layer, gelu (the hard case) + softplus.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
for R in 8 32 64 128; do
    echo ""; echo "########## two-sided rank=$R ##########"; date
    python -u -m experiments.gradient_bridge.phase2_image \
        --activations gelu softplus --two_sided --rank $R \
        --n_train 8000 --n_eval 96 --epochs 60 --device cuda
done
echo "=== DONE $(date) ==="
