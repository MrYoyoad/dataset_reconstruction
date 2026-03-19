#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o sprint2b_p56_%J.out
#BSUB -e sprint2b_p56_%J.err
#BSUB -J sprint2b_p56

# Sprint 2b Phase 5-6: LR realism ablations (RE-SUBMIT on long-gpu)
# Previous run (job 794136) killed at 6-hour limit on short-gpu.
# Phase 5: LR magnitude (4 LRs × 5 T × {full, r=8, r=32} = 60 configs)
# Phase 6: Cosine SGD (4 T × {full, r=8, r=32} = 12 configs)
# Total: 72 configs

# Activate conda
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado

# Add dataset_reconstruction to path
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting Sprint 2b Phases 5-6 (LR Realism Ablations) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python -u -m experiments.run_sprint2b_sweep \
    --phase 5 6 \
    --finetune_activation leaky_relu \
    --device cuda \
    --seed 42

echo "=== Sprint 2b Phases 5-6 Complete ==="
echo "Date: $(date)"
echo "Results saved in results/"
