#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o sprint2b_p7_%J.out
#BSUB -e sprint2b_p7_%J.err
#BSUB -J sprint2b_p7

# Sprint 2b Phase 7: Linear LR schedule
# T ∈ {1,2,5,10,20,50,100,500,1000} × {full, LoRA r=8} = 18 configs
# Linear decay: lr_t = lr * (1 - t/T)

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting Sprint 2b Phase 7 (Linear LR Schedule) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python -u -m experiments.run_sprint2b_sweep \
    --phase 7 \
    --finetune_activation leaky_relu \
    --device cuda \
    --seed 42

echo "=== Sprint 2b Phase 7 Complete ==="
echo "Date: $(date)"
