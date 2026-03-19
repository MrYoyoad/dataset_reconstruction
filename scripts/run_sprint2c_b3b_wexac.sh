#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o sprint2c_b3b_%J.out
#BSUB -e sprint2c_b3b_%J.err
#BSUB -J sprint2c_b3b

# Sprint 2c Track B3b: Scale best LoRA optimizer×activation across T
# Depends on B3a results — update --best_optimizer and --best_activation
# 5 T × 2 ranks = 10 configs

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting Sprint 2c B3b (best LoRA combo across T) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# UPDATE THESE after analyzing B3a results:
python -u -m experiments.run_sprint2c_sweep \
    --track B3b \
    --best_optimizer lbfgs \
    --best_activation leaky_relu \
    --device cuda \
    --seed 42

echo "=== Sprint 2c B3b Complete ==="
echo "Date: $(date)"
