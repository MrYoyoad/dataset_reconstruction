#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o sprint2c_b3a_b4_%J.out
#BSUB -e sprint2c_b3a_b4_%J.err
#BSUB -J sprint2c_b3a_b4

# Sprint 2c Track B3a + B4 (re-run: B2 already done, short-gpu killed B3a+B4)
# B3a: Optimizer × activation for LoRA (12 configs)
# B4: N sweep (8 configs)

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting Sprint 2c B3a + B4 (re-run on long-gpu) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python -u -m experiments.run_sprint2c_sweep \
    --track B3a B4 \
    --device cuda \
    --seed 42

echo "=== Sprint 2c B3a + B4 Complete ==="
echo "Date: $(date)"
