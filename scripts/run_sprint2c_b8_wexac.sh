#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o sprint2c_b8_%J.out
#BSUB -e sprint2c_b8_%J.err
#BSUB -J sprint2c_b8

# Sprint 2c Track B8: Loss Weight Ablation (box_weight + extended CW)
# Sub-sweep 1: CW ∈ {0.01, 0.5, 2.0, 10.0, 100.0}, BW=5.0 (20 configs)
# Sub-sweep 2: BW ∈ {0, 0.5, 1.0, 5.0, 20.0}, CW=1.0 (20 configs, minus 1 dup)
# Sub-sweep 3: CW=0, BW=0 bare baseline (4 configs)
# ~43 unique configs total

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting Sprint 2c B8 (Loss Weight Ablation) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python -u -m experiments.run_sprint2c_sweep \
    --track B8 \
    --device cuda \
    --seed 42

echo "=== Sprint 2c B8 Complete ==="
echo "Date: $(date)"
