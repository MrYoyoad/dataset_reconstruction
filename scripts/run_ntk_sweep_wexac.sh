#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o ntk_sweep_%J.out
#BSUB -e ntk_sweep_%J.err
#BSUB -J ntk_sweep

# NTK Multi-Step Sweep (Experiment B)
# Sweeps T ∈ {1,2,5,10,20,50,100,500,1000} × rank ∈ {full,1,4,8,32,64}
# Expected runtime: ~1-2 hours on L40S with L-BFGS extraction

# Activate conda
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado

# Add dataset_reconstruction to path
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting NTK Multi-Step Sweep ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Run the sweep with L-BFGS extraction (1000 epochs is plenty for L-BFGS)
# L-BFGS does 20 function evals per step, so 1000 epochs = 20K effective evals
python -m experiments.run_sweep \
    --experiment b \
    --seed 42 \
    --extraction_epochs 1000 \
    --device cuda

echo "=== Sweep Complete ==="
echo "Date: $(date)"
echo "Results saved in results/"
