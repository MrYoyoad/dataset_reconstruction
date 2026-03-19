#\!/bin/bash -l
# Experiment B on GPU — with explicit logging

LOG=~/exp_b_log.txt

echo "=== Starting Experiment B on GPU ===" | tee $LOG
echo "Time: $(date)" | tee -a $LOG

# Load conda
module load miniconda 2>/dev/null
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

echo "Python: $(which python)" | tee -a $LOG
echo "PyTorch: $(python -c "import torch; print(torch.__version__)")" | tee -a $LOG
echo "CUDA: $(python -c "import torch; print(torch.cuda.is_available())")" | tee -a $LOG
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a $LOG

cd ~

python -m experiments.run_experiment_b \
    --n_steps 1 --rank 8 --n_per_class 1 \
    --device cuda 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Experiment B finished at $(date) ===" | tee -a $LOG
