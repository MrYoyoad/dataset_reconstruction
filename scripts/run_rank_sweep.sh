#\!/bin/bash -l
LOG=~/rank_sweep_log.txt

echo "=== Rank Ablation Sweep (seed=42, digits 5+0) ===" | tee $LOG
echo "Time: $(date)" | tee -a $LOG

module load miniconda 2>/dev/null
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

echo "CUDA: $(python -c "import torch; print(torch.cuda.is_available())")" | tee -a $LOG
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a $LOG

cd ~

for RANK in 16 32; do
    echo "" | tee -a $LOG
    echo "=== rank=$RANK, seed=42 ===" | tee -a $LOG
    python -m experiments.run_experiment_b \
        --n_steps 1 --rank $RANK --n_per_class 1 --seed 42 \
        --device cuda 2>&1 | tee -a $LOG
done

echo "" | tee -a $LOG
echo "=== Sweep finished at $(date) ===" | tee -a $LOG
