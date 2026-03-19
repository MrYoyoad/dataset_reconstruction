#\!/bin/bash -l
LOG=~/exp_a_log.txt

echo "=== Starting Experiment A on GPU ===" | tee $LOG
echo "Time: $(date)" | tee -a $LOG

module load miniconda 2>/dev/null
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

echo "Python: $(which python)" | tee -a $LOG
echo "PyTorch: $(python -c "import torch; print(torch.__version__)")" | tee -a $LOG
echo "CUDA: $(python -c "import torch; print(torch.cuda.is_available())")" | tee -a $LOG
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a $LOG

cd ~
python run_exp_a_gpu.py 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Experiment A finished at $(date) ===" | tee -a $LOG
