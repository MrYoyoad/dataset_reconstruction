#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o t_sweep_examples_%J.out
#BSUB -e t_sweep_examples_%J.err
#BSUB -J t_sweep_examples

# Re-run best-rank extraction for each T and generate PDF + saved tensors.
# Expected runtime: ~30 min on L40S (9 configs × 1000 L-BFGS epochs each)

# Activate conda
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado

# Add dataset_reconstruction to path
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting T-Sweep Examples Generation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python -u -m experiments.gen_t_sweep_examples \
    --device cuda \
    --extraction_epochs 1000

echo "=== Done ==="
echo "Date: $(date)"
echo "Results: results/t_sweep_examples.pth"
echo "Figure:  figures/t_sweep_examples.pdf"
