#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o sprint2c_b5_b6_b7_%J.out
#BSUB -e sprint2c_b5_b6_b7_%J.err
#BSUB -J sprint2c_b5_b6_b7

# Sprint 2c Realism Probes (EXP-19, EXP-20, EXP-21)
# B5: N fine-tuning sweep — n_per_class ∈ {1,3,5,10}, T ∈ {1,10} (16 configs)
# B6: AdamW fine-tuning probe — {sgd,adamw} × T ∈ {1,10} × {full,r=8} (8 configs)
# B7: Consistency weight × T — α ∈ {0,0.1,1,5} × T ∈ {1,10,100,1000} × {full,r=8} (32 configs)

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting Sprint 2c B5 + B6 + B7 (Realism Probes) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python -u -m experiments.run_sprint2c_sweep \
    --track B5 B6 B7 \
    --device cuda \
    --seed 42

echo "=== Sprint 2c B5 + B6 + B7 Complete ==="
echo "Date: $(date)"
