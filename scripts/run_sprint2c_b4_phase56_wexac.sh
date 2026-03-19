#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o sprint2c_b4_ph56_%J.out
#BSUB -e sprint2c_b4_ph56_%J.err
#BSUB -J sprint2c_b4_ph56

# Combined job:
# 1. Sprint 2c Track B4: N sweep at extraction time (8 configs)
#    - Re-run after fixing tensor size mismatch bug in ntk_extraction.py
# 2. Sprint 2b Phase 5: LR magnitude ablation (60 configs)
#    - LR ∈ {0.001, 0.005, 0.01, 0.05} × T ∈ {1, 5, 10, 50, 100} × {full, r=8, r=32}
# 3. Sprint 2b Phase 6: Cosine SGD schedule (12 configs)
#    - T ∈ {10, 50, 100, 500} × {full, r=8, r=32}

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== Starting Combined Job: B4 + Phase 5 + Phase 6 ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# --- Track B4: N sweep (bugfix re-run) ---
echo ""
echo "========================================"
echo "PART 1/3: Sprint 2c Track B4 (N sweep)"
echo "========================================"
python -u -m experiments.run_sprint2c_sweep \
    --track B4 \
    --device cuda \
    --seed 42

echo ""
echo "========================================"
echo "PART 2/3: Sprint 2b Phase 5 (LR magnitude)"
echo "========================================"
python -u -m experiments.run_sprint2b_sweep \
    --phase 5 \
    --device cuda \
    --finetune_activation leaky_relu \
    --seed 42

echo ""
echo "========================================"
echo "PART 3/3: Sprint 2b Phase 6 (Cosine schedule)"
echo "========================================"
python -u -m experiments.run_sprint2b_sweep \
    --phase 6 \
    --device cuda \
    --finetune_activation leaky_relu \
    --seed 42

echo ""
echo "=== All 3 parts complete ==="
echo "Date: $(date)"
