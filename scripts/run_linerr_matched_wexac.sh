#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/linerr_matched_%J.out
#BSUB -e scripts/wexac_logs/linerr_matched_%J.err
#BSUB -J linerr_matched

# Follow-up A: close Lemma B. Calibrate LR per activation to a common weight_change (0.05, T=10),
# then compute the function-space linearization error. Prediction: at matched ||delta|| the lin-error
# follows sigma'' (gelu/high-beta > softplus/tanh; kinked highest), reversing the raw softplus anomaly.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
python -u -m experiments.linerr_matched_wchg --target_wc 0.05 --n_steps 10 --device cuda
echo "=== DONE $(date) ==="
