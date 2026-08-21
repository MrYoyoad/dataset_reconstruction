#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768]"
#BSUB -gpu "num=1:j_exclusive=yes:gmodel=NVIDIAA100_SXM4"
#BSUB -o scripts/wexac_logs/gb_pbox_fashion_%J.out
#BSUB -e scripts/wexac_logs/gb_pbox_fashion_%J.err
#BSUB -J gb_pbox_fashion
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
python -u -m experiments.gradient_bridge.phase2_e2e \
    --dataset fashion --npc_list 1 2 5 --activations softplus gelu --pixel_box \
    --n_train 12000 --dec_epochs 70 --ext_epochs 10000 --rank 8 --device cuda
echo "=== DONE $(date) ==="
