#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0 && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gb_pbox_flowers_%J.out
#BSUB -e scripts/wexac_logs/gb_pbox_flowers_%J.err
#BSUB -J gb_pbox_flowers
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.gradient_bridge.phase2_e2e \
    --dataset flowers32 --npc_list 1 2 --activations softplus gelu --pixel_box \
    --n_train 12000 --dec_epochs 70 --ext_epochs 10000 --rank 8 --device cuda
echo "=== DONE $(date) ==="
