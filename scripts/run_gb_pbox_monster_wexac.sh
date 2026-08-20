#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=65536] select[ngpus>0 && hname!='hgn46']"
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -o scripts/wexac_logs/gb_pbox_monster_%J.out
#BSUB -e scripts/wexac_logs/gb_pbox_monster_%J.err
#BSUB -J gb_pbox_monster
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.gradient_bridge.phase2_e2e \
    --dataset cifar10 --npc_list 1 --activations softplus gelu --pixel_box \
    --n_train 6000 --dec_epochs 60 --ext_epochs 8000 --rank 8 --device cuda
echo "=== DONE $(date) ==="
