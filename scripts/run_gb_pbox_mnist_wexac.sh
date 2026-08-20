#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0 && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gb_pbox_mnist_%J.out
#BSUB -e scripts/wexac_logs/gb_pbox_mnist_%J.err
#BSUB -J gb_pbox_mnist
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.gradient_bridge.phase2_e2e \
    --dataset mnist --npc_list 1 2 5 --activations softplus gelu --pixel_box \
    --n_train 15000 --dec_epochs 90 --ext_epochs 12000 --rank 8 --device cuda
echo "=== DONE $(date) ==="
