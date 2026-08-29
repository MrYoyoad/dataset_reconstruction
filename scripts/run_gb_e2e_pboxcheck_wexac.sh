#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0 && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gb_e2e_pboxcheck_%J.out
#BSUB -e scripts/wexac_logs/gb_e2e_pboxcheck_%J.err
#BSUB -J gb_pboxcheck
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
# verify pixel_box: clip should drop ~0, raw ssim should become trustworthy (near ssim_norm)
python -u -m experiments.gradient_bridge.phase2_e2e \
    --dataset mnist --npc_list 1 --activations softplus --pixel_box \
    --n_train 6000 --dec_epochs 50 --ext_epochs 8000 --rank 8 --device cuda
echo "=== DONE $(date) ==="
