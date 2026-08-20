#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=49152] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gb_e2e_monster_%J.out
#BSUB -e scripts/wexac_logs/gb_e2e_monster_%J.err
#BSUB -J gb_e2e_monster
# Bridge attack on the MONSTER: wide+deep CIFAR-10 net (3072-2048x4-1, 19M, 5 layers). CIFAR 50k proxy
# = abundant (no flowers-style starvation), so this isolates the NETWORK-SCALE variable.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
python -u -m experiments.gradient_bridge.phase2_e2e \
    --dataset cifar10 --npc_list 1 --activations softplus gelu \
    --n_train 8000 --dec_epochs 60 --ext_epochs 8000 --rank 8 --device cuda
echo "=== DONE $(date) ==="
