#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=65536]"
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -o scripts/wexac_logs/gb_e2e_monster_%J.out
#BSUB -e scripts/wexac_logs/gb_e2e_monster_%J.err
#BSUB -J gb_e2e_monster
# Bridge on the MONSTER (3072-2048x4-1, 5 layers). EXCLUSIVE GPU (the 2048-wide decoders need ~3GB each
# and the shared card OOM'd). CIFAR 50k proxy -> use n_train 20000 for the hard 3072-dim input decoder.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
python -u -m experiments.gradient_bridge.phase2_e2e \
    --dataset cifar10 --npc_list 1 --activations softplus gelu \
    --n_train 20000 --dec_epochs 60 --ext_epochs 8000 --rank 8 --dec_batch 64 --device cuda
echo "=== DONE $(date) ==="
