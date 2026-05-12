#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=49152] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 16:00
#BSUB -o wexac_logs/phase0_face_n3_same_%J.out
#BSUB -e wexac_logs/phase0_face_n3_same_%J.err
#BSUB -J phase0_face_n3_same

# N=3 same-person reconstruction at the D3 winner config.
# face1.jpg, face2.jpg, face3.jpg are all photos of the same person.
# All labels=0 (single-class threat model — attacker fine-tunes a face-recognition
# LoRA on multiple photos of one identity). The interesting question: do we
# recover three distinct photos or a single superposed face?

set -e
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado

echo "=== Phase 0: N=3 same-person face reconstruction (D3 winner config) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

python -u -m experiments.phase0_vit_inversion \
    --mode full \
    --image_path "data/faces/face1.jpg,data/faces/face2.jpg,data/faces/face3.jpg" \
    --optimizer signAdam \
    --tv_weight 1e-1 \
    --tv_norm l2 \
    --lr 0.05 \
    --n_iters 30000 \
    --n_restarts 8 \
    --freq_weight 1e-3 \
    --lpips_weight 0 \
    --device cuda \
    --seed 42 \
    --run_tag face_n3_same_d3winner

echo ""
echo "=== N=3 same-person reconstruction complete: $(date) ==="
