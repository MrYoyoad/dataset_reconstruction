#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 36:00
#BSUB -o wexac_logs/phase0_face_sweep_%J.out
#BSUB -e wexac_logs/phase0_face_sweep_%J.err
#BSUB -J phase0_face_sweep

set -e

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
mkdir -p results figures scripts/wexac_logs

pip install -q scipy 2>/dev/null

echo "=== Phase 0: Face Hyperparameter Sweep (face1 only) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

FACE="data/faces/face1.jpg"

# F1: D1 winner baseline (signAdam, tv=1e-2, 10K iters)
echo ""
echo "=== F1: signAdam, tv=1e-2, 10K iters (D1 winner) ==="
echo "Date: $(date)"
python -u -m experiments.phase0_vit_inversion \
    --mode full --image_path "$FACE" \
    --n_iters 10000 --n_restarts 8 \
    --optimizer signAdam --tv_weight 1e-2 --tv_norm l2 \
    --device cuda --seed 42

# F2: Stronger TV (faces are smoother than flowers)
echo ""
echo "=== F2: signAdam, tv=5e-2, 10K iters ==="
echo "Date: $(date)"
python -u -m experiments.phase0_vit_inversion \
    --mode full --image_path "$FACE" \
    --n_iters 10000 --n_restarts 8 \
    --optimizer signAdam --tv_weight 5e-2 --tv_norm l2 \
    --device cuda --seed 42

# F3: More iterations (faces have more detail)
echo ""
echo "=== F3: signAdam, tv=1e-2, 30K iters ==="
echo "Date: $(date)"
python -u -m experiments.phase0_vit_inversion \
    --mode full --image_path "$FACE" \
    --n_iters 30000 --n_restarts 8 \
    --optimizer signAdam --tv_weight 1e-2 --tv_norm l2 \
    --device cuda --seed 42

# F4: Stronger TV + more iterations
echo ""
echo "=== F4: signAdam, tv=5e-2, 30K iters ==="
echo "Date: $(date)"
python -u -m experiments.phase0_vit_inversion \
    --mode full --image_path "$FACE" \
    --n_iters 30000 --n_restarts 8 \
    --optimizer signAdam --tv_weight 5e-2 --tv_norm l2 \
    --device cuda --seed 42

echo ""
echo "=== Face Sweep Complete ==="
echo "Date: $(date)"
