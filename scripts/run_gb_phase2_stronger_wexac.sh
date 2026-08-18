#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gb_phase2_stronger_%J.out
#BSUB -e scripts/wexac_logs/gb_phase2_stronger_%J.err
#BSUB -J gb_phase2_stronger

# Follow-up 1: attack the GB-Phase 2 input-layer bottleneck (softplus baseline decoded only 0.637 ->
# image SSIM 0.02). Stronger SINGLE-sample measurements (m=1 keeps it one image, no superposition):
#   - two-sided (nonzero A0: observes col(B0)+row(A0)) at r=8
#   - higher measurement rank r=32, r=64 (col(B0) ceiling rises as sqrt(r/out))
# If any lifts the input-layer decode + image SSIM, the bridge's failure is fixable; if it stays ~0.6,
# the input-layer bottleneck is fundamental.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

echo "########## ARM 1: two-sided, r=8 (softplus/gelu/relu) ##########"; date
python -u -m experiments.gradient_bridge.phase2_image \
    --activations softplus gelu relu --two_sided --rank 8 \
    --n_train 12000 --n_eval 128 --epochs 100 --device cuda

echo "########## ARM 2: higher rank single-sided (softplus) ##########"; date
for R in 32 64; do
    echo "-- r=$R --"
    python -u -m experiments.gradient_bridge.phase2_image \
        --activations softplus --rank $R \
        --n_train 12000 --n_eval 128 --epochs 100 --device cuda
done
echo "=== DONE $(date) ==="
