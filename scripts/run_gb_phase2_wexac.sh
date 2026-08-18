#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gb_phase2_%J.out
#BSUB -e scripts/wexac_logs/gb_phase2_%J.err
#BSUB -J gb_phase2

# GB-Phase 2: the end-to-end bridge attack -- decoded input-layer gradient -> actual IMAGE.
# For layer 0 the single-sample gradient is rank-1 (dL/dW0 = g_err (x) x^T), so the decoded gradient's
# top right singular vector IS the image. Tests whether the bridge duality (softplus decodes best)
# actually yields a recovered image. softplus (expected high) vs gelu/relu.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# STAGE 0 smoke (new code): tiny run must produce a number, else abort
echo "########## STAGE 0 smoke ##########"
python -u -m experiments.gradient_bridge.phase2_image --activations softplus \
    --n_train 500 --n_eval 32 --epochs 5 --device cuda || { echo "STAGE 0 FAILED"; exit 1; }
echo "STAGE 0 PASSED"

echo "########## FULL RUN ##########"
python -u -m experiments.gradient_bridge.phase2_image \
    --activations softplus gelu relu --n_train 12000 --n_eval 128 --epochs 100 --device cuda
echo "=== DONE $(date) ==="
