#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gb_phase2_e2e_%J.out
#BSUB -e scripts/wexac_logs/gb_phase2_e2e_%J.err
#BSUB -J gb_phase2_e2e
# CULMINATION of the bridge track: real decoded per-layer delta_w -> Experiment-B extraction (NOT SVD).
# The corruption proxy predicted SSIM ~0.5 from the model-based inverter; this uses the REAL trained
# per-layer decoders end-to-end. Arms: TRUE delta_w (ceiling) / DECODED all-layers (the attack) /
# DECODED input-only / TRUE input-only. If DECODED all-layers nears the TRUE ceiling, the bridge closes.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
python -u -m experiments.gradient_bridge.phase2_e2e \
    --activations softplus gelu --npc 1 --seed 42 \
    --n_train 15000 --dec_epochs 90 --ext_epochs 12000 --rank 8 --device cuda
echo "=== DONE $(date) ==="
