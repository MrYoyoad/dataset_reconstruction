#!/bin/bash
#BSUB -q long-gpu
#BSUB -gpu "num=1:j_exclusive=yes:gmodel=NVIDIAA100_SXM4"
#BSUB -R "rusage[mem=49152]"
#BSUB -W 4:00
#BSUB -o scripts/wexac_logs/realrank_%J.out
#BSUB -e scripts/wexac_logs/realrank_%J.err
#BSUB -J realrank

# =====================================================================
# Real-encoder test of the depth rank law (T5.2 vs F11's corrected law) on the deep 15-layer backbone.
# THEORY test, not an attack config (pixel arm k_1 ~ 692 is ~10x the k<=66 identifiability cap).
# Zero-drift certificate construction (no LoRA training -> no divergence gate to inherit).
# Dedicated EXCLUSIVE A100 for clean FP64: the rank ladder runs to 1e-12 and a shared GPU's FP64 noise would
# corrupt it. See experiments/multilayer_cert/real_encoder_ranklaw.py for the pre-registration.
# =====================================================================
set +u                                   # conda activate breaks under set -u in this env (LESSONS_LEARNED)
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado

python -u -m experiments.multilayer_cert.real_encoder_ranklaw \
    --model models/exact_inversion/mnist_mlp_d15w1000.pth \
    --N 8 --r 108 --first 1 3 --charts pixel chart66 --maxL 8 --seed 1 \
    --out "results/multilayer_cert/real_encoder_ranklaw_${LSB_JOBID:-manual}.jsonl"
