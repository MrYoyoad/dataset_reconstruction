#!/bin/bash
# Multi-seed face1 reconstruction at the D3 winner config.
# Establishes canonical SSIM mean±std for face1 (replaces the seed=42 anecdote).
# Also enables per-pixel median fusion across seeds — knocks down high-freq
# color speckle for free.
#
# 5 seeds × signAdam, tv=1e-1, lr=0.05, freq=1e-3, 30K iters × 8 restarts,
# full ViT gradient, face1.jpg. Each seed is a separate bsub job.
#
# Output: results/phase0_full_r8_n1_s<seed>_<timestamp>_face1_mseed_s<seed>.pth
# Logs:   wexac_logs/phase0_face1_mseed_s<seed>_<jobid>.out

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p wexac_logs

SEEDS=(7 13 42 99 2026)

for seed in "${SEEDS[@]}"; do
    tag="face1_mseed_s${seed}"
    jobname="phase0_${tag}"

    echo "Submitting seed ${seed}"
    bsub -q long-gpu \
         -R "rusage[mem=32768] select[ngpus>0]" \
         -gpu "num=1" \
         -W 12:00 \
         -o "wexac_logs/${jobname}_%J.out" \
         -e "wexac_logs/${jobname}_%J.err" \
         -J "${jobname}" \
         bash -c "
            source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
            conda activate /home/projects/galvardi/yoado/.conda/envs/rec
            cd /home/projects/galvardi/yoado
            python -u -m experiments.phase0_vit_inversion \\
                --mode full \\
                --image_path data/faces/face1.jpg \\
                --optimizer signAdam \\
                --tv_weight 1e-1 \\
                --tv_norm l2 \\
                --lr 0.05 \\
                --n_iters 30000 \\
                --n_restarts 8 \\
                --freq_weight 1e-3 \\
                --lpips_weight 0 \\
                --device cuda \\
                --seed ${seed} \\
                --run_tag ${tag}
         "
    sleep 20
done

echo ""
echo "Submitted 5 multi-seed face1 jobs (seeds: ${SEEDS[*]})."
echo "Track: bjobs -J 'phase0_face1_mseed_*'"
