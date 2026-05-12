#!/bin/bash
# Chroma-coupled TV (LAB-space) on face1.jpg — attacks the colored speckle
# that spatial-only TV leaves behind. LAB-space TV with heavier penalty on
# a/b (chroma) than L (luminance). Natural images have smooth chroma;
# colored speckle violates this where pure RGB-TV cannot see it.
#
# Same backbone as D3 winner: signAdam, tv=1e-1, lr=0.05, freq=1e-3,
# 30K iters × 8 restarts, full ViT gradient, face1.jpg, seed=42.
# Only diff: --tv_norm=lab and --tv_chroma_weight (sweep 2 values for safety).

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p wexac_logs

submit() {
    local arm="$1"; local chroma="$2"
    local tag="face1_chroma_tv_cw${chroma}"
    local jobname="phase0_${tag}"

    echo "Submitting arm ${arm}: chroma_weight=${chroma}"
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
                --tv_norm lab \\
                --tv_chroma_weight ${chroma} \\
                --lr 0.05 \\
                --n_iters 30000 \\
                --n_restarts 8 \\
                --freq_weight 1e-3 \\
                --lpips_weight 0 \\
                --device cuda \\
                --seed 42 \\
                --run_tag ${tag}
         "
    sleep 20
}

# Two chroma-weight settings: 5 (default) and 20 (aggressive). Both bracket
# the natural range. If 20 over-flattens, 5 is the fallback.
submit C5  5
submit C20 20

echo ""
echo "Submitted 2 chroma-TV jobs on face1.jpg."
echo "Track: bjobs -J 'phase0_face1_chroma_tv_*'"
