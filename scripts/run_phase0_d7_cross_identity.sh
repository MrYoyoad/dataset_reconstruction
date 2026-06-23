#!/bin/bash
# D7: Cross-identity N>1 ablation.
#
# Tests whether the N>1 reconstruction win we saw with the same-person
# (D-N3 → SSIM 0.662) generalizes to different identities, where the
# gradient-mixing symmetry becomes nontrivial.
#
# 3 arms, each N=2, otherwise identical to the D3 winner config
# (signAdam, tv=1e-1, lr=0.05, freq=1e-3, 30K iters × 8 restarts, full
# ViT gradient, seed=42).
#
#   A — face1.jpg + flowers102:42 (cross-identity, BOTH label=0)
#       Tests H3 (same-class defuses) vs H2 (mixing wins).
#   B — face1.jpg + flowers102:42 (cross-identity, OPPOSITE labels 0/1)
#       Strongest stress test for mixing symmetry. Tests H2.
#   C — flowers102:42 + flowers102:100 (cross-subject, same-class)
#       Cleanest test of same-class-defuses without face-manifold help.
#
# Output: results/phase0_full_r8_n*_s42_<ts>_d7_<arm>.pth
# Logs:   wexac_logs/phase0_d7_<arm>_<jobid>.{out,err}

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p wexac_logs

# Common phase0 args (no image_path / image_labels — those are per-arm)
COMMON='--mode full --optimizer signAdam --tv_weight 1e-1 --tv_norm l2 --lr 0.05 --n_iters 30000 --n_restarts 8 --freq_weight 1e-3 --lpips_weight 0 --device cuda --seed 42'

submit() {
    # $1=arm tag, $2=image_path tokens, $3=image_labels (or "")
    local arm="$1" paths="$2" labels="$3"
    local tag="d7_${arm}"
    local jobname="phase0_${tag}"

    # Build the optional --image_labels arg (omitted when labels is empty)
    local label_arg=""
    if [[ -n "${labels}" ]]; then
        label_arg="--image_labels ${labels}"
    fi

    echo "Submitting arm ${arm}: paths='${paths}' labels='${labels:-(default 0,0)}'"
    bsub -q long-gpu \
         -R "rusage[mem=32768] select[ngpus>0]" \
         -gpu "num=1" \
         -W 18:00 \
         -o "wexac_logs/${jobname}_%J.out" \
         -e "wexac_logs/${jobname}_%J.err" \
         -J "${jobname}" \
         bash -c "
            source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
            conda activate /home/projects/galvardi/yoado/.conda/envs/rec
            cd /home/projects/galvardi/yoado
            python -u -m experiments.phase0_vit_inversion \\
                ${COMMON} \\
                --image_path '${paths}' \\
                ${label_arg} \\
                --run_tag ${tag}
         "
    sleep 20
}

# Arm A: cross-identity, both label=0
submit A "data/faces/face1.jpg,flowers102:42" ""

# Arm B: cross-identity, opposite labels
submit B "data/faces/face1.jpg,flowers102:42" "0,1"

# Arm C: same-class, two different flower subjects
submit C "flowers102:42,flowers102:100" ""

echo ""
echo "Submitted 3 D7 cross-identity jobs."
echo "Track:  bjobs -J 'phase0_d7_*'"
echo "Logs:   wexac_logs/phase0_d7_*_<jobid>.out"
echo ""
echo "After all complete, analyze with the N=3 cross-matrix renderer "
echo "(generalize from 3x3 to 2x2)."
