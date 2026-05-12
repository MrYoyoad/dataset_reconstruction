#!/bin/bash
# Submit the Phase 0 face-prior ablation grid on face1.jpg.
#
# 9 unique configs (E2/E3/F2 dedup against C/D/D):
#   A  control     tv=1e-1, cos=1.0, face=0       (D3 winner re-run)
#   B  face-only   tv=0,    cos=1.0, face=1e-1
#   C  TV+face/low tv=1e-1, cos=1.0, face=1e-2    (= E2)
#   D  TV+face/hi  tv=1e-1, cos=1.0, face=1e-1    (= E3 = F2)
#   E1 strength    tv=1e-1, cos=1.0, face=1e-3
#   E4 strength    tv=1e-1, cos=1.0, face=1.0
#   F1 cos sweep   tv=1e-1, cos=0.5, face=1e-1
#   F3 cos sweep   tv=1e-1, cos=2.0, face=1e-1
#   F4 cos sweep   tv=1e-1, cos=5.0, face=1e-1
#
# Each arm: 30k iters, 8 restarts, signAdam, lr=0.05, freq=1e-3, full ViT
# gradient, seed 42 on face1.jpg. Each is a separate bsub job (parallel GPUs).
#
# Usage:
#   ./scripts/run_phase0_face_prior_sweep.sh
#
# Output:
#   results/phase0_full_r8_n1_s42_<timestamp>_face_prior_<arm>.pth (per arm)
#   wexac_logs/phase0_face_prior_<arm>_<jobid>.out
#
# After all arms finish, run:
#   python -m experiments.analyze_face_prior_sweep
# to generate figures and the metrics CSV.

set -euo pipefail

cd "$(dirname "$0")/.."

# Common bsub args
BSUB_COMMON='-q long-gpu -R "rusage[mem=32768] select[ngpus>0]" -gpu "num=1" -W 12:00'

# Common phase0 args
COMMON_ARGS='--mode full --image_path data/faces/face1.jpg --optimizer signAdam --tv_norm l2 --lr 0.05 --n_iters 30000 --n_restarts 8 --freq_weight 1e-3 --lpips_weight 0 --device cuda --seed 42'

submit() {
    # $1=arm tag, $2=tv_weight, $3=cos_weight, $4=face_weight
    local arm="$1" tv="$2" cosw="$3" facew="$4"
    local tag="face_prior_${arm}"
    local jobname="phase0_${tag}"
    local out="wexac_logs/${jobname}_%J.out"
    local err="wexac_logs/${jobname}_%J.err"

    echo "Submitting arm ${arm}: tv=${tv}, cos=${cosw}, face=${facew}"
    bsub -q long-gpu -R "rusage[mem=32768] select[ngpus>0]" -gpu "num=1" -W 48:00 \
         -o "${out}" -e "${err}" -J "${jobname}" \
         bash -c "
            source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
            conda activate /home/projects/galvardi/yoado/.conda/envs/rec
            cd /home/projects/galvardi/yoado
            python -u -m experiments.phase0_vit_inversion \\
                ${COMMON_ARGS} \\
                --tv_weight ${tv} \\
                --cos_weight ${cosw} \\
                --face_weight ${facew} \\
                --face_layout_weight 1.0 \\
                --face_sym_weight 0.5 \\
                --face_warmup_iters 5000 \\
                --face_ramp_iters 2000 \\
                --face_model auto \\
                --run_tag ${tag}
         "
    # Spread jobs across hosts: give LSF a moment to refresh host load
    # between submissions so jobs don't all land on the same node.
    sleep 30
}

mkdir -p wexac_logs

# Arm: tv      cos    face
submit A    1e-1   1.0   0
submit B    0      1.0   1e-1
submit C    1e-1   1.0   1e-2
submit D    1e-1   1.0   1e-1
submit E1   1e-1   1.0   1e-3
submit E4   1e-1   1.0   1.0
submit F1   1e-1   0.5   1e-1
submit F3   1e-1   2.0   1e-1
submit F4   1e-1   5.0   1e-1

echo ""
echo "Submitted 9 face-prior ablation jobs."
echo "Track: bjobs -J 'phase0_face_prior_*'"
echo "Logs:  wexac_logs/phase0_face_prior_*_<jobid>.out"
echo ""
echo "After all complete, run:"
echo "  python -m experiments.analyze_face_prior_sweep"
