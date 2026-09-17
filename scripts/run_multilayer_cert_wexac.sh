#!/bin/bash
#BSUB -q short
#BSUB -R "rusage[mem=16384]"
#BSUB -W 2:00
#BSUB -o scripts/wexac_logs/mlcert_%J.out
#BSUB -e scripts/wexac_logs/mlcert_%J.err
#BSUB -J mlcert

# =====================================================================
# Multilayer certificate track (theory/T1..T6).
#
#   checks    numerical sanity checks for every theorem in theory/*.md. Each prints PASS/FAIL against a
#             PRE-STATED tolerance. The load-bearing one is T2 (Prop. A): the FULL certificate must annihilate
#             H_l^0 to machine zero at LARGE drift wherever rank B_T == N'. A FAIL there falsifies the track's
#             main claim -- do not "fix" it by loosening the tolerance.
#   survival  M1/M2/M3: layerwise drift (total AND orthogonal), training-span dimension N', certificate residual
#             (full and truncated), and the stacked chart-Jacobian rank, swept over lr x T.
#
# CPU is deliberate: everything here is small and FP64, and FP64 on a shared GPU is slower and noisier.
# =====================================================================
set +u                                   # conda activate breaks under set -u in this env (LESSONS_LEARNED)
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado

STAGE="${1:-checks}"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p results/multilayer_cert

case "$STAGE" in
  checks)
    python -u -m experiments.multilayer_cert.theory_checks --check all --device cpu \
        --out "results/multilayer_cert/theory_checks_${LSB_JOBID:-$TS}.jsonl"
    ;;
  survival)
    python -u -m experiments.multilayer_cert.survival --sweep --save --seeds 3 --device cpu \
        --out "results/multilayer_cert/survival_${LSB_JOBID:-$TS}.jsonl"
    ;;
  *)
    echo "usage: $0 {checks|survival}" ; exit 2 ;;
esac
