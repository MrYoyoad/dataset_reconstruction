#!/bin/bash
# =====================================================================
# The graded comparison with a PRE-REGISTERED VALIDITY BAND (yoado-cd/81, §22 follow-up).
# Job 280255 scored a baseline of AUC 1.000, which cannot be beaten, only tied -- so it measured the baseline's
# ceiling rather than our statistic. The fix is to fix the band in advance: the trivial loss-threshold baseline
# must land in 0.6-0.9 AUC for a cell to COUNT. Outside it the cell is VOID and is not scored, and the band is
# reported beside every row so a reader can see which regime it sat in.
# Sweeps private-set size and step count to LAND in the band rather than hoping to, and runs the HEAD arm --
# the one non-weight-shared module, where the certificate itself is non-vacuous and therefore serves as the
# positive control for the graded statistic as well as being the deployment-relevant attack cell.
# The graded score is a MEMBERSHIP statistic; its comparator is the membership literature's baselines, not any
# reconstruction metric.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=49152] select[ngpus>0]" -J ei103_banded ...
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START graded, banded, head + qkv $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
for MOD in head qkv; do
  for CELL in "64 5 0.002" "64 10 0.005" "128 5 0.002" "128 10 0.002"; do
    set -- $CELL
    echo "=== module=$MOD N=$1 T=$2 lr=$3 ==="
    python -u -m experiments.exact_inversion.graded_imprint --module $MOD --block 6 --r 8 16 64 \
        --N $1 --n-nonmember 128 --T $2 --lr $3 --seed 1 --baseline-band 0.6 0.9 \
        --out $OUT/step103_banded_${LSB_JOBID}.jsonl
  done
done
echo "=== DONE $(date) ==="
