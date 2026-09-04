#!/bin/bash
# =====================================================================
# THE HEAD COMPARISON AS A TRAJECTORY, replacing the band that voided every cell of job 285127.
# There, the certificate scored AUC 1.000 while the trivial loss baseline sat at chance, because the release had
# barely trained -- so every cell fell outside the pre-registered 0.6-0.9 band and was VOID. The band's lower
# bound was justified as "nothing is detectable", which is false in that cell. Amending the rule after seeing the
# numbers would be post-hoc, so the cells stand VOID and this is the replacement design.
# Sweep the training length so the baseline WALKS from chance, through the informative band, to saturation, and
# report the certificate's AUC along the whole path. The comparison is read where the band says it is informative;
# the rest of the curve is shown rather than discarded.
# PRE-REGISTERED before the run: certificate near 1.0 while the baseline passes through 0.6-0.9 => a statistic
# from the deterministic channel beats the standard membership baseline where that baseline is informative.
# Certificate degrading as the baseline improves => the two measure the same thing and there is no claim.
# N = 32 at r = 64 keeps the certificate non-vacuous (span N < r); N = 64 runs as the vacuous control.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=49152] select[ngpus>0]" -J ei105_headtraj ...
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START head trajectory $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
for T in 5 20 50 100 200 400; do
  for N in 32 64; do
    echo "=== N=$N T=$T lr=0.02 ==="
    python -u -m experiments.exact_inversion.graded_imprint --module head --r 64 \
        --N $N --n-nonmember 128 --T $T --lr 0.02 --seed 1 --baseline-band 0.6 0.9 \
        --out $OUT/step105_headtraj_${LSB_JOBID}.jsonl
  done
done
echo "=== DONE $(date) ==="
