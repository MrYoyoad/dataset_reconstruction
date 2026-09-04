#!/bin/bash
# =====================================================================
# THE SURVIVING CONTRIBUTION, TURNED FROM AN ARGUMENT INTO A MEASUREMENT (yoado-cd).
# Job 287241 gave the shadow attack EXACT distributional access -- its shadows came from the same pool as the
# target. That is the right control and the wrong description of a real attacker, so the whole applicability
# argument currently rests on a caveat rather than a number. This measures the degradation.
# Four levels of mismatch in the attacker's own pool, with the certificate unchanged throughout because it uses
# none of it:
#   same   shadows from the candidate pool          (exact access; the control)
#   near   same distribution, DISJOINT photographs  (a well-resourced attacker with a similar public pool)
#   far    CIFAR-100                                (natural photographs, different content and resolution)
#   gross  FashionMNIST                             (greyscale, not natural images)
# The candidate is present in the shadows at every level -- an attacker testing an image has that image. The
# mismatch is in the CO-TRAINING data around it, which is what they cannot obtain.
# PRE-REGISTERED: the certificate is FLAT across all four (it cannot move; it reads only the release). The shadow
# attack degrades monotonically with mismatch. The reported quantity is the mismatch level at which the shadow
# attack falls BELOW the certificate. If it never falls, the applicability claim is weak and we say so plainly --
# an attacker with a merely similar public pool would then have no reason to use this channel.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=49152] select[ngpus>0]" -J ei107_mismatch ...
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START LiRA distributional mismatch $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.head_lira --pool 256 --N 32 --r 64 --shadows 256 --k-cand 16 \
    --shadow-sources same near far gross --Ts 5 50 200 --lr 0.02 --seed 1 \
    --out $OUT/step107_mismatch_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
