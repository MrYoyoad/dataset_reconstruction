#!/bin/bash
# =====================================================================
# THE DEPTH-OF-FIRST-ADAPTATION CAP (yoado-cd, §18). Everything measured so far starts adaptation at the
# PIXEL-INPUT layer. Real LoRA does not -- adapters sit on blocks whose input is already a feature. The measured
# encoder profile (784, 784, 692, 219, 187, 138, 104, ... 19) then says the whole stack is capped by what the
# frozen path below the FIRST adapted layer transmits, whatever the rank and however many layers follow.
# Adapt layers d..d+3 with everything below FROZEN, for d = 1, 2, 4, 7.
# PRE-REGISTERED: the saturating pixel count equals the encoder's transmitted rank at depth d -- 784, 784, ~219,
# ~104 -- and NOT 784 in the deep cells. If it holds, the honest headline is "the release determines the image
# only when adaptation reaches near the input", which is actionable for a defender. sigma_min is reported beside
# every saturation number: a cap reached at 1e-10 and one reached at 1e-2 are not the same result.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=49152] select[ngpus>0]" -J ei97_depthfirst \
#        -o scripts/wexac_logs/ei97_depthfirst_%J.out -e scripts/wexac_logs/ei97_depthfirst_%J.err \
#        bash scripts/run_depth_of_first_adaptation_wexac.sh
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START depth-of-first-adaptation $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
MODEL=models/exact_inversion/mnist_mlp_d15w1000.pth
for D in 1 2 4 7; do
  echo "=== first adapted layer d=$D (layers $D..$((D+3)) adapted, everything below FROZEN) ==="
  python -u -m experiments.exact_inversion.layer_curve --model $MODEL --N 8 --r 256 --k 16 --T 100 --lr 0.01 \
      --seed 1 --adapt $D $((D+1)) $((D+2)) $((D+3)) --layers $D $((D+1)) $((D+2)) $((D+3)) \
      --out $OUT/step97_depthfirst_${LSB_JOBID}.jsonl
done
echo "=== DONE $(date) ==="
