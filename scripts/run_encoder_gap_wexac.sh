#!/bin/bash
# =====================================================================
# WHAT THE ENCODER COSTS (audit from yoado-cd; job 202172 withdrawn as an identity).
# With the adapted layer taking RAW PIXELS the condition C x = 0 is linear, so its pixel Jacobian IS C and the
# count is rank(C) = r - N' by construction. Nothing is measured. Put a FROZEN NONLINEAR ENCODER below the
# adapted layer and the condition becomes C phi(x), the pixel Jacobian is C.Dphi(x), and its rank can be strictly
# less than r - N'. THAT GAP is the measurement.
# Each cell adapts exactly ONE layer with everything else frozen, so the encoder is frozen and nonlinear and the
# adapted layer's input cannot drift. rank Dphi is measured at the truth in the same run and printed beside it.
# PRE-REGISTERED: gap = 0 wherever Dphi is full rank at the truth, and grows with any rank deficiency in the
# encoder. Since rank(C.Dphi) <= min(r - N', rank Dphi), the gap MUST open once r - N' exceeds rank Dphi -- so the
# informative cells are the high ranks, and the number to report is where the pixel rank stops tracking r - N'.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=32768] select[ngpus>0]" -J ei95_encgap \
#        -o scripts/wexac_logs/ei95_encgap_%J.out -e scripts/wexac_logs/ei95_encgap_%J.err \
#        bash scripts/run_encoder_gap_wexac.sh
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START encoder-gap $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
MODEL=models/exact_inversion/mnist_mlp_d15w1000.pth
for LAYER in 2 3 5 8; do
  for R in 64 256 600 900; do
    echo "=== adapt layer $LAYER, r=$R (encoder = $((LAYER-1)) frozen nonlinear layers) ==="
    python -u -m experiments.exact_inversion.layer_curve --model $MODEL --N 8 --r $R --k 16 --T 100 --lr 0.01 \
        --seed 1 --adapt $LAYER --layers $LAYER --out $OUT/step95_encgap_${LSB_JOBID}.jsonl
  done
done
echo "=== control: adapt layer 1 (NO encoder, the withdrawn identity cell) ==="
python -u -m experiments.exact_inversion.layer_curve --model $MODEL --N 8 --r 600 --k 16 --T 100 --lr 0.01 \
    --seed 1 --adapt 1 --layers 1 --out $OUT/step95_encgap_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
