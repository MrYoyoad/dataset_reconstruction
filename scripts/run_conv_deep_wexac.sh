#!/bin/bash
# =====================================================================
# CONV, corrected scope (audit 02b93e8 from yoado-cd/81): saturation is a COMPUTABLE CONDITION, not an
# architectural fact -- the patch span fills the layer only where N x positions >= C_in.k^2, which is true in
# early conv layers and FALSE in deep ones. So the shallow cells measured one regime and the claim must not be
# stated unqualified. Two things are separated here, which the earlier run conflated:
#   SATURATION  the recorded patch span fills the input dimension        -> no margin at any rank
#   DRIFT       A_0 h leaves row(B_T) because upstream layers were adapted -> margin exists, condition FAILS
# The solo arms adapt ONE layer with everything else frozen, so the adapted layer's input never moves and drift
# is switched off by construction. deep spec: d = 9, 576, 1152, 2304 against N*P = 1568, 392, 128, 32.
# PRE-REGISTERED: in the solo arms, layers 3 and 4 (N*P < d) are NON-VACUOUS and their certificate HOLDS at the
# truth to ~1e-15, with margin ~ r - patch_span_rank; layer 1 stays vacuous at every rank (span 9 = d = 9). In the
# all-adapted arms the deep layers keep a margin but the residual FAILS, as in the shallow run. If the solo deep
# layers also fail, saturation was never the mechanism and the conv claim rests on drift alone.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=32768] select[ngpus>0]" -J ei93_convdeep \
#        -o scripts/wexac_logs/ei93_convdeep_%J.out -e scripts/wexac_logs/ei93_convdeep_%J.err \
#        bash scripts/run_conv_deep_wexac.sh
# =====================================================================
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START conv deep-regime $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.conv_certificate --spec deep --arms solo all \
    --N 8 --ranks 16 64 128 256 --T 200 --lr 0.01 --epochs 8 --seed 1 \
    --out $OUT/step93_convdeep_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
