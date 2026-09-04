#!/bin/bash
# =====================================================================
# The EXTENDED-LAYER CURVE: independent constraints on RAW PIXELS vs adapted depth.
# Three points (58 -> 102 -> 158 at 1/2/3 adapted layers) are three points; this builds the curve.
#   bash scripts/run_layer_curve_wexac.sh [depth] [width]
# Stage 1 trains a deep frozen MNIST backbone (plain GELU; retries with --residual if it will not train),
# stage 2 measures the stacked certificate Jacobian's pixel rank at L = 1,2,3,4,6,8,10,12,14,depth,
# beside rank DF_1 (the architecture's ceiling), sigma_min at the rank, and the usable rank.
# Submit:
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=16384] select[ngpus>0]" -J ei89_curve \
#        -o scripts/wexac_logs/ei89_curve_%J.out -e scripts/wexac_logs/ei89_curve_%J.err \
#        bash scripts/run_layer_curve_wexac.sh 15 1000
# =====================================================================
set +u
DEPTH=${1:-15}; WIDTH=${2:-1000}
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
python -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"} torch={torch.__version__}')"
echo "=== START layer_curve depth=$DEPTH width=$WIDTH $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="

MODEL=models/exact_inversion/mnist_mlp_d${DEPTH}w${WIDTH}.pth
python -u -m experiments.exact_inversion.train_deep_backbone --depth $DEPTH --width $WIDTH \
    --out-dir models/exact_inversion
RC=$?
if [ $RC -eq 3 ]; then
  echo "=== plain depth $DEPTH would not train; retrying WITH residual (changes the pixel Jacobian, flagged) ==="
  python -u -m experiments.exact_inversion.train_deep_backbone --depth $DEPTH --width $WIDTH --residual \
      --out-dir models/exact_inversion
  MODEL=models/exact_inversion/mnist_mlp_d${DEPTH}w${WIDTH}_res.pth
elif [ $RC -ne 0 ]; then
  echo "=== backbone training FAILED (rc=$RC) -- no curve ==="; exit $RC
fi

echo "=== CURVE on $MODEL $(date) ==="
python -u -m experiments.exact_inversion.layer_curve --model $MODEL --N 8 --r 64 --k 16 --T 400 --lr 0.01 \
    --seed 1 --out $OUT/step89_layercurve_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
