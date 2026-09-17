#!/bin/bash
# WP0 (notes/plan_2026-09-18_cnn_ranklaw_newclass_charts.md): the base-training gate and the `_full` twins.
#
#   bash scripts/run_base_training_gate_wexac.sh gate [ckpt ...]   short-gpu   measure every WP0 checkpoint (or the listed ones)
#   bash scripts/run_base_training_gate_wexac.sh mlp_full           long-gpu    train models/exact_inversion/mnist_mlp_strong_full.pth
#   bash scripts/run_base_training_gate_wexac.sh conv_full          long-gpu    train models/exact_inversion/mnist_conv_deep_full.pth
#   bash scripts/run_base_training_gate_wexac.sh bottleneck         long-gpu    train models/exact_inversion/mnist_conv_bottleneck.pth from scratch
#   bash scripts/run_base_training_gate_wexac.sh d15_full           long-gpu    train models/exact_inversion/mnist_mlp_d15w1000_full.pth
#
# The gate rows go to results/base_training_gate.jsonl (one JSON line per checkpoint, appended, job id in the log name).
STAGE="${1:-gate}"; shift || true
cd /home/projects/galvardi/yoado
case "$STAGE" in
  gate)      Q=short-gpu; MEM=16384; JN=base_gate ;;
  mlp_full)  Q=long-gpu;  MEM=16384; JN=mlp_full ;;
  conv_full) Q=long-gpu;  MEM=16384; JN=conv_full ;;
  bottleneck) Q=long-gpu; MEM=16384; JN=bottleneck ;;
  d15_full)  Q=long-gpu;  MEM=24576; JN=d15_full ;;
  *) echo "usage: $0 {gate [ckpt ...]|mlp_full|conv_full}"; exit 2 ;;
esac
bsub -q "$Q" -gpu "num=1" -R "rusage[mem=${MEM}] select[ngpus>0]" -J "$JN" \
     -o "scripts/wexac_logs/${JN}_%J.out" -e "scripts/wexac_logs/${JN}_%J.err" <<JOB
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "# host \$(hostname)  job \$LSB_JOBID  git \$(git rev-parse --short HEAD)  stage ${STAGE}  \$(date)"
case "${STAGE}" in
  gate)
    python -u -m experiments.exact_inversion.base_training_gate --out results/base_training_gate.jsonl $@
    ;;
  mlp_full)
    python -u -m experiments.exact_inversion.train_strong_backbone \
        --init-from models/exact_inversion/mnist_mlp_strong.pth --out models/exact_inversion/mnist_mlp_strong_full.pth \
        --target-train-acc 0.995 --min-train-loss 1e-2 --max-epochs 300 --seed 0
    ;;
  conv_full)
    python -u -m experiments.exact_inversion.train_conv_backbone --spec deep \
        --init-from models/exact_inversion/mnist_conv_deep.pth --out models/exact_inversion/mnist_conv_deep_full.pth \
        --target-train-acc 0.995 --min-train-loss 1e-2 --max-epochs 300 --seed 1
    ;;
  bottleneck)
    python -u -m experiments.exact_inversion.train_conv_backbone --spec bottleneck \
        --out models/exact_inversion/mnist_conv_bottleneck.pth \
        --target-train-acc 0.995 --min-train-loss 1e-2 --max-epochs 300 --seed 1
    ;;
  d15_full)
    python -u -m experiments.exact_inversion.train_deep_backbone \
        --init-from models/exact_inversion/mnist_mlp_d15w1000.pth --out models/exact_inversion/mnist_mlp_d15w1000_full.pth \
        --target-train-acc 0.995 --min-train-loss 1e-2 --max-epochs 300 --seed 0
    ;;
esac
echo "# done \$(date)"
JOB
