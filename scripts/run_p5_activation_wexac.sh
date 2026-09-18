#!/bin/bash
# P5 (notes/plan_2026-09-18_multilayer_parameter_program.md, audit item 8): ReLU / tanh twins of the two MNIST
# backbones the multilayer track uses, trained from scratch to the base gate (BASE_TRAINING_GATE.md) with the
# activation recorded in the checkpoint as "act".  Same seed / lr / bs as the originals, only the activation differs:
#   strong  784-1000-1000-10   seed 0, Adam 1e-3, bs 128   (mnist_mlp_strong.pth: 30 epochs, no rule)
#   d15     depth 15 width 1000 seed 0, Adam 3e-4, bs 128   (mnist_mlp_d15w1000.pth: 40 epochs, no rule)
#
#   bash scripts/run_p5_activation_wexac.sh p5_train [name ...]    long-gpu  one job per name; default all four:
#                                                                  strong_relu strong_tanh d15_relu d15_tanh
#       EXTRA="--lr 1e-3" bash scripts/run_p5_activation_wexac.sh p5_train d15_tanh     # lr override for a stalled twin
#   bash scripts/run_p5_activation_wexac.sh p5_gate [ckpt ...]     short-gpu base_training_gate on the twins that exist
#
# Outputs: models/exact_inversion/mnist_mlp_strong_{relu,tanh}.pth, mnist_mlp_d15w1000_{relu,tanh}_full.pth;
# gate rows appended to results/base_training_gate.jsonl; logs scripts/wexac_logs/p5_<name>_<jobid>.{out,err}.
STAGE="${1:-p5_train}"; shift || true
cd /home/projects/galvardi/yoado
RULE="--target-train-acc 0.995 --min-train-loss 1e-2 --max-epochs 300 --seed 0"
MODELS=models/exact_inversion

submit() {   # submit <job-name> <queue> <mem MB> <gpu string> <command...>
  local JN="$1" Q="$2" MEM="$3" GPU="$4"; shift 4
  bsub -q "$Q" -gpu "$GPU" -R "rusage[mem=${MEM}] select[ngpus>0]" -J "$JN" \
       -o "scripts/wexac_logs/${JN}_%J.out" -e "scripts/wexac_logs/${JN}_%J.err" <<JOB
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "# host \$(hostname)  job \$LSB_JOBID  git \$(git rev-parse --short HEAD)  stage ${STAGE}  \$(date)"
$@
echo "# done \$(date)"
JOB
}

case "$STAGE" in
  p5_train)
    NAMES="${@:-strong_relu strong_tanh d15_relu d15_tanh}"
    for NAME in $NAMES; do
      case "$NAME" in
        strong_relu) CMD="python -u -m experiments.exact_inversion.train_strong_backbone --act relu --out $MODELS/mnist_mlp_strong_relu.pth $RULE"; MEM=16384 ;;
        strong_tanh) CMD="python -u -m experiments.exact_inversion.train_strong_backbone --act tanh --out $MODELS/mnist_mlp_strong_tanh.pth $RULE"; MEM=16384 ;;
        d15_relu)    CMD="python -u -m experiments.exact_inversion.train_deep_backbone --depth 15 --width 1000 --act relu --out $MODELS/mnist_mlp_d15w1000_relu_full.pth $RULE"; MEM=24576 ;;
        d15_tanh)    CMD="python -u -m experiments.exact_inversion.train_deep_backbone --depth 15 --width 1000 --act tanh --out $MODELS/mnist_mlp_d15w1000_tanh_full.pth $RULE"; MEM=24576 ;;
        *) echo "unknown twin '$NAME' (strong_relu|strong_tanh|d15_relu|d15_tanh)"; exit 2 ;;
      esac
      submit "p5_$NAME" long-gpu "$MEM" "num=1:gmem=20G" "$CMD ${EXTRA:-}"
    done ;;
  p5_gate)
    if [ $# -gt 0 ]; then CKPTS="$@"; else
      CKPTS=""; for F in mnist_mlp_strong_relu mnist_mlp_strong_tanh mnist_mlp_d15w1000_relu_full mnist_mlp_d15w1000_tanh_full; do
        [ -f "$MODELS/$F.pth" ] && CKPTS="$CKPTS $MODELS/$F.pth" || echo "# p5_gate: $MODELS/$F.pth not present, skipped"; done
    fi
    [ -z "$CKPTS" ] && { echo "p5_gate: no checkpoints to gate"; exit 1; }
    submit p5_gate short-gpu 16384 "num=1" "python -u -m experiments.exact_inversion.base_training_gate --out results/base_training_gate.jsonl --ckpts $CKPTS" ;;
  *) echo "usage: $0 {p5_train [name ...]|p5_gate [ckpt ...]}"; exit 2 ;;
esac
