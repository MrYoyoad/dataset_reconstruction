#!/bin/bash
# One bsub per (example, chart) cell: identical solver and 400 starts everywhere, nothing tuned between cells, and
# the convolutional example's charts run in parallel instead of serialising into a ~17-hour job.
#   bash experiments/oracle_ladder/submit_ladder.sh [example ...]        default: mlp_motorcycle cnn_keyboard
#   examples: mlp_motorcycle cnn_keyboard mlp_letter_a mlp_letter_a_full
# Environment overrides (for smoke cells): CHARTS="oracle:0" restricts the charts (space-separated list of
#   oracle:<eps> | pca | wrong), EXTRA="--starts 20 --out-dir results/oracle_ladder/smoke" is appended to every cell,
#   QUEUE forces the queue (default: long-gpu for cnn_keyboard, short-gpu otherwise).
cd /home/projects/galvardi/yoado; mkdir -p results/oracle_ladder scripts/wexac_logs
EPS="0 0.01 0.02 0.03 0.05 0.075 0.10 0.15 0.20 0.30 0.40 0.60"
EXAMPLES=${@:-mlp_motorcycle cnn_keyboard}
DEFAULT_CHARTS="$(for E in $EPS; do echo -n "oracle:$E "; done)pca wrong"
CHARTS=${CHARTS:-$DEFAULT_CHARTS}

submit() {   # submit <example> <jobtag> <cell args...>
  local EX=$1 TAG=$2; shift 2
  local Q=${QUEUE:-$([ "$EX" = "cnn_keyboard" ] && echo long-gpu || echo short-gpu)}
  bsub -q $Q -gpu "num=1" -R "rusage[mem=24576] select[ngpus>0]" -J ol_${EX}_${TAG} \
       -o scripts/wexac_logs/ol_${EX}_${TAG}_%J.out -e scripts/wexac_logs/ol_${EX}_${TAG}_%J.err <<JOB | grep -o "Job <[0-9]*>"
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
python -u -m experiments.oracle_ladder.ladder_cell --example ${EX} $@ ${EXTRA}
JOB
}

for EX in $EXAMPLES; do
  for C in $CHARTS; do
    case $C in
      oracle:*) submit $EX e${C#oracle:} --chart $C ;;
      pca)      submit $EX pca --chart pca ;;
      wrong)    submit $EX wr --chart oracle:0 --wrong_release ;;
      *)        echo "unknown chart $C" >&2 ;;
    esac
  done
done
