#!/bin/bash
# Bootstrap / iterative chart (plan 2026-09-18 WP4, round2 TEST 8): experiments/bootstrap_chart/bootstrap.py
#
#   bash scripts/run_bootstrap_chart_wexac.sh smoke   # MNIST only, 10 starts, 1 round of each variant, short-gpu (minutes)
#   bash scripts/run_bootstrap_chart_wexac.sh full    # one long-gpu job per release (mnist, cifar), 200 starts per global chart
#
# Base model (WP0 / audit 2026-09-18): mnist_mlp_strong.pth passed the base gate in job 355833 (99.83 % train, CE 6.3e-3)
# and is used as is -- no `_full` twin. The CNN checkpoint is measured in-job (train / test / CE stamped on every row).
# Budget (measured per-start cost, ladder jobs 435295 / 435321): MLP ~0.75 s, CNN ~25 s per LM start -> the CNN's local
# (variant-B) charts get 2 random starts + 1 warm start per slot; the MLP gets 8 + 1.
cd /home/projects/galvardi/yoado; mkdir -p results/bootstrap_chart figures/bootstrap_chart scripts/wexac_logs
STAGE="${1:-smoke}"

submit() {   # $1 = job name, $2 = queue, $3 = mem MB, $4... = python args
  local NAME=$1 Q=$2 MEM=$3; shift 3
  bsub -q "$Q" -gpu "num=1" -R "rusage[mem=$MEM] select[ngpus>0]" -J "$NAME" \
       -o "scripts/wexac_logs/${NAME}_%J.out" -e "scripts/wexac_logs/${NAME}_%J.err" <<JOB | grep -o "Job <[0-9]*>"
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "# host \$(hostname)  job \$LSB_JOBID  git \$(git rev-parse --short HEAD)  stage $STAGE  \$(date)"
python -m py_compile experiments/bootstrap_chart/bootstrap.py || exit 3
python -u -m experiments.bootstrap_chart.bootstrap $* --out results/bootstrap_chart/rounds_\${LSB_JOBID}.jsonl
echo "# DONE \$(date)"
JOB
}

case "$STAGE" in
  smoke)
    submit bsc_smoke short-gpu 16384 --releases mnist --starts 10 --local-starts 2 --a-rounds 1 --b-rounds 1 --clf-epochs 1 --K 200
    ;;
  full)
    submit bsc_mnist long-gpu 32768 --releases mnist --starts 200 --local-starts 8 --a-rounds 2 --b-rounds 4 --K 200
    submit bsc_cifar long-gpu 32768 --releases cifar --starts 200 --local-starts 2 --a-rounds 2 --b-rounds 4 --K 200
    ;;
  *)
    echo "usage: $0 {smoke|full}"; exit 2 ;;
esac
