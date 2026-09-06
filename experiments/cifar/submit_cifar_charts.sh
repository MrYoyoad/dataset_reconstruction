#!/bin/bash
# One bsub job per cell of experiments/cifar/cifar_charts.py.
#   bash experiments/cifar/submit_cifar_charts.sh <layer> <chart> <solver> <private> <k> [extra args]
L=$1; CH=$2; SO=$3; PR=$4; K=$5; shift 5; EXTRA="$@"
TAG=$(echo "$EXTRA" | tr -d " -" ); NAME=cc_L${L}_${CH}_${SO}_${PR}_k${K}${TAG:+_$TAG}; OUT=experiments/cifar/charts/L${L}_${CH}_${SO}_${PR}_k${K}${TAG:+_$TAG}
cd /home/projects/galvardi/yoado; mkdir -p experiments/cifar/charts
bsub -q long-gpu -gpu "num=1" -R "rusage[mem=16384] select[ngpus>0]" -J ${NAME} \
     -o scripts/wexac_logs/${NAME}_%J.out -e scripts/wexac_logs/${NAME}_%J.err <<JOB
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "# host \$(hostname)  job \$LSB_JOBID  git \$(git rev-parse --short HEAD)  ${NAME} ${EXTRA}"
python -u experiments/cifar/cifar_charts.py --layer ${L} --chart ${CH} --solver ${SO} --private ${PR} --k ${K} --out ${OUT} ${EXTRA}
JOB
