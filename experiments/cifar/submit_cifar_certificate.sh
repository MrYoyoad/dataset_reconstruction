#!/bin/bash
# CIFAR-10 certificate replica, one bsub job per chart dimension k.
#   bash experiments/cifar/submit_cifar_certificate.sh 32            # cifar_certificate.py (repo root), raw private images
#   bash experiments/cifar/submit_cifar_certificate.sh 32 onchart    # experiments/cifar/cifar_certificate_onchart.py, on-chart control
K=${1:-32}; MODE=${2:-raw}
cd /home/projects/galvardi/yoado
if [ "$MODE" = "onchart" ]; then
  SCRIPT=experiments/cifar/cifar_certificate_onchart.py; OUT=experiments/cifar/k${K}_onchart; NAME=cifar_cert_k${K}_onchart
else
  SCRIPT=cifar_certificate.py; OUT=experiments/cifar/k${K}; NAME=cifar_cert_k${K}
fi
bsub -q long-gpu -gpu "num=1" -R "rusage[mem=16384] select[ngpus>0]" -J ${NAME} \
     -o scripts/wexac_logs/${NAME}_%J.out -e scripts/wexac_logs/${NAME}_%J.err <<JOB
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "# host \$(hostname)  job \$LSB_JOBID  git \$(git rev-parse --short HEAD)  k=${K} mode=${MODE}"
nvidia-smi --query-gpu=name --format=csv,noheader
python -u ${SCRIPT} --N 8 --r 64 --k ${K} --T 200 --starts 400 --out ${OUT}
JOB
