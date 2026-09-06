#!/bin/bash
# Head-to-head NTK vs certificate on one release. One bsub per cell.
#   bash experiments/cifar/submit_ntk_vs_cert.sh cifar_keyboard
#   bash experiments/cifar/submit_ntk_vs_cert.sh cifar_mixed      # keyboard + apple, two new head rows
#   bash experiments/cifar/submit_ntk_vs_cert.sh mnist_a
#   bash experiments/cifar/submit_ntk_vs_cert.sh mnist_mixed      # letters a + t
CELL=${1:-cifar_keyboard}
cd /home/projects/galvardi/yoado; mkdir -p results/ntk_vs_cert
case $CELL in
  cifar_keyboard) ARGS="--dataset cifar --newclass cifar100:keyboard" ;;
  cifar_apple)    ARGS="--dataset cifar --newclass cifar100:apple" ;;
  cifar_mixed)    ARGS="--dataset cifar --newclass cifar100:keyboard --newclass2 cifar100:apple" ;;
  mnist_a)        ARGS="--dataset mnist --letter a" ;;
  mnist_mixed)    ARGS="--dataset mnist --letter a --letter2 t" ;;
  *) echo "unknown cell $CELL"; exit 1 ;;
esac
bsub -q long-gpu -gpu "num=1" -R "rusage[mem=24576] select[ngpus>0]" -J nvc_${CELL} \
     -o scripts/wexac_logs/nvc_${CELL}_%J.out -e scripts/wexac_logs/nvc_${CELL}_%J.err <<JOB
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "# host \$(hostname)  job \$LSB_JOBID  git \$(git rev-parse --short HEAD)  cell ${CELL}"
python -u -m experiments.cifar.ntk_vs_certificate ${ARGS} --starts 200 --out results/ntk_vs_cert/nvc_${CELL}_\$LSB_JOBID.jsonl
JOB
