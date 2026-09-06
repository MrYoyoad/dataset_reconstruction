#!/bin/bash
# Head-to-head NTK (free coefficients) vs certificate, swept over chart type x k x T. One bsub per cell-family.
#   bash experiments/cifar/submit_ntk_vs_cert.sh cifar_keyboard
#   bash experiments/cifar/submit_ntk_vs_cert.sh cifar_mixed      # keyboard + apple, two new head rows
#   bash experiments/cifar/submit_ntk_vs_cert.sh mnist_a
#   bash experiments/cifar/submit_ntk_vs_cert.sh mnist_mixed      # letters a + t, two new head rows
CELL=${1:-cifar_keyboard}
CHARTS=${2:-"pca ae"}; KS=${3:-"16 32 48"}; TS=${4:-"1 400"}; EXTRA=${5:-}; DEP=${6:-}
cd /home/projects/galvardi/yoado; mkdir -p results/ntk_vs_cert
case $CELL in
  cifar_keyboard) ARGS="--dataset cifar --newclass cifar100:keyboard" ;;
  cifar_apple)    ARGS="--dataset cifar --newclass cifar100:apple" ;;
  cifar_mixed)    ARGS="--dataset cifar --newclass cifar100:keyboard --newclass2 cifar100:apple" ;;
  mnist_a)        ARGS="--dataset mnist --letter a" ;;
  mnist_t)        ARGS="--dataset mnist --letter t" ;;
  mnist_mixed)    ARGS="--dataset mnist --letter a --letter2 t" ;;
  *) echo "unknown cell $CELL"; exit 1 ;;
esac
bsub -q long-gpu -gpu "num=1" -R "rusage[mem=24576] select[ngpus>0]" ${DEP:+-w "done($DEP)"} -J nvc_${CELL}${EXTRA:+_diag} \
     -o scripts/wexac_logs/nvc2_${CELL}${EXTRA:+_diag}_%J.out -e scripts/wexac_logs/nvc2_${CELL}${EXTRA:+_diag}_%J.err <<JOB
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "# host \$(hostname)  job \$LSB_JOBID  git \$(git rev-parse --short HEAD)  cell ${CELL}  charts '${CHARTS}'  ks '${KS}'  Ts '${TS}'"
python -u -m experiments.cifar.ntk_vs_certificate ${ARGS} --charts ${CHARTS} --ks ${KS} --Ts ${TS} --starts 200 ${EXTRA} \
       --out results/ntk_vs_cert/sweep_${CELL}${EXTRA:+_diag}_\$LSB_JOBID.jsonl
JOB
