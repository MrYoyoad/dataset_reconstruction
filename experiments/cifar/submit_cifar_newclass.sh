#!/bin/bash
# One bsub per ARCHITECTURE; the classes run sequentially inside it so the backbone is trained once (no checkpoint race).
#   bash experiments/cifar/submit_cifar_newclass.sh mlp
#   bash experiments/cifar/submit_cifar_newclass.sh cnn
ARCH=${1:-mlp}
CLASSES=${2:-"cifar100:keyboard cifar100:skyscraper cifar100:mushroom flowers102"}
EXTRA=${3:-}          # e.g. --overtrain
cd /home/projects/galvardi/yoado
bsub -q long-gpu -gpu "num=1" -R "rusage[mem=24576] select[ngpus>0]" -J nc_${ARCH}${EXTRA:+_ot} \
     -o scripts/wexac_logs/nc_${ARCH}${EXTRA:+_ot}_%J.out -e scripts/wexac_logs/nc_${ARCH}${EXTRA:+_ot}_%J.err <<JOB
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "# host \$(hostname)  job \$LSB_JOBID  git \$(git rev-parse --short HEAD)  arch ${ARCH}"
OUT=results/cifar_newclass/nc_${ARCH}${EXTRA:+_ot}_\$LSB_JOBID.jsonl
mkdir -p results/cifar_newclass
for C in ${CLASSES}; do
  echo "##### \$C  (on-chart)"
  python -u -m experiments.cifar.cifar_newclass --arch ${ARCH} --newclass \$C --k 32 ${EXTRA} --out \$OUT
done
echo "##### wrong-release control (keyboard)"
python -u -m experiments.cifar.cifar_newclass --arch ${ARCH} --newclass cifar100:keyboard --k 32 --wrong_release ${EXTRA} --out \$OUT
echo "##### raw privates (keyboard)"
python -u -m experiments.cifar.cifar_newclass --arch ${ARCH} --newclass cifar100:keyboard --k 32 --private raw ${EXTRA} --out \$OUT
JOB
