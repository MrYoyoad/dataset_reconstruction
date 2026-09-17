#!/bin/bash
# Head-to-head NTK (free coefficients) vs certificate, swept over chart type x k x T. One bsub per cell-family
# (or one per T with SPLIT_T=1, the CNN-arm default).
#   bash experiments/cifar/submit_ntk_vs_cert.sh <cell> [backbone]
#     cells      cifar_keyboard cifar_apple cifar_mixed | cifar_motorcycle cifar_bottle cifar_mixed_mb cifar_mixed_mb_samerow
#                mnist_a mnist_t mnist_mixed mnist_mixed_samerow
#     backbone   CIFAR only: cnn | mlp_overtrained   (the weak cifar10_mlp_newclass.pth is NOT selectable here; legacy rows only)
#   env overrides: CHARTS KS TS STARTS QUEUE GPU_REQ MEM EXTRA DEP SPLIT_T SPLIT_CHART TAG NTK_ARMS DRYRUN (=1: print, no bsub)
#     TAG=smoke   -> job/jsonl names carry _smoke and outputs go to results|figures/ntk_vs_cert/smoke/
# Defaults follow the 2026-09-18 plan + audit (WP2): Ts "1 5 20 100 400", ks "16 32 48", 200 starts, charts "pca ae"
# (+ "pca_perclass" on mixed cells); mnist_a / mnist_mixed default to Ts "5 20 100" because T in {1,400} exist as jobs
# 302279 / 302280 / 304349 (reuse, do not repeat). CNN arm: one job per (T, chart) (SPLIT_T=1 SPLIT_CHART=1), gmem=20000, main NTK form only
# (--ntk-forms lora --ntk-solvers varpro): its LM arm alone is ~41 min per cell (log nc_cnn_293438) and each NTK arm
# through the FP64 conv stack is ~5 h (smoke 355918: 3 s per 30 iters x 32 images); the MLP arm keeps all four NTK arms.
#   smoke:  TAG=smoke QUEUE=short-gpu KS=32 TS="1 400" STARTS=20 bash experiments/cifar/submit_ntk_vs_cert.sh mnist_mixed_samerow
CELL=${1:-cifar_keyboard}; BB=${2:-}
cd /home/projects/galvardi/yoado; mkdir -p results/ntk_vs_cert scripts/wexac_logs
MIXED=0; TS_DEFAULT="1 5 20 100 400"
case $CELL in
  cifar_keyboard)         ARGS="--dataset cifar --newclass cifar100:keyboard" ;;
  cifar_apple)            ARGS="--dataset cifar --newclass cifar100:apple" ;;
  cifar_mixed)            ARGS="--dataset cifar --newclass cifar100:keyboard --newclass2 cifar100:apple"; MIXED=1 ;;
  cifar_motorcycle)       ARGS="--dataset cifar --newclass cifar100:motorcycle" ;;
  cifar_bottle)           ARGS="--dataset cifar --newclass cifar100:bottle" ;;
  cifar_mixed_mb)         ARGS="--dataset cifar --newclass cifar100:motorcycle --newclass2 cifar100:bottle"; MIXED=1 ;;
  cifar_mixed_mb_samerow) ARGS="--dataset cifar --newclass cifar100:motorcycle --newclass2 cifar100:bottle --same-row"; MIXED=1 ;;
  mnist_a)                ARGS="--dataset mnist --letter a"; TS_DEFAULT="5 20 100" ;;
  mnist_t)                ARGS="--dataset mnist --letter t" ;;
  mnist_mixed)            ARGS="--dataset mnist --letter a --letter2 t"; MIXED=1; TS_DEFAULT="5 20 100" ;;
  mnist_mixed_samerow)    ARGS="--dataset mnist --letter a --letter2 t --same-row"; MIXED=1 ;;
  *) echo "unknown cell $CELL"; exit 1 ;;
esac
NAME=$CELL
if [[ $CELL == cifar_* ]]; then
  case $BB in
    cnn)             ARGS="$ARGS --backbone cnn"; GPU_REQ=${GPU_REQ:-"num=1:gmem=20000"}; SPLIT_T=${SPLIT_T:-1}; SPLIT_CHART=${SPLIT_CHART:-1}
                     NTK_ARMS=${NTK_ARMS:-"--ntk-forms lora --ntk-solvers varpro"} ;;
    mlp_overtrained) ARGS="$ARGS --backbone mlp_overtrained" ;;
    *) echo "CIFAR cells need a backbone: cnn | mlp_overtrained"; exit 1 ;;
  esac
  NAME=${CELL}_${BB}
fi
CHARTS=${CHARTS:-"pca ae$( [[ $MIXED == 1 ]] && echo " pca_perclass")"}
KS=${KS:-"16 32 48"}; TS=${TS:-$TS_DEFAULT}; STARTS=${STARTS:-200}; QUEUE=${QUEUE:-long-gpu}
GPU_REQ=${GPU_REQ:-"num=1"}; MEM=${MEM:-24576}; SPLIT_T=${SPLIT_T:-0}; SPLIT_CHART=${SPLIT_CHART:-0}; NTK_ARMS=${NTK_ARMS:-}
SAVE_DIR=results/ntk_vs_cert${TAG:+/$TAG}; FIG_DIR=figures/ntk_vs_cert${TAG:+/$TAG}
NAME=${NAME}${TAG:+_$TAG}
ARGS="$ARGS --ks $KS --starts $STARTS $NTK_ARMS --save-dir $SAVE_DIR --fig-dir $FIG_DIR $EXTRA"

submit() {   # $1 = job-name suffix, $2 = Ts for this job, $3 = charts for this job
  local JN=${NAME}$1
  if [[ ${DRYRUN:-0} == 1 ]]; then echo "DRYRUN nvc_${JN}: -q $QUEUE -gpu '$GPU_REQ' charts '$3' Ts '$2' ks '$KS' starts $STARTS $NTK_ARMS -> ${SAVE_DIR}/sweep_${JN}_<jobid>.jsonl"; return; fi
  bsub -q $QUEUE -gpu "$GPU_REQ" -R "rusage[mem=$MEM] select[ngpus>0]" ${DEP:+-w "done($DEP)"} -J nvc_${JN} \
       -o scripts/wexac_logs/nvc2_${JN}_%J.out -e scripts/wexac_logs/nvc2_${JN}_%J.err <<JOB
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "# host \$(hostname)  job \$LSB_JOBID  git \$(git rev-parse --short HEAD)  cell ${CELL}  backbone '${BB}'  charts '$3'  ks '${KS}'  Ts '$2'  starts ${STARTS}  ntk_arms '${NTK_ARMS:-default}'"
python -u -m experiments.cifar.ntk_vs_certificate ${ARGS} --charts $3 --Ts $2 --out ${SAVE_DIR}/sweep_${JN}_\$LSB_JOBID.jsonl
JOB
}
TS_JOBS=$( [[ $SPLIT_T == 1 ]] && echo "$TS" || echo "$TS" | tr ' ' ',' )          # one job per T, or all Ts in one job
CH_JOBS=$( [[ $SPLIT_CHART == 1 ]] && echo "$CHARTS" || echo "$CHARTS" | tr ' ' ',' )
for T in $TS_JOBS; do for CH in $CH_JOBS; do
  submit "$( [[ $SPLIT_T == 1 ]] && echo _T$T )$( [[ $SPLIT_CHART == 1 ]] && echo _$CH )" "${T//,/ }" "${CH//,/ }"
done; done
