#!/bin/bash
# WP3 -- pretrained decoder (sd-vae-ft-mse) as a chart: fidelity check, no inversion.  short-gpu, 1 GPU, 24 GB.
#   bash scripts/run_decoder_chart_wexac.sh smoke                 # one image set, k 16, K 64, 200 Adam steps (minutes)
#   bash scripts/run_decoder_chart_wexac.sh full [image_set]      # one job per (image set, anchor, K): 18 jobs, ~2 h each
# Sizing (smoke 355910, shared A40): 4.18 s per Adam step for 8 images x 3 restarts at 256^2, so the plan's 2000 steps x
# 24 combinations would be ~56 h per image set. Deviation, recorded in RESULT.md: 400 steps (the smoke trace is flat from
# step ~80 under the same cosine schedule) and the (anchor, K) split, 4 k-values per job -> ~1.9 h per job at that rate.
# diffusers lives OUTSIDE the shared rec env (audit 2026-09-18): .conda/extra_pkgs, added to PYTHONPATH here only.
# HF_HUB_OFFLINE=1 + local_files_only=True: the job fails loudly if the cached weights are missing; it never downloads.
cd /home/projects/galvardi/yoado; mkdir -p results/decoder_chart figures/decoder_chart scripts/wexac_logs
MODE=${1:-smoke}
submit() {   # $1 = job name, rest = python args
  NAME=$1; shift
  bsub -q short-gpu -gpu "num=1" -R "rusage[mem=24576] select[ngpus>0]" -J "$NAME" \
       -o scripts/wexac_logs/${NAME}_%J.out -e scripts/wexac_logs/${NAME}_%J.err <<JOB | grep -o "Job <[0-9]*>"
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH=/home/projects/galvardi/yoado/.conda/extra_pkgs:\$PYTHONPATH
export HF_HUB_OFFLINE=1 HF_HOME=/home/projects/galvardi/yoado/.cache/huggingface
python -u -m experiments.decoder_chart.fidelity $@
JOB
}
case "$MODE" in
  smoke) submit dc_smoke --image-sets mlp_motorcycle --ks 16 --Ks 64 --steps 200 --tag smoke ;;
  full)  for S in ${2:-mlp_motorcycle cnn_keyboard mnist_letter_a}; do for A in proxy_nn truth_nn truth_latent; do for K in 64 256; do
           submit dc_full_${S}_${A}_K${K} --image-sets $S --anchors $A --Ks $K --steps 400 --tag ${A}_K${K}
         done; done; done ;;
  *) echo "usage: $0 smoke|full [image_set]"; exit 1 ;;
esac
