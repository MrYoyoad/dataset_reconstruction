#!/bin/bash
# WP3 -- pretrained decoder (sd-vae-ft-mse) as a chart: fidelity check, no inversion.  short-gpu, 1 GPU, 24 GB.
#   bash scripts/run_decoder_chart_wexac.sh smoke                 # one image set, k 16, K 64, 200 Adam steps (minutes)
#   bash scripts/run_decoder_chart_wexac.sh full [image_set]      # one job per (image set, anchor, K): 18 jobs, ~2 h each
#   bash scripts/run_decoder_chart_wexac.sh full2000 [image_set]  # SAME 18-way split at the plan's 2000 steps, on long-gpu
#   bash scripts/run_decoder_chart_wexac.sh one2000 <set> <anchor> <K> [--chunk 4]   # resubmit one 2000-step cell
# Sizing (smoke 355910, shared A40): 4.18 s per Adam step for 8 images x 3 restarts at 256^2, so the plan's 2000 steps x
# 24 combinations would be ~56 h per image set. Deviation, recorded in RESULT.md: 400 steps (the smoke trace is flat from
# step ~80 under the same cosine schedule) and the (anchor, K) split, 4 k-values per job -> ~1.9 h per job at that rate.
# 2000-step mode (2026-09-18, at Yoad's request): restores the plan's step count and keeps the SAME 18-way
# (image set, anchor, K) split -- 4 k-combinations per job, 72 in total, which is under the plan's 24-per-job and in
# the 10-20 job band. It must NOT go to short-gpu: the 400-step jobs measured 2052-5840 s, and 5x the Adam steps puts
# the slowest near 7 h against that queue's 360 min cap, so full2000 forces long-gpu. Tag s2000_* keeps the rows in
# separate files from the 400-step ones. EXPECTATION, pre-registered: the 400-step traces are converged by step ~80
# (last 80% of the run moves the error 0.01-0.61%), so this should REPRODUCE the 400-step numbers and thereby close
# the recorded deviation. A move larger than ~1% would instead mean the 400-step rows were under-converged.
# diffusers lives OUTSIDE the shared rec env (audit 2026-09-18): .conda/extra_pkgs, added to PYTHONPATH here only.
# HF_HUB_OFFLINE=1 + local_files_only=True: the job fails loudly if the cached weights are missing; it never downloads.
# gmem=20G: the first full submission (356034-356051, plain num=1) lost 9 of 18 jobs to CUDA OOM on SHARED A40s (other
# users' processes at 13-20 GB of 44 GB); the fit needs ~12 GB at 256^2 with chunk 8. Resubmit a single cell with
#   bash scripts/run_decoder_chart_wexac.sh one <image_set> <anchor> <K> [--chunk 4]
# Three survivors of the first batch (356034, 356035, 356045) also OOM'd at ~1650 s with the process itself at 18 GB
# (chunk 8, fragmentation); their resubmissions use --chunk 4.
cd /home/projects/galvardi/yoado; mkdir -p results/decoder_chart figures/decoder_chart scripts/wexac_logs
MODE=${1:-smoke}
QUEUE=${QUEUE:-short-gpu}     # short-gpu caps at 360 min; the 2000-step mode overrides this to long-gpu
submit() {   # $1 = job name, rest = python args
  NAME=$1; shift
  bsub -q $QUEUE -gpu "num=1:gmem=20G" -R "rusage[mem=24576] select[ngpus>0]" -J "$NAME" \
       -o scripts/wexac_logs/${NAME}_%J.out -e scripts/wexac_logs/${NAME}_%J.err <<JOB | grep -o "Job <[0-9]*>"
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH=/home/projects/galvardi/yoado/.conda/extra_pkgs:\$PYTHONPATH
export HF_HUB_OFFLINE=1 HF_HOME=/home/projects/galvardi/yoado/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u -m experiments.decoder_chart.fidelity $@
JOB
}
case "$MODE" in
  smoke) submit dc_smoke --image-sets mlp_motorcycle --ks 16 --Ks 64 --steps 200 --tag smoke ;;
  full)  for S in ${2:-mlp_motorcycle cnn_keyboard mnist_letter_a}; do for A in proxy_nn truth_nn truth_latent; do for K in 64 256; do
           submit dc_full_${S}_${A}_K${K} --image-sets $S --anchors $A --Ks $K --steps 400 --tag ${A}_K${K}
         done; done; done ;;
  one)   submit dc_full_${2}_${3}_K${4} --image-sets $2 --anchors $3 --Ks $4 --steps 400 --tag ${3}_K${4} "${@:5}" ;;   # extras, e.g. --chunk 4
  full2000) QUEUE=long-gpu; for S in ${2:-mlp_motorcycle cnn_keyboard mnist_letter_a}; do for A in proxy_nn truth_nn truth_latent; do for K in 64 256; do
           submit dc2k_${S}_${A}_K${K} --image-sets $S --anchors $A --Ks $K --steps 2000 --tag s2000_${A}_K${K}
         done; done; done ;;
  one2000) QUEUE=long-gpu; submit dc2k_${2}_${3}_K${4} --image-sets $2 --anchors $3 --Ks $4 --steps 2000 --tag s2000_${3}_K${4} "${@:5}" ;;
  *) echo "usage: $0 smoke|full [image_set]|one <image_set> <anchor> <K>"; exit 1 ;;
esac
