#!/bin/bash
# One bsub per (example, chart) cell: identical solver and 400 starts everywhere, nothing tuned between cells, and
# the convolutional example's charts run in parallel instead of serialising into a ~17-hour job.
#   bash experiments/oracle_ladder/submit_ladder.sh
cd /home/projects/galvardi/yoado; mkdir -p results/oracle_ladder
EPS="0 0.01 0.02 0.03 0.05 0.075 0.10 0.15 0.20 0.30 0.40 0.60"
for EX in mlp_motorcycle cnn_keyboard; do
  Q=$([ "$EX" = "cnn_keyboard" ] && echo long-gpu || echo short-gpu)
  for E in $EPS; do
    bsub -q $Q -gpu "num=1" -R "rusage[mem=24576] select[ngpus>0]" -J ol_${EX}_e${E} \
         -o scripts/wexac_logs/ol_${EX}_e${E}_%J.out -e scripts/wexac_logs/ol_${EX}_e${E}_%J.err <<JOB | grep -o "Job <[0-9]*>"
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
python -u -m experiments.oracle_ladder.ladder_cell --example ${EX} --chart oracle:${E}
JOB
  done
  bsub -q $Q -gpu "num=1" -R "rusage[mem=24576] select[ngpus>0]" -J ol_${EX}_pca \
       -o scripts/wexac_logs/ol_${EX}_pca_%J.out -e scripts/wexac_logs/ol_${EX}_pca_%J.err <<JOB | grep -o "Job <[0-9]*>"
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
python -u -m experiments.oracle_ladder.ladder_cell --example ${EX} --chart pca
JOB
  bsub -q $Q -gpu "num=1" -R "rusage[mem=24576] select[ngpus>0]" -J ol_${EX}_wr \
       -o scripts/wexac_logs/ol_${EX}_wr_%J.out -e scripts/wexac_logs/ol_${EX}_wr_%J.err <<JOB | grep -o "Job <[0-9]*>"
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
python -u -m experiments.oracle_ladder.ladder_cell --example ${EX} --chart oracle:0 --wrong_release
JOB
done
