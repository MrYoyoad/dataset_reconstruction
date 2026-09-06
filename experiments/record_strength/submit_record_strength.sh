#!/bin/bash
# Record strength vs recovery: two GPU jobs (letters, confident) and a CPU plot job that waits for both.
#   bash experiments/record_strength/submit_record_strength.sh
cd /home/projects/galvardi/yoado
mkdir -p results/record_strength figures/record_strength
submit_cell () {
  CELL=$1
  bsub -q long-gpu -gpu "num=1" -R "rusage[mem=16384] select[ngpus>0]" -J rs_${CELL} \
       -o scripts/wexac_logs/rs_${CELL}_%J.out -e scripts/wexac_logs/rs_${CELL}_%J.err <<JOB | grep -o "Job <[0-9]*>" | grep -o "[0-9]*"
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "# host \$(hostname)  job \$LSB_JOBID  git \$(git rev-parse --short HEAD)  cell ${CELL}"
nvidia-smi --query-gpu=name --format=csv,noheader
python -u -m experiments.record_strength.record_strength --cells ${CELL} \
       --out results/record_strength/rs_${CELL}_\$LSB_JOBID.jsonl --save-dir results/record_strength
JOB
}
J1=$(submit_cell letters); J2=$(submit_cell confident)
echo "letters job $J1, confident job $J2"
bsub -q short -R "rusage[mem=8192]" -w "done($J1) && done($J2)" -J rs_plot \
     -o scripts/wexac_logs/rs_plot_%J.out -e scripts/wexac_logs/rs_plot_%J.err <<JOB
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
python -u -m experiments.record_strength.record_strength --plot "results/record_strength/rs_letters_${J1}.jsonl" "results/record_strength/rs_confident_${J2}.jsonl"
JOB
