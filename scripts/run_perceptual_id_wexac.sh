#!/bin/bash
# Perceptual-identification tier over saved cells: experiments/utils/perceptual_id_sweep.py (CPU only, queue short).
#   bash scripts/run_perceptual_id_wexac.sh smoke   # the two finished letters ladders: mlp_letter_a oracle eps 0 / 0.03 / 0.10 + pca
#   bash scripts/run_perceptual_id_wexac.sh full    # everything present on disk now (rerun later with the same command)
cd /home/projects/galvardi/yoado; mkdir -p results/perceptual_id figures/perceptual_id scripts/wexac_logs
STAGE="${1:-smoke}"
case "$STAGE" in
  smoke) ARGS="--sources oracle_ladder --include mlp_letter_a_eps0.pth mlp_letter_a_eps0.03.pth mlp_letter_a_eps0.1.pth mlp_letter_a_pca.pth --figures --tag smoke" ;;
  full)  ARGS="--figures" ;;
  *) echo "usage: $0 {smoke|full}"; exit 2 ;;
esac
bsub -q short -n 4 -R "rusage[mem=16384] span[hosts=1]" -J pid_${STAGE} \
     -o scripts/wexac_logs/pid_${STAGE}_%J.out -e scripts/wexac_logs/pid_${STAGE}_%J.err <<JOB | grep -o "Job <[0-9]*>"
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
echo "# host \$(hostname)  job \$LSB_JOBID  git \$(git rev-parse --short HEAD)  stage $STAGE  \$(date)"
python -m py_compile experiments/utils/perceptual_id.py experiments/utils/perceptual_id_sweep.py || exit 3
python -u -m experiments.utils.perceptual_id_sweep $ARGS
echo "# DONE \$(date)"
JOB
