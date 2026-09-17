#!/bin/bash
#BSUB -q short
#BSUB -J archive_coeff_check
#BSUB -o scripts/wexac_logs/archive_coeff_check_%J.out
#BSUB -e scripts/wexac_logs/archive_coeff_check_%J.err
#BSUB -R "rusage[mem=2048]"
#
# Independent FP64 evaluation of the three hand-computed coefficients in the archive's multilayer note
# (approver ruling C4, commit e0a2bb1). CPU only, seconds. No GPU requested on purpose.
set +u   # `set -u` breaks this repo's conda activate (ADDR2LINE unbound) -- see LESSONS_LEARNED
source /home/projects/galvardi/yoado/miniforge3/etc/profile.d/conda.sh 2>/dev/null || \
  source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rec
cd /home/projects/galvardi/yoado
python -u -m experiments.archive_checks.check_hand_coefficients
