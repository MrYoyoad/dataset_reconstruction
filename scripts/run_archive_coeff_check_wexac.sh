#!/bin/bash
#BSUB -q short
#BSUB -R "rusage[mem=2048]"
#BSUB -W 0:10
#BSUB -o scripts/wexac_logs/archive_coeff_%J.out
#BSUB -e scripts/wexac_logs/archive_coeff_%J.err
#BSUB -J archive_coeff

# =====================================================================
# Independent FP64 evaluation of the three hand-computed coefficients in the archive's multilayer note
# (notes/gal_2026-09/multilayer_lora_theory_source.tex): Ex 3.5 -3/8, Ex 5.5 eta/16, Sec 8.6 -1/3.
# Ruled C4 by the approver lane at commit e0a2bb1: all three carry a PROVED status and none has ever been
# evaluated here, while the existing theory_checks T3 cell validates a DIFFERENT instance.
#
# Each cell re-derives its coefficient from the dynamics the note specifies rather than re-evaluating the
# note's own closed forms, and checks the note's exact intermediate matrices FIRST -- if those disagree the
# coefficient verdict is withheld and the cell reports "construction mismatch", because comparing a
# coefficient against a different dynamical system is a coincidence test, not a comparison.
# Tolerances are pre-stated in the module docstring. Do not loosen one to make a cell pass.
#
# CPU on purpose: FP64 numpy, three cells, seconds.
# =====================================================================
set +u                                   # conda activate breaks under set -u in this env (LESSONS_LEARNED)
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado

python -u -m experiments.archive_checks.check_hand_coefficients
