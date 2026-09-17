#!/bin/bash
# Is the invertibility of I + M_t G load-bearing for the certificate theorem? The derivation says no
# -- the closure induction needs only containment, and Pi A_T = Pi A_0 needs only Pi A_0 H = 0, which
# follows from rank B_T = q alone. This stresses it numerically: push the step size until I + M_t G is
# far from the identity and see whether the identity or the rank claim ever breaks.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
python -u experiments/exact_inversion/h3_invertibility_check.py
