#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/h_spotcheck_%J.out
#BSUB -e scripts/wexac_logs/h_spotcheck_%J.err
#BSUB -J h_spotcheck
# §III.0 CHEAP H SPOT-CHECK — the de-risking gate that runs BEFORE all expensive scaling.
# Behavioral leave-one-out (Feldman-style, on the margin) memorization score for the EXACT 12
# arm-D (image, context) cells with saved whitened sensitivities (job 245964), rank-correlated
# against those sensitivities. PRE-REGISTERED: positive rho expected; rho<=0 = KILL (downstream
# re-labels to "parametric detectability only"). CHEAP: 2*12*K_loo=240 adapter trainings
# (T=1000 tiny MLP, ~1.5 s each per job-245964 calibration) + 8 stage-0 trainings => ~15 min
# total, comfortably inside short-gpu.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== H-SPOTCHECK START $(date) on $(hostname) ==="
echo "--- gate A: import sanity ---"
python -c "import experiments.dataset_sensitivity.h_spotcheck" \
  || { echo "FATAL import gate"; exit 1; }
echo "--- gate B: stage-0 (1 target, m=4 only, K_loo=4, finite-mem assert) ---"
python -u -m experiments.dataset_sensitivity.h_spotcheck --stage0 --device cuda
rc=$?
if [ $rc -ne 0 ]; then echo "FATAL: stage-0 crashed (rc=$rc)."; exit 1; fi
echo "--- full run: 12 cells x K_loo=10 paired LOO trainings ---"
python -u -m experiments.dataset_sensitivity.h_spotcheck --device cuda \
  --K_loo 10 --lr 0.5 --T 1000 --rank 8
rc=$?
echo "=== H-SPOTCHECK DONE $(date) (exit $rc) ==="
if [ $rc -ne 0 ]; then echo "FATAL: full run crashed (rc=$rc)."; exit 1; fi
echo "READ: rho(mem,sens)>+0.4 => de-risked, scale §III.1/§III.2 (III.3 full gate still required)."
echo "      rho<=0 => KILL: downstream re-labels to 'parametric detectability only'."
