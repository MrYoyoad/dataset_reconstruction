#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/margin_vs_sens_%J.out
#BSUB -e scripts/wexac_logs/margin_vs_sens_%J.err
#BSUB -J margin_vs_sens
# MARGIN-vs-SENSITIVITY analysis arm (support-vector/max-margin hypothesis for LoRA leakage).
# CHEAP: 13 reference adapter trainings (12 arm-D contexts + 1 arm-C balanced set) + forwards.
# Correlates base-margin / gradnorm / lambda-proxy against the ALREADY-MEASURED arm-D
# per_target_sens (results/arm_d_context/arm_d_summary.json) — no new sensitivity runs.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== MARGIN-vs-SENS START $(date) on $(hostname) ==="
echo "--- gate A: import sanity ---"
python -c "import experiments.dataset_sensitivity.margin_vs_sensitivity" \
  || { echo "FATAL import gate"; exit 1; }
echo "--- gate B: stage-0 (base margins + one adapter at m=4, finiteness asserts) ---"
python -u -m experiments.dataset_sensitivity.margin_vs_sensitivity --stage0 --device cuda
rc=$?
if [ $rc -ne 0 ]; then echo "FATAL: stage-0 crashed (rc=$rc)."; exit 1; fi
echo "--- full run ---"
python -u -m experiments.dataset_sensitivity.margin_vs_sensitivity --device cuda \
  --T 1000 --lr 0.5 --rank 8
rc=$?
echo "=== MARGIN-vs-SENS DONE $(date) (exit $rc) ==="
if [ $rc -ne 0 ]; then echo "FATAL: full run crashed (rc=$rc)."; exit 1; fi
