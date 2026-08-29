#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/arm_b_stage0_%J.out
#BSUB -e scripts/wexac_logs/arm_b_stage0_%J.err
#BSUB -J arm_b_stage0

# ARM B — STAGE-0 ONLY (N=8, K=3, 3 targets). Verify the floor decomposition + no crashes
# BEFORE the full ~450-retrain sweep. yoado-a2 audits this log.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== STAGE-0 START $(date) on $(hostname) ==="
python -u -m experiments.dataset_sensitivity.arm_b_dilution --stage0 --device cuda
rc=$?
echo "=== STAGE-0 DONE $(date) (exit $rc) ==="
if [ $rc -ne 0 ]; then echo "FATAL: Stage-0 crashed (rc=$rc)."; exit 1; fi
