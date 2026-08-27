#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/arm_c_stage0_%J.out
#BSUB -e scripts/wexac_logs/arm_c_stage0_%J.err
#BSUB -J arm_c_stage0
# ARM C (class imbalance) STAGE-0 plumbing gate: 3-way metric self-test + arm_c tiny sanity.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== STAGE-0 START $(date) on $(hostname) ==="
echo "--- gate A: 3-way metric self-test ---"
python -u -m experiments.dataset_sensitivity.whitened_metric || { echo "FATAL metric self-test"; exit 1; }
echo "--- gate B: arm_c tiny sanity ---"
python -u -m experiments.dataset_sensitivity.arm_c_imbalance --stage0 --device cuda
rc=$?
echo "=== STAGE-0 DONE $(date) (exit $rc) ==="
if [ $rc -ne 0 ]; then echo "FATAL: arm_c Stage-0 crashed (rc=$rc)."; exit 1; fi
