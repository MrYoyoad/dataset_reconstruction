#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/ood_injection_stage0_%J.out
#BSUB -e scripts/wexac_logs/ood_injection_stage0_%J.err
#BSUB -J ood_inj_stage0
# OOD-STYLE DIGIT INJECTION — STAGE-0 plumbing gate:
#   gate A = USPS import/download check (compute nodes may lack internet -> pre-download on login node),
#   gate B = 3-way whitened metric self-test,
#   gate C = ood_injection tiny sanity (N=12, K=12, 1 OOD member; asserts finite metric).
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== STAGE-0 START $(date) on $(hostname) ==="
echo "--- gate A: USPS load/download check ---"
python -u -c "from experiments.dataset_sensitivity.ood_injection import load_usps; ds = load_usps(); print(f'USPS OK: {len(ds)} test images')" \
    || { echo "FATAL: USPS unavailable — pre-download on the LOGIN node (see the error above)"; exit 1; }
echo "--- gate B: 3-way metric self-test ---"
python -u -m experiments.dataset_sensitivity.whitened_metric || { echo "FATAL metric self-test"; exit 1; }
echo "--- gate C: ood_injection tiny sanity ---"
python -u -m experiments.dataset_sensitivity.ood_injection --stage0 --device cuda
rc=$?
echo "=== STAGE-0 DONE $(date) (exit $rc) ==="
if [ $rc -ne 0 ]; then echo "FATAL: ood_injection Stage-0 crashed (rc=$rc)."; exit 1; fi
