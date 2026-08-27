#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/arm_b_fashion_%J.out
#BSUB -e scripts/wexac_logs/arm_b_fashion_%J.err
#BSUB -J arm_b_fashion

# =====================================================================
# ARM B — per-image dilution SENSITIVITY vs N, on FASHION-MNIST (base
# checkpoint models/weights-fashion_gelu.pth). Same 3-way whitened metric
# as the mnist reconfirm; --dataset fashion swaps only the base/data loader.
# Results land in results/arm_b_dilution/ tagged with _fashion so they do
# NOT overwrite the mnist run.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## GATE 0a: 3-way metric self-test (on-node) ##########"; date
python -u -m experiments.dataset_sensitivity.whitened_metric
if [ $? -ne 0 ]; then echo "FATAL: whitened_metric self-test failed on node. Aborting."; exit 1; fi
echo "GATE 0a PASSED (4/4 incl K-convergence)."

echo ""; echo "########## GATE 0b: arm-b fashion sanity (N=8, K=12, 2 targets) ##########"; date
python -u -m experiments.dataset_sensitivity.arm_b_dilution --stage0 --dataset fashion --device cuda
if [ $? -ne 0 ]; then echo "FATAL: arm-b Stage-0 (fashion) failed. Aborting."; exit 1; fi
echo "GATE 0b PASSED."

echo ""; echo "########## K=50 : N in {4,8,16,32} (fashion) ##########"; date
python -u -m experiments.dataset_sensitivity.arm_b_dilution \
    --N_list 4 8 16 32 --K 50 --n_targets 4 --lr 0.5 --T 1000 --rank 8 --dataset fashion --device cuda

echo ""; echo "=== DONE $(date) ==="
echo "READ: sensitivity(N) FLAT across N with robust p; compare magnitude/shape to the mnist arm-b run."
