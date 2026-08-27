#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/arm_c_imbalance_%J.out
#BSUB -e scripts/wexac_logs/arm_c_imbalance_%J.err
#BSUB -J arm_c_imbalance
# ARM C — CLASS IMBALANCE: does a MINORITY-class example leave a larger per-example imprint on a
# LoRA adapter than a MAJORITY-class one, and does that gap widen as the minority class gets rarer?
# Per-example single-image swap (arm-B style) in an imbalanced set; sens_min vs sens_maj at each m.
# notes BATTERY TRACKER, arm C.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
echo "########## GATE: metric self-test + arm_c sanity ##########"
python -u -m experiments.dataset_sensitivity.whitened_metric || { echo FATAL metric; exit 1; }
python -u -m experiments.dataset_sensitivity.arm_c_imbalance --stage0 --device cuda || { echo FATAL stage0; exit 1; }
echo "########## FULL: m {1,2,4,8}, N=16, K=50 ##########"; date
python -u -m experiments.dataset_sensitivity.arm_c_imbalance \
    --m_list 1 2 4 8 --N 16 --K 50 --n_targets_per_class 3 --lr 0.5 --T 1000 --rank 8 --device cuda
echo "=== DONE $(date) ==="
echo "READ: ratio=sens_min/sens_maj > 1 AND growing as m falls => rarer minority example more identifiable."
