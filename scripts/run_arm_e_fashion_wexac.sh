#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/arm_e_fashion_%J.out
#BSUB -e scripts/wexac_logs/arm_e_fashion_%J.err
#BSUB -J arm_e_fashion
# ARM E — DUPLICATION scaling on FASHION-MNIST (base checkpoint
# models/weights-fashion_gelu.pth). Fixed-prevalence copies-vs-distinct, Σ frozen
# across k, empirical rank-null. Does the low-rank bottleneck SATURATE duplication
# imprint? β(sensitivity) at r=8 (bottleneck) vs r=32 (full). --dataset fashion
# swaps only the base/data loader; results tagged _fashion (no mnist overwrite).
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
echo "########## GATE: metric self-test + arm_e fashion sanity ##########"
python -u -m experiments.dataset_sensitivity.whitened_metric || { echo FATAL metric; exit 1; }
python -u -m experiments.dataset_sensitivity.arm_e_duplication --stage0 --dataset fashion --device cuda || { echo FATAL stage0; exit 1; }
echo "########## FULL: rank {8,32} x k {1,2,4,8}, N=16, K=50 (fashion) ##########"; date
python -u -m experiments.dataset_sensitivity.arm_e_duplication \
    --rank_list 8 32 --k_list 1 2 4 8 --N 16 --K 50 --n_targets 4 --lr 0.5 --T 1000 --dataset fashion --device cuda
echo "=== DONE $(date) ==="
echo "READ: β(sensitivity) r=8 vs r=32. β(low)<β(high) => low-rank SATURATES duplication (protective)."
