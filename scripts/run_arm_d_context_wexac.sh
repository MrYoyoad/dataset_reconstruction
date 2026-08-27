#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/arm_d_context_%J.out
#BSUB -e scripts/wexac_logs/arm_d_context_%J.err
#BSUB -J arm_d_context
# ARM D — CONTEXT RARITY (image-controlled): does the SAME fixed image T leak more per-example when it
# is the LONE minority (rare context) than when it has many same-class peers (typical context)? Holds
# T and its held-out control T' FIXED across m, varying ONLY T's class-1 peer count — removes arm C's
# image-identity / class-identity confound. notes BATTERY TRACKER, arm D.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
echo "########## GATE: metric self-test + arm_d sanity ##########"
python -u -m experiments.dataset_sensitivity.whitened_metric || { echo FATAL metric; exit 1; }
python -u -m experiments.dataset_sensitivity.arm_d_context --stage0 --device cuda || { echo FATAL stage0; exit 1; }
echo "########## FULL: m {1,2,4,8}, N=16, K=50, 3 fixed targets ##########"; date
python -u -m experiments.dataset_sensitivity.arm_d_context \
    --m_list 1 2 4 8 --N 16 --K 50 --n_targets 3 --lr 0.5 --T 1000 --rank 8 --device cuda
echo "=== DONE $(date) ==="
echo "READ: sens(m) FALLING as m grows => clean context-rarity effect; rarity gain sens(m=1)/sens(m=8)>1 => lone-minority T leaks more."
