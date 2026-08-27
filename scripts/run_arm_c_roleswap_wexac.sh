#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/arm_c_roleswap_%J.out
#BSUB -e scripts/wexac_logs/arm_c_roleswap_%J.err
#BSUB -J arm_c_roleswap
# ARM C ROLE-SWAP CONTROL: minority = class-0 (was class-1). If the balanced-point ratio was pure
# class identity, it should INVERT (3.3 -> ~0.30) while the RARITY effect (normalized ratio rising as
# m shrinks) REPLICATES. Proves the 3.3x is class identity, not an artifact, and rarity is symmetric.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.dataset_sensitivity.whitened_metric || { echo FATAL metric; exit 1; }
python -u -m experiments.dataset_sensitivity.arm_c_imbalance --stage0 --minority_class 0 --device cuda || { echo FATAL stage0; exit 1; }
echo "########## ROLE-SWAP FULL: minority_class=0, m in {1,2,4,8}, N=16, K=50 ##########"; date
python -u -m experiments.dataset_sensitivity.arm_c_imbalance \
    --m_list 1 2 4 8 --N 16 --K 50 --n_targets_per_class 3 --lr 0.5 --T 1000 --rank 8 \
    --minority_class 0 --device cuda
echo "=== DONE $(date) ==="
echo "READ: balanced(m=8) ratio should be ~1/3.3=0.30 (identity inverts); rarity effect (ratio/balanced"
echo "      rising as m shrinks) should REPLICATE the minority_class=1 run (~2x at m=1)."
