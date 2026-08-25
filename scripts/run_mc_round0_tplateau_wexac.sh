#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_round0_%J.out
#BSUB -e scripts/wexac_logs/mc_round0_%J.err
#BSUB -J mc_round0

# =====================================================================
# ROUND 0 — T-PLATEAU CONVERGENCE CONTROL (load-bearing; gates the headline).
# The "multi-class amplifies leakage" headline (r_J 99/107 -> 160) was measured at
# T=50 where the binary base may be UNDERFIT -> the gap could be CE converging
# FASTER, not leaking more. Settle it: sweep T PAST memorization for BOTH bases at
# the SAME lr, read the r_J-vs-T CURVE. Does binary r_J climb toward 160 (=> "2x"
# is a training-speed artifact) or stay ~99 (=> real measurement gap)?
#   both bases (num_classes 2 & 10) x {mnist,fashion} x k{8 headline, 32 headroom}
#   fixed lr=0.1, Ts swept to plateau. r_J-only (S-invariant -> cheap). GELU exact-J.
# Headline stays PROVISIONAL until this reads out. (hgn46 ECC-faulty, lgn28 slow: excluded.)
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## STAGE 0: AD gate (binary + CE) ##########"; date
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: AD gate failed. Aborting."; exit 1; fi

# k=8 (Nk=160, the headline config): full T-ladder past memorization, both bases.
echo ""; echo "########## Round 0 k=8 (Nk=160, headline): r_J-vs-T, both bases ##########"; date
for DS in mnist fashion; do
  for NC in 2 10; do
    echo ""; echo "-- rigor $DS num_classes=$NC N=20 k=8 lr=0.1 (T-plateau) --"; date
    python -u -m experiments.jacobian_spectrum --rigor \
        --dataset $DS --activation gelu --num_classes $NC \
        --N 20 --k 8 --rank 8 --tangent qr \
        --Ts 5 50 200 500 --lr 0.1 --seed 42 --save --device cuda \
        --tag round0_${DS}_nc${NC}_k8
  done
done

# k=32 (Nk=640, headroom): lighter ladder — checks whether r_J needs larger T once
# it's not pinned at the domain ceiling (feeds the Round A k-ladder read).
echo ""; echo "########## Round 0 k=32 (Nk=640, headroom): r_J-vs-T check ##########"; date
for DS in mnist fashion; do
  for NC in 2 10; do
    echo ""; echo "-- rigor $DS num_classes=$NC N=20 k=32 lr=0.1 (headroom T-check) --"; date
    python -u -m experiments.jacobian_spectrum --rigor \
        --dataset $DS --activation gelu --num_classes $NC \
        --N 20 --k 32 --rank 8 --tangent qr \
        --Ts 5 50 200 --lr 0.1 --seed 42 --save --device cuda \
        --tag round0_${DS}_nc${NC}_k32
  done
done

echo ""; echo "=== DONE $(date) ==="
