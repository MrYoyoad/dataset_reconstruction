#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/leakage_multiclass_%J.out
#BSUB -e scripts/wexac_logs/leakage_multiclass_%J.err
#BSUB -J leak_mc

# =====================================================================
# TIER B leakage: does K-class CE AMPLIFY leakage vs binary? (yoado-8a's
# hypothesis: CE injects a (K-1)-dim per-sample residual vs BCE's 1-dim -> col(J)
# gains rank, q_eff rises; saturates at r_J; needs frozen[out] well-conditioned.)
#
# PRIMARY (clean, base-confound removed): hold the 10-class base θ0 FIXED, vary
# classes-PRESENT K_eff in {2,5,10} at fixed N=20, read q_eff vs K_eff. Only the
# output-residual width changes. run_j1 also prints frozen[out] cond/eff_rank
# (the gate on whether the extra directions can inject).
# SECONDARY (base-confounded, the "realistic" number): binary-θ0 vs 10-class-θ0
# at matched N — labeled as confounded (different frozen weights).
# Needs job 231594's 10-class bases (weights-<ds>10_<act>.pth). GELU (exact-J).
# SUBMIT ONLY AFTER 231594 completes AND its 10-class test accuracies verified.
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

echo ""; echo "########## A) PRIMARY: q_eff vs K_eff on FIXED 10-class base (N=20) ##########"; date
for DS in mnist fashion; do
  for KEFF in 2 5 10; do
    echo ""; echo "-- j1 $DS gelu N=20 num_classes=10 classes_present=$KEFF T=50 --"; date
    python -u -m experiments.jacobian_spectrum --j1 \
        --dataset $DS --activation gelu --tangent qr \
        --num_classes 10 --classes_present $KEFF \
        --N 20 --k 8 --T 50 --rank 8 --S_list 64 --shrink_list 0.01 \
        --eps_list 0.1 0.3 1.0 3.0 10.0 --save --device cuda \
        --tag ${DS}_gelu_N20_K10_keff${KEFF}
  done
done

echo ""; echo "########## B) SECONDARY (base-confounded): binary vs 10-class, matched N ##########"; date
for DS in mnist fashion; do
  for NC in 2 10; do
    echo ""; echo "-- j1 $DS gelu N=20 num_classes=$NC T=50 (base-confounded) --"; date
    python -u -m experiments.jacobian_spectrum --j1 \
        --dataset $DS --activation gelu --tangent qr --num_classes $NC \
        --N 20 --k 8 --T 50 --rank 8 --S_list 64 --shrink_list 0.01 \
        --eps_list 0.1 0.3 1.0 3.0 10.0 --save --device cuda \
        --tag ${DS}_gelu_N20_nc${NC}
  done
done

echo ""; echo "########## C) rigor: leakage+memorization+held-acc vs T (10-class, N=20) ##########"; date
for DS in mnist fashion; do
  echo ""; echo "-- rigor $DS gelu num_classes=10 N=20 lr=0.1 --"; date
  python -u -m experiments.jacobian_spectrum --rigor \
      --dataset $DS --activation gelu --num_classes 10 \
      --N 20 --k 8 --rank 8 --tangent qr \
      --Ts 5 20 50 100 200 --lr 0.1 --seed 42 --save --device cuda \
      --tag ${DS}_gelu_nc10
done

echo ""; echo "=== DONE $(date) ==="
