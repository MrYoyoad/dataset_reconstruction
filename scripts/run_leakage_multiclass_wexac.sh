#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/leakage_multiclass_%J.out
#BSUB -e scripts/wexac_logs/leakage_multiclass_%J.err
#BSUB -J leak_mc

# =====================================================================
# TIER B leakage: does K-class CE AMPLIFY leakage vs binary BCE? (yoado-8a's
# hypothesis: CE injects a (K-1)-dim per-sample residual vs BCE's 1-dim -> col(J)
# can gain rank, q_eff can RISE). Runs on the 10-class honest bases (job 231594:
# weights-<ds>10_<act>.pth). GELU only for the exact-J arm. N-SWEEP up to N>=K=10
# (the effect needs classes represented). Compare eff_rank/q_eff to the binary
# tables (STATUS "Orthonormal-vs-PCA-variant secret").
# SUBMIT ONLY AFTER 231594 completes AND its 10-class test accuracies are verified.
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

echo ""; echo "########## A) J1 leakage: multi-class vs binary, N-sweep ##########"; date
for DS in mnist fashion; do
  for NC in 2 10; do          # 2 = binary baseline (same code), 10 = multi-class
    for N in 10 20; do        # N >= K so classes are represented
      echo ""; echo "-- j1 $DS gelu N=$N num_classes=$NC T=50 tangent=qr --"; date
      python -u -m experiments.jacobian_spectrum --j1 \
          --dataset $DS --activation gelu --tangent qr --num_classes $NC \
          --N $N --k 8 --T 50 --rank 8 --S_list 64 --shrink_list 0.01 \
          --eps_list 0.1 0.3 1.0 3.0 10.0 --save --device cuda \
          --tag ${DS}_gelu_N${N}_nc${NC}
    done
  done
done

echo ""; echo "########## B) rigor: leakage+memorization+held-acc vs T (10-class) ##########"; date
for DS in mnist fashion; do
  echo ""; echo "-- rigor $DS gelu num_classes=10 N=20 lr=0.1 --"; date
  python -u -m experiments.jacobian_spectrum --rigor \
      --dataset $DS --activation gelu --num_classes 10 \
      --N 20 --k 8 --rank 8 --tangent qr \
      --Ts 5 20 50 100 200 --lr 0.1 --seed 42 --save --device cuda \
      --tag ${DS}_gelu_nc10
done

echo ""; echo "=== DONE $(date) ==="
