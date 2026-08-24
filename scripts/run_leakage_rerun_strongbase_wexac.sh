#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/leakage_rerun_strongbase_%J.out
#BSUB -e scripts/wexac_logs/leakage_rerun_strongbase_%J.err
#BSUB -J leak_rerun

# =====================================================================
# RE-RUN the leakage suites on the TIER-A STRONG base (job 224204: full-data +
# healthy-init binary bases, ~97-98%). Same experiments as the 88%-base runs
# (215013 qr / 215289 pca-variants / 201658+201904 rigor) so we can see whether
# a strong base changes leakage. Loads canonical models/weights-<ds>_<act>.pth
# (now the strong base) UNCHANGED — no code change (still binary, output_dim=1).
# SUBMIT ONLY AFTER 224204 completes AND its test accuracies are verified.
#   datasets {mnist,fashion} x acts {gelu,relu} x T {5,50}; rigor at lr=0.1.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## STAGE 0: AD gate ##########"; date
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: AD gate failed. Aborting."; exit 1; fi

echo ""; echo "########## A) qr leakage (orthonormal secret, STRONG base) ##########"; date
for DS in mnist fashion; do for ACT in gelu relu; do for T in 5 50; do
  echo ""; echo "-- j1 $DS $ACT T=$T tangent=qr [strongbase] --"; date
  python -u -m experiments.jacobian_spectrum --j1 \
      --dataset $DS --activation $ACT --tangent qr \
      --N 4 --k 8 --T $T --rank 8 --S_list 64 --shrink_list 0.01 \
      --eps_list 0.1 0.3 1.0 3.0 10.0 --save --device cuda
done; done; done

echo ""; echo "########## B) pca-variants leakage (STRONG base) ##########"; date
for DS in mnist fashion; do for ACT in gelu relu; do for T in 5 50; do
  echo ""; echo "-- h1 $DS $ACT T=$T (pca variants + qr) [strongbase] --"; date
  python -u -m experiments.jacobian_spectrum --h1 \
      --dataset $DS --activation $ACT --N 4 --k 8 --T $T --rank 8 \
      --h1_methods pca difference pca_tail residual qr \
      --eps_list 0.1 1.0 10.0 --seed 42 --save --device cuda
done; done; done

echo ""; echo "########## C) rigor: leakage+memorization+held-acc vs T (STRONG base, lr=0.1) ##########"; date
for DS in mnist fashion; do for ACT in gelu relu; do
  echo ""; echo "-- rigor $DS $ACT lr=0.1 [strongbase] --"; date
  python -u -m experiments.jacobian_spectrum --rigor \
      --dataset $DS --activation $ACT --N 4 --k 8 --rank 8 --tangent qr \
      --Ts 5 20 50 100 200 --lr 0.1 --seed 42 --save --device cuda
done; done

echo ""; echo "=== DONE $(date) ==="
