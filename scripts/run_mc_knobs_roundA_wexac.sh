#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_roundA_%J.out
#BSUB -e scripts/wexac_logs/mc_roundA_%J.err
#BSUB -J mc_roundA

# =====================================================================
# ROUND A — the binary-vs-multiclass MASTER TABLE across all 3 knobs, r_J-only
# (S-invariant → ONE minimal S=16, do NOT sweep S; Round B does clean q_eff).
# Knob 1 DIRECTIONS (run_h1), Knob 2 SCHEMES (run_schemes, now multi-class-wired),
# Knob 3 N-SWEEP + k-LADDER (run_j1). All at GELU exact-J, N=20 headline (+ N-sweep),
# num_classes {2 binary, 10}. k-ladder {8,16,32} at N=20 = the DOMAIN-vs-MEASUREMENT
# axis (Round 0 showed k=8 saturates to r_J=Nk=160 for BOTH bases by T=200 → need
# headroom). READ r_J & q_eff, NEVER eff_rank (shape only). (hgn46 ECC / lgn28 slow excluded.)
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

echo ""; echo "########## Knob 3: N-SWEEP (k=8) — N-collapse read, both bases ##########"; date
for DS in mnist fashion; do for NC in 2 10; do for N in 2 4 10 20; do
  echo ""; echo "-- j1 $DS nc=$NC N=$N k=8 (N-sweep) --"; date
  python -u -m experiments.jacobian_spectrum --j1 --dataset $DS --activation gelu \
      --num_classes $NC --N $N --k 8 --T 50 --rank 8 --tangent qr \
      --S_list 16 --shrink_list 0.01 --eps_list 0.1 1.0 3.0 10.0 --save \
      --tag roundA_nsweep_${DS}_nc${NC}_N${N}_k8 --device cuda
done; done; done

echo ""; echo "########## Knob 3: k-LADDER at N=20 — DOMAIN-vs-MEASUREMENT, both bases ##########"; date
for DS in mnist fashion; do for NC in 2 10; do for K in 16 32; do
  echo ""; echo "-- j1 $DS nc=$NC N=20 k=$K (k-ladder headroom) --"; date
  python -u -m experiments.jacobian_spectrum --j1 --dataset $DS --activation gelu \
      --num_classes $NC --N 20 --k $K --T 50 --rank 8 --tangent qr \
      --S_list 16 --shrink_list 0.01 --eps_list 0.1 1.0 3.0 10.0 --save \
      --tag roundA_kladder_${DS}_nc${NC}_N20_k${K} --device cuda
done; done; done

echo ""; echo "########## Knob 1: DIRECTIONS (5 methods) at N=20 k=8, both bases ##########"; date
for DS in mnist fashion; do for NC in 2 10; do
  echo ""; echo "-- h1 $DS nc=$NC N=20 k=8 (directions) --"; date
  python -u -m experiments.jacobian_spectrum --h1 --dataset $DS --activation gelu \
      --num_classes $NC --N 20 --k 8 --T 50 --rank 8 \
      --h1_methods pca difference pca_tail residual qr \
      --eps_list 0.1 1.0 10.0 --seed 42 --save \
      --tag roundA_dir_${DS}_nc${NC} --device cuda
done; done

echo ""; echo "########## Knob 2: SCHEMES (DIFFERENT/SAME/MIXTURE) at N=20 k=8, both bases ##########"; date
for DS in mnist fashion; do for NC in 2 10; do
  echo ""; echo "-- schemes $DS nc=$NC N=20 k=8 --"; date
  python -u -m experiments.jacobian_spectrum --schemes --dataset $DS --activation gelu \
      --num_classes $NC --N 20 --k 8 --T 50 --rank 8 --tangent qr --save \
      --tag roundA_schemes_${DS}_nc${NC} --device cuda
done; done

echo ""; echo "=== DONE $(date) ==="
