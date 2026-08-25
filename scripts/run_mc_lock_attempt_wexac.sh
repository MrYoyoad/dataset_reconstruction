#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_lock_%J.out
#BSUB -e scripts/wexac_logs/mc_lock_%J.err
#BSUB -J mc_lock

# =====================================================================
# LOCK ATTEMPT — the lr-window probe (481883) found NO strictly-clean-converged lr
# (lr=0.6 FD-clean but hard sample 1.26e-3; lr=0.7 chaotic FD=1.0). Two remaining
# clean angles (yoado-35), both with FULL-CONFIG FD gates (guard #1) and BOTH bases
# checked (guard #2):
#  A) lr=0.6/T=500, N=20 — the MOST-converged FD-clean N=20 recipe (1.26e-3 < 0.5's 1.8e-3).
#  B) N=10 corroboration (--N 10, Nk=80): 10-class 1/class -> far easier to memorize
#     -> likely BOTH bases converge cleanly at the SAFE lr=0.5. Different-N support for
#     the reversal as a general training-MAP property (S>=4*80=320).
# Read: reversal (10-class q_eff < binary) holding at (A) more-converged N=20 AND (B)
# both-converged N=10 => robust despite the exact-lock wall.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: AD gate failed."; exit 1; fi

# ---- FULL-CONFIG FD gate helper (real [14272,Nk] J, 3 coords) ----
fdgate () { # args: NC N k T lr
  python -u -c "
import torch, experiments.jacobian_spectrum as j
torch.set_default_dtype(torch.float64)
ctx,_,_,_ = j._mnist_ctx(N=$2,k=$3,T=$4,rank=8,activation='gelu',seed=42,device='cuda',
                         tangent_method='qr',dataset='mnist',num_classes=$1,
                         classes_present=None,lr=$5)
Nk=$2*$3; a0=torch.zeros(Nk,dtype=torch.float64,device='cuda')
J=j.exact_jacobian(a0,ctx,method='jvp_double')
cs=[0,Nk//2,Nk-1]; fd=j.finite_difference_jacobian(a0,ctx,cs,eps=1e-5)
rel=max((J[:,c]-fd[c]).norm().item()/(fd[c].norm().item()+1e-30) for c in cs)
nan=bool(torch.isnan(J).any() or torch.isinf(J).any())
print(f'  FULL-FD nc=$1 N=$2 k=$3 T=$4 lr=$5: {rel:.3e} NaN={nan} -> {\"CLEAN\" if rel<1e-4 and not nan else \"CHAOTIC\"}')
assert rel<1e-4 and not nan, 'full-config J FAILS FD'
"
}

echo ""; echo "########## A) lr=0.6/T=500, N=20 — full-FD both bases ##########"; date
fdgate 2 20 8 500 0.6 && fdgate 10 20 8 500 0.6 || { echo "A: full-FD failed at lr=0.6 — skip A"; }
for NC in 2 10; do
  echo ""; echo "-- rigor+j1 $NC N=20 k=8 lr=0.6 T=500 --"; date
  python -u -m experiments.jacobian_spectrum --rigor --dataset mnist --activation gelu \
      --num_classes $NC --N 20 --k 8 --rank 8 --tangent qr --Ts 500 --lr 0.6 --seed 42 --device cuda --tag lockA_memchk_nc${NC}
  python -u -m experiments.jacobian_spectrum --j1 --dataset mnist --activation gelu \
      --num_classes $NC --N 20 --k 8 --T 500 --rank 8 --tangent qr --lr 0.6 \
      --S_list 640 --shrink_list 0.01 --eps_list 0.1 0.3 1.0 3.0 10.0 --save --tag lockA_mnist_nc${NC} --device cuda
done

echo ""; echo "########## B) N=10 corroboration, lr=0.5/T=1000 — full-FD both bases ##########"; date
fdgate 2 10 8 1000 0.5 && fdgate 10 10 8 1000 0.5 || { echo "B: full-FD failed — skip B"; }
for NC in 2 10; do
  echo ""; echo "-- rigor+j1 $NC N=10 k=8 lr=0.5 T=1000 --"; date
  python -u -m experiments.jacobian_spectrum --rigor --dataset mnist --activation gelu \
      --num_classes $NC --N 10 --k 8 --rank 8 --tangent qr --Ts 1000 --lr 0.5 --seed 42 --device cuda --tag lockB_memchk_nc${NC}
  python -u -m experiments.jacobian_spectrum --j1 --dataset mnist --activation gelu \
      --num_classes $NC --N 10 --k 8 --T 1000 --rank 8 --tangent qr --lr 0.5 \
      --S_list 320 --shrink_list 0.01 --eps_list 0.1 0.3 1.0 3.0 10.0 --save --tag lockB_mnist_nc${NC} --device cuda
done

echo ""; echo "=== DONE $(date) ==="
