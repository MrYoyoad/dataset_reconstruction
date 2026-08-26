#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=24576] select[ngpus>0 && hname!='lgn28' && hname!='hgn46' && hname!='hgn45']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/fashion_mc_rerun_%J.out
#BSUB -e scripts/wexac_logs/fashion_mc_rerun_%J.err
#BSUB -J fashion_r16_rerun

# =====================================================================
# FASHION 10-class RE-RUN — the two cells the rank sweep (581629) missed:
#  - fashion r=8  nc=10: FD-CHAOTIC (NaN) -> expected to re-bound-out (deterministic).
#  - fashion r=16 nc=10: rigor CONVERGED but q_eff crashed on a raw-cloud SVD
#    (cuSOLVER err 319). SVD diagnostic now has a gesvd fallback + skip-on-fail
#    (q_eff uses col(J) whitening, unaffected) -> this cell should now yield q_eff.
# Same locked recipe as the sweep (N=10 k=8 T=1000 lr=0.5 S=320 gelu qr seed42),
# so the numbers are directly comparable to the mnist reversal table + fashion binary.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: AD gate failed."; exit 1; fi

fdgate () { # DS NC N k T lr rank
  python -u -c "
import torch, experiments.jacobian_spectrum as j
torch.set_default_dtype(torch.float64)
ctx,_,_,_ = j._mnist_ctx(N=$3,k=$4,T=$5,rank=$7,activation='gelu',seed=42,device='cuda',
                         tangent_method='qr',dataset='$1',num_classes=$2,classes_present=None,lr=$6)
Nk=$3*$4; a0=torch.zeros(Nk,dtype=torch.float64,device='cuda')
J=j.exact_jacobian(a0,ctx,method='jvp_double')
cs=[0,Nk//2,Nk-1]; fd=j.finite_difference_jacobian(a0,ctx,cs,eps=1e-5)
rel=max((J[:,c]-fd[c]).norm().item()/(fd[c].norm().item()+1e-30) for c in cs)
nan=bool(torch.isnan(J).any() or torch.isinf(J).any())
print(f'  FULL-FD $1 nc=$2 N=$3 k=$4 T=$5 lr=$6 rank=$7: dimY={J.shape[0]} {rel:.3e} NaN={nan} -> {\"CLEAN\" if rel<1e-4 and not nan else \"CHAOTIC\"}')
assert rel<1e-4 and not nan, 'full-config J FAILS FD'
"
}

run_cell () { # rank
  local R=$1
  fdgate fashion 10 10 8 1000 0.5 $R || { echo "[fashion nc=10 rank=$R FD-CHAOTIC -> BOUNDED OUT (deterministic, matched recipe)]"; return; }
  echo "-- rigor+j1 fashion nc=10 N=10 k=8 rank=$R --"; date
  python -u -m experiments.jacobian_spectrum --rigor --dataset fashion --activation gelu \
      --num_classes 10 --N 10 --k 8 --rank $R --tangent qr --Ts 1000 --lr 0.5 --seed 42 --device cuda \
      --tag fashionrerun_memchk_r${R}
  python -u -m experiments.jacobian_spectrum --j1 --dataset fashion --activation gelu \
      --num_classes 10 --N 10 --k 8 --T 1000 --rank $R --tangent qr --lr 0.5 \
      --S_list 320 --shrink_list 0.01 --eps_list 0.1 0.3 1.0 3.0 10.0 --save \
      --tag fashionrerun_r${R} --device cuda
}

for R in 16; do
  echo ""; echo "===== fashion nc=10 rank=$R ====="; date
  run_cell $R
done

echo ""; echo "=== DONE $(date) ==="
echo "READ: fashion r16 nc10 q_eff@ε1 vs fashion r16 nc2 (35) -> reversal on fashion? r8 nc10 expected bound-out."
