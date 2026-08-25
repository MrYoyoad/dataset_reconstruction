#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_roundB_plateau_%J.out
#BSUB -e scripts/wexac_logs/mc_roundB_plateau_%J.err
#BSUB -J mc_Bplat

# =====================================================================
# ROUND B PLATEAU CHECK — the lr=1.0 cross-check FAILED its FD gate (chaotic J,
# rel err 2.7e-2), so we can't converge the 10-class hard sample without breaking
# the exact Jacobian. Instead address yoado-35's ACTUAL concern (10-class q_eff
# "still RISING 73->81, not plateaued") in the CLEAN lr=0.5 regime: push T DEEPER
# (T=1500, 2000) and see if q_eff PLATEAUS. If 10-class q_eff flattens (~81-85) and
# stays FAR below binary (~117) -> the reversal is robust to the residual underfit
# (the rise stops; it does NOT climb toward binary). FD-gate at lr=0.5/T=2000 first
# (lr=0.5/T=1000 passed at 2.5e-8; deeper may or may not). Also re-check iso_ratio.
# k=8, S=640, mnist, lr=0.5. Binary T=2000 as the converged reference (should be flat).
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

echo ""; echo "########## DEEP-FD GATE (lr=0.5, T=2000) — is the deeper clean-lr J still valid? ##########"; date
python -u -c "
import torch, experiments.jacobian_spectrum as j
torch.set_default_dtype(torch.float64)
ctx,_,_,_ = j._mnist_ctx(N=4,k=4,T=2000,rank=8,activation='gelu',seed=42,device='cuda',
                         tangent_method='qr',dataset='mnist',num_classes=10,classes_present=2,lr=0.5)
a0=torch.zeros(16,dtype=torch.float64,device='cuda')
J=j.exact_jacobian(a0,ctx,method='jvp_double')
coords=[0,5,10]; fd=j.finite_difference_jacobian(a0,ctx,coords,eps=1e-5)
rel=max((J[:,c]-fd[c]).norm().item()/(fd[c].norm().item()+1e-30) for c in coords)
nan=bool(torch.isnan(J).any() or torch.isinf(J).any())
print(f'DEEP-FD rel err (lr=0.5,T=2000,N4k4): {rel:.3e}  NaN/Inf={nan}')
assert rel<1e-4 and not nan, 'lr=0.5/T=2000 J FAILS FD — deeper clean-lr unroll also chaotic'
print('DEEP-FD GATE PASSED (lr=0.5/T=2000 J clean)')
"
if [ $? -ne 0 ]; then echo "FATAL: lr=0.5/T=2000 FD failed — even the clean lr goes chaotic this deep; plateau unreachable."; exit 1; fi

echo ""; echo "########## q_eff PLATEAU: lr=0.5, T{1500,2000}, S=640, both bases ##########"; date
for NC in 2 10; do for T in 1500 2000; do
  echo ""; echo "-- j1 mnist nc=$NC N=20 k=8 lr=0.5 T=$T S=640 (plateau check) --"; date
  python -u -m experiments.jacobian_spectrum --j1 --dataset mnist --activation gelu \
      --num_classes $NC --N 20 --k 8 --T $T --rank 8 --tangent qr --lr 0.5 \
      --S_list 640 --shrink_list 0.01 --eps_list 0.1 0.3 1.0 3.0 10.0 --save \
      --tag Bplat_mnist_nc${NC}_T${T} --device cuda
done; done

echo ""; echo "=== DONE $(date) ==="
