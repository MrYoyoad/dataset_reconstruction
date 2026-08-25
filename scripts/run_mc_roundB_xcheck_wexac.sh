#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_roundB_xcheck_%J.out
#BSUB -e scripts/wexac_logs/mc_roundB_xcheck_%J.err
#BSUB -J mc_Bxchk

# =====================================================================
# ROUND B CROSS-CHECK (yoado-35) — the reversal (10-class q_eff LOWER than binary)
# was measured at lr=0.5 where the 10-class arm is UNDERFIT (one hard sample; q_eff
# still RISING 73->81 with T, not plateaued) while binary IS converged. Same
# underfit-arm trap we killed the amplification for -> the reversal could be the
# asymmetry. Re-measure at lr=1.0/T=500 where the PROBE showed BOTH bases memorize
# (binary YES@T200, 10-class YES@T500). FD-GATE the lr=1.0 J first (high lr = chaos
# risk); if it fails, this recipe is unusable and the reversal stays lr=0.5-only.
# Read: reversal HOLDS at matched convergence -> "CE self-protects under noise" real;
# NARROWS/vanishes -> it was the convergence asymmetry. Also re-check iso_ratio at
# matched convergence. k=8 (Nk=160) S={640,1280}, both bases, mnist.
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

echo ""; echo "########## DEEP-UNROLL FD GATE (lr=1.0, T=500) — reject chaotic high-lr J ##########"; date
python -u -c "
import torch, inspect, experiments.jacobian_spectrum as j
torch.set_default_dtype(torch.float64)
ctx,_,_,_ = j._mnist_ctx(N=4,k=4,T=500,rank=8,activation='gelu',seed=42,device='cuda',
                         tangent_method='qr',dataset='mnist',num_classes=10,classes_present=2,lr=1.0)
a0=torch.zeros(16,dtype=torch.float64,device='cuda')
J=j.exact_jacobian(a0,ctx,method='jvp_double')
coords=[0,5,10]; fd=j.finite_difference_jacobian(a0,ctx,coords,eps=1e-5)
rel=max((J[:,c]-fd[c]).norm().item()/(fd[c].norm().item()+1e-30) for c in coords)
nan=bool(torch.isnan(J).any() or torch.isinf(J).any())
print(f'DEEP-FD rel err (lr=1.0,T=500,N4k4): {rel:.3e}  NaN/Inf={nan}')
assert rel<1e-4 and not nan, 'lr=1.0 J FAILS FD/NaN (chaotic) — cross-check recipe unusable'
print('DEEP-FD GATE PASSED (lr=1.0 J is clean)')
"
if [ $? -ne 0 ]; then echo "FATAL: lr=1.0 deep-FD gate failed — high-lr J is chaotic, cannot trust its q_eff. Reversal stays lr=0.5-provisional."; exit 1; fi

echo ""; echo "########## max_bce check: lr=1.0/T=500 both bases (both 20/20?) ##########"; date
for NC in 2 10; do
  echo ""; echo "-- rigor mnist nc=$NC N=20 k=8 lr=1.0 T=500 (max_bce) --"; date
  python -u -m experiments.jacobian_spectrum --rigor --dataset mnist --activation gelu \
      --num_classes $NC --N 20 --k 8 --rank 8 --tangent qr \
      --Ts 500 --lr 1.0 --seed 42 --save --device cuda --tag Bxchk_memchk_nc${NC}
done

echo ""; echo "########## CROSS-CHECK q_eff: lr=1.0, T=500, S{640,1280}, both bases ##########"; date
for NC in 2 10; do for S in 640 1280; do
  echo ""; echo "-- j1 mnist nc=$NC N=20 k=8 lr=1.0 T=500 S=$S (cross-check q_eff) --"; date
  python -u -m experiments.jacobian_spectrum --j1 --dataset mnist --activation gelu \
      --num_classes $NC --N 20 --k 8 --T 500 --rank 8 --tangent qr --lr 1.0 \
      --S_list $S --shrink_list 0.01 --eps_list 0.1 0.3 1.0 3.0 10.0 --save \
      --tag Bxchk_mnist_nc${NC}_S${S} --device cuda
done; done

echo ""; echo "=== DONE $(date) ==="
