#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_roundB_%J.out
#BSUB -e scripts/wexac_logs/mc_roundB_%J.err
#BSUB -J mc_roundB

# =====================================================================
# ROUND B — CLEAN q_eff at CONVERGENCE (the last fundamental test after Round 0
# retracted the r_J amplification). Recipe (memprobe 399120 + yoado-35 audit):
# lr=0.5, T∈{500,1000}, SAME both bases (lowest well-conditioned lr; lr=1.0 is the
# chaotic reserve). q_eff-PLATEAU criterion (stable across T500↔T1000), NOT strict
# BCE. k=8 (Nk=160) S∈{640,1280}=4·Nk,8·Nk for the adequacy+stability gates.
# Value-only Σ_seed unroll (create_graph=False) makes S=1280 feasible.
# Quote q_eff ONLY if: (i) deep-FD gate passes, (ii) eff_rank(Σ_seed)≳r_J, (iii)
# q_eff stable T500↔T1000 AND S640↔S1280. THE question: at convergence (both bases
# full r_J), does CE q_eff > binary? If yes → conditional lr=1.0 cross-check (reserve).
# If no → the fundamental-CE story is fully dead. (hgn46/lgn28 excluded.)
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

echo ""; echo "########## DEEP-UNROLL FD GATE (lr=0.5, T=1000) — reject chaotic J ##########"; date
python -u -c "
import torch, inspect, experiments.jacobian_spectrum as j
assert 'lr' in inspect.signature(j.run_j1).parameters, 'run_j1 lr NOT wired — Round B would use default lr'
print('run_j1 lr-wire OK')
torch.set_default_dtype(torch.float64)
ctx,_,_,_ = j._mnist_ctx(N=4,k=4,T=1000,rank=8,activation='gelu',seed=42,device='cuda',
                         tangent_method='qr',dataset='mnist',num_classes=10,classes_present=2,lr=0.5)
a0=torch.zeros(16,dtype=torch.float64,device='cuda')
J=j.exact_jacobian(a0,ctx,method='jvp_double')
coords=[0,5,10]; fd=j.finite_difference_jacobian(a0,ctx,coords,eps=1e-5)
rel=max((J[:,c]-fd[i]).norm().item()/(fd[i].norm().item()+1e-30) for i,c in enumerate(coords))
nan=bool(torch.isnan(J).any() or torch.isinf(J).any())
print(f'DEEP-FD rel err (lr=0.5,T=1000,N4k4): {rel:.3e}  NaN/Inf={nan}')
assert rel<1e-4 and not nan, 'deep-unroll J FAILS FD/NaN — reject lr=0.5/T=1000 recipe'
print('DEEP-FD GATE PASSED')
"
if [ $? -ne 0 ]; then echo "FATAL: deep-unroll FD gate failed — do NOT trust Round B q_eff at this recipe."; exit 1; fi

echo ""; echo "########## STEP 1: max_bce at lr=0.5/T=1000 (is 10-class 20/20 = clean recipe?) ##########"; date
for NC in 2 10; do
  echo ""; echo "-- rigor mnist nc=$NC N=20 k=8 lr=0.5 Ts{500,1000} (max_bce check) --"; date
  python -u -m experiments.jacobian_spectrum --rigor --dataset mnist --activation gelu \
      --num_classes $NC --N 20 --k 8 --rank 8 --tangent qr \
      --Ts 500 1000 --lr 0.5 --seed 42 --save --device cuda --tag roundB_memchk_nc${NC}
done

echo ""; echo "########## ROUND B: clean q_eff, lr=0.5, T{500,1000} x S{640,1280}, both bases ##########"; date
for NC in 2 10; do for T in 500 1000; do for S in 640 1280; do
  echo ""; echo "-- j1 mnist nc=$NC N=20 k=8 lr=0.5 T=$T S=$S (clean q_eff) --"; date
  python -u -m experiments.jacobian_spectrum --j1 --dataset mnist --activation gelu \
      --num_classes $NC --N 20 --k 8 --T $T --rank 8 --tangent qr --lr 0.5 \
      --S_list $S --shrink_list 0.01 --eps_list 0.1 0.3 1.0 3.0 10.0 --save \
      --tag roundB_mnist_nc${NC}_T${T}_S${S} --device cuda
done; done; done

echo ""; echo "=== DONE $(date) ==="
