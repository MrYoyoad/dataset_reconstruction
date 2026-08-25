#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=24576] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_rank_sweep_%J.out
#BSUB -e scripts/wexac_logs/mc_rank_sweep_%J.err
#BSUB -J mc_ranksweep

# =====================================================================
# LoRA RANK SWEEP (notes/lora_rank_sweep_plan.md, yoado-19 + yoado-35 audit-clean).
# Is the CONFIRMED reversal (10-class q_eff < binary at convergence; iso 10>bin)
# rank-robust across r ∈ {2,4,8,16,32}? Headline split at r=N=10 (r>=16 ~ full-FT,
# Jang 2024). Pure param sweep of the LOCKED config: N=10 k=8 (Nk=80) T=1000 lr=0.5
# S=320(=4·Nk) gelu qr seed42, both bases. Anchor r=8 MUST reproduce iso 0.49/0.68,
# q_eff|col(J) ε=1 = 59/36. Gates: r threaded EVERYWHERE incl FD (gate #0); r_J-per-r
# master validity (r_J=80→raw counts, r_J<80→fraction q_eff/r_J); max_bce<1e-3 per r
# (r=2 may not memorize); FD-gate each config abort-on-fail — r=32 FD-fail = BOUND OUT,
# do NOT change recipe; {S,2S} stability at r=8 AND r=32. mnist full 5-r; fashion r∈{8,16}.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: AD gate failed."; exit 1; fi

# FULL-config FD gate — threads dataset AND rank (gate #0). args: DS NC N k T lr rank
fdgate () {
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

run_cell () { # DS NC rank SLIST
  local DS=$1 NC=$2 R=$3 SL=$4
  fdgate $DS $NC 10 8 1000 0.5 $R || { echo "[$DS nc=$NC rank=$R FD-CHAOTIC -> BOUNDED OUT]"; return; }
  echo "-- rigor+j1 $DS nc=$NC N=10 k=8 rank=$R S={$SL} --"; date
  python -u -m experiments.jacobian_spectrum --rigor --dataset $DS --activation gelu \
      --num_classes $NC --N 10 --k 8 --rank $R --tangent qr --Ts 1000 --lr 0.5 --seed 42 --device cuda \
      --tag ranksweep_memchk_${DS}_nc${NC}_r${R}
  python -u -m experiments.jacobian_spectrum --j1 --dataset $DS --activation gelu \
      --num_classes $NC --N 10 --k 8 --T 1000 --rank $R --tangent qr --lr 0.5 \
      --S_list $SL --shrink_list 0.01 --eps_list 0.1 0.3 1.0 3.0 10.0 --save \
      --tag ranksweep_${DS}_nc${NC}_r${R} --device cuda
}

echo ""; echo "########## MNIST full sweep: r ∈ {2,4,8,16,32} ##########"; date
for R in 2 4 8 16 32; do
  echo ""; echo "===== rank=$R (r vs N=10: $( [ $R -ge 10 ] && echo 'r>=N ~full-FT' || echo 'r<N low-rank' )) ====="; date
  # {S,2S} stability at the anchor (r=8) and stress case (r=32); else S=320 only.
  if [ $R -eq 8 ] || [ $R -eq 32 ]; then SL="320 640"; else SL="320"; fi
  for NC in 2 10; do run_cell mnist $NC $R "$SL"; done
done

echo ""; echo "########## FASHION crossing check: r ∈ {8,16} ##########"; date
for R in 8 16; do
  echo ""; echo "===== fashion rank=$R ====="; date
  for NC in 2 10; do run_cell fashion $NC $R "320"; done
done

echo ""; echo "=== DONE $(date) ==="
echo "READ: r_J-per-r FIRST (80=raw counts, <80=fraction q_eff/r_J); anchor r=8 must give iso 0.49/0.68 q_eff@ε1 59/36; reversal HOLD vs BREAK at r>=N."
