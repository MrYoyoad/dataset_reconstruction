#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_lowrank_conv_%J.out
#BSUB -e scripts/wexac_logs/mc_lowrank_conv_%J.err
#BSUB -J mc_lowrank_conv

# =====================================================================
# LOW-RANK CONVERGENCE FOLLOW-UP to the rank sweep (job 581629).
# Loose end: at the locked recipe (lr=0.5/T=1000) the 10-class base is UNDERFIT at
# r=2 (max_bce 6.1e-3) and r=4 (1.9e-3) — so the r<8 reversal AT CONVERGENCE is unknown
# (those cells were convergence-confounded, correctly excluded from the headline).
# GOAL: drive low-rank 10-class to memorization (max_bce<1e-3) with MORE optimization,
# then measure q_eff — does the reversal exist / strengthen at deep low rank?
#
# TENSION (the real science): the clean-J differentiable island is lr<=0.6 / T<=1000
# (characterized). Converging low-rank 10-class needs MORE optimization -> pushes OUT of
# the island -> FD-chaos risk. So each (lr,T) rung is FD-GATED; a rung that goes chaotic
# is BOUNDED OUT (recipe unchanged, per the comparability rule). Two honest outcomes:
#   (a) a FD-CLEAN rung converges 10-class -> measure the r=2,4 reversal at convergence;
#   (b) all converging rungs are FD-chaotic (or 10-class can't memorize 10 imgs at low
#       rank at all = a capacity wall) -> low-rank 10-class not cleanly measurable, a
#       legitimate bounded-out finding that closes the loose end either way.
# FAIRNESS: both bases run at the SAME (lr,T) per rung (no per-config recipe change).
# Harness (fdgate/run_cell/rigor/j1) copied VERBATIM from the audited 581629 script;
# only lr,T are parameterized (they were already fdgate args) -> no harness drift.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: AD gate failed."; exit 1; fi

# FULL-config FD gate — threads dataset, rank, lr AND T (gate #0). args: DS NC N k T lr rank
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

run_cell () { # DS NC rank lr T SLIST
  local DS=$1 NC=$2 R=$3 LR=$4 T=$5 SL=$6
  fdgate $DS $NC 10 8 $T $LR $R || { echo "[$DS nc=$NC r=$R lr=$LR T=$T FD-CHAOTIC -> BOUNDED OUT]"; return; }
  echo "-- rigor+j1 $DS nc=$NC N=10 k=8 rank=$R lr=$LR T=$T S={$SL} --"; date
  python -u -m experiments.jacobian_spectrum --rigor --dataset $DS --activation gelu \
      --num_classes $NC --N 10 --k 8 --rank $R --tangent qr --Ts $T --lr $LR --seed 42 --device cuda \
      --tag lowrankconv_memchk_${DS}_nc${NC}_r${R}_lr${LR}_T${T}
  python -u -m experiments.jacobian_spectrum --j1 --dataset $DS --activation gelu \
      --num_classes $NC --N 10 --k 8 --T $T --rank $R --tangent qr --lr $LR \
      --S_list $SL --shrink_list 0.01 --eps_list 0.1 0.3 1.0 3.0 10.0 --save \
      --tag lowrankconv_${DS}_nc${NC}_r${R}_lr${LR}_T${T} --device cuda
}

# (lr,T) escalation ladder at the island edge lr=0.6, deepening T; FD-gate bounds out chaos.
echo ""; echo "########## LOW-RANK CONVERGENCE: r ∈ {2,4}, mnist, both bases ##########"; date
for R in 2 4; do
  for LRT in "0.6 2000" "0.6 4000"; do
    set -- $LRT; LR=$1; T=$2
    echo ""; echo "===== r=$R lr=$LR T=$T (drive low-rank 10-class to memorize) ====="; date
    for NC in 2 10; do run_cell mnist $NC $R $LR $T "320"; done
  done
done

echo ""; echo "=== DONE $(date) ==="
echo "READ: per (r,rung) — did 10-class reach max_bce<1e-3 (converged) AND FD-clean? If yes,"
echo "compare q_eff|col(J) ε=1 binary vs 10-class at the SAME (lr,T) -> the r<8 reversal at"
echo "convergence. If all converging rungs FD-chaotic OR 10-class never memorizes -> bounded out."
