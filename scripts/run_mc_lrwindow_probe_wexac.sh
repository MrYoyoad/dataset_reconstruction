#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_lrwindow_%J.out
#BSUB -e scripts/wexac_logs/mc_lrwindow_%J.err
#BSUB -J mc_lrwin

# =====================================================================
# lr-WINDOW PROBE (yoado-35 #4) — is there a recipe that CONVERGES the 10-class
# hard sample AND keeps the exact Jacobian FD-clean? The wall so far:
#   lr=0.5 T=1000 CLEAN but 10-class STUCK (max_bce ~1.8e-3, 19/20)
#   lr=1.0 T=500  CHAOTIC (FD 2.7e-2)  | lr=0.5 T=2000 CHAOTIC (deep unroll)
# So try INTERMEDIATE lr at SHALLOW T=500 (T=500<1000 is depth-safe; the open
# question is whether lr in 0.6-0.8 memorizes the hard sample WITHOUT breaking FD).
# For each lr: (a) DEEP-FD gate at lr/T=500 (N4k4), (b) 10-class N=20 T=500 max_bce.
# A window with BOTH FD-clean AND max_bce<1e-3 => run the clean q_eff lock there.
# If NO lr has both => the AD-vs-aggressive-lr wall is genuinely fundamental
# (a reportable methodological boundary; Metz "Gradients Are Not All You Need").
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

for LR in 0.6 0.7 0.8; do
  echo ""; echo "########## lr=$LR / T=500 : FD + 10-class max_bce ##########"; date
  echo "-- FD gate at lr=$LR/T=500 --"
  python -u -c "
import torch, experiments.jacobian_spectrum as j
torch.set_default_dtype(torch.float64)
ctx,_,_,_ = j._mnist_ctx(N=4,k=4,T=500,rank=8,activation='gelu',seed=42,device='cuda',
                         tangent_method='qr',dataset='mnist',num_classes=10,classes_present=2,lr=$LR)
a0=torch.zeros(16,dtype=torch.float64,device='cuda')
J=j.exact_jacobian(a0,ctx,method='jvp_double')
coords=[0,5,10]; fd=j.finite_difference_jacobian(a0,ctx,coords,eps=1e-5)
rel=max((J[:,c]-fd[c]).norm().item()/(fd[c].norm().item()+1e-30) for c in coords)
nan=bool(torch.isnan(J).any() or torch.isinf(J).any())
print(f'  lr=$LR/T=500 FD rel err: {rel:.3e}  NaN/Inf={nan}  ->  {\"FD-CLEAN\" if (rel<1e-4 and not nan) else \"FD-CHAOTIC\"}')
"
  echo "-- 10-class max_bce at lr=$LR/T=500 --"
  python -u -m experiments.jacobian_spectrum --rigor --dataset mnist --activation gelu \
      --num_classes 10 --N 20 --k 8 --rank 8 --tangent qr \
      --Ts 500 --lr $LR --seed 42 --device cuda --tag lrwin_nc10_lr${LR}
done

echo ""; echo "=== DONE $(date) ==="
echo "READ: pick the LOWEST lr with BOTH FD-CLEAN and 10-class max_bce<1e-3 (memorized YES) -> clean lock lr."
