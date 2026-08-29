#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -W 4:00
#BSUB -o scripts/wexac_logs/atlas_zoo_%J.out
#BSUB -e scripts/wexac_logs/atlas_zoo_%J.err
#BSUB -J atlas_zoo

# =====================================================================
# Adapter-space atlas — FACTORIAL ZOO build (user-authorized 2026-08-29).
# {activation}×{composition=digit-pair}×{lr}×{init-seed}, saving raw (B,A) per cell so BOTH the ΔW-product
# and the raw-(B,A) clustering methods run off one balanced multi-seed-per-cell population.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# ---------- STAGE 0: checkpoints exist + 1-cell smoke (abort on fail) ----------
echo "########## STAGE 0: checkpoints + smoke ##########"
python - <<'PY' || { echo "STAGE 0 FAILED"; exit 1; }
import os, torch; torch.set_default_dtype(torch.float64)
from experiments.configs import MODELS_DIR
for a in ["gelu","relu","softplus"]:
    f=os.path.join(MODELS_DIR, f"weights-mnist_{a}.pth")
    assert os.path.exists(f), f"MISSING checkpoint {f}"
    print("  OK", f)
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0, build_set
dev="cuda" if torch.cuda.is_available() else "cpu"
act=make_activation("gelu")
xr,yr,_=build_set(2, seed=42, device=dev, dataset="mnist")
_,frozen,b0,_,ds_mean=_honest_target(xr,yr,50,8,"gelu",0.5,dev,"mnist",num_classes=2)
out_f=frozen[0].shape[0]
x_ft,y_ft,digits=build_set(2, seed=0, device=dev, dataset="mnist")
A,B,mbce,dW=train_adapter(frozen,b0,draw_B0(100,out_f,8,dev),x_ft-ds_mean,y_ft,0.5,50,act,8)
print(f"  SMOKE OK: A{tuple(A[0].shape)} B{tuple(B[0].shape)} max_bce={mbce:.4f} digits={sorted(set(int(d) for d in digits))}")
PY
echo "STAGE 0 PASSED"

# ---------- STAGE 1: full factorial zoo ----------
echo "########## STAGE 1: factorial zoo ##########"; date
python -u -m experiments.dataset_sensitivity.atlas_zoo --save --device cuda --dataset mnist

echo "=== DONE $(date) ==="
ls -la results/atlas_zoo/zoo_bank.pth 2>/dev/null
