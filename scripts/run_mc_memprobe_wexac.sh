#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_memprobe_%J.out
#BSUB -e scripts/wexac_logs/mc_memprobe_%J.err
#BSUB -J mc_memprobe

# =====================================================================
# ROUND B MEMORIZATION PROBE (yoado-35 protocol) — find the (lr,T) that reaches
# TRUE memorization (max_bce<1e-3) at N=20 for BOTH bases with the SAME recipe,
# BEFORE measuring q_eff. Round 0 showed lr=0.1 only reaches max_bce~6e-3 by T=500
# at N=20 -> need a MODERATE lr BUMP (preferred over deep T: shallower unroll, less
# meta-grad chaos, cheaper per S-sample). Sweep lr {0.3,0.5,1.0} x Ts {100,200,500}
# at k=8 (cheap J), read the max_bce column. Pick the smallest (lr,T) with
# max_bce<1e-3 for BOTH binary AND 10-class -> that (lr,T) sets the Round B q_eff
# convergence regime (measured at T and ~2T to confirm the q_eff plateau).
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

echo ""; echo "########## memorization probe: lr x T, both bases, N=20 k=8 (read max_bce) ##########"; date
for LR in 0.3 0.5 1.0; do
  for NC in 2 10; do
    echo ""; echo "-- probe mnist nc=$NC N=20 k=8 lr=$LR (find max_bce<1e-3) --"; date
    python -u -m experiments.jacobian_spectrum --rigor \
        --dataset mnist --activation gelu --num_classes $NC \
        --N 20 --k 8 --rank 8 --tangent qr \
        --Ts 100 200 500 --lr $LR --seed 42 --device cuda \
        --tag memprobe_mnist_nc${NC}_lr${LR}
  done
done

echo ""; echo "=== DONE $(date) ==="
