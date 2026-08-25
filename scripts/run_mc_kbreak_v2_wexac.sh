#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=65536] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1:gmodel=NVIDIAA100_SXM4"
#BSUB -o scripts/wexac_logs/mc_kbreak_v2_%J.out
#BSUB -e scripts/wexac_logs/mc_kbreak_v2_%J.err
#BSUB -J mc_kbreak2

# =====================================================================
# "MAKE IT BREAK" v2 — reach the r_J < Nk break. v1 (399105) OOM'd on the 10-class
# k=128 cell: the multi-class CE unroll builds a ~10x wider double-backward graph
# (10 logits vs 1) -> exact_jacobian OOM on a normal GPU. Fixes: A100 (80GB) +
# T=200 (smaller create_graph graph; the dimY ceiling is T-robust). Push k PAST the
# ambient ceiling dimY = rank*(in+hidden) = 8*(784+1000) = 14272:
#   k=512 -> Nk=10240 (< dimY: expect r_J = Nk, FULL)
#   k=768 -> Nk=15360 (> dimY: expect r_J caps ~14272 = THE BREAK)
#   k=1024-> Nk=20480 (>> dimY: r_J pinned ~14272, break unmistakable)
# THE question: do binary and 10-class cap at the SAME value (~dimY, loss-independent
# ambient ceiling = confirms the retraction) or DIFFERENT (a real rank-capacity gap)?
# Each cell = fresh `python -m` process (isolated; one OOM won't kill the rest).
# r_J-only (no Sigma_seed). mnist, both bases, lr=0.1.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}'); print(torch.cuda.get_device_name(0))"
echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## STAGE 0: AD gate (binary + CE) ##########"; date
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: AD gate failed. Aborting."; exit 1; fi

echo ""; echo "########## k-BREAK v2: N=20, k in {512,768,1024} straddling dimY=14272 ##########"; date
for K in 512 768 1024; do
  for NC in 2 10; do
    NK=$((20*K))
    echo ""; echo "-- rigor mnist num_classes=$NC N=20 k=$K lr=0.1 T=200 (Nk=$NK vs dimY=14272; BREAK) --"; date
    python -u -m experiments.jacobian_spectrum --rigor \
        --dataset mnist --activation gelu --num_classes $NC \
        --N 20 --k $K --rank 8 --tangent qr \
        --Ts 200 --lr 0.1 --seed 42 --save --device cuda \
        --tag kbreak2_mnist_nc${NC}_k${K} || echo "[CELL FAILED nc=$NC k=$K — continuing]"
  done
done

echo ""; echo "=== DONE $(date) ==="
