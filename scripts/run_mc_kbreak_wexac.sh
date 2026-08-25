#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=65536] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_kbreak_%J.out
#BSUB -e scripts/wexac_logs/mc_kbreak_%J.err
#BSUB -J mc_kbreak

# =====================================================================
# "MAKE IT BREAK" — push k until the adapter can NO LONGER record every secret
# (r_J < Nk). Round 0: r_J = Nk (FULL) up to Nk=640 for BOTH bases -> domain-limited.
# yoado-35 insight: J is [dimY x Nk] with dimY = rank*(in+hidden) = 8*(784+1000) =
# 14272, and rank <= min(dimY, Nk). So r_J MUST break once Nk crosses ~14272
# (N=20 -> k~714). k-ladder 128/256/512/768 -> Nk = 2560/5120/10240/15360 STRADDLES
# the dimY ceiling. THE question: when it breaks, do binary and 10-class cap at the
# SAME ceiling (~dimY, loss-independent = the trivial ambient limit) or DIFFERENT
# (a real CE measurement-capacity advantage)? Single T=500 (well-trained; r_J fills
# fast). r_J-only (cheap, no Sigma_seed). mnist, both bases. (hgn46/lgn28 excluded;
# 64GB for the big SVD: J at Nk=15360 is 14272x15360 float64 ~1.75GB + workspace.)
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

echo ""; echo "########## k-BREAK: N=20, k in {128,256,512,768}, both bases (dimY=14272 ceiling) ##########"; date
for K in 128 256 512 768; do
  for NC in 2 10; do
    NK=$((20*K))
    echo ""; echo "-- rigor mnist num_classes=$NC N=20 k=$K lr=0.1 (Nk=$NK vs dimY=14272; break-hunt) --"; date
    python -u -m experiments.jacobian_spectrum --rigor \
        --dataset mnist --activation gelu --num_classes $NC \
        --N 20 --k $K --rank 8 --tangent qr \
        --Ts 500 --lr 0.1 --seed 42 --save --device cuda \
        --tag kbreak_mnist_nc${NC}_k${K}
  done
done

echo ""; echo "=== DONE $(date) ==="
