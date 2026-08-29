#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/arm_b_lam_k2k_%J.out
#BSUB -e scripts/wexac_logs/arm_b_lam_k2k_%J.err
#BSUB -J arm_b_lam_k2k
# {K,2K} on lambda[0]: decomp at K=100 for N=32,64 (where lam[0] is smallest). If lam[0] RISES
# vs the K=50 values (0.009, 0.012) -> downward MP bias confirmed -> d2(N) growth was ARTIFACT.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
echo "K=100 (compare lam[0] to K=50: N=32 was 0.0091, N=64 was 0.0118)"
python -u -m experiments.dataset_sensitivity.arm_b_decomp_diag --N_list 32 64 --K 100 --lr 0.5 --T 1000 --rank 8 --device cuda
echo "=== DONE $(date) ==="
