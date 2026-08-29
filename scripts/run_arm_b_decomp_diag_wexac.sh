#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/arm_b_decomp_diag_%J.out
#BSUB -e scripts/wexac_logs/arm_b_decomp_diag_%J.err
#BSUB -J arm_b_decomp
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.dataset_sensitivity.arm_b_decomp_diag --stage0 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: decomp Stage-0 crashed."; exit 1; fi
echo "########## FULL DECOMP (K=50) ##########"; date
python -u -m experiments.dataset_sensitivity.arm_b_decomp_diag --N_list 2 4 8 16 32 64 --K 50 --lr 0.5 --T 1000 --rank 8 --device cuda
echo "=== DONE $(date) ==="
