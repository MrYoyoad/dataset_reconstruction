#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/arm_b_null_diag_%J.out
#BSUB -e scripts/wexac_logs/arm_b_null_diag_%J.err
#BSUB -J arm_b_null_diag
# NULL-ONLY d2(N): reseed-vs-reseed (no swap). MUST be flat ~0; growing => the swap d2(N) is DRIFT.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
echo "########## STAGE 0 ##########"
python -u -m experiments.dataset_sensitivity.arm_b_null_diag --stage0 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: null-diag Stage-0 crashed."; exit 1; fi
echo "########## FULL NULL-DIAG (K=50) ##########"; date
python -u -m experiments.dataset_sensitivity.arm_b_null_diag --N_list 2 4 8 16 32 64 --K 50 --lr 0.5 --T 1000 --rank 8 --device cuda
echo "=== DONE $(date) ==="
