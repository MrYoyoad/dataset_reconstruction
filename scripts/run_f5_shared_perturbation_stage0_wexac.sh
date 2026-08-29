#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0 && hname!='lgn28' && hname!='hgn46' && hname!='hgn45' && hname!='lgn13']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/f5_shared_perturbation_stage0_%J.out
#BSUB -e scripts/wexac_logs/f5_shared_perturbation_stage0_%J.err
#BSUB -J f5_shared_perturbation_stage0
# F5 STAGE-0 (TINY GATE): known-recipe ΔW-matching recovery of a SHARED transform parameter.
# N=6, 1 p_true-draw, coarse 5-pt rotation grid, K_seeds=2. Asserts a FINITE skill and a REAL
# peak in the ΔW-cosine-vs-p curve (max > mean), then exits. NOT the science run — just proves
# the fine-tune + attack + metric pipeline fires end-to-end before the full long-gpu job.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== F5 STAGE-0 START $(date) ==="
python -u -m experiments.dataset_sensitivity.fig_f5_shared_perturbation_scaffold \
    --stage0 --transform rotation --device cuda
rc=$?
if [ $rc -ne 0 ]; then echo "FATAL: F5 stage-0 gate FAILED (rc=$rc)"; exit 1; fi
echo "=== F5 STAGE-0 OK $(date) — pipeline fires; clear to submit the full run ==="
