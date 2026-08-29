#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0 && hname!='lgn28' && hname!='hgn46' && hname!='hgn45' && hname!='lgn13']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/f5_shared_perturbation_%J.out
#BSUB -e scripts/wexac_logs/f5_shared_perturbation_%J.err
#BSUB -J f5_shared_perturbation
# F5 FULL RUN: known-recipe ΔW-matching recovery of a SHARED transform parameter, BOTH transforms
# (rotation -40..40 step5 [17 pts]; blur sigma 0.5..4.0 step0.5 [8 pts]), N=16, >=8 independent
# p_true-draws, K_seeds=3. The proxy candidate grid is an attacker asset -> computed ONCE per
# transform and reused across draws (bounded compute). Emits figures/meeting/f5_shared_perturbation.png
# + results/shared_perturbation/f5_{rot,blur}.{pth,json}. RECOVERABLE iff skill CI lower bound > 0.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== F5 FULL START $(date) — both transforms, N=16, n_draws=8, K_seeds=3 ==="
python -u -m experiments.dataset_sensitivity.fig_f5_shared_perturbation_scaffold \
    --transform both --N 16 --n_draws 8 --K_seeds 3 --T 1000 --lr 0.5 --rank 8 \
    --device cuda --out figures/meeting/f5_shared_perturbation.png
rc=$?
if [ $rc -ne 0 ]; then echo "FATAL: F5 full run FAILED (rc=$rc)"; exit 1; fi
echo "=== F5 FULL DONE $(date) — read skill±CI per transform + the ΔW-cosine curves ==="
