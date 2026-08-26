#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=8192] select[hname!='lgn28' && hname!='hgn46' && hname!='hgn45']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/rank_sweep_plots_%J.out
#BSUB -e scripts/wexac_logs/rank_sweep_plots_%J.err
#BSUB -J rank_sweep_plots

# Rank-sweep figures (notes/rank_sweep_plots_plan.md rev2, audit-clean yoado-30).
# Fig1 headline (A q_eff-vs-r converged | B gap-vs-r | C iso-vs-r decouple | D max_bce gate),
# Fig2 eps small-multiples, Fig3 sigma-spectrum r8. Reads .pth (not hardcoded);
# ANCHOR SELF-CHECK = hard abort; fashion read-before-cite. bsub-only.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.plot_rank_sweep
echo "=== EXIT $? $(date) ==="
