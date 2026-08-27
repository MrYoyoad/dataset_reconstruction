#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=8192] select[hname!='lgn28' && hname!='hgn46' && hname!='hgn45' && hname!='lgn13']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/recon_baseline_plot_%J.out
#BSUB -e scripts/wexac_logs/recon_baseline_plot_%J.err
#BSUB -J recon_baseline_plot
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.plot_reconstruction_baseline
echo "=== EXIT $? $(date) ==="
