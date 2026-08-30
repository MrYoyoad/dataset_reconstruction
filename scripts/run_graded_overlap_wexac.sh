#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46' && hname!='lgn28' && hname!='hgn45' && hname!='lgn13']"
#BSUB -gpu "num=1"
#BSUB -W 0:30
#BSUB -o scripts/wexac_logs/graded_overlap_%J.out
#BSUB -e scripts/wexac_logs/graded_overlap_%J.err
#BSUB -J graded_overlap
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) ==="
echo "##### build 100%-overlap point #####"
python -u -m experiments.dataset_sensitivity.full_zoo --save --device cuda
echo "##### graded-overlap curve (0/50/100%) #####"
python -u -m experiments.dataset_sensitivity.graded_overlap
echo "=== DONE $(date) ==="
