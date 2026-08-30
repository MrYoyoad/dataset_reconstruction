#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='hgn46' && hname!='lgn28' && hname!='hgn45' && hname!='lgn13']"
#BSUB -gpu "num=1"
#BSUB -W 0:30
#BSUB -o scripts/wexac_logs/open_regimes_zoo_%J.out
#BSUB -e scripts/wexac_logs/open_regimes_zoo_%J.err
#BSUB -J open_regimes_zoo
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) ==="
echo "##### A: instance-level atlas (same-digits) #####"
python -u -m experiments.dataset_sensitivity.instance_zoo --save --device cuda
echo "##### B: ecosystem partial-overlap (anchor-digit) #####"
python -u -m experiments.dataset_sensitivity.partial_zoo --save --device cuda
echo "=== DONE $(date) ==="
