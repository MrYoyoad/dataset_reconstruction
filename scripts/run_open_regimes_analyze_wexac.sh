#!/bin/bash
#BSUB -q short
#BSUB -R "rusage[mem=8192]"
#BSUB -W 0:15
#BSUB -o scripts/wexac_logs/open_regimes_analyze_%J.out
#BSUB -e scripts/wexac_logs/open_regimes_analyze_%J.err
#BSUB -J open_regimes_analyze
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) ==="
echo "##### A: INSTANCE-LEVEL atlas recovery (same-digits {0,1}, composition=image-sample) #####"
python -u -m experiments.dataset_sensitivity.atlas_analyze --bank results/instance_zoo/instance_bank.pth --out figures/atlas/atlas_instance.png
echo ""
echo "##### B: ECOSYSTEM PARTIAL-OVERLAP GAIN (anchor-digit {0,x}) #####"
python -u -m experiments.dataset_sensitivity.eco_analyze --bank results/partial_zoo/partial_bank.pth --tag partial
echo "=== DONE $(date) ==="
