#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/vit_lora_scaled_%J.out
#BSUB -e scripts/wexac_logs/vit_lora_scaled_%J.err
#BSUB -J vit_lora_scaled
# ViT+LoRA SCALED: firm up the stage-0 "single image detectable" (p=0.03 at K=10) toward the p=0.002
# floor. N=16 private, K=50 seeds, 3 swap targets, vit_tiny rank-4 LoRA on blocks 0-2 qkv.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import timm, peft; print(f'timm={timm.__version__} peft={peft.__version__}')" || { echo FATAL timm; exit 1; }
echo "=== START $(date) on $(hostname) ==="
python -u -m experiments.dataset_sensitivity.vit_lora_sensitivity \
    --N 16 --K 50 --steps 300 --lr 5e-3 --rank 4 --classes 3 8 --n_targets 3 --device cuda
echo "=== DONE $(date) ==="
echo "READ: sensitivity>0 & pvalue small across targets => single image robustly detectable in ViT LoRA."
