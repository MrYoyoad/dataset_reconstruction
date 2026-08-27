#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/similarity_ladder_%J.out
#BSUB -e scripts/wexac_logs/similarity_ladder_%J.err
#BSUB -J similarity_ladder
# SIMILARITY-GRADED SWAP — does swap sensitivity depend on how VISUALLY SIMILAR the replacement is?
# Fixed private D (N=16), fixed class-1 target T; ladder of replacements T' at graded visual distance
# (5 parametric perturbations of T + encoder-ranked NN/median/far same-digit retrievals + one
# cross-digit same-parity far anchor); standard arm-B/D paired-per-seed whitened swap measurement per
# rung against ONE shared baseline ensemble. Encoder: DINO ViT-S/16 with vit_tiny ImageNet fallback.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
echo "########## GATE: metric self-test + similarity_ladder sanity ##########"
python -u -m experiments.dataset_sensitivity.whitened_metric || { echo FATAL metric; exit 1; }
python -u -m experiments.dataset_sensitivity.similarity_ladder --stage0 --device cuda || { echo FATAL stage0; exit 1; }
echo "########## FULL: N=16, K=50, 2 targets, 9 rungs, bank=200 ##########"; date
python -u -m experiments.dataset_sensitivity.similarity_ladder \
    --N 16 --K 50 --n_targets 2 --lr 0.5 --T 1000 --rank 8 --bank 200 --device cuda
echo "=== DONE $(date) ==="
echo "READ: sens RISING with distance + near-duplicate ~null => adapter records the CONCEPT not the instance; flat-high => instance-level memorization; compare pooled rhos (d_encoder vs d_pixel vs |dg0|) for the best predictor."
