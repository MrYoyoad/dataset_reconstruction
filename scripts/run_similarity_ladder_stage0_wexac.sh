#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/similarity_ladder_stage0_%J.out
#BSUB -e scripts/wexac_logs/similarity_ladder_stage0_%J.err
#BSUB -J similarity_ladder_stage0
# SIMILARITY LADDER STAGE-0 plumbing gate: 3-way metric self-test + similarity_ladder tiny sanity
# (N=12, K=12, 1 target, rungs = tiny-noise + encoder-NN + encoder-far; exercises the timm encoder
# fallback chain, the bank retrieval, and the whitened swap measurement end to end).
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== STAGE-0 START $(date) on $(hostname) ==="
echo "--- gate A: 3-way metric self-test ---"
python -u -m experiments.dataset_sensitivity.whitened_metric || { echo "FATAL metric self-test"; exit 1; }
echo "--- gate B: similarity_ladder tiny sanity ---"
python -u -m experiments.dataset_sensitivity.similarity_ladder --stage0 --device cuda
rc=$?
echo "=== STAGE-0 DONE $(date) (exit $rc) ==="
if [ $rc -ne 0 ]; then echo "FATAL: similarity_ladder Stage-0 crashed (rc=$rc)."; exit 1; fi
