#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers_freec_qb_%J.out
#BSUB -e scripts/wexac_logs/flowers_freec_qb_%J.err
#BSUB -J flowers_freec_qb

# Q-B (Phase D) overlap-vs-novel in the REALISTIC free-coefficient attack (verified recipe).
# Holdout theta_0 (species 0..19 held out); ReLU fine-tune; seen vs novel x 3 seeds.

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
HOLDOUT="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
CKPT=dataset_reconstruction/models/weights-flowers32_holdout.pth
FREEC="--free_coefficients --optimizer sgd --relu_alpha 10000 --consistency_weight 1.0 --n_restarts 5 --extraction_epochs 40000"
echo "=== START $(date) on $(hostname) ==="

python - <<'PY' || { echo "STAGE 0 FAILED"; exit 1; }
from experiments.configs import DATASET_SPECS
from experiments.run_experiment_b import load_pretrained
spec = DATASET_SPECS['flowers32']
load_pretrained(device='cpu', pretrained_path='dataset_reconstruction/models/weights-flowers32_holdout.pth',
                input_dim=spec['input_dim'], hidden=spec['hidden'])
print("STAGE 0 OK: holdout theta_0 loads")
PY

for SRC in seen novel; do
  for SEED in 42 43 44; do
    echo ""; echo "########## flowers32 free-c Q-B source=$SRC seed=$SEED ##########"; date
    python -u -m experiments.run_experiment_b --dataset flowers32 --pretrained_path "$CKPT" \
      --n_steps 1 --rank 8 --seed "$SEED" --lr 0.01 --source "$SRC" --holdout_species $HOLDOUT \
      $FREEC --no_baseline --save_results --skip_if_exists --device cuda
  done
done
echo ""; echo "=== FLOWERS FREE-C Q-B COMPLETE $(date) ==="
