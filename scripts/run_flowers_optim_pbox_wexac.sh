#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers_optim_pbox_%J.out
#BSUB -e scripts/wexac_logs/flowers_optim_pbox_%J.err
#BSUB -J flowers_optim_pbox

# =====================================================================
# RE-DO of the optimizer axis. The old adamw configs were TRIPLY confounded: (1) lbfgs extraction
# (overfits the free coefficients -- our recipe lesson), (2) weight_change=0.60 (far out of the NTK
# band, vs sgd's ~0.11), (3) ~30% clipping. This re-do removes all three:
#   - validated free-c extraction (sgd + a10000 + consistency + restarts) for ALL arms,
#   - --pixel_box (+ vw5) so nothing clips,
#   - an adamw fine-tune-lr LADDER {0.002, 0.005} to bracket sgd@0.01's in-band wc~0.11, so the
#     sgd-vs-adamw comparison is at MATCHED signal (pick the adamw lr whose wc ~= 0.11 in analysis).
# Loss l2 vs cosine on each (SimuDy predicts cosine >> l2). N=2, r=8, seed 42 (matched-wc probe,
# not a multi-seed headline). The sgd/lr0.01/l2 arm == the N-sweep npc=1 config (reused via skip).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
FREEC="--free_coefficients --optimizer sgd --relu_alpha 10000 --consistency_weight 1.0 --n_restarts 5 --extraction_epochs 40000"
CKPT=dataset_reconstruction/models/weights-flowers32.pth

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# ft_optimizer + fine-tune lr pairs (adamw lr ladder brackets sgd@0.01's wc)
for PAIR in "sgd 0.01" "adamw 0.002" "adamw 0.005"; do
  set -- $PAIR; FTOPT=$1; LRV=$2
  for LOSS in l2 cosine; do
    LOSSFLAG=""; [ "$LOSS" = cosine ] && LOSSFLAG="--loss_type cosine"
    echo ""; echo "########## flowers32 optim ft=$FTOPT lr=$LRV loss=$LOSS (pbox) ##########"; date
    python -u -m experiments.run_experiment_b \
      --dataset flowers32 --pretrained_path "$CKPT" \
      --n_steps 1 --rank 8 --seed 42 --lr "$LRV" --finetune_optimizer "$FTOPT" \
      --pixel_box --verify_weight 5.0 $LOSSFLAG \
      $FREEC --no_baseline --save_results --skip_if_exists --device cuda
  done
done
echo ""; echo "=== FLOWERS OPTIMIZER pbox (matched-wc) COMPLETE $(date) ==="
