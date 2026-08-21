#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1:gmem=4000"
#BSUB -o scripts/wexac_logs/flowers_nsweep_pbox_%J.out
#BSUB -e scripts/wexac_logs/flowers_nsweep_pbox_%J.err
#BSUB -J flowers_nsweep_pbox

# =====================================================================
# Honest N-sweep (default free-c recipe), CORRECTED design.  Two findings drove this:
#   (1) The clip contamination the audit saw (0.10-0.47) was the SOFTPLUS N-sweep (nrank_665601).
#       The DEFAULT-recipe N-sweep (main_427349) was already clip-clean at N>=4 (npc=4 clip 0.009,
#       npc=8 clip 0.001). So only the moderately-clipped JOINT config (npc=2, clip 0.104) needs a box.
#   (2) --pixel_box at verify_weight 5.0 OVER-CONSTRAINS the low-signal sequential-peel path: it drives
#       the per-source coefficients to ~0 (c~0.005 vs true ~0.06) and blanks the reconstruction
#       (margin +0.045/+0.085 -> ~0). The peel path was already clip-clean, so it must NOT be boxed.
# Therefore: BOX the joint configs (npc 1,2), do NOT box the peel configs (npc 4,8). 3 seeds.
# The box genuinely HELPS npc=2 (clip 0.104->0.045, margin +0.028 -> +0.176 at seed 42).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
FREEC="--free_coefficients --optimizer sgd --relu_alpha 10000 --consistency_weight 1.0 --n_restarts 5 --extraction_epochs 40000"
CKPT=dataset_reconstruction/models/weights-flowers32.pth

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

for SEED in 42 43 44; do
  for NPC in 1 2 4 8; do
    if [ "$NPC" -ge 4 ]; then
      # peel path — already clip-clean; boxing blanks it, so NO --pixel_box here
      EXTRA="--sequential_peel --verify_weight 5.0"
    else
      # joint free-c — box the moderately-clipped small-N configs
      EXTRA="--pixel_box --verify_weight 5.0"
    fi
    echo ""; echo "########## flowers32 N-sweep npc=$NPC seed=$SEED (${EXTRA}) ##########"; date
    python -u -m experiments.run_experiment_b \
      --dataset flowers32 --pretrained_path "$CKPT" \
      --n_steps 1 --rank 8 --seed "$SEED" --lr 0.01 --n_per_class "$NPC" \
      $EXTRA $FREEC --no_baseline --save_results --skip_if_exists --device cuda
  done
done
echo ""; echo "=== FLOWERS N-SWEEP (corrected: box joint, no-box peel) COMPLETE $(date) ==="
