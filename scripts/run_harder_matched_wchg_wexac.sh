#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/harder_matched_wchg_%J.out
#BSUB -e scripts/wexac_logs/harder_matched_wchg_%J.err
#BSUB -J harder_matched_wchg

# =====================================================================
# Follow-up to the Step-2 batch (STATUS.md 2026-08-13): the Fashion/Flowers activation comparison
# at lr=0.01 was DEGENERATE — 11/13 activations gave weight_change=0.000 (the MNIST-pretrained net
# emits a ~zero BCE gradient on transfer data at one step). This job re-runs the spectrum at HIGHER
# LR {0.1, 0.3} to push weight_change into a NON-degenerate, matchable band, so we can read a FAIR
# matched-weight_change comparison post-hoc (rescore + interpolate on the saved per-run weight_change)
# and settle whether softplus wins or loses on harder data — or whether the kinked nets simply
# saturate (weight_change stays ~0 even at high LR = a real transfer finding, not an activation loss).
# N=2, verify_weight 5.0 (curb clipping at larger updates), fashion + flowers-28x28. Smooth-first.
#
# NOTE: this is still the MNIST-theta_0 TRANSFER proxy; the definitive test is the flowers-NATIVE
# theta_0 track (parallel session). Keep that framing.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

ACTS="sigmoid tanh gelu silu softplus mish gelu_tanh elu celu selu hardswish leaky_relu relu"
LRS="0.1 0.3"

for DS in fashion flowers; do
    for ACT in $ACTS; do
        for LR in $LRS; do
            echo ""; echo "########## $DS N=2 act=$ACT lr=$LR T=1 r=8 vw=5 ##########"; date
            python -u -m experiments.run_experiment_b \
                --dataset "$DS" --n_steps 1 --rank 8 --seed 42 --lr "$LR" --verify_weight 5.0 \
                --finetune_activation "$ACT" \
                --no_baseline --save_results --skip_if_exists --device cuda
        done
    done
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
