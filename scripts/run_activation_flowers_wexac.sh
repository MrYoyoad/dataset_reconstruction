#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/activation_flowers_%J.out
#BSUB -e scripts/wexac_logs/activation_flowers_%J.err
#BSUB -J activation_flowers

# =====================================================================
# The SAME cookbook (MNIST-MLP, LoRA r=8, T=1, NTK/oracle reconstruction, the activation spectrum)
# applied to FLOWERS102 rendered as 28x28 grayscale -- real natural-image structure through the
# same 784-MLP the whole study uses. theta_0 stays MNIST-pretrained (transfer/PEFT attack).
# Much harder than MNIST/Fashion: the question is whether the softplus>>...>elu ranking and the
# smoothness->linearization story survive on real image structure (even downsampled). Same
# 13-activation smooth->kinked spectrum, lr=0.01, at N=2 and N=10 (verify_weight 5.0). ~26 configs.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# ---------- STAGE 0: validate the flowers data path on the GPU node ----------
echo ""; echo "########## STAGE 0: flowers loader check ##########"
python - <<'PY' || { echo "STAGE 0 FAILED: flowers loader"; exit 1; }
from experiments.data_utils import get_finetuning_data, get_control_images_in_distribution
xf, yf, c, _ = get_finetuning_data(1, seed=42, dataset='flowers')
assert tuple(xf.shape) == (2, 1, 28, 28), xf.shape
get_control_images_in_distribution(c, dataset='flowers')
print("flowers loader OK", tuple(xf.shape), "classes", c)
PY
echo "STAGE 0 PASSED"

ACTS="sigmoid tanh gelu silu softplus mish gelu_tanh elu celu selu hardswish leaky_relu relu"

for ACT in $ACTS; do
    echo ""; echo "########## FLOWERS N=2 act=$ACT lr=0.01 T=1 r=8 ##########"; date
    python -u -m experiments.run_experiment_b \
        --dataset flowers --n_steps 1 --rank 8 --seed 42 --lr 0.01 \
        --finetune_activation "$ACT" \
        --no_baseline --save_results --skip_if_exists --device cuda
done

for ACT in $ACTS; do
    echo ""; echo "########## FLOWERS N=10 act=$ACT lr=0.01 T=1 r=8 ##########"; date
    python -u -m experiments.run_experiment_b \
        --dataset flowers --n_steps 1 --rank 8 --seed 42 --lr 0.01 --n_per_class 5 --verify_weight 5.0 \
        --finetune_activation "$ACT" \
        --no_baseline --save_results --skip_if_exists --device cuda
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
