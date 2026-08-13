#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/activation_fashion_%J.out
#BSUB -e scripts/wexac_logs/activation_fashion_%J.err
#BSUB -J activation_fashion

# =====================================================================
# Step 4 (Addition 1, harder data) x Step 2a: the activation spectrum on FASHION-MNIST.
# Fashion-MNIST is 28x28x1 -> drops into the 784-MLP; unlike MNIST-N=2 its dataset mean is NOT
# ~= each image, so control-margin / retrieval speak for themselves. theta_0 stays the
# MNIST-pretrained base -> a realistic transfer/PEFT attack (foundation model pretrained on public
# data, fine-tuned on private Fashion data). Question: does the softplus>>gelu>...>elu ranking
# from MNIST TRANSFER to harder image structure? Same 13-activation smooth->kinked spectrum,
# lr=0.01, at N=2 (n_per_class=1) and N=10 (n_per_class=5, --verify_weight 5.0). ~26 configs.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

# ---------- STAGE 0: validate the fashion data path on the GPU node ----------
echo ""; echo "########## STAGE 0: fashion loader check ##########"
python - <<'PY' || { echo "STAGE 0 FAILED: fashion loader"; exit 1; }
from experiments.data_utils import get_finetuning_data, get_control_images_in_distribution
xf, yf, c, _ = get_finetuning_data(1, seed=42, dataset='fashion')
assert tuple(xf.shape) == (2, 1, 28, 28), xf.shape
get_control_images_in_distribution(c, dataset='fashion')
print("fashion loader OK", tuple(xf.shape), "classes", c)
PY
echo "STAGE 0 PASSED"

ACTS="sigmoid tanh gelu silu softplus mish gelu_tanh elu celu selu hardswish leaky_relu relu"

# ---------- N=2 ----------
for ACT in $ACTS; do
    echo ""; echo "########## FASHION N=2 act=$ACT lr=0.01 T=1 r=8 ##########"; date
    python -u -m experiments.run_experiment_b \
        --dataset fashion --n_steps 1 --rank 8 --seed 42 --lr 0.01 \
        --finetune_activation "$ACT" \
        --no_baseline --save_results --skip_if_exists --device cuda
done

# ---------- N=10 (verify_weight 5.0 to curb clipping) ----------
for ACT in $ACTS; do
    echo ""; echo "########## FASHION N=10 act=$ACT lr=0.01 T=1 r=8 ##########"; date
    python -u -m experiments.run_experiment_b \
        --dataset fashion --n_steps 1 --rank 8 --seed 42 --lr 0.01 --n_per_class 5 --verify_weight 5.0 \
        --finetune_activation "$ACT" \
        --no_baseline --save_results --skip_if_exists --device cuda
done

echo ""; echo "=== ALL STAGES COMPLETE $(date) ==="
