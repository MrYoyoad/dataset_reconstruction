#!/bin/bash
# =====================================================================
# P2(ii) (plan 2026-09-18, Audit item 7): the ZERO-DRIFT conv rank law on a SECOND real architecture -- ResNet-18 on
# CIFAR-10, trained to the base gate, LoRA slots on the four 3x3 convs of ONE residual stage (torchvision `layer3`,
# the stage whose INPUT is the 128-channel map and whose convs run at 8x8 positions: p_l = 128*9 = 1152 for
# block-1 conv1, 256*9 = 2304 for the other three; P_l = 64; N*P_l = 512 at N = 8).
# Trainer:  experiments/exact_inversion/train_resnet_backbone.py       (FP32, SGD/cosine/aug, gate measured every epoch)
# Harness:  experiments/multilayer_cert/resnet_ranklaw.py               (FP64; PRE-REGISTRATION in its docstring)
# Gate:     experiments/exact_inversion/base_training_gate.py --ckpts   (family cifar_resnet; run at the end of `train`)
#
#   bash scripts/run_resnet_ranklaw_wexac.sh submit train   # short-gpu, plain GPU: train to the gate + gate row
#   bash scripts/run_resnet_ranklaw_wexac.sh submit smoke   # short-gpu, plain GPU: r 64 (+512 1024 code-path), k 32 128, block 1 only
#   bash scripts/run_resnet_ranklaw_wexac.sh submit full    # long-gpu, A100 exclusive (FP64 SVDs): r 64 256 512, k 32..3072, L 1..4
#
# PRE-SUBMISSION ARITHMETIC (stated before any row; the harness header recomputes it from the MEASURED spans):
#   conv (torchvision name) | input | p_l   | P_l | N*P_l | live iff N'_l < min(r, p_l)  -> at r = 64 / 256 / 512
#   layer3.0.conv1 (s2)     | 128@16x16 | 1152 | 64 | 512 | N' <= 512 < p_l: NEVER p_l-saturated; live iff N' < r
#   layer3.0.conv2          | 256@8x8   | 2304 | 64 | 512 | same
#   layer3.1.conv1          | 256@8x8   | 2304 | 64 | 512 | same
#   layer3.1.conv2          | 256@8x8   | 2304 | 64 | 512 | same
#   Consequence: the p_l-saturation death of the MNIST CNN's conv 1 (N' = p_l = 9) cannot happen here; what CAN
#   happen is the r-side vacuity N' >= r.  With 512 patch vectors per conv, r = 64 and r = 256 are vacuous unless
#   the patch span is degenerate (N' < r), and r = 512 is live iff the 512 patch vectors are linearly DEPENDENT
#   (N' < 512).  r = 1024 > N*P_l is live by arithmetic whatever N' is (rank C >= 512) -- it is in the SMOKE as a
#   code-path exerciser and is the recommended 4th rank for `full` if the smoke measures N' = 512 everywhere.
#   Weight-sharing budget q_l = min(rank C_l * P_l, d_l), d_l <= min(k, input-rep width 32768 / 16384).
#
# JACOBIAN SIZES (FP64, k = 3072 pixel chart): input-representation Jacobian 81 920 x 3072 = 2.0 GB per image;
# patch Jacobian per conv (unfold of the columns) <= 147 456 x 3072 = 3.6 GB transient; certified block per conv
# <= rank C * 64 x 3072 <= 32 768 x 3072 = 0.8 GB at r = 512; stacked <= 131 072 x 3072 = 3.2 GB; forward-mode
# intermediates ~6-8 MB per tangent -> chunk 256 (default --jac-chunk) ~2 GB.  Peak ~15-20 GB -> gmem >= 40 GB.
# The SVDs are FP64 (~1e12-5e12 flop each): the A100 (19.5 TFLOPS FP64) is requested for `full`; a plain card is
# fine for `smoke` (k <= 128).
# =====================================================================
STAGE="${1:-smoke}"
if [ "$STAGE" = "submit" ]; then
    WHAT="${2:-smoke}"
    cd /home/projects/galvardi/yoado
    mkdir -p scripts/wexac_logs results/multilayer_cert models/exact_inversion
    if [ "$WHAT" = "train" ]; then
        bsub -q short-gpu -gpu "num=1:gmem=16000" -R "rusage[mem=32768] select[ngpus>0]" -W 4:00 \
             -J resnet_train -o scripts/wexac_logs/resnet_train_%J.out -e scripts/wexac_logs/resnet_train_%J.err \
             bash scripts/run_resnet_ranklaw_wexac.sh train
    elif [ "$WHAT" = "smoke" ]; then
        bsub -q short-gpu -gpu "num=1:gmem=16000" -R "rusage[mem=32768] select[ngpus>0]" -W 1:30 \
             -J resnet_smoke -o scripts/wexac_logs/resnet_smoke_%J.out -e scripts/wexac_logs/resnet_smoke_%J.err \
             bash scripts/run_resnet_ranklaw_wexac.sh smoke
    elif [ "$WHAT" = "full" ]; then
        bsub -q long-gpu -gpu "num=1:j_exclusive=yes:gmodel=NVIDIAA100_SXM4" -R "rusage[mem=65536] select[ngpus>0]" -W 12:00 \
             -J resnet_full -o scripts/wexac_logs/resnet_full_%J.out -e scripts/wexac_logs/resnet_full_%J.err \
             bash scripts/run_resnet_ranklaw_wexac.sh full
    else
        echo "unknown stage $WHAT"; exit 2
    fi
    exit 0
fi

set +u                                   # conda activate breaks under set -u in this env (LESSONS_LEARNED)
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/multilayer_cert; mkdir -p $OUT
CK=models/exact_inversion/cifar10_resnet18.pth
echo "=== START resnet_ranklaw stage=$STAGE $(date) on $(hostname) git=$(git rev-parse --short HEAD) job=${LSB_JOBID:-manual} ==="
python -c "import torch, torchvision; print(f'CUDA={torch.cuda.is_available()} torch={torch.__version__} tv={torchvision.__version__} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
sed -n '/^# PRE-SUBMISSION ARITHMETIC/,/^# =====/p' scripts/run_resnet_ranklaw_wexac.sh

if [ "$STAGE" = "train" ]; then
    if [ -f "$CK" ]; then echo "# $CK exists -- refusing to overwrite a base checkpoint"; exit 3; fi
    python -u -m experiments.exact_inversion.train_resnet_backbone --out $CK --seed 1 \
        --epochs 120 --max-epochs 200 --target-train-acc 0.995 --min-train-loss 1e-2 --data-root data
    echo "=== independent gate measurement (base_training_gate, family cifar_resnet) ==="
    python -u -m experiments.exact_inversion.base_training_gate --ckpts $CK --cifar-root data --out results/base_training_gate.jsonl
elif [ "$STAGE" = "smoke" ]; then
    if [ ! -f "$CK" ]; then echo "# $CK absent -- train first"; exit 3; fi
    # r 64 = the brief's smoke rank; 512 and 1024 exercise the certified-stack code path (1024 is live by arithmetic)
    python -u -m experiments.multilayer_cert.resnet_ranklaw --ckpt $CK --label smoke \
        --N 8 --r 64 512 1024 --ks 32 128 --maxL 2 --seed 1 --jac-chunk 256 --verify-patch-rank \
        --out "$OUT/resnet_ranklaw_smoke_${LSB_JOBID:-manual}.jsonl"
elif [ "$STAGE" = "full" ]; then
    if [ ! -f "$CK" ]; then echo "# $CK absent -- refusing to run the full stage"; exit 3; fi
    # Coordinator 2026-09-18 after smoke 366250: at N = 8 every stage-3 conv has N'_l = N*P_l = 512, so r <= 512 is
    # r-side VACUOUS (certificate rank 0) -- the brief's r-set tests nothing.  Live arms: r = 1024 at N = 8 (rank C = 512
    # per conv), and r = 512 at N = 4 (N' = 256 < 512).  The vacuous r = 64 / 256 rows are kept at N = 8 as the control.
    python -u -m experiments.multilayer_cert.resnet_ranklaw --ckpt $CK --label p2ii-full-N8 \
        --N 8 --r 64 256 1024 --ks 32 128 384 1024 3072 --maxL 4 --seed 1 --jac-chunk 256 \
        --out "$OUT/resnet_ranklaw_N8_${LSB_JOBID:-manual}.jsonl"
    python -u -m experiments.multilayer_cert.resnet_ranklaw --ckpt $CK --label p2ii-full-N4 \
        --N 4 --r 256 512 1024 --ks 32 128 384 1024 3072 --maxL 4 --seed 1 --jac-chunk 256 \
        --out "$OUT/resnet_ranklaw_N4_${LSB_JOBID:-manual}.jsonl"
else
    echo "unknown stage $STAGE"; exit 2
fi
echo "=== DONE $(date) ==="
