#!/bin/bash
# =====================================================================
# WP1 (plan 2026-09-18, as amended by its Audit section): the multilayer certificate rank law on a CONV encoder
# plus the T arm.  Harness: experiments/multilayer_cert/conv_encoder_ranklaw.py (port of real_encoder_ranklaw.py).
#
#   bash scripts/run_conv_ranklaw_wexac.sh submit smoke   # short-gpu, plain GPU: deep ckpt (smoke-only) + bottleneck mini
#   bash scripts/run_conv_ranklaw_wexac.sh submit full    # long-gpu, plain GPU (gmem>=16000), 32 GB, bottleneck spec
#
# PRE-SUBMISSION ARITHMETIC (bottleneck spec 1->64->128->8->256 s2 + dense 1024->1000 + head; sides 28->14->7->4->2;
# patch dims p_l = 9/576/1152/72, dense 1024, head 1000; positions P_l = 196/49/16/4/1/1; representation widths
# 784 -> 12544 -> 6272 -> 128 -> 1024 -> 1000 -> 10, so rank M_l is bounded by min(k, 784, 784, 784, 128, 128, 128)).
# Recorded counts N'_l are the patch-span ranks MEASURED IN THE JOB (the python header recomputes everything below
# from them); the ESTIMATES here use job 205887's deep-net spans 9/232/117 for convs 1-3, N*P_4 = 32 for conv 4, N = 8
# for dense/head.  r = 256: cert rank min(r,p_l)-N'_l ~ 0 / 24 / 139 / 40 / 248 / 248.
#   (i)  weight-sharing count q_l = min(rank C_l * P_l, d_l): q ~ [0, min(1176,k), min(2224,k), min(160,128), 128, 128]
#        first=1, k=784: T5.2 = min(784, 2352) = 784, corrected = min(784, 784+0, 784+784, 128+1568, ...) = 784 -> NO
#        first=3, k=784: d_1 = 784 = q_1 -> both 784 -> NO;   first=5: d = [128,128], both 128 -> control, NO.
#        Under (i) NO config discriminates at any k in {128,256,384,784}: conv 2 (or conv 3) alone pins the chart.
#   (ii) audit dense-style count q_l = min(rank C_l, d_l): first=1, k=784: q ~ [0, 24, 139, 40, 128, 128], sum 459,
#        T5.2 = min(784, 459) = 459, corrected at j=conv4: 128 + 0 + 24 + 139 = 291 < 459 -> DISCRIMINATES (k >= 291);
#        k=256: T5.2 = 256, corrected = min(256, 256+0, 256+24, 128+163=291, ...) = 256 -> NO; k=384: T5.2 = 384,
#        corrected = 291 -> YES.  first=3, k=784: q ~ [139, 40, 128, 128], T5.2 = min(784, 435) = 435, corrected =
#        min(784, 128+139=267, ...) = 267 -> YES.  first=5: control, coincide.
# Which count is real is settled by the measured q_l against q_l_formula (i) / q_l_formula_dense_style (ii) on
# every row, BEFORE the law is read.  Under (i) the pre-registered outcome is VACUOUS; under (ii) DISCRIMINATION or
# FALSIFIED with first=5 as the control.
#
# JACOBIAN SIZES (FP64, k = 784): patch Jacobian rows 9*196+576*49+1152*16+72*4+1024+1000 = 50 732 -> 318 MB per
# image (one image at a time, never cached); stacked rows <= 256*(49+16+4+1+1) = 18 176 (conv 1 vacuous) -> 114 MB;
# forward-mode intermediates ~0.5 MB/tangent -> ~0.4 GB for 784 tangents in one chunk.  A plain long-gpu card with
# gmem >= 16 GB is ample (audit: A100 exclusive not required).
# =====================================================================
STAGE="${1:-smoke}"
if [ "$STAGE" = "submit" ]; then
    WHAT="${2:-smoke}"
    cd /home/projects/galvardi/yoado
    mkdir -p scripts/wexac_logs results/multilayer_cert
    if [ "$WHAT" = "smoke" ]; then
        bsub -q short-gpu -gpu "num=1:gmem=16000" -R "rusage[mem=24576] select[ngpus>0]" -W 1:30 \
             -J convrank_smoke -o scripts/wexac_logs/convrank_smoke_%J.out -e scripts/wexac_logs/convrank_smoke_%J.err \
             bash scripts/run_conv_ranklaw_wexac.sh smoke
    elif [ "$WHAT" = "full" ]; then
        bsub -q long-gpu -gpu "num=1:gmem=16000" -R "rusage[mem=32768] select[ngpus>0]" -W 10:00 \
             -J convrank_full -o scripts/wexac_logs/convrank_full_%J.out -e scripts/wexac_logs/convrank_full_%J.err \
             bash scripts/run_conv_ranklaw_wexac.sh full
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
CK_BOTTLENECK=models/exact_inversion/mnist_conv_bottleneck.pth
CK_DEEP=models/exact_inversion/mnist_conv_deep.pth
echo "=== START conv_encoder_ranklaw stage=$STAGE $(date) on $(hostname) git=$(git rev-parse --short HEAD) job=${LSB_JOBID:-manual} ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()} torch={torch.__version__} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
sed -n '/^# PRE-SUBMISSION ARITHMETIC/,/^# =====/p' scripts/run_conv_ranklaw_wexac.sh

if [ "$STAGE" = "smoke" ]; then
    # (1) code-path exercise on the deep checkpoint (SMOKE-ONLY: the deep spec is vacuous by arithmetic)
    python -u -m experiments.multilayer_cert.conv_encoder_ranklaw --spec deep --ckpt $CK_DEEP --label smoke-only-deep \
        --N 8 --r 256 --ks 16 32 --first 1 --maxL 2 --T-arm 1 5 --lr 0.01 --seed 1 \
        --out "$OUT/conv_ranklaw_smoke_deep_${LSB_JOBID:-manual}.jsonl"
    # (2) the dense slot + the full module list on the bottleneck checkpoint, tiny charts
    if [ -f "$CK_BOTTLENECK" ]; then
        python -u -m experiments.multilayer_cert.conv_encoder_ranklaw --spec bottleneck --ckpt $CK_BOTTLENECK --label smoke-only-bottleneck \
            --N 8 --r 256 --ks 16 32 --first 1 5 --maxL 6 --T-arm 1 5 --lr 0.01 --seed 1 \
            --out "$OUT/conv_ranklaw_smoke_bottleneck_${LSB_JOBID:-manual}.jsonl"
    else
        echo "# $CK_BOTTLENECK absent: bottleneck smoke skipped"
    fi
elif [ "$STAGE" = "full" ]; then
    if [ ! -f "$CK_BOTTLENECK" ]; then echo "# $CK_BOTTLENECK absent -- refusing to run the full stage"; exit 3; fi
    python -u -m experiments.multilayer_cert.conv_encoder_ranklaw --spec bottleneck --ckpt $CK_BOTTLENECK --label wp1-full \
        --N 8 --r 256 --ks 16 32 66 128 256 384 512 784 --first 1 3 5 --maxL 6 --T-arm 1 5 20 100 400 --lr 0.01 --seed 1 \
        --out "$OUT/conv_ranklaw_${LSB_JOBID:-manual}.jsonl"
else
    echo "unknown stage $STAGE"; exit 2
fi
echo "=== DONE $(date) ==="
