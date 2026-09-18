#!/bin/bash
# =====================================================================
# P3 / P6 (plan 2026-09-18 + its Audit items 1-4): drift of an adapted layer's input, both certificates at the truth,
# and the per-image certificate solve.  Harness: experiments/multilayer_cert/drift_cert.py (read its docstring: the
# pre-registered outcomes are there).  Modelled on run_conv_ranklaw_wexac.sh: set +u, conda activate, python -u,
# job-id-suffixed outputs, git hash + script sha on every row.
#
#   bash scripts/run_drift_cert_wexac.sh submit smoke              # 1 job, long-gpu, < 25 min
#   bash scripts/run_drift_cert_wexac.sh submit p3                 # 15 jobs: twin targets {2,4,8} x seeds {1,2,3} + strong targets {2,3} x seeds
#   bash scripts/run_drift_cert_wexac.sh submit p6                 # 2 jobs: drift arm + zero-drift arm (seeds inside)
#   DRY=1 bash scripts/run_drift_cert_wexac.sh submit p3           # print the bsub lines, submit nothing
#   CERT=both bash scripts/run_drift_cert_wexac.sh submit p3       # solve with the full AND the truncated certificate (2x time)
#   PRIVATE=onchart bash scripts/run_drift_cert_wexac.sh submit p3  # privates = their PCA-32 projections (Rule B's eps_land needs this; see DRIFT_NOTES)
# Job bodies (what bsub runs):  smoke | p3 <twin|strong> <target> <seed> | p6 <drift|ctrl>
#
# P3 cells per job: the zero-drift CONTROL (--adapt = target only; eps_land is measured there) at T x lr (15 cells),
# then the drift cells with layer target-1 also adapted at r_lower in {4,16,64} (45 cells) -- 60 cells per job.
# Strong MLP: target 2 (layer 1 adapted) and target 3 = the softmax head, PRE-REGISTERED contaminated (audit item 1).
# P6: twin target 2 with layer 1 adapted at the best (T, lr, r_lower) from P3 -- PLACEHOLDERS below, the coordinator
# fills them -- momentum {0, 0.9} x wd {0, eta*lambda*T in {0.01,0.1,1,3,10,20}, the eta*lambda = 1 corner}, seeds 1 2 3,
# and the same grid at zero drift (audit item 4).  lambda is derived in-job from P6_LR and P6_T.
# =====================================================================
TWIN=models/exact_inversion/mnist_mlp_d15w1000_full.pth
STRONG=models/exact_inversion/mnist_mlp_strong.pth
CERT=${CERT:-full}                      # full | trunc | both  (which certificate the solve uses)
PRIVATE=${PRIVATE:-raw}                 # raw | onchart  (onchart: the truth is in the PCA-k chart -> tier 1 can land, eps_land is a landing floor; Rule B)
STARTS=${STARTS:-400}
# ---- P6 placeholders: the coordinator fills these from the P3 read-out (best (T, lr, r_lower) for twin target 2)
P6_T=${P6_T:-400}
P6_LR=${P6_LR:-FILL_ME}
P6_RL=${P6_RL:-FILL_ME}
P6_ETALT="0.01 0.1 1 3 10 20"           # eta*lambda*T grid (log-spaced), plus the eta*lambda = 1 corner

STAGE="${1:-smoke}"
GPU_RES='-gpu num=1:gmem=20G -R rusage[mem=24576]'
if [ "$STAGE" = "submit" ]; then
    WHAT="${2:-smoke}"
    cd /home/projects/galvardi/yoado
    mkdir -p scripts/wexac_logs results/multilayer_cert figures/multilayer_cert/drift_cert
    run() { echo "bsub $*"; [ -z "$DRY" ] && bsub "$@"; }
    case "$WHAT" in
      smoke)
        run -q long-gpu -gpu "num=1:gmem=20G" -R "rusage[mem=24576] select[ngpus>0]" -W 0:30 -J driftcert_smoke \
            -o scripts/wexac_logs/driftcert_smoke_%J.out -e scripts/wexac_logs/driftcert_smoke_%J.err \
            bash scripts/run_drift_cert_wexac.sh smoke ;;
      p3)
        for SEED in 1 2 3; do
          for TGT in 2 4 8; do
            run -q long-gpu -gpu "num=1:gmem=20G" -R "rusage[mem=24576] select[ngpus>0]" -W 36:00 -J driftcert_p3_twin_t${TGT}_s${SEED} \
                -o scripts/wexac_logs/driftcert_p3_twin_t${TGT}_s${SEED}_%J.out -e scripts/wexac_logs/driftcert_p3_twin_t${TGT}_s${SEED}_%J.err \
                bash scripts/run_drift_cert_wexac.sh p3 twin $TGT $SEED
          done
          for TGT in 2 3; do
            run -q long-gpu -gpu "num=1:gmem=20G" -R "rusage[mem=24576] select[ngpus>0]" -W 36:00 -J driftcert_p3_strong_t${TGT}_s${SEED} \
                -o scripts/wexac_logs/driftcert_p3_strong_t${TGT}_s${SEED}_%J.out -e scripts/wexac_logs/driftcert_p3_strong_t${TGT}_s${SEED}_%J.err \
                bash scripts/run_drift_cert_wexac.sh p3 strong $TGT $SEED
          done
        done ;;
      p6)
        if [ "$P6_LR" = "FILL_ME" ] || [ "$P6_RL" = "FILL_ME" ]; then echo "P6_LR / P6_RL are placeholders: fill them from the P3 read-out first"; exit 2; fi
        for ARM in drift ctrl; do
          run -q long-gpu -gpu "num=1:gmem=20G" -R "rusage[mem=24576] select[ngpus>0]" -W 24:00 -J driftcert_p6_${ARM} \
              -o scripts/wexac_logs/driftcert_p6_${ARM}_%J.out -e scripts/wexac_logs/driftcert_p6_${ARM}_%J.err \
              bash scripts/run_drift_cert_wexac.sh p6 $ARM
        done ;;
      *) echo "unknown stage $WHAT"; exit 2 ;;
    esac
    exit 0
fi

set +u                                   # conda activate breaks under set -u in this env (LESSONS_LEARNED)
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUTDIR=results/multilayer_cert; mkdir -p $OUTDIR figures/multilayer_cert/drift_cert
JOB=${LSB_JOBID:-manual}
echo "=== START drift_cert stage=$STAGE args=$* $(date) on $(hostname) git=$(git rev-parse --short HEAD) job=$JOB cert=$CERT private=$PRIVATE ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()} torch={torch.__version__} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
python -m py_compile experiments/multilayer_cert/drift_cert.py || exit 3
RUN="python -u -m experiments.multilayer_cert.drift_cert"
TS="1 5 20 100 400"; LRS="0.003 0.01 0.03"

case "$STAGE" in
  smoke)
    OUT=$OUTDIR/drift_cert_smoke_${JOB}.jsonl
    # (1) zero-drift control: layer 2 alone (its input is the frozen layer-1 output) -- rho_full MUST be at the FP64 floor
    $RUN --model $TWIN --adapt 2 --target 2 --r-per-layer 64 --T 1 20 --lr 0.01 --momentum 0 --wd 0 --starts 20 --seed 1 --label smoke-control --out $OUT
    # (2) the drift cell of the spec: layer 1 adapted below the target layer 2
    $RUN --model $TWIN --adapt 1 2 --target 2 --r-per-layer 16 64 --T 1 20 --lr 0.01 --momentum 0 --wd 0 --starts 20 --seed 1 --label smoke-drift --out $OUT
    # (3) the strong-MLP loader path, no solve (seconds); and the head target pre-registered contaminated
    $RUN --model $STRONG --adapt 1 2 --target 2 --r-per-layer 16 64 --T 20 --lr 0.01 --starts 5 --seed 1 --no-solve --label smoke-strong-loader --out $OUT
    $RUN --model $STRONG --adapt 2 3 --target 3 --r-per-layer 16 64 --T 20 --lr 0.01 --starts 5 --seed 1 --no-solve --label smoke-strong-head --out $OUT
    # (4) the option code paths: momentum, weight decay, --stack-below, both certificates (5 starts)
    $RUN --model $TWIN --adapt 1 2 --target 2 --r-per-layer 16 64 --T 20 --lr 0.01 --momentum 0.9 --wd 1e-3 --starts 5 --seed 1 --stack-below --solve-cert both --label smoke-options --out $OUT
    ;;
  p3)
    MODEL_TAG=$2; TGT=$3; SEED=$4
    case "$MODEL_TAG" in twin) MODEL=$TWIN ;; strong) MODEL=$STRONG ;; *) echo "model twin|strong"; exit 2 ;; esac
    LOWER=$((TGT - 1))
    OUT=$OUTDIR/drift_cert_p3_${MODEL_TAG}_tgt${TGT}_s${SEED}_${PRIVATE}_${JOB}.jsonl
    $RUN --model $MODEL --adapt $TGT --target $TGT --r-per-layer 64 --T $TS --lr $LRS --momentum 0 --wd 0 --starts $STARTS --seed $SEED \
         --solve-cert $CERT --private $PRIVATE --label p3-control-$PRIVATE --out $OUT
    for RL in 4 16 64; do
      $RUN --model $MODEL --adapt $LOWER $TGT --target $TGT --r-per-layer $RL 64 --T $TS --lr $LRS --momentum 0 --wd 0 --starts $STARTS --seed $SEED \
           --solve-cert $CERT --private $PRIVATE --label p3-drift-$PRIVATE --out $OUT
    done
    ;;
  p6)
    ARM=$2
    LAMBDAS=$(python -c "lr,T=$P6_LR,$P6_T; print(' '.join(repr(x/(lr*T)) for x in [$(echo $P6_ETALT | sed 's/ /,/g')]) + ' ' + repr(1.0/lr))")
    echo "# P6 wd grid (lambda) at lr=$P6_LR T=$P6_T: 0 $LAMBDAS  (last = the eta*lambda = 1 corner)"
    OUT=$OUTDIR/drift_cert_p6_${ARM}_${PRIVATE}_${JOB}.jsonl
    if [ "$ARM" = "drift" ]; then ADAPT="1 2"; RPL="$P6_RL 64"; else ADAPT="2"; RPL="64"; fi
    $RUN --model $TWIN --adapt $ADAPT --target 2 --r-per-layer $RPL --T $P6_T --lr $P6_LR --momentum 0 0.9 --wd 0 $LAMBDAS --starts $STARTS --seed 1 2 3 \
         --solve-cert $CERT --private $PRIVATE --label p6-$ARM-$PRIVATE --out $OUT
    ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "=== DONE $(date) ==="
