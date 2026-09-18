#!/bin/bash
# =====================================================================
# Multilayer parameter program, packages P1 (rank r sweep) and P4 (layer subsets) of
# notes/plan_2026-09-18_multilayer_parameter_program.md, on the two rank-law harnesses
#   experiments/multilayer_cert/real_encoder_ranklaw.py   (d15 MLP twin, models/exact_inversion/mnist_mlp_d15w1000_full.pth)
#   experiments/multilayer_cert/conv_encoder_ranklaw.py   (bottleneck CNN, models/exact_inversion/mnist_conv_bottleneck.pth)
# Modelled on scripts/run_conv_ranklaw_wexac.sh: set +u, conda activate, python -u, job-id-suffixed outputs in
# results/multilayer_cert/, attested git hash + script sha on every row (multilayer_cert.common.provenance).
#
#   bash scripts/run_ranklaw_program_wexac.sh submit smoke   # one job: both harnesses, twin ckpts, r 16/64, k 32/128,
#                                                            #   patterns prefix alternate suffix:2, maxL 3, no T arm,
#                                                            #   PLUS the identity check (original d15 model, r=108,
#                                                            #   k=128, first=1, maxL=2, prefix) diffed against
#                                                            #   results/multilayer_cert/real_encoder_ranklaw_365681.jsonl
#   bash scripts/run_ranklaw_program_wexac.sh submit p4      # two jobs (p4_mlp, p4_cnn): layer subsets at k 128/384/784, seeds 1 2 3
#   bash scripts/run_ranklaw_program_wexac.sh submit p1      # two jobs (p1_mlp, p1_cnn): r sweep, prefix, k 32/128/384/784, seeds 1 2 3
#   bash scripts/run_ranklaw_program_wexac.sh submit p2i     # one job: conv harness on the DEEP conv twin (audit item 11)
#
# Queue/GPU per the program brief: -q long-gpu -gpu "num=1:gmem=20G".  Zero-drift only (no --T-arm) in every stage.
# The `single:8` pattern is vacuous on the 6-module CNN (skipped with a printed note); `single:1` is skipped for
# first > 1 on both harnesses -- by design, never silently remapped.
#
# p1 BRIDGE CELL: the d15 TWIN at --r 108 (inside the p1_mlp list) is the bridge to job 365681, which ran the
# ORIGINAL d15 model at r = 108 (same seed-1 truths, same A_0 draw; only the backbone weights differ).
#
# p1 CNN PRE-REGISTERED LIVE-MODULE SET PER r (audit item 7), from the measured zero-drift patch spans of job 355907
# N'_l = [9, 232, 117, 32, 8, 8] on modules [conv1 p=9, conv2 p=576, conv3 p=1152, conv4 p=72, dense p=1024, head
# p=1000]; certificate rank = min(r, p_l) - N'_l, a module is LIVE when that is > 0:
#   r = 8   : none live (dense/head r - N = 0) -> the cell tests NOTHING; kept as the control
#   r = 16  : dense (8), head (8)                                  conv1-4 vacuous
#   r = 32  : dense (24), head (24)                                conv1-4 vacuous
#   r = 64  : conv4 (32), dense (56), head (56)                    conv1-3 vacuous
#   r = 128 : conv3 (11), conv4 (40), dense (120), head (120)      conv1-2 vacuous
#   r = 256 : conv2 (24), conv3 (139), conv4 (40), dense (248), head (248)   conv1 vacuous (p_l = 9 = N'_1 at every r)
# i.e. conv1 vacuous always, conv2 vacuous for r <= 232, conv3 for r <= 117, conv4 for r <= 32.  The row's
# `vacuous` / `layers_dropped_vacuous` fields record what actually happened against this table.
# =====================================================================
STAGE="${1:-smoke}"
REPO=/home/projects/galvardi/yoado
if [ "$STAGE" = "submit" ]; then
    WHAT="${2:-smoke}"
    cd $REPO
    mkdir -p scripts/wexac_logs results/multilayer_cert
    GPU='num=1:gmem=20G'
    case "$WHAT" in
        smoke)
            bsub -q long-gpu -gpu "$GPU" -R "rusage[mem=32768] select[ngpus>0]" -W 1:00 \
                 -J ranklaw_smoke -o scripts/wexac_logs/ranklaw_smoke_%J.out -e scripts/wexac_logs/ranklaw_smoke_%J.err \
                 bash scripts/run_ranklaw_program_wexac.sh smoke ;;
        p4)
            bsub -q long-gpu -gpu "$GPU" -R "rusage[mem=49152] select[ngpus>0]" -W 8:00 \
                 -J ranklaw_p4_mlp -o scripts/wexac_logs/ranklaw_p4_mlp_%J.out -e scripts/wexac_logs/ranklaw_p4_mlp_%J.err \
                 bash scripts/run_ranklaw_program_wexac.sh p4_mlp
            bsub -q long-gpu -gpu "$GPU" -R "rusage[mem=49152] select[ngpus>0]" -W 8:00 \
                 -J ranklaw_p4_cnn -o scripts/wexac_logs/ranklaw_p4_cnn_%J.out -e scripts/wexac_logs/ranklaw_p4_cnn_%J.err \
                 bash scripts/run_ranklaw_program_wexac.sh p4_cnn ;;
        p2i)
            bsub -q long-gpu -gpu "$GPU" -R "rusage[mem=49152] select[ngpus>0]" -W 8:00 \
                 -J ranklaw_p2i_cnn -o scripts/wexac_logs/ranklaw_p2i_cnn_%J.out -e scripts/wexac_logs/ranklaw_p2i_cnn_%J.err \
                 bash scripts/run_ranklaw_program_wexac.sh p2i_cnn ;;
        p1)
            bsub -q long-gpu -gpu "$GPU" -R "rusage[mem=49152] select[ngpus>0]" -W 10:00 \
                 -J ranklaw_p1_mlp -o scripts/wexac_logs/ranklaw_p1_mlp_%J.out -e scripts/wexac_logs/ranklaw_p1_mlp_%J.err \
                 bash scripts/run_ranklaw_program_wexac.sh p1_mlp
            bsub -q long-gpu -gpu "$GPU" -R "rusage[mem=49152] select[ngpus>0]" -W 10:00 \
                 -J ranklaw_p1_cnn -o scripts/wexac_logs/ranklaw_p1_cnn_%J.out -e scripts/wexac_logs/ranklaw_p1_cnn_%J.err \
                 bash scripts/run_ranklaw_program_wexac.sh p1_cnn ;;
        *) echo "unknown stage $WHAT (smoke|p4|p1|p2i)"; exit 2 ;;
    esac
    exit 0
fi

set +u                                   # conda activate breaks under set -u in this env (LESSONS_LEARNED)
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd $REPO
OUT=results/multilayer_cert; mkdir -p $OUT
JOB=${LSB_JOBID:-manual}
CK_MLP_TWIN=models/exact_inversion/mnist_mlp_d15w1000_full.pth     # PASSES the WP0 gate (BASE_TRAINING_GATE.md, 355870/355873)
CK_MLP_ORIG=models/exact_inversion/mnist_mlp_d15w1000.pth          # the model of job 365681 (identity check only)
CK_CNN=models/exact_inversion/mnist_conv_bottleneck.pth
CK_CNN_DEEP=models/exact_inversion/mnist_conv_deep_full.pth       # PASSES the WP0 gate (355840/355842); deep spec = 4 convs + head
BASELINE=results/multilayer_cert/real_encoder_ranklaw_365681.jsonl
P4_PATTERNS="prefix suffix:2 suffix:4 middle:2 middle:4 alternate random:2:3 random:4:3 single:1 single:3 single:5 single:8"
for f in $CK_MLP_TWIN $CK_MLP_ORIG $CK_CNN; do
    if [ ! -f "$f" ]; then echo "# $f absent -- refusing to run stage $STAGE"; exit 3; fi
done
echo "=== START ranklaw_program stage=$STAGE $(date) on $(hostname) git=$(git rev-parse --short HEAD) job=$JOB ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()} torch={torch.__version__} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
T0=$(date +%s)

MLP="python -u -m experiments.multilayer_cert.real_encoder_ranklaw --N 8"
CNN="python -u -m experiments.multilayer_cert.conv_encoder_ranklaw --spec bottleneck --ckpt $CK_CNN --N 8"
SEEDS3="--seed 1 2 3"                    # three seeds where a number is quoted (plan common ground); the smoke stays at seed 1

case "$STAGE" in
    smoke)
        $MLP --seed 1 --model $CK_MLP_TWIN --r 16 64 --ks 32 128 --first 1 3 --maxL 3 --layers prefix alternate suffix:2 \
             --out "$OUT/ranklaw_program_smoke_mlp_${JOB}.jsonl"
        echo "=== smoke MLP twin done at +$(( $(date +%s) - T0 ))s ==="
        $CNN --seed 1 --label smoke --r 16 64 --ks 32 128 --first 1 3 5 --maxL 3 --layers prefix alternate suffix:2 \
             --out "$OUT/ranklaw_program_smoke_cnn_${JOB}.jsonl"
        echo "=== smoke CNN done at +$(( $(date +%s) - T0 ))s ==="
        # identity check: the ORIGINAL model at 365681's settings restricted to (k=128, first=1, L<=2), default flags
        $MLP --seed 1 --model $CK_MLP_ORIG --r 108 --ks 128 --first 1 --maxL 2 --layers prefix \
             --out "$OUT/ranklaw_program_smoke_identity_${JOB}.jsonl"
        echo "=== identity run done at +$(( $(date +%s) - T0 ))s; diffing against $BASELINE ==="
        python - "$BASELINE" "$OUT/ranklaw_program_smoke_identity_${JOB}.jsonl" <<'PY'
import json, sys
IGNORE = {"git", "script_sha", "host", "cmd"}          # job/host/time/provenance fields
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
old = [r for r in load(sys.argv[1]) if r["config"]["k"] == 128 and r["config"]["first_adapted"] == 1 and r["n_layers"] <= 2]
new = load(sys.argv[2])
key = lambda r: (r["config"]["k"], r["config"]["first_adapted"], r["config"]["r"], r["n_layers"])
newd = {key(r): r for r in new}
ndiff, nnoise = 0, 0
def close(a, b, rel=1e-9):
    if isinstance(a, bool) or isinstance(b, bool): return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300) or (a == 0 and b == 0)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b): return all(close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys(): return all(close(a[k], b[k], rel) for k in a)
    return a == b
print(f"IDENTITY DIFF: {len(old)} baseline rows, {len(new)} new rows")
for o in old:
    n = newd.get(key(o))
    if n is None:
        print(f"  MISSING new row for {key(o)}"); ndiff += 1; continue
    for k, v in o.items():
        if k in IGNORE: continue
        if k not in n:
            print(f"  {key(o)} field {k!r}: present in baseline (= {v!r}), ABSENT in new row"); ndiff += 1
        elif n[k] != v:
            fp = close(v, n[k]); nnoise += int(fp); ndiff += int(not fp)
            print(f"  {key(o)} field {k!r} [{'FP-NOISE rel<=1e-9' if fp else 'REAL'}]: baseline {v!r} != new {n[k]!r}")
    added = [k for k in n if k not in o]
    print(f"  {key(o)}: fields added in new row: {added}")
print(f"IDENTITY DIFF RESULT: {ndiff} REAL differences on pre-existing fields (+{nnoise} float diffs within rel 1e-9, "
      f"i.e. GPU/host noise; ignoring {sorted(IGNORE)}).  gap_at_corrected is a ratio of null-space singular values "
      f"(~1e-16) and is expected to move by O(1) between hosts; the ranks and ladders must not.")
PY
        ;;
    p4_mlp)
        $MLP $SEEDS3 --model $CK_MLP_TWIN --r 108 --ks 128 384 784 --first 1 3 --maxL 8 --layers $P4_PATTERNS \
             --out "$OUT/ranklaw_p4_mlp_${JOB}.jsonl" ;;
    p4_cnn)
        $CNN $SEEDS3 --label p4 --r 256 --ks 128 384 784 --first 1 3 5 --maxL 6 --layers $P4_PATTERNS \
             --out "$OUT/ranklaw_p4_cnn_${JOB}.jsonl" ;;
    p1_mlp)
        $MLP $SEEDS3 --model $CK_MLP_TWIN --r 8 16 32 64 108 256 --ks 32 128 384 784 --first 1 3 --maxL 8 --layers prefix \
             --out "$OUT/ranklaw_p1_mlp_${JOB}.jsonl" ;;
    p1_cnn)
        $CNN $SEEDS3 --label p1 --r 8 16 32 64 128 256 --ks 32 128 384 784 --first 1 3 5 --maxL 6 --layers prefix \
             --out "$OUT/ranklaw_p1_cnn_${JOB}.jsonl" ;;
    p2i_cnn)
        if [ ! -f "$CK_CNN_DEEP" ]; then echo "# $CK_CNN_DEEP absent -- refusing to run stage $STAGE"; exit 3; fi
        python -u -m experiments.multilayer_cert.conv_encoder_ranklaw --spec deep --ckpt $CK_CNN_DEEP --N 8 --seed 1 --label p2i \
             --r 256 --ks 16 32 66 128 256 384 512 784 --first 1 3 --maxL 5 --layers prefix alternate \
             --out "$OUT/ranklaw_p2i_cnn_${JOB}.jsonl" ;;
    p4repl)
        # Replication of the ONE law-separating GAPPED cell from p4_mlp (job 366146): the random non-contiguous set
        # [3,4,10,15] predicts 260 (corrected) vs 287 (T5.2), had a real gap (6e5/8e5) and measured 260 at k=384 and
        # 784 -- but on a SINGLE seed, and it is the only such cell in 522 rows.  Here: that set plus five neighbours,
        # all three seeds, both widths.  If it replicates, it is the MLP's first gapped refutation of T5.2.
        $MLP $SEEDS3 --model $CK_MLP_TWIN --r 108 --ks 384 784 --first 1 --maxL 8 \
             --layers explicit:3,4,10,15 explicit:3,4,10,14 explicit:3,4,9,15 explicit:3,5,10,15 explicit:2,4,10,15 explicit:3,4,11,15 \
             --out "$OUT/ranklaw_p4repl_${JOB}.jsonl" ;;
    *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "=== DONE $(date) wall=$(( $(date +%s) - T0 ))s ==="
