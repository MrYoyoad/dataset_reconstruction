#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/fullft_valley_core2_%J.out
#BSUB -e scripts/wexac_logs/fullft_valley_core2_%J.err
#BSUB -J fullft_valley_core2

# =====================================================================
# FULL-FT VALLEY COMPARISON — core wave, PART 2 of 2.
# Plan: notes/fullft_valley_comparison_plan.md (v1.2 FINAL). REQUIRES part 1's
# results/fullft_valley/calibration.json (submit with -w 'done(<part1 jobid>)').
# Arms in sequence: D (PRIMARY, all-layer full FT + per-layer readout, saves the
# per-layer Delta-theta/v stacks for the B1 multi-fraction CPU rescore), E_b0 +
# E_eps (the P4 noise-exchangeability pair), B2 (§4.0 SGD-order gate; 1 target,
# {p00, p0_noise, mid, r_cross} — the mid rung is the d*-determining crossing).
# S6c: if part 1's calib printed a mid-rung recommendation != p3_rot15, add
# --mid_rung <rec> to the arm D and B2 lines below before submitting.
# HEADLINE READ ONLY AFTER B1 + B2 PASS (plan §6 step 6).
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

if [ ! -f results/fullft_valley/calibration.json ]; then
    echo "FATAL: results/fullft_valley/calibration.json missing — run part 1 first"
    echo "(bsub < scripts/run_fullft_valley_wexac.sh), then submit this with -w 'done(<jobid>)'."
    exit 1
fi
# S5 BAND ENFORCEMENT: freeze lr per regime from calibration.json (the arms read it via
# resolve_config) and ABORT if any full-FT regime's tuned lr is off the memorization band.
for R in D B2; do
    OK=$(python -c "import json;print(json.load(open('results/fullft_valley/calibration.json'))['lr']['$R']['in_band'])")
    if [ "$OK" != "True" ]; then
        echo "FATAL (S5): regime $R lr NOT in band [1e-4,1e-3) — retune LR_GRID + rerun calib"
        echo "before quoting the headline. Aborting."; exit 1
    fi
done
echo "S5 band check: D + B2 lr in-band (frozen from calibration.json)."

# BRACKET ENFORCEMENT (ε-linearity gate, auditor-unanimous): never run the PRIMARY arm D on a
# failing/unmeasurable ε-bracket. Require bracket['D'] to have PASSED both gates (α∈[-2.3,-1.7]
# AND d*-ratio<2) at the final (possibly ε-shrunk) eps, and measurable != False (adequacy floor).
DPASS=$(python -c "import json;g=json.load(open('results/fullft_valley/calibration.json'))['bracket']['D']['gate'];print(int(bool(g.get('passed')) and g.get('measurable', True)))")
if [ "$DPASS" != "1" ]; then
  echo "FATAL: arm-D ε-bracket did NOT pass linearity+adequacy (calibration.json bracket['D'].gate)."
  echo "       Either no ε in the shrink loop was linear-AND-adequate (arm D UNMEASURABLE — an honest"
  echo "       finding, report it), or calib predates the shrink loop. NOT running the primary arm on a"
  echo "       failing ε. Aborting the wave."; exit 1
fi
echo "arm-D ε-bracket PASS (linearity + adequacy, post-shrink) — proceeding to the primary arm."

echo ""; echo "########## ARM D — FULL FT all layers [PRIMARY], eps-noise, per-layer readout ##########"; date
python -u -m experiments.dataset_sensitivity.fullft_valley --arm D \
    --K 50 --n_targets 2 --T 1000 --rank 8 --N 16 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: arm D FAILED."; exit 1; fi

echo ""; echo "########## ARM E_b0 — LoRA r=8, B0-reseed noise (P4 reference half) ##########"; date
python -u -m experiments.dataset_sensitivity.fullft_valley --arm E_b0 \
    --K 50 --n_targets 2 --T 1000 --rank 8 --N 16 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: arm E_b0 FAILED."; exit 1; fi

echo ""; echo "########## ARM E_eps — LoRA r=8, fixed B0 + theta0 eps-perturb (P4 eps half) ##########"; date
python -u -m experiments.dataset_sensitivity.fullft_valley --arm E_eps \
    --K 50 --n_targets 2 --T 1000 --rank 8 --N 16 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: arm E_eps FAILED."; exit 1; fi

echo ""; echo "########## ARM B2 — §4.0 GATE: full FT under SGD minibatch-order noise ##########"; date
python -u -m experiments.dataset_sensitivity.fullft_valley --arm B2 \
    --K 50 --n_targets 1 --T 1000 --rank 8 --N 16 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: arm B2 FAILED."; exit 1; fi

echo ""; echo "########## ARM F — leave-one-out (§2.1), full-all vs LoRA r=8 + g0 piggyback ##########"; date
python -u -m experiments.dataset_sensitivity.fullft_valley --arm F \
    --K 50 --n_targets 6 --T 1000 --rank 8 --N 16 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: arm F FAILED."; exit 1; fi

echo ""; echo "########## ARM B1 — §4.0 GATE: dimension-invariance rescore of arm-D stacks (CPU) ##########"; date
python -u -m experiments.dataset_sensitivity.fullft_valley --arm B1 \
    --arm_d_tag "" --b1_fractions 25000 100000 450000 1785000 --device cpu
if [ $? -ne 0 ]; then echo "FATAL: arm B1 FAILED."; exit 1; fi

echo ""; echo "=== DONE $(date) ==="
echo "READ ORDER (plan §6): (1) every arm's p00_identity ~0 + p(r_cross)<0.05 gates;"
echo "(2) P4: E_eps vs E_b0 per-rung s within CIs (power gate §4.1.7 from saved stacks"
echo "— a pass without 2x-detection power is INCONCLUSIVE); (3) B2 vs D s/d* at the"
echo "d*-determining rungs (disagreement kills the headline); (4) B1 dimension-"
echo "invariance = CPU rescore of the saved arm-D v/Delta-theta stacks (float32 on"
echo "disk — UPCAST to float64 before the metric) at {~25k, ~100k, ~450k, 1.8M};"
echo "(5) ONLY after B1+B2 pass: the P1/P1b/P2 headline read vs arm A (printed in the"
echo "arm-D read block). TF7 conditionality applies to every cross-regime statement."
