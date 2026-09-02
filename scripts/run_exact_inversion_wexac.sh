#!/bin/bash
# =====================================================================
# Exact LoRA inversion (framework Rev 10, Primitive 3, exact form) on WEXAC.
# Usage:  bash scripts/run_exact_inversion_wexac.sh <stage> [arg]
#   step1               validation cells vs the finite-difference table (results_rev9 S3b) + CPU/GPU timing
#   step2_near <noise>  basin study, init=near, one --init-noise value, 5 seeds, 8 restarts
#   step2_init <init>   basin study with an attacker-available initialiser: random | span | cert | spananchor
#   step3               Adam release, init-noise sweep
#   step4 <N>           (N,k) phase diagram with backprop, one N column (k in 2..14)
# Submit with e.g.
#   bsub -q long-gpu -gpu "num=1" -R "rusage[mem=8192] select[ngpus>0]" -J ei_step1 \
#        -o scripts/wexac_logs/ei_step1_%J.out -e scripts/wexac_logs/ei_step1_%J.err \
#        bash scripts/run_exact_inversion_wexac.sh step1
# Every python call appends a JSON line (seed, git hash, cmdline, host) to results/exact_inversion/*.jsonl.
# =====================================================================
set +u
STAGE=${1:-step1}; ARG=${2:-}
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
PY="python -u experiments/exact_inversion/lora_exact_inversion.py"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"} torch={torch.__version__}')"
echo "=== START $STAGE $ARG $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="

case $STAGE in
  step1)
    # cells of results_rev9.pdf S3b (finite differences):  (k,N,T,lr,start)
    for cell in "12 8 400 0.01 0.10" "6 12 400 0.01 0.24" "8 10 400 0.01 0.05" "12 8 1500 0.03 0.04" "12 8 1500 0.03 0.10"; do
      set -- $cell
      echo "##### step1 cell k=$1 N=$2 T=$3 lr=$4 init-noise=$5 (cuda)"
      $PY --release sgd --k $1 --N $2 --T $3 --lr $4 --init near --init-noise $5 --seed 1 \
          --lm-iters 60 --device cuda --out $OUT/step1_validation.jsonl --save-prefix $OUT/step1
    done
    echo "##### step1 timing: same first cell on CPU"
    $PY --release sgd --k 12 --N 8 --T 400 --lr 0.01 --init near --init-noise 0.10 --seed 1 \
        --lm-iters 10 --device cpu --out $OUT/step1_timing_cpu.jsonl
    ;;
  step2_near)
    $PY --release sgd --k 12 --N 8 --T 1500 --lr 0.03 --init near --init-noise $ARG --seeds 1 2 3 4 5 \
        --restarts 8 --restart-noise 0.1 --lm-iters 60 --device cuda \
        --out $OUT/step2_basin_near_${LSB_JOBID:-local}.jsonl --save-prefix $OUT/step2 --quiet
    ;;
  step2_init)
    $PY --release sgd --k 12 --N 8 --T 1500 --lr 0.03 --init $ARG --seeds 1 2 3 4 5 \
        --restarts 8 --restart-noise 0.1 --lm-iters 60 --init-iters 400 --device cuda \
        --out $OUT/step2_basin_init_${LSB_JOBID:-local}.jsonl --save-prefix $OUT/step2 --quiet
    ;;
  step3)
    # Adam: unknowns are the latents PLUS the whole A_0 (r x n = 1536 here), so an LM Jacobian is
    # infeasible (>5 min per Jacobian, measured); LBFGS needs one backward per closure instead.
    # The certificate does not exist here (rank B_T = r makes C identically zero) -- read cert_vacuous.
    for noise in 0.05 0.10 0.20 0.30; do
      echo "##### step3 adam init-noise=$noise"
      $PY --release adam --k 12 --N 8 --T 800 --lr 0.003 --init near --init-noise $noise --seeds 1 2 3 \
          --solver lbfgs --outer 40 --lbfgs-iter 40 --restarts 2 --restart-noise 0.1 --device cuda \
          --out $OUT/step3_adam_${LSB_JOBID:-local}.jsonl --save-prefix $OUT/step3
    done
    ;;
  step5_rescue)
    # The 15 (k,N) cells that failed in step4 at restarts=1, re-run post-fix with 4 restarts that now
    # re-seed the WHOLE unknown vector.  Separates "basin/schedule artefact" from "hard cell".
    for cell in "14 2" "12 4" "8 6" "10 8" "2 10" "12 10" "14 10" "2 12" "4 12" "6 12" "10 12" "2 14" "6 14" "8 14" "12 14"; do
      set -- $cell
      echo "##### step5 rescue k=$1 N=$2"
      $PY --release sgd --k $1 --N $2 --T 400 --lr 0.01 --init near --init-noise 0.10 --seed 1 \
          --restarts 4 --restart-noise 0.15 --lm-iters 60 --device cuda \
          --out $OUT/step5_rescue_${LSB_JOBID:-local}.jsonl --save-prefix $OUT/step5 --quiet
    done
    ;;
  step4)
    $PY --sweep --release sgd --T 400 --lr 0.01 --init near --init-noise 0.10 --seed 1 \
        --sweep-Ns $ARG --lm-iters 60 --device cuda \
        --out $OUT/step4_sweep_${LSB_JOBID:-local}.jsonl --sweep-out $OUT/step4_sweep_N$ARG.json
    ;;
  *) echo "unknown stage $STAGE"; exit 1;;
esac
echo "=== DONE $STAGE $ARG $(date) ==="
