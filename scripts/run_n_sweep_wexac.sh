#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/n_sweep_%J.out
#BSUB -e scripts/wexac_logs/n_sweep_%J.err
#BSUB -J n_sweep

# =====================================================================
# Larger-N sweep — the point is to make SSIM meaningful again.
#
# At N=2 the ds_mean baseline is ~0.76 (mostly-black MNIST + tiny N), so
# SSIM can't tell a real reconstruction from a blur. As N grows the mean
# becomes an unrecognisable smudge and the baseline drops, opening room
# for a real attack to prove itself. We sweep N and, for each, read the
# RECONSTRUCTION vs the ds_mean BASELINE (both now in compute_all_metrics).
#
# Held fixed (default activation = the known-good path that gave full=0.99,
# so this isolates the N effect and stays comparable to all historical runs):
#   T=1, rank in {full, 8}, seeds 42/43/44. NO --finetune_activation.
# Resumable via --skip_if_exists (npc=1 points already exist -> reused free).
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -u -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "=== START $(date) on $(hostname) ==="

# For the FULL model, the reconstruction IS the "baseline" run, so it must NOT get --no_baseline
# (rank=None + --no_baseline -> "Nothing to run"). For LoRA we pass --no_baseline to skip the full
# reference and reconstruct only the adapter.
run_full () {  # run_full <label> <extra args...>
    local label="$1"; shift
    echo ""; echo "=== ${label} ==="; date
    python -u -m experiments.run_experiment_b \
        --n_steps 1 --save_results --skip_if_exists --device cuda "$@"
}
run_lora () {  # run_lora <label> <extra args...>
    local label="$1"; shift
    echo ""; echo "=== ${label} ==="; date
    python -u -m experiments.run_experiment_b \
        --n_steps 1 --no_baseline --save_results --skip_if_exists --device cuda "$@"
}

# N = 2 * n_per_class (binary odd/even). npc 1..16 -> N = 2..32.
for NPC in 1 2 4 8 16; do
    for SEED in 42 43 44; do
        run_full "N-sweep FULL npc=$NPC seed=$SEED"  --n_per_class $NPC --seed $SEED
        run_lora "N-sweep LoRA r8 npc=$NPC seed=$SEED" --rank 8 --n_per_class $NPC --seed $SEED
    done
done

echo ""; echo "=== N-SWEEP COMPLETE $(date) ==="
echo "Re-score + read the recon-vs-baseline gap with:"
echo "  python -m experiments.recompute_metrics --glob 'results/exp_b_T1_*npc*.pth'"
