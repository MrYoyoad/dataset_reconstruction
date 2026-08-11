#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 8:00
#BSUB -o scripts/wexac_logs/gb_improve_%J.out
#BSUB -e scripts/wexac_logs/gb_improve_%J.err
#BSUB -J gb_improve

# =====================================================================
# GB-Phase 1 IMPROVE — three levers to push layer-1 full-cosine past 0.685.
#
# The r8 hidden-layer decoder plateaus at full-cos 0.685 (ceiling 0.086), FLAT
# across rank -> recovery is prior/decoder-limited, not measurement-limited. We
# test three orthogonal levers, all on LAYER 1 (out=1000, in=1000):
#
#   (i)   baseline r8        : control, reproduces the 0.685 reference.
#   (ii)  multi-sample m=8   : realistic rank-8 batch gradient (Sum_i g_err_i (x) g_inp_i).
#   (iii) two-sided a0=0.1   : add a small A0 -> measure via col(B0) AND row(A0)
#                              (2r-dim subspace, lifts the honest ceiling).
#   (iv)  big decoder        : hidden 2048, depth 3, out_rank 32, 250 ep
#                              -> directly tests the "decoder-limited" hypothesis.
#
# Honest read for every arm: full-cosine vs projection-cosine (the col(B0) or
# two-sided ceiling). Beating the ceiling = hallucinating out-of-subspace grad.
# Sized to stay well under ~1.5 GPU-hr total.
# =====================================================================

source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec

cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -u -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "=== START $(date) on $(hostname) ==="

NP=12000

# ---------------------------------------------------------------------
# Arm (i) — baseline r8 (control): reproduce the 0.685 reference.
# ---------------------------------------------------------------------
echo ""; echo "########## ARM (i): layer 1 baseline r8 ##########"; date
python -u -m experiments.gradient_bridge.generate_pairs \
    --layer 1 --rank 8 --n_pairs $NP --device cuda \
    --save results/gb_pairs_L1_r8_improve.pth
if [ $? -ne 0 ]; then echo "FATAL: pair gen (i) failed."; exit 1; fi
python -u -m experiments.gradient_bridge.train_decoder \
    --bank results/gb_pairs_L1_r8_improve.pth --epochs 100 \
    --out_mode lowrank --out_rank 16 --batch 128 --device cuda \
    --tag L1_r8_baseline

# ---------------------------------------------------------------------
# Arm (ii) — multi-sample m=8 (rank-8 batch gradient). Far fewer pairs since each
# consumes m=8 informative proxy samples and survival ~50% (need ~NP*m/0.5 from
# 60000 MNIST train): 3500 pairs -> ~56000 seen, comfortably inside the pool.
# ---------------------------------------------------------------------
echo ""; echo "########## ARM (ii): layer 1 multi-sample m=8 ##########"; date
python -u -m experiments.gradient_bridge.generate_pairs \
    --layer 1 --rank 8 --n_pairs 3500 --samples_per_pair 8 --device cuda \
    --save results/gb_pairs_L1_r8_m8.pth
if [ $? -ne 0 ]; then echo "FATAL: pair gen (ii) failed."; exit 1; fi
python -u -m experiments.gradient_bridge.train_decoder \
    --bank results/gb_pairs_L1_r8_m8.pth --epochs 100 \
    --out_mode lowrank --out_rank 16 --batch 128 --device cuda \
    --tag L1_r8_m8

# ---------------------------------------------------------------------
# Arm (iii) — two-sided measurement (A0 = 0.1*randn). Both channels + bases.
# ---------------------------------------------------------------------
echo ""; echo "########## ARM (iii): layer 1 two-sided a0=0.1 ##########"; date
python -u -m experiments.gradient_bridge.generate_pairs \
    --layer 1 --rank 8 --n_pairs $NP --two_sided --a_init_scale 0.1 --device cuda \
    --save results/gb_pairs_L1_r8_twosided.pth
if [ $? -ne 0 ]; then echo "FATAL: pair gen (iii) failed."; exit 1; fi
python -u -m experiments.gradient_bridge.train_decoder \
    --bank results/gb_pairs_L1_r8_twosided.pth --epochs 100 \
    --out_mode lowrank --out_rank 16 --batch 128 --device cuda \
    --tag L1_r8_twosided

# ---------------------------------------------------------------------
# Arm (iv) — big decoder on the baseline r8 bank (reuse arm (i)'s pairs).
# ---------------------------------------------------------------------
echo ""; echo "########## ARM (iv): layer 1 big decoder ##########"; date
python -u -m experiments.gradient_bridge.train_decoder \
    --bank results/gb_pairs_L1_r8_improve.pth --epochs 250 \
    --out_mode lowrank --out_rank 32 --hidden 2048 --depth 3 --batch 128 --device cuda \
    --tag L1_r8_bigdec

echo ""; echo "=== ALL ARMS COMPLETE $(date) ==="
