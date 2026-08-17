#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/bridge_by_activation_%J.out
#BSUB -e scripts/wexac_logs/bridge_by_activation_%J.err
#BSUB -J bridge_by_activation

# Exp B: the BRIDGE CONNECTION (never run). Does the same smoothness/gate-rank that governs
# direct-inversion leakage also govern the gradient-bridge DECODABILITY? Train the R2F-style decoder
# f_phi:(A,B)->grad_W L on the hidden layer (L1, r=8) per activation, and read the decoded full-cosine
# vs the col(B0) projection ceiling. Then correlate the decoded cosine with eff_rank(M) / smoothness
# (gate-matrix test). Hypothesis: smooth activations give collinear/low-rank per-sample gradients ->
# a more predictable proxy manifold -> the decoder hallucinates the out-of-subspace part more easily
# (higher decoded cosine); kinked (relu) -> diverse gradients -> harder. A duality with direct
# inversion (where collinearity HURTS leakage) would be the interesting result.
#
# Spectrum: softplus (smoothest winner) -> silu -> gelu -> relu (kink), + softplus_b50 (near-ReLU).
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"

NP=12000
for ACT in softplus silu gelu softplus_b50 relu; do
    SAFE=$(echo "$ACT" | tr -d '.')
    echo ""; echo "########## bridge decoder act=$ACT (L1, r=8) ##########"; date
    python -u -m experiments.gradient_bridge.generate_pairs \
        --layer 1 --rank 8 --n_pairs $NP --activation "$ACT" --device cuda \
        --save "results/gb_pairs_L1_r8_${SAFE}.pth"
    if [ $? -ne 0 ]; then echo "FATAL: pair gen ($ACT) failed."; continue; fi
    python -u -m experiments.gradient_bridge.train_decoder \
        --bank "results/gb_pairs_L1_r8_${SAFE}.pth" --epochs 100 \
        --out_mode lowrank --out_rank 16 --batch 128 --device cuda \
        --tag "L1_r8_${SAFE}"
done
echo "=== ALL DONE $(date) ==="
