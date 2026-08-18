#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gb_phase2_full_%J.out
#BSUB -e scripts/wexac_logs/gb_phase2_full_%J.err
#BSUB -J gb_phase2_full

# GB-Phase 2 full-pipeline test (Exp 2): decoded per-layer gradients -> full ΔW -> Experiment-B inverter.
# Sensitivity form: corrupt the true ΔW per layer to the MEASURED decoder cosine and run extraction.
# Key question: does the full extraction (using ALL layers + known θ_0) recover x when the input layer
# is degraded to 0.64 but the hidden layer is 0.997? Arms isolate which layer's fidelity matters.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
for ACT in softplus gelu; do
    echo ""; echo "########## activation=$ACT ##########"; date
    python -u -m experiments.gradient_bridge.phase2_full --activation $ACT --npc 1 --seed 42 --device cuda
done
echo "=== DONE $(date) ==="
