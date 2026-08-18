#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/gb_phase2_tv_%J.out
#BSUB -e scripts/wexac_logs/gb_phase2_tv_%J.err
#BSUB -J gb_phase2_tv

# GB-Phase 2 + TV lever: apply Phase-0's prior lesson (the prior is the lever, not the cosine) to the
# bridge's two-sided decoded input-layer gradient. Sweep the TV weight to find the MNIST-scale sweet
# spot (Phase 0 found tv=1e-1 on 224x224 ViT). Prediction: SSIM jumps from the SVD baseline (lambda=0)
# toward Phase-0-style recovery. Grids: true / raw-SVD / TV-prior.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
echo "=== START $(date) on $(hostname) ==="
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
for ACT in softplus gelu; do
    echo ""; echo "########## $ACT ##########"; date
    python -u -m experiments.gradient_bridge.phase2_tv --activation $ACT --device cuda
done
echo "=== DONE $(date) ==="
