#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/jacobian_honest_smoke_%J.out
#BSUB -e scripts/wexac_logs/jacobian_honest_smoke_%J.err
#BSUB -J jac_honest_smoke

# =====================================================================
# Validate R0b: the honest-theta0 pipeline end-to-end (no swap). Uses the freshly
# trained models/weights-mnist_gelu.pth. Toy-AD FD gate (synthetic) + real MNIST
# smoke (now via _honest_target loading the GELU-trained weights UNDER gelu) + a
# tiny J0 on the honest gelu model. Also checks the modifiedrelu J-guard fires.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## STAGE 0+1: AD gate + HONEST MNIST-GELU smoke ##########"; date
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: gate or honest smoke failed."; exit 1; fi

echo ""; echo "########## STAGE 2: tiny J0 on honest gelu theta0 ##########"; date
python -u -m experiments.jacobian_spectrum --j0 --dataset mnist --N 2 --k 8 \
    --T 5 --rank 8 --activation gelu --tangent qr \
    --eps_list 0.001 0.01 0.1 --device cuda

echo ""; echo "=== DONE $(date) ==="