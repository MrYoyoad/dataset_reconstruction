#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/fullft_jacobian_stage0_%J.out
#BSUB -e scripts/wexac_logs/fullft_jacobian_stage0_%J.err
#BSUB -J fullft_jac_stage0

# =====================================================================
# Arm G (full-FT valley comparison §2.2) — STAGE-0 GATE.
# Full-param Jacobian J_full = ∂vec(Δθ)/∂a vs the LoRA J_LoRA, same tangents.
#
#   Gate 1  jacobian_spectrum toy-AD gate + MNIST smoke (the machinery this arm
#           imports MUST pass first — aborts on fail).
#   Gate 2  arm-G --stage0: assert J_full finite + central-FD spot-check of the
#           jvp double-backward (rel err <1e-4) on the tiny config
#           (N=2,k=8,T=5,layer-0,gelu,float64) + valley-basis build check.
#
# GELU-only, float64 (the create_graph unroll needs C^∞ — never modified_relu).
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## GATE 1: jacobian_spectrum toy-AD gate + MNIST smoke ##########"; date
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: jacobian_spectrum gate failed. Aborting."; exit 1; fi

echo ""; echo "########## GATE 2: arm-G stage-0 (J_full FD spot-check) ##########"; date
python -u -m experiments.dataset_sensitivity.fullft_jacobian --stage0 --device cuda
if [ $? -ne 0 ]; then echo "FATAL: arm-G stage-0 gate failed. Aborting."; exit 1; fi

echo ""; echo "=== ALL STAGE-0 GATES PASSED $(date) ==="
