#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/margin_at_scale_%J.out
#BSUB -e scripts/wexac_logs/margin_at_scale_%J.err
#BSUB -J margin_at_scale
# SIII.1 MARGIN AT SCALE — the headline figure (WHO leaks). Upgrades the n=12 margin MVP
# (job 260171, rho(sens,g0)=+0.857, CI ~ +/-0.4) to 24 targets stratified across the g0
# spectrum, both classes, with permutation-p + bootstrap 95% CI (pre-reg: rho>+0.6,
# half-width<0.15; KILL rho<+0.3), the MANDATORY theta_0-independent typicality control
# (partial rho), and the lazy/NTK diagnostic (spearman(g0,gT) + per-module ||dW||/||W0||).
# ~24 x 2 x 50 = 2400 trainings (arm-D scale x2); the script prints its own runtime estimate.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
echo "########## GATE 1: whitened-metric acceptance self-test ##########"
python -u -m experiments.dataset_sensitivity.whitened_metric || { echo FATAL metric; exit 1; }
echo "########## GATE 2: margin_at_scale --stage0 (4 targets, K=12, assert finite) ##########"
python -u -m experiments.dataset_sensitivity.margin_at_scale --stage0 --device cuda || { echo FATAL stage0; exit 1; }
echo "########## FULL: 24 targets (12/class, g0-stratified), N=16, K=50 ##########"; date
python -u -m experiments.dataset_sensitivity.margin_at_scale \
    --n_targets 24 --N 16 --K 50 --lr 0.5 --T 1000 --rank 8 \
    --n_perm_rho 10000 --n_boot 10000 --device cuda
echo "=== DONE $(date) ==="
echo "READ: verdict block at the end — PASS = rho(sens,g0)>+0.6 AND CI half-width<0.15, no tercile"
echo "sign flip; KILL = rho<+0.3. Then: PARTIAL rho(sens,g0|atypicality) (does theta_0 geometry"
echo "predict beyond intrinsic atypicality?) and spearman(g0,gT) + ||dW||/||W0|| (NTK mechanism)."
