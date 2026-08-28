#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=8192] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/meeting_figures_%J.out
#BSUB -e scripts/wexac_logs/meeting_figures_%J.err
#BSUB -J meeting_figs

# =====================================================================
# MEETING FIGURES RENDER — F2 (similarity ladder) + F3 (who-leaks / g0).
#
# Both figures only READ existing results and render PNGs (CPU-bound
# matplotlib; a GPU slot is requested only for env parity — no training,
# no reconstruction, no fine-tuning fires here).
#   F2  <- results/similarity_ladder/ladder_t*.pth   (job 268959)
#   F3  <- results/margin_at_scale/summary.json      (job 272504)
#
# F5 (shared-perturbation) is COMPUTE-GATED and is NOT touched here — its
# compute path refuses to run without --approved and awaits Gal's sign-off.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## F2: similarity ladder (s-vs-d companion + blur d fix) ##########"; date
python -u -m experiments.dataset_sensitivity.fig_f2_similarity_ladder

echo ""; echo "########## F3: who-leaks / g0 (retitle + honest CI/tercile) ##########"; date
python -u -m experiments.dataset_sensitivity.fig_f3_margin

echo ""; echo "=== DONE $(date) ==="
echo "Outputs:"
echo "  figures/similarity_ladder/f2_similarity_ladder.png"
echo "  figures/margin_at_scale/f3_margin_who_leaks.png"
