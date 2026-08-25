#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_roundB_%J.out
#BSUB -e scripts/wexac_logs/mc_roundB_%J.err
#BSUB -J mc_roundB

# =====================================================================
# ROUND B — CLEAN q_eff (S≥4·Nk) on the headline cells only. This is where q_eff
# becomes a QUOTABLE number (job 247834's 97 was S=64<r_J=160 undersampled). N=20
# k=8 → Nk=160 → S=640. Value-only Σ_seed unroll (create_graph=False, now wired)
# makes S=640 feasible on long-gpu. Both bases at the PLATEAU-T from Round 0 (set
# TPLAT below once Round 0 reads out — the r_J plateau, ~200+). Adequacy gate:
# only quote q_eff once eff_rank(Σ_seed) ≳ r_J AND q_eff stable across {S,2S}.
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

TPLAT=200   # plateau-T from Round 0 (r_J saturates by T=200 at k=8, lr=0.1); adjust if needed

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname)  TPLAT=$TPLAT ==="

echo ""; echo "########## STAGE 0: AD gate (binary + CE) ##########"; date
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: AD gate failed. Aborting."; exit 1; fi

# S=640 (=4·Nk) AND S=1280 (2S) for the stability check the adequacy gate requires.
echo ""; echo "########## Round B: clean q_eff, headline cells (both bases), S={640,1280} ##########"; date
for DS in mnist fashion; do for NC in 2 10; do for S in 640 1280; do
  echo ""; echo "-- j1 $DS nc=$NC N=20 k=8 T=$TPLAT S=$S (clean q_eff) --"; date
  python -u -m experiments.jacobian_spectrum --j1 --dataset $DS --activation gelu \
      --num_classes $NC --N 20 --k 8 --T $TPLAT --rank 8 --tangent qr \
      --S_list $S --shrink_list 0.01 --eps_list 0.1 0.3 1.0 3.0 10.0 --save \
      --tag roundB_${DS}_nc${NC}_S${S} --device cuda
done; done; done

echo ""; echo "=== DONE $(date) ==="
