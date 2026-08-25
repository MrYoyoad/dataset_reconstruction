#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_roundC_%J.out
#BSUB -e scripts/wexac_logs/mc_roundC_%J.err
#BSUB -J mc_roundC

# =====================================================================
# ROUND C — HEAD-WIDTH scaling law (the mechanistic test of "driver = CE head width").
# Fixed 10-class base θ0, classes_present=2, CE over first K′ logits (--subhead_k),
# K′∈{2,3,5,10}, at k=16 HEADROOM (so r_J is NOT pinned at the domain ceiling — Round 0
# showed k=8 saturates). S=1280 (=4·Nk at Nk=320) for clean q_eff; value-only Σ_seed
# unroll makes it feasible. Reads: DOMAIN-limited (r_J tracks Nk regardless of K′) vs
# MEASUREMENT-limited (r_J plateaus below Nk at a K′-dependent value = head-width drives it).
# NEGATIVE CONTROL: sweep classes_present {2,5,10} at the SAME k=16 — r_J must stay FLAT
# (moves under K′, flat under classes_present ⇒ driver is head-width, not class coverage).
# Overtraining T-sweep was front-loaded into Round 0 (done there). (hgn46/lgn28 excluded.)
# =====================================================================
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"

python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="

echo ""; echo "########## STAGE 0: AD gate (binary + CE) ##########"; date
python -u -m experiments.jacobian_spectrum --smoke --device cuda
if [ $? -ne 0 ]; then echo "FATAL: AD gate failed. Aborting."; exit 1; fi

echo ""; echo "########## Round C: HEAD-WIDTH sweep K' (fixed 10-class base, classes_present=2, k=16) ##########"; date
for DS in mnist fashion; do for KP in 2 3 5 10; do
  echo ""; echo "-- j1 $DS num_classes=10 classes_present=2 subhead_k=$KP N=20 k=16 (head-width) --"; date
  python -u -m experiments.jacobian_spectrum --j1 --dataset $DS --activation gelu \
      --num_classes 10 --classes_present 2 --subhead_k $KP \
      --N 20 --k 16 --T 200 --rank 8 --tangent qr \
      --S_list 1280 --shrink_list 0.01 --eps_list 0.1 0.3 1.0 3.0 10.0 --save \
      --tag roundC_headwidth_${DS}_Kp${KP} --device cuda
done; done

echo ""; echo "########## Round C: NEGATIVE CONTROL — classes_present sweep at k=16 (r_J must stay FLAT) ##########"; date
for DS in mnist fashion; do for KE in 2 5 10; do
  echo ""; echo "-- j1 $DS num_classes=10 classes_present=$KE (full 10-head) N=20 k=16 (neg control) --"; date
  python -u -m experiments.jacobian_spectrum --j1 --dataset $DS --activation gelu \
      --num_classes 10 --classes_present $KE \
      --N 20 --k 16 --T 200 --rank 8 --tangent qr \
      --S_list 16 --shrink_list 0.01 --eps_list 0.1 1.0 10.0 --save \
      --tag roundC_negctrl_${DS}_Ke${KE} --device cuda
done; done

echo ""; echo "=== DONE $(date) ==="
