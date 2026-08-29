#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0 && hname!='lgn28' && hname!='hgn46' && hname!='hgn45' && hname!='lgn13']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/fullft_valley_scaleup_%J.out
#BSUB -e scripts/wexac_logs/fullft_valley_scaleup_%J.err
#BSUB -J fullft_valley_scaleup
# VALLEY SCALE-UP: firm the d*_full≈d*_LoRA headline — dial arms E_b0(=LoRA arm A)/C/D at n_targets=6
# (was 2). Reuses the committed calibration.json (shrunk eps_D). Bracket guard: never run D on a
# failing eps.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
if [ ! -f results/fullft_valley/calibration.json ]; then echo "FATAL: calibration.json missing"; exit 1; fi
DPASS=$(python -c "import json;g=json.load(open('results/fullft_valley/calibration.json'))['bracket']['D']['gate'];print(int(bool(g.get('passed')) and g.get('measurable',True)))")
if [ "$DPASS" != "1" ]; then echo "FATAL: arm-D bracket not measurable — abort"; exit 1; fi
echo "=== START $(date) — dial scale-up n_targets=6 ==="
for A in E_b0 C D; do
  echo ""; echo "########## ARM $A (n_targets=6) ##########"; date
  python -u -m experiments.dataset_sensitivity.fullft_valley --arm $A \
      --K 50 --n_targets 6 --T 1000 --rank 8 --N 16 --device cuda --tag _n6
  if [ $? -ne 0 ]; then echo "FATAL: arm $A failed"; exit 1; fi
done
echo ""; echo "=== DONE $(date) — read A(E_b0)/C/D d* at n=6; headline d*_full vs d*_LoRA ==="
