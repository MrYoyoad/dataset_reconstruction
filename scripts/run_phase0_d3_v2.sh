#!/bin/bash
# Submit 7 D3 configs as individual bsub jobs.
# Uses --run_tag for collision-proof filenames; freq/lpips wiring fixed.
set -e

cd /home/projects/galvardi/yoado
mkdir -p wexac_logs

# D3 configs: (idx, run_tag, freq_weight, lpips_weight)
configs=(
  "0 freq1e-3        1e-3 0"
  "1 freq1e-2        1e-2 0"
  "2 freq1e-1        1e-1 0"
  "3 lpips1e-3       0    1e-3"
  "4 lpips1e-2       0    1e-2"
  "5 freq1e-2_lpips1e-3 1e-2 1e-3"
  "6 freq1e-2_lpips1e-2 1e-2 1e-2"
)

for cfg in "${configs[@]}"; do
  read idx tag fw lw <<< "$cfg"

  bsub <<EOF
#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 6:00
#BSUB -o wexac_logs/phase0_d3v2_${idx}_${tag}_%J.out
#BSUB -e wexac_logs/phase0_d3v2_${idx}_${tag}_%J.err
#BSUB -J phase0_d3v2_${idx}

set -e
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado

echo "=== D3 v2 Config ${idx}: ${tag} (freq=${fw}, lpips=${lw}) ==="
echo "Date: \$(date)"
echo "Host: \$(hostname)"

python -u -m experiments.phase0_vit_inversion \
    --mode full \
    --optimizer signAdam \
    --tv_weight 1e-1 \
    --tv_norm l2 \
    --lr 0.05 \
    --n_iters 30000 \
    --n_restarts 2 \
    --freq_weight ${fw} \
    --lpips_weight ${lw} \
    --device cuda \
    --seed 42 \
    --run_tag d3v2_idx${idx}_${tag}

echo "=== D3 v2 Config ${idx} Complete: \$(date) ==="
EOF

done

echo "All 7 D3 v2 jobs submitted."
