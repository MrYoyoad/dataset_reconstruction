#!/bin/bash
# After all 7 D3 v2 jobs finish: find winner by SSIM, submit face job using
# winning config (freq_weight, lpips_weight) on top of D3 baseline.
set -e
cd /home/projects/galvardi/yoado

echo "=== D3 v2 analysis ==="
echo "Date: $(date)"

# Collect SSIM from each D3 v2 log
best_ssim="0.0"
best_idx=""
best_tag=""
best_fw="0"
best_lw="0"

# idx, tag, fw, lw must match scripts/run_phase0_d3_v2.sh
configs=(
  "0 freq1e-3            1e-3 0"
  "1 freq1e-2            1e-2 0"
  "2 freq1e-1            1e-1 0"
  "3 lpips1e-3           0    1e-3"
  "4 lpips1e-2           0    1e-2"
  "5 freq1e-2_lpips1e-3  1e-2 1e-3"
  "6 freq1e-2_lpips1e-2  1e-2 1e-2"
)

echo ""
echo "idx | tag                  | freq | lpips | SSIM   | cos_sim | status"
echo "----+----------------------+------+-------+--------+---------+--------"

for cfg in "${configs[@]}"; do
  read idx tag fw lw <<< "$cfg"
  # Find the most recent log for this idx (by job id)
  log=$(ls -t wexac_logs/phase0_d3v2_${idx}_${tag}_*.out 2>/dev/null | head -1)
  if [ -z "$log" ]; then
    printf "%3s | %-20s | %4s | %5s | (no log)\n" "$idx" "$tag" "$fw" "$lw"
    continue
  fi
  ssim=$(grep -oE "SSIM=0\.[0-9]+" "$log" | head -1 | grep -oE "0\.[0-9]+")
  cos=$(grep -oE "cos_sim=0\.[0-9]+" "$log" | tail -1 | grep -oE "0\.[0-9]+")
  done_marker=$(grep -c "Successfully completed\|Complete: " "$log")
  status="DONE"
  if [ -z "$ssim" ]; then status="RUNNING/FAILED"; ssim="-"; cos="-"; fi
  printf "%3s | %-20s | %4s | %5s | %s | %s | %s\n" \
    "$idx" "$tag" "$fw" "$lw" "$ssim" "$cos" "$status"
  # Track best
  if [ -n "$ssim" ] && [ "$ssim" != "-" ]; then
    if awk "BEGIN{exit !($ssim > $best_ssim)}"; then
      best_ssim=$ssim; best_idx=$idx; best_tag=$tag; best_fw=$fw; best_lw=$lw
    fi
  fi
done

echo ""
echo "WINNER: idx=$best_idx ($best_tag), SSIM=$best_ssim"
echo "        freq_weight=$best_fw, lpips_weight=$best_lw"

if [ -z "$best_idx" ]; then
  echo "ERROR: no D3 v2 winner found — check job statuses with bjobs."
  exit 1
fi

# Submit face job with winning config
echo ""
echo "=== Submitting face job with winning D3 config ==="

bsub <<EOF
#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=32768] select[ngpus>0]"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -o wexac_logs/phase0_face_d3winner_${best_tag}_%J.out
#BSUB -e wexac_logs/phase0_face_d3winner_${best_tag}_%J.err
#BSUB -J phase0_face_d3w_${best_idx}

set -e
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado

echo "=== Face reconstruction with D3 winner: idx=${best_idx} (${best_tag}) ==="
echo "Date: \$(date)"
echo "Host: \$(hostname)"
echo "Config: signAdam, tv=1e-1, lr=0.05, 30K iters, 8 restarts"
echo "        freq_weight=${best_fw}, lpips_weight=${best_lw}"

python -u -m experiments.phase0_vit_inversion \
    --mode full \
    --image_path data/faces/face1.jpg \
    --optimizer signAdam \
    --tv_weight 1e-1 \
    --tv_norm l2 \
    --lr 0.05 \
    --n_iters 30000 \
    --n_restarts 8 \
    --freq_weight ${best_fw} \
    --lpips_weight ${best_lw} \
    --device cuda \
    --seed 42 \
    --run_tag face_d3winner_${best_tag}

echo "=== Face D3-winner reconstruction complete: \$(date) ==="
EOF

echo "Submitted face job."
