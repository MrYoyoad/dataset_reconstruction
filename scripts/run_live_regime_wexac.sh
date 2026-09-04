#!/bin/bash
# TEST 2: the certificate in the LIVE regime of the counting rule, on a real pretrained ResNet-18.
# Scoring locked by b9; design scoped by c9 (activation-space, member side unscored, freezing is the hypothesis,
# SGD gate); controls from 7e (population of single-member releases, shared non-member pool disclosed, and a
# per-draw never-trained null of the SAME image under the previous draw's release).
# The Adam arm is a NEGATIVE control and must come back void at every draw: under Adam rank B_T = r and no
# certificate exists, so a non-void Adam row would mean the harness is measuring something else.
set +u
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
OUT=results/exact_inversion; mkdir -p $OUT
echo "=== START live regime $(date) on $(hostname) git=$(git rev-parse --short HEAD) ==="
python -u -m experiments.exact_inversion.live_regime --model resnet18 --stage 4 --r 64 \
    --draws 20 --nonmembers 1000 --T 200 --lr 0.05 --optimiser sgd --seed 1 \
    --out $OUT/step116_live_${LSB_JOBID}.jsonl
echo "=== NEGATIVE CONTROL: Adam, must be void at every draw ==="
python -u -m experiments.exact_inversion.live_regime --model resnet18 --stage 4 --r 64 \
    --draws 6 --nonmembers 1000 --T 200 --lr 0.05 --optimiser adam --seed 1 \
    --out $OUT/step116_live_${LSB_JOBID}.jsonl
echo "=== DONE $(date) ==="
