#!/bin/bash
# =====================================================================
# Chart program (plan 2026-09-18 P7 / P8, audit corrections 9-10). Analysis on SAVED artifacts; no new training.
#
#   p8 <mnist|cifar> [b-arms...]   experiments/bootstrap_chart/handover_jacobian.py on the bootstrap tensors
#                                  (jobs 355987 MNIST, 355988 CIFAR) -> results/bootstrap_chart/handover_<jobid>.jsonl
#   p7 [charts...]                 experiments/cifar/chart_conditioning.py on the CIFAR CNN motorcycle head release
#                                  -> results/cifar/chart_conditioning_<jobid>.jsonl
#
# Submit (the CNN cells need a GPU: one LM oracle start on the CNN is ~40 s on a GPU, minutes on CPU; the MNIST MLP is CPU):
#   bsub -q short     -n 4 -R "rusage[mem=16384]" -W 2:00 -J chart_p8_mnist -o scripts/wexac_logs/chartprog_%J.out -e scripts/wexac_logs/chartprog_%J.err \
#        bash scripts/run_chart_program_wexac.sh p8 mnist
#   bsub -q short-gpu -gpu "num=1" -R "rusage[mem=24576]" -W 3:00 -J chart_p8_cifar -o ... bash scripts/run_chart_program_wexac.sh p8 cifar recovery
#   bsub -q short-gpu -gpu "num=1" -R "rusage[mem=24576]" -W 3:00 -J chart_p7_pca   -o ... bash scripts/run_chart_program_wexac.sh p7 pca pca_perclass
#   bsub -q short-gpu -gpu "num=1" -R "rusage[mem=24576]" -W 3:00 -J chart_p7_ae    -o ... bash scripts/run_chart_program_wexac.sh p7 ae
#   bsub -q short-gpu -gpu "num=1" -R "rusage[mem=24576]" -W 3:00 -J chart_p7_local -o ... bash scripts/run_chart_program_wexac.sh p7 local
# =====================================================================
set +u                                   # conda activate breaks under set -u in this env (LESSONS_LEARNED)
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONUNBUFFERED=1

STAGE="${1:-p8}"; shift
mkdir -p results/bootstrap_chart results/cifar
python -m py_compile experiments/bootstrap_chart/handover_jacobian.py experiments/cifar/chart_conditioning.py || exit 3

case "$STAGE" in
  p8)
    REL="${1:-mnist}"; shift
    if [ "$REL" = "mnist" ]; then JOB=355987; else JOB=355988; fi
    ARMS="${@:-recovery random_anchor oracle_anchor}"
    python -u -m experiments.bootstrap_chart.handover_jacobian --release "$REL" --job "$JOB" --b-arms $ARMS
    ;;
  p7)
    CHARTS="${@:-pca ae local pca_perclass}"
    python -u -m experiments.cifar.chart_conditioning --charts $CHARTS --ks 16 32 48
    ;;
  *)
    echo "usage: $0 {p8 <mnist|cifar> [b-arms]|p7 [charts]}" ; exit 2 ;;
esac
