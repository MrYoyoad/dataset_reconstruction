#!/bin/bash
#BSUB -q long-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/ood_injection_%J.out
#BSUB -e scripts/wexac_logs/ood_injection_%J.err
#BSUB -J ood_injection
# OOD-STYLE DIGIT INJECTION (full): USPS digits (visually MNIST-like, different handwriting
# style/source, 16->28 bilinear) injected into the N=16 MNIST LoRA fine-tune set. Do the OOD
# members leak MORE (paired-per-seed whitened sensitivity, USPS->held-out-USPS swap vs the
# matched MNIST member's MNIST->held-out-MNIST swap), and does the base-model gradient norm
# g0 PREDICT it? Plus the style-only cross swap (USPS -> original MNIST, same digit/slot).
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
echo "########## GATE A: USPS load/download check ##########"
python -u -c "from experiments.dataset_sensitivity.ood_injection import load_usps; ds = load_usps(); print(f'USPS OK: {len(ds)} test images')" \
    || { echo "FATAL: USPS unavailable — pre-download on the LOGIN node (see the error above)"; exit 1; }
echo "########## GATE B: metric self-test + stage0 sanity ##########"
python -u -m experiments.dataset_sensitivity.whitened_metric || { echo FATAL metric; exit 1; }
python -u -m experiments.dataset_sensitivity.ood_injection --stage0 --device cuda || { echo FATAL stage0; exit 1; }
echo "########## FULL: N=16, K=50, n_ood=2, rank 8, T=1000, lr 0.5 ##########"; date
python -u -m experiments.dataset_sensitivity.ood_injection \
    --N 16 --K 50 --n_ood 2 --lr 0.5 --T 1000 --rank 8 --device cuda
echo "=== DONE $(date) ==="
echo "READ: amplification = sens(USPS swap)/sens(matched MNIST swap) > 1 => OOD members leak more;"
echo "      predictor verdict = does the g0 direction (printed BEFORE training) match the measured ratio;"
echo "      style-only swap ~ within-USPS swap => the STYLE itself is what the adapter notices."
