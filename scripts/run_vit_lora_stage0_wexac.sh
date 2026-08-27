#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/vit_lora_stage0_%J.out
#BSUB -e scripts/wexac_logs/vit_lora_stage0_%J.err
#BSUB -J vit_lora_stage0
# ViT LoRA dataset-sensitivity STAGE-0 plumbing gate:
#   gate A: timm+peft importable (fail loudly if not)
#   gate B: whitened metric self-test (architecture-agnostic metric still sound)
#   gate C: vit_lora_sensitivity --stage0 (tiny end-to-end: fit ViT LoRA, extract ΔW, run metric)
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== STAGE-0 START $(date) on $(hostname) ==="
echo "--- gate A: timm + peft importable ---"
python -c "import timm, peft; print(f'timm={timm.__version__} peft={peft.__version__}')" \
  || { echo "FATAL: timm/peft missing in this env"; exit 1; }
echo "--- gate B: whitened metric self-test ---"
python -u -m experiments.dataset_sensitivity.whitened_metric \
  || { echo "FATAL: metric self-test"; exit 1; }
echo "--- gate C: vit_lora_sensitivity tiny end-to-end ---"
python -u -m experiments.dataset_sensitivity.vit_lora_sensitivity --stage0 --device cuda
rc=$?
echo "=== STAGE-0 DONE $(date) (exit $rc) ==="
if [ $rc -ne 0 ]; then echo "FATAL: vit_lora Stage-0 crashed (rc=$rc)."; exit 1; fi
