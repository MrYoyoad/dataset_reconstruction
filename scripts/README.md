# scripts/

28 WEXAC LSF job submission scripts for GPU experiments.

---

## Script Index

### Sprint 1: Proof of Concept

| Script | Experiment | Queue | Status |
|--------|-----------|-------|--------|
| `run_exp_a_gpu.sh` | Experiment A (KKT) | interactive-gpu | Complete |
| `run_exp_b_gpu.sh` | Experiment B (NTK) | interactive-gpu | Complete |
| `run_rank_sweep.sh` | Rank sweep (1-64) | short-gpu | Complete |

### Sprint 2: Free Coefficients & Ablations

| Script | Experiment | Queue | Status |
|--------|-----------|-------|--------|
| `run_sprint2_track0.sh` | Activation ablation | short-gpu | Complete |
| `run_sprint2_track2.sh` | Free-coefficient extraction | short-gpu | Complete |
| `run_activation_ablation_wexac.sh` | Extended activation sweep | short-gpu | Complete |
| `run_lora_free_coeff_wexac.sh` | LoRA free-c rank sweep | short-gpu | Complete |
| `run_ntk_sweep_wexac.sh` | NTK step sweep | short-gpu | Complete |

### Sprint 2b: Multi-Step & Scaling

| Script | Experiment | Queue | Status |
|--------|-----------|-------|--------|
| `run_sprint2b_wexac.sh` | Phases 0-4 (activation, multi-step, restarts) | long-gpu | Complete |
| `run_sprint2b_phase56_wexac.sh` | Phases 5-6 (LR magnitude, warm-start) | long-gpu | Complete |
| `run_sprint2b_phase7_wexac.sh` | Phase 7 (progressive warm-start) | long-gpu | Complete |

### Sprint 2c: KKT & NTK Ablations

| Script | Experiment | Queue | Status |
|--------|-----------|-------|--------|
| `run_sprint2c_track_a_wexac.sh` | Track A: KKT N-sweep (short) | short-gpu | Complete |
| `run_sprint2c_track_a_split.sh` | Track A: KKT N-sweep (long) | long-gpu | CLOSED |
| `run_sprint2c_track_b_wexac.sh` | Track B: combined NTK ablations | long-gpu | Complete |
| `run_sprint2c_b1_wexac.sh` | B1: LR scheduling + warm-start | long-gpu | Complete |
| `run_sprint2c_b3a_b4_wexac.sh` | B3a: optimizer x activation, B4: N-sweep | long-gpu | Complete |
| `run_sprint2c_b3b_wexac.sh` | B3b: optimizer x activation (extended) | long-gpu | Complete |
| `run_sprint2c_b4_wexac.sh` | B4: N-sweep (NTK) | long-gpu | Complete |
| `run_sprint2c_b4_phase56_wexac.sh` | B4: Phases 5-6 | long-gpu | Complete |
| `run_sprint2c_b5_b6_b7_wexac.sh` | B5-B7: additional ablations | long-gpu | Complete |
| `run_sprint2c_b8_wexac.sh` | B8: fine-tuning optimizer | long-gpu | Complete |

### Diagnostics & Validation

| Script | Experiment | Queue | Status |
|--------|-----------|-------|--------|
| `run_diagnostic_wexac.sh` | NTK regime diagnostics | short-gpu | Complete |
| `run_seed_fix_ablation_wexac.sh` | Seed variance study | short-gpu | Complete |
| `run_b6_extra_seeds.sh` | Extra seeds for B6 track | short-gpu | Complete |
| `run_t_sweep_examples_wexac.sh` | Generate T-sweep figures | short-gpu | Complete |
| `run_overnight_sprint2_cleanup.sh` | Multi-seed validation (50+30 seeds) | long-gpu | Complete |

### Phase 0: ViT Gradient Inversion

| Script | Experiment | Queue | Status |
|--------|-----------|-------|--------|
| `run_phase0_wexac.sh` | Phase 0 (original, buggy) | short-gpu | Failed |
| `run_phase0_fixed_wexac.sh` | Phase 0 (bug-fixed) | long-gpu | Pending |

---

## Naming Convention

- `run_<experiment>_wexac.sh` — batch submission via `bsub`
- `run_<experiment>.sh` — interactive or simple scripts
- Sprint-specific: `run_sprint2c_b3a_b4_wexac.sh` = Sprint 2c, Tracks B3a and B4

---

## How to Submit

```bash
ssh wexac                           # Requires Weizmann VPN
cd /home/projects/galvardi/yoado
bsub < scripts/run_<name>.sh       # Submit to LSF queue
bjobs                               # Monitor
bpeek <job_id>                      # Tail output
```

---

## LSF Directives Reference

All scripts begin with `#BSUB` directives:

```bash
#BSUB -q long-gpu                           # Queue (4h/24h/168h)
#BSUB -R "rusage[mem=16384] select[ngpus>0]" # 16GB RAM + GPU
#BSUB -gpu "num=1"                           # 1 GPU
#BSUB -W 24:00                               # Wall time
#BSUB -o wexac_logs/name_%J.out              # stdout (%J=job ID)
#BSUB -e wexac_logs/name_%J.err              # stderr
#BSUB -J job_name                            # Job name
```

---

## Standard Script Preamble

All scripts share this pattern:

```bash
#!/bin/bash
set -e
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
mkdir -p results figures wexac_logs

python -u -m experiments.<module> --device cuda [args...]
```

---

## Job Logs

Output goes to `scripts/wexac_logs/` (and sometimes root `wexac_logs/`). Named `<experiment>_<jobid>.out/.err`.

---

## Creating a New Script

1. Copy an existing script (e.g., `run_exp_b_gpu.sh`)
2. Update `#BSUB` directives (queue, wall time, job name, log paths)
3. Update the `python -u -m experiments.<module>` command
4. Add entry to this README
5. Submit: `bsub < scripts/run_<name>.sh`
