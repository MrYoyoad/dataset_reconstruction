# Reconstructing Private Data from LoRA Adapters

**MSc thesis · Weizmann Institute of Science · Advisor: [Gal Vardi](https://scholar.google.co.il/citations?user=LVk3xE4AAAAJ&hl=en)**

![Python](https://img.shields.io/badge/python-3.8-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.4.1-ee4c2c) ![License](https://img.shields.io/badge/license-research--only-lightgrey)

LoRA adapters are published openly on HuggingFace and CivitAI. This project asks what those released weights still
contain about the private images they were fine-tuned on — and shows that, for a single adapted layer, the answer
is an exact algebraic object you can compute from the adapter alone.

Extends [Haim et al., NeurIPS 2022](https://arxiv.org/abs/2206.07758) — *Reconstructing Training Data From Trained
Neural Networks* — to foundation models and parameter-efficient fine-tuning.

---

## Overview

Fine-tuning from `B₀ = 0` with an SGD-class optimizer makes the released adapter a deterministic function of the
private data. From the released factors `(A_T, B_T)` alone we build a **certificate**:

```
C = Π⊥_row(B_T) · A_T        satisfies        C · h = 0
```

for every private representation `h` the adapter recorded. No knowledge of the training recipe, no labels, no
access to the random seed, no shadow models. The private images are then searched for inside a low-dimensional
public image family (a **chart**), where the certificate turns into a small system of equations.

Two properties of the certificate:

- **It is exact, not approximate.** Where every representation is recorded, the residual sits at machine precision.
- **It says what the adapter recorded, not how many photos there were.** The batch size is not recoverable.

---

## Results

Private images are recovered **exactly** — to 15 decimal places, from the adapter alone, starting from random
initialisations:

| | Measured |
|---|---|
| Private images recovered, exactly | **8 of 8**, on 16 of 60 random starts |
| Recovery error on those starts | `2e-15` — machine precision (19 of 60 clear a `1e-2` bar) |
| Certificate residual `C·h` | machine precision, no recipe or labels used |
| Telling a true recovery from a false one | precision **1.000** against a 0.000 null (base rate 0.324) |
| Ambiguity removed by a 12-dimensional chart | **232 dimensions → 0** |
| Depths at which the theory is exact | nullity **80 = predicted 80**, depths 2–8 |

Three findings behind those numbers:

- **The adapter can be inverted without knowing how it was trained.** The certificate `C·h = 0` is built from the
  released factors and nothing else — no training recipe, no labels, no random seed, no shadow models. Attacks in
  this space normally assume at least one of these.
- **An attacker can verify their own answers.** Ranking candidates blind by final objective separates true
  recoveries from false ones perfectly on the cell we measured, so recovery does not depend on already knowing the
  private data.
- **What leaks is what the model had to learn.** Each example is recorded at the scale of the error it still
  carried, which predicts *which* images leak from the public model alone — and makes the amount of leakage a
  property the defender can reason about.

Full numbers, conditions and job ids: **[results/CLAIMS_LEDGER.md](results/CLAIMS_LEDGER.md)**.
Current position and open problems: **[notes/research_overview_2026-09-17.md](notes/research_overview_2026-09-17.md)**.

**Next:** multilayer adapters · other chart families · representations to pixels · a first attempt at text.

---

## Installation

```bash
git clone https://github.com/MrYoyoad/dataset_reconstruction.git yoado && cd yoado
conda env create -f dataset_reconstruction/environment_macos.yaml
conda activate rec
```

Python 3.8 · PyTorch 2.4.1 · timm · peft · kornia. Experiments run on the WEXAC cluster (NVIDIA L40S / A100);
`dataset_reconstruction/settings.py` sets the dataset, results, and model paths.

## Usage

```bash
# certificate + replay inversion  (FP64 throughout)
python experiments/exact_inversion/lora_exact_inversion.py --release sgd --init random --solver lm
python experiments/exact_inversion/analyze_exact_inversion.py          # tables + figures

# multilayer: does the certificate survive when a layer's inputs drift?
bash scripts/run_multilayer_cert_wexac.sh checks

# original Haim et al. pipeline
cd dataset_reconstruction
python Main.py --run_mode=train --problem=cifar10_vehicles_animals ...
```

Batch jobs go through LSF:

```bash
bsub -q long-gpu -gpu "num=1" bash scripts/run_exact_inversion_wexac.sh step1
```

---

## Repository

```
├── experiments/
│   ├── exact_inversion/     certificate + replay inversion (the main track)
│   ├── multilayer_cert/     does the certificate survive at depth
│   ├── oracle_ladder/       how good does a chart have to be
│   ├── dataset_sensitivity/ the earlier whitened-Jacobian identifiability ruler
│   └── gradient_bridge/     LoRA -> full-gradient decoder
├── theory/                  one proposed theorem per file, with its proof status
├── notes/                   research overview, technical record, meeting records
├── results/                 metrics (.jsonl/.csv tracked; .pth git-ignored)
├── figures/  scripts/  papers/
└── dataset_reconstruction/  original Haim et al. codebase (separate git)
```

---

## How results are reported

Research code is easy to fool, so this repo keeps a **[claims ledger](results/CLAIMS_LEDGER.md)**: one row per
claim, with the cell it was measured on, the conditions it holds under, and a column for *what it does not show*.
A claim needs two independent checks before it leaves the repository, a number read from our own write-up never
counts as one of them, and withdrawn claims stay in the file with the reason instead of being deleted. Failures
are reported as failures — an optimizer that did not converge and an answer that converged to the wrong image are
recorded as different outcomes, because they are.

## Documentation

| | |
|---|---|
| [notes/research_overview_2026-09-17.md](notes/research_overview_2026-09-17.md) | Start here — the current position and what would change it |
| [notes/technical_record_2026-09-17.md](notes/technical_record_2026-09-17.md) | Definitions, assumptions, proved statements, open problems |
| [results/CLAIMS_LEDGER.md](results/CLAIMS_LEDGER.md) | Every claim with its conditions and job ids |
| [STATUS.md](STATUS.md) | Latest results and pending work |
| [LESSONS_LEARNED.md](LESSONS_LEARNED.md) | Pitfalls and design decisions |
| [theory/README.md](theory/README.md) | Theorem index and notation |

---

## Citation

This thesis builds on:

```bib
@inproceedings{haim2022reconstructing,
  author = {Haim, Niv and Vardi, Gal and Yehudai, Gilad and Shamir, Ohad and Irani, Michal},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {Reconstructing Training Data From Trained Neural Networks},
  volume = {35},
  pages = {22911--22924},
  year = {2022}
}
```

Key external anchors: Jang et al. (ICML 2024, LoRA NTK, r ≳ √N); Putterman/Lim et al. (ICLR 2025, Learning on LoRAs); Tian et al. (ICLR 2025, SimuDy).

---

## License

Research use only. Based on the [Haim et al. implementation](https://github.com/nivha/dataset_reconstruction).
