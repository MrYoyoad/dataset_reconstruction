# Reconstructing Private Data from LoRA Adapters

**MSc Thesis, Weizmann Institute of Science**
Advisor: [Gal Vardi](https://scholar.google.co.il/citations?user=LVk3xE4AAAAJ&hl=en)

> Extending Haim et al. (NeurIPS 2022) — *"Reconstructing Training Data From Trained Neural Networks"* — to Foundation Models and Parameter-Efficient Fine-Tuning.

**Status — 2026-09-18.** The direction was agreed at the 2026-09-15 supervision meeting. The object of study is
no longer a reconstruction pipeline but a **certificate**: an exact, recipe-free, label-free linear functional of
the released adapter that annihilates every recorded representation. Four fronts are open, in the order agreed:

| # | Front | Where it stands |
|---|-------|-----------------|
| 1 | **Multilayer** — does the certificate survive when a layer's inputs drift? (the advisor's own main question) | runs exist, **§M is UNAUDITED** — see [Methodology](#methodology--how-a-claim-gets-out-of-this-repository) |
| 2 | **Other chart families** — public PCA is answered and it is a *no* | §G, §H, §0.1 |
| 3 | **Improved reconstruction** — from identifiability to pixels | §N2 is the live split |
| 4 | **First text attempt** | not started |

> **Posture — observe, don't conclude.** Every leakage number here bounds **the channel it was measured on**, not
> what a stronger attacker could recover. Every claim in this file carries a `§` pointer into
> [results/CLAIMS_LEDGER.md](results/CLAIMS_LEDGER.md); a sentence with no ledger line does not go out, here or
> anywhere else.

---

## The object

Fine-tuning is a deterministic map from private data to a released pair of factors. For a single adapted layer
trained from `B_0 = 0` by an SGD-class rule, the trajectory **closes** on two batch-sized coefficient matrices:

```
B_t = P_t (A_0 H)^T ,        A_t = A_0 (I + H M_t H^T)
```

where `H = [h_1 … h_N]` are the private representations at the adapted layer and `P_t, M_t` are `N`-by-`N`. Two
consequences, and they are the whole frame:

- The release is a function of the data and of the seed **only through** `X = A_0 U`, `U` an orthonormal basis of
  `row(H)`. The unknown seed contribution is `rank × span-dimension`, not `rank × feature-dimension`. **§A1, §A2**
- What the release determines is `q = rank H` — the dimension of the private span — **never** the number of
  photographs `N`. **§B2**

### The certificate

```
C = Π⊥_{row(B_T)} · A_T       ⟹      C h_i = 0   for every recorded h_i
```

No recipe, no labels, no seed, no shadow models. **§A3.** Exactness is conditional: at `q = N` the residual sits
near `1e-16`; under partial recording the worst representation climbs toward order one while the median stays
small — the loss concentrates in a handful, it is not uniform. **§A4**

What a representation contributes scales with that example's accumulated error, so *what leaks is what the model
had to learn* — a model that already fits its data records nothing. **§B1**

### Three objects, and they are nested

```
{truth}  ⊆  {replay residual = 0}  ⊆  {certificate = 0}
```

The inclusion is **proved**; the content of the measurement is that it is **strict**. On one release the
certificate recovers **0 of 60** starts while **19 of 60** replay starts recover every image at the `1e-2` landing
criterion (18 at a `2.2e-15` worst-image bar, 16 at `2.0e-15`). **§D1, §D2.** The airtight statement is the
negative:

> **Identifiability is not determined by the release and the chart alone.** The route matters. Any claim of the
> form *"cannot identify by **either** route"* is false. **§D3**

### The chart, and why it is the load-bearing component

A chart is the low-dimensional parametrisation the attacker solves in. Constraining the features to a `k = 12`
chart takes the nullity of the residual Jacobian **at the truth** from **232 to 0**: the chart restores
identifiability by removing exactly the directions that trade against the unknown seed. **§N1, §N3.** The decision
line is *partial*, not yes/no — invertible with a known seed, not invertible with a free seed, restored by a
chart. **§N2**

But identifiability is measured with the release's own **oracle** chart. For a *buildable* public chart the
fidelity wall stands: chart error over the landing gate never approaches 1 at any width, under any convention —
**5.9×** at the single most generous reading available anywhere in the grid. **§G, §0.1.** A foundation-model
embedding does not dissolve it; the shared-concept chart is *worse*. **§H**

### Counting, and its one-sidedness

```
margin = min(r, d) − min(q·p, d)          p = adapted positions per image
```

evaluable from architecture and batch size before any release exists. It is **ONE-SIDED**: sound when it says the
channel is closed, **silent when it says open**. **§C1.** It is necessary and **not** sufficient — where the
chart-to-input map is affine, every blend of the private representations is an exact solution at any chart
dimension, so a larger `k` cannot help. **§C2.** The capacity line for the reduced channel is measured **sharp to
one unit of `k`**; for the full factor pair it is a candidate, not a result. **§C4.** The single-layer tangent rank
is proved; the **stacked** rank across layers is open, and it is the quantity the count actually needs. **§C5**

> **The standing prohibition most likely to slip:** *a raw equation count is never evidence of identifiability.*
> Only the rank of the stacked Jacobian on the chart answers it. **§C3** — and §N5 is that rule demonstrated
> rather than asserted: an approver lane derived a **32**-dimensional solution family by counting where the
> measured nullity is **232**. Counting was wrong by a factor of seven; one small job corrected it.

---

## Methodology — how a claim gets out of this repository

This is the part of the project that is deliberate, and it is worth stating on the front page.

**1. The ledger is the gate.** [results/CLAIMS_LEDGER.md](results/CLAIMS_LEDGER.md) holds one row per claim, with
columns `text` · `measured on` (the cell, with `q` and `N` side by side) · `holds under` (conditions, precision
included) · **`NOT shown`** (what a reader would wrongly infer) · `job ids` · `register` · `status`. If the
`NOT shown` column covers what a sentence implies, **the sentence is wrong even when its number is right.**

**2. Registers — where a number came from.**

| register | meaning |
|---|---|
| `read-rows` | read from the result rows |
| `read-function` | read from the code that computed it |
| `derived` | follows from a stated derivation |
| `read-prose` | read from our own write-up — **never counts as a PASS** |

A number sourced only to our own prose is not a measurement. Several corrections in this project's history were
exactly that failure, including one where a claim was quoted from an artifact's own text rather than from the job.

**3. Two independent PASSes before a claim ships.** `SETTLED` = two · `HELD` = awaiting the second · `UNAUDITED` =
none. An audit that exists only in chat messages **does not exist**; it has to be on disk. The multilayer group
§M is listed as UNAUDITED for precisely this reason — the write-up has stood complete and uncertified, and it is
the document answering the question the advisor named as his main one.

**4. Three outcomes, never two.** A cell that tested nothing is distinct from a pass and a fail. Reconstruction
verdicts separate `optimisation failure (residual not zero)` — a **basin/solver** problem — from
`alias (residual zero, wrong image)` — an **information** problem. Merging them into "it didn't work" destroys the
only diagnostic that matters.

**5. Withdrawn claims stay in place with their reason.** §F is a standing list, never deleted — including a claim
that was real mathematics but **vacuous as a defence**, because the remedy belonged to the attacker, who picks the
chart.

**6. Standing rules, enforced in the ledger.**
- A count is never identifiability.
- A landed count without its start budget is not a quantity — yield rises with the budget. **§E6**
- A comparison carries the construction of **both** sides; a target-conditioned figure may not be set against a
  universal one.
- Consistency with the release is **not** evidence of correctness — compare to ground truth only.
- For a published object, the source is the **live** object, not a local copy.

---

## What is measured, and what is not

| | | ledger |
|---|---|---|
| The trajectory closes; the seed enters only through the private span | ✅ settled | §A1, §A2 |
| The certificate annihilates recorded representations, recipe-free | ✅ settled | §A3, §A4 |
| The release gives `q`, never `N` | ✅ settled | §B2 |
| The counting rule, as a one-sided closure test | ✅ settled | §C1–§C5 |
| Route-dependence of identifiability (`certificate 0/60` vs `replay 19/60`) | ✅ settled | §D1–§D3 |
| Residual as a sound witness below the line; `precision 1.000` vs a `0.000` disjoint-release null | ⏳ one PASS | §E1–§E3 |
| Breadth frontier — the 8th image only where precision falls to `0.648` | ⏳ one PASS | §E4 |
| A chart takes the seed-free nullity `232 → 0` | ⏳ one PASS | §N1–§N4 |
| Public-chart fidelity shortfall never approaches 1 | ✅ settled | §G, §0.1 |
| Multilayer survival under drift (the advisor's main question) | ❗ **UNAUDITED — zero PASSes** | §M1–§M3 |
| Depth additivity; the corrected rank law on a real encoder | ⏳ one PASS | §M4–§M6 |
| **Whether a recovered representation is recognisable as an image** | ❗ **unmeasured — the load-bearing gap** | §Open |

---

## Repository

```
yoado/
├── README.md                       <- this file
├── results/CLAIMS_LEDGER.md        <- THE GATE. Every outgoing sentence traces to a row here
├── STATUS.md                       <- landed results, pending tasks, known issues
├── LESSONS_LEARNED.md              <- insights and pitfalls (how the corrections happened)
├── STYLE_GUIDE.md  style_guide/    <- doc/slide/LaTeX/plot rules + visual guardrails
│
├── theory/                         <- one proposed theorem per file: T1..T6
│                                      Statement | Assumptions | Proof | Where each assumption
│                                      is used | Counterexample search | Status | Sanity check.
│                                      A numerical check agreeing NEVER promotes a status to PROVED.
│
├── experiments/
│   ├── exact_inversion/            <- the certificate + replay testbed (FP64, synthetic + MNIST)
│   │   ├── lora_exact_inversion.py <-   solve for ({w_i}, X) by simulating the recipe
│   │   └── train_precision.py      <-   landing gate; at_floor indexes the LANDED image's floor
│   ├── multilayer_cert/            <- does CH = 0 survive at depth (front 1)  [UNAUDITED]
│   ├── oracle_ladder/              <- chart fidelity vs the landing gate (front 2)
│   ├── dataset_sensitivity/        <- the earlier whitened-Jacobian identifiability ruler
│   ├── gradient_bridge/            <- LoRA -> full-gradient decoder (supplies an INITIALISER)
│   └── tests/                      <- pytest suite
│
├── scripts/                        <- WEXAC (LSF bsub) job submission; scripts/deck/ = pptx generator
├── notes/                          <- framework + audits (see Documentation below)
├── results/ figures/ papers/       <- rows (.jsonl/.csv tracked; .pth git-ignored), plots, PDFs
└── dataset_reconstruction/         <- original Haim et al. codebase (separate git)
```

---

## Quick Start

```bash
conda env create -f dataset_reconstruction/environment_macos.yaml && conda activate rec

# the certificate / replay testbed  (FP64 throughout -- never silently downcast)
python experiments/exact_inversion/lora_exact_inversion.py --help
python experiments/exact_inversion/analyze_exact_inversion.py

# multilayer survival + the per-theorem sanity checks with PRE-STATED tolerances
bash scripts/run_multilayer_cert_wexac.sh checks
```

**All serious compute runs on WEXAC** (NVIDIA L40S / A100, CUDA 12.x; `rec` env = PyTorch 2.4.1+cu121). Nothing is
run locally, not even a smoke test.

```bash
cd dataset_reconstruction && ./wexac_connect.sh shell   # interactive GPU shell
bsub -q long-gpu -gpu "num=1" ... bash scripts/run_exact_inversion_wexac.sh step1
```

Ground rules for the inversion tracks: **FP64 everywhere** · **never change the recipe silently** (simulator and
release change identically, and the run is relabelled) · **read `fwd_check` first** — until the simulator
reproduces the release at the ground truth to machine precision, no downstream number means anything.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [results/CLAIMS_LEDGER.md](results/CLAIMS_LEDGER.md) | **Start here.** Every claim, its conditions, and what it does *not* show |
| [STATUS.md](STATUS.md) | Landed results, pending tasks, known issues |
| [LESSONS_LEARNED.md](LESSONS_LEARNED.md) | Insights and pitfalls — the record of how corrections happened |
| [notes/exact_lora_inversion_framework.md](notes/exact_lora_inversion_framework.md) | The framework: the quotient certificate, the capacity line, inversion by simulating the recipe |
| [notes/audit_inverting_a_finetune_2026-09-05.md](notes/audit_inverting_a_finetune_2026-09-05.md) | Full audit of our own claims, and the comparison to SimuDy and the advisor's papers |
| [notes/meeting_summary_2026-09-15.md](notes/meeting_summary_2026-09-15.md) | The supervision meeting that set the current four fronts |
| [theory/README.md](theory/README.md) | Theorem index + notation table |
| [notes/next_experiment_plan.md](notes/next_experiment_plan.md) | Actionable to-do |
| [STYLE_GUIDE.md](STYLE_GUIDE.md) / [style_guide/](style_guide/) | Formatting rules and visual guardrails |

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
