# Exact LoRA inversion (framework Rev 10, Primitive 3, exact form)

Synthetic FP64 testbed for the "recipe is the forward model" claim: the released LoRA factors
`(A_T, B_T)` are a deterministic function of the private data and of `X = A_0 U`; simulate the
public recipe on candidate data and backprop through the unrolled training loop to solve for
`({w_i}, X)`.  Theory: `framework_rev10.pdf` (Sec. 5); finite-difference baseline numbers:
`results_rev9.pdf` (Sec. 3b).  Both PDFs live on the Mac bundle (`~/Documents/Weizmann/Thesis/`);
sync them into `papers/` when convenient.

Script: `lora_exact_inversion.py` (see its docstring).  Job runner: `scripts/run_exact_inversion_wexac.sh`.
Results: `results/exact_inversion/*.jsonl` (one JSON line per cell/seed with seed, git hash, command line, host),
tensors `*.pth`, write-ups `RESULTS.md` / `NOTES.md` in this directory.

Ground rules (from the task spec)
1. FP64 everywhere.
2. Never change the recipe silently; simulator and release must change identically and the run labelled.
3. Every number is provisional (†) until it comes from the committed script with a recorded seed.
4. Report failures as failures; distinguish "residual not zero" (optimisation) from
   "residual zero, wrong image" (non-identifiability) — the `verdict` field does this.
5. Do not touch the theory documents; disagreements go to `NOTES.md` with evidence.

Steps: 1 validate vs finite differences (SGD) · 2 basin study (near / random / span / cert / spananchor)
· 3 Adam release · 4 (N,k) phase diagram with backprop · 5 write-up.
