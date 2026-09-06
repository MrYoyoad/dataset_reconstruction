#!/usr/bin/env python3
"""Bundle the 2026-09-06 results (record strength + the CIFAR certificate study) into one zip with figures.

  python -m experiments.cifar.make_bundle [--out /home/projects/galvardi/yoado/results_2026-09-06.zip]

Contents: the two RESULT.md write-ups, every figure they cite, the generated tables, the per-cell JSON rows (small),
the scripts that produced them, and an INDEX.md that says what each file is and which job produced it.
"""
import argparse, glob, json, os, zipfile

FIG_DIRS = ["figures/record_strength", "figures/cifar_charts", "figures/cifar_newclass", "figures/cifar_study", "figures/ntk_vs_cert"]
FIG_FILES = ["experiments/cifar/k32/cifar_certificate.png", "experiments/cifar/k48/cifar_certificate.png",
             "experiments/cifar/k32_onchart/cifar_certificate_onchart.png", "experiments/cifar/k48_onchart/cifar_certificate_onchart.png",
             "figures/exact_inversion/letters_recovery_k32_760909.png", "figures/exact_inversion/certificate_recovery_k6_706721.png"]
DOCS = ["experiments/record_strength/RESULT.md", "experiments/cifar/RESULT.md"]
EXTRA_DOCS = [("notes/ntk_vs_certificate_comparison.md", "EQUIVALENCE_linearised_vs_certificate.md")]
SCRIPTS = ["cifar_certificate.py", "experiments/cifar/cifar_certificate_onchart.py", "experiments/cifar/cifar_charts.py",
           "experiments/cifar/cifar_trained_newclass.py", "experiments/cifar/cifar_newclass.py", "experiments/cifar/replot_grids.py",
           "experiments/record_strength/record_strength.py", "experiments/record_strength/sigma_decomposition.py",
           "experiments/cifar/submit_cifar_certificate.sh", "experiments/cifar/submit_cifar_charts.sh",
           "experiments/cifar/submit_cifar_newclass.sh", "experiments/record_strength/submit_record_strength.sh"]
TABLES = ["figures/record_strength/table.md", "figures/cifar_charts/table.md", "results/record_strength/sigma_decomposition.json"]

INDEX = """# Results bundle — 2026-09-06

Two studies, both run on the WEXAC GPU cluster. Every number here comes from a job whose id is given; nothing is
carried over from an earlier write-up without being re-measured.

## 1. Record strength vs recovery  (`record_strength/RESULT.md`)

Does an example's recovery from the certificate search depend on how strongly the adapter recorded it, or is
recording a threshold? Answer: a threshold in exact arithmetic, over eleven orders of magnitude of record strength;
a graded effect appears only when the release itself is trained in reduced precision.

Reproduction gate: every landing count reproduces the original jobs (760909, 764976, 706721) exactly.

Figures: `figures/record_strength/`
  recovery_error_vs_sigma.png    the threshold — two bands, nothing between
  basin_vs_sigma.png             how many random starts find each example
  sigma_vs_u.png                 record strength against the accumulated softmax error
  letters_precision_overlay.png  the same eight letters from releases trained in fp64 / fp32 / bf16 / fp16
Tables: `tables/record_strength_table.md` (per example), `tables/sigma_decomposition.json`
Jobs: 255095 (letters), 255098 (confident digits), 255107 (plots), 279342 (decomposition + replot)

## 2. The CIFAR certificate study  (`cifar/RESULT.md`)

The supplied CIFAR replica does not recover its private images. The cause is structural: with the adapter on the
pixel layer the certificate is linear in the image, so every blend of the private images is an exact zero and the
search returns a blend. Moving the adapter behind a nonlinearity fixes it, and then random starts recover every
private image, with the attacker able to tell which starts succeeded from the residual alone.

Figures:
  cifar_replica/            the replica as supplied (jobs 252897, 252898) and its on-chart control (257893, 257895)
  cifar_charts/             one panel per cell of the layer x chart x solver x privates grid, plus table.md
  cifar_newclass/           weird added-on classes on fully trained backbones (MLP, CNN, over-trained)
  mnist_reference/          the MNIST cells this replicates: letters (760909) and the k=6 digits (706721)

## 3. THE COMPARISON TO THE NTK-REGIME ROUTE  (`EQUIVALENCE_linearised_vs_certificate.md`)

**If you are looking for the comparison to the NTK regime, this is it.** It is filed under "equivalence" rather than
"comparison" because the result changed: the two routes turn out to have the same zero set, so a comparison of which
recovers more would have been measuring solvers rather than information. The rename was our decision and the
document opens by withdrawing the earlier comparison framing.

## 3b. What that document says

`EQUIVALENCE_linearised_vs_certificate.md`. This is NOT a comparison of which route recovers more, and an earlier
draft that was has been withdrawn. Two results: the LoRA-aware linearised model is never mis-specified, at any step
count, as a corollary of the closure lemma; and the two routes have the SAME ZERO SET, the certificate being the
per-candidate form of the condition and the linearised representer its joint form plus an independence clause. So
the equations cannot be the difference between them, and every measured gap is solver and search arity. The document
also records the two solver corrections found in review, one of which was a handicap that favoured the certificate.

## 4. THE TWO-TYPES-OF-IMAGES FIGURES

Three different pairings could be meant by this, so all three are named here and each is one search away.

**(i) Two kinds of image in ONE private batch** — the mixed-class cells, which is what was asked for in the session:
CIFAR keyboards *and* apples fine-tuned together with a new output row each, and MNIST letters *a* and *t* together.
Files: `figures/ntk_vs_cert/cifar_keyboard+apple_*.png` and `figures/ntk_vs_cert/mnist_letter_a+letter_t_*.png`.
What the pair shows: the certificate recovers 8 of 8 from a heterogeneous private batch, so mixing two unrelated
added classes does not degrade it.

**(ii) The raw private image against its chart projection** — every panel in `figures/cifar_newclass/` and
`figures/cifar_charts/` has three rows: the raw private image, its chart projection, and what the attack returned.
What the pair shows: the attack returns the chart's projection of the private image, not the image itself, which is
the fidelity caveat. It is also why identification rests on image error rather than on a similarity score.

**(iii) Held-out classes against a different corpus** — `figures/cifar_newclass/mlp_keyboard_*`,
`mlp_skyscraper_*`, `mlp_mushroom_*` are held-out CIFAR-100 classes; `mlp_flowers102_*` are photographs from a
different dataset entirely. What the pair shows: the recovery does not depend on the private images coming from the
same corpus as anything the model saw.

## 5. THE STUDY FIGURES  (`figures/cifar_study/`)

  F1_blend_degeneracy.png     the certificate residual along the affine hull of the private images: FLAT at machine
                              precision for an affine composition, which is the lemma with nothing left to argue;
                              a bowl touching the floor only at the truths once a nonlinearity is in the way
  F2_coverage_vs_precision.png  the pixel-layer cell fails at coverage, not precision
  F3_equivalence.png          the certificate residual against the linearised fit's floor: they vanish together
  F4_verdict_axis.png         alias against search failure on one axis, with the project's existing threshold
  F5_solver_handicap.png      a handicap in our OWN comparison, found and removed
  F6_isolation_test.png       the isolation test, read one-sided: full rank certifies, deficiency certifies nothing

## 6. Scripts

`scripts/` holds every script used, including the job submitters. `cifar_certificate.py` is the file as supplied,
unmodified; everything else is new and lives under `experiments/`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/projects/galvardi/yoado")
    ap.add_argument("--out", default="/home/projects/galvardi/yoado/results_2026-09-06.zip")
    a = ap.parse_args(); os.chdir(a.root)
    n = 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("INDEX.md", INDEX); n += 1
        for d in DOCS:
            if os.path.exists(d): z.write(d, os.path.join(os.path.basename(os.path.dirname(d)), "RESULT.md")); n += 1
        for src, dst in EXTRA_DOCS:
            if os.path.exists(src): z.write(src, dst); n += 1
        for d in FIG_DIRS:
            for f in sorted(glob.glob(os.path.join(d, "*.png"))): z.write(f, os.path.join("figures", os.path.basename(d), os.path.basename(f))); n += 1
        for f in FIG_FILES:
            if not os.path.exists(f): continue
            sub = "mnist_reference" if "exact_inversion" in f else "cifar_replica"
            name = os.path.basename(f) if sub == "mnist_reference" else f.split("/")[2] + "_" + os.path.basename(f)
            z.write(f, os.path.join("figures", sub, name)); n += 1
        for f in TABLES:
            if os.path.exists(f):
                stem = "record_strength_table.md" if "record_strength/table" in f else os.path.basename(f)
                if "cifar_charts" in f: stem = "cifar_charts_table.md"
                z.write(f, os.path.join("tables", stem)); n += 1
        for f in SCRIPTS:
            if os.path.exists(f): z.write(f, os.path.join("scripts", os.path.basename(f))); n += 1
        # per-cell rows: small json only (tensors stay on the cluster)
        for f in sorted(glob.glob("experiments/cifar/charts/*/result.json")):
            z.write(f, os.path.join("rows", "charts_" + f.split("/")[-2] + ".json")); n += 1
        for f in sorted(glob.glob("results/cifar_newclass/*.jsonl")) + sorted(glob.glob("results/record_strength/*.jsonl")):
            z.write(f, os.path.join("rows", os.path.basename(f))); n += 1
    print(f"# wrote {a.out} ({n} files, {os.path.getsize(a.out)/1e6:.1f} MB)")
    with zipfile.ZipFile(a.out) as z:
        for i in z.namelist(): print("   ", i)


if __name__ == "__main__":
    main()
