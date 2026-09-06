#!/usr/bin/env python3
"""Bundle the 2026-09-06 results (record strength + the CIFAR certificate study) into one zip with figures.

  python -m experiments.cifar.make_bundle [--out /home/projects/galvardi/yoado/results_2026-09-06.zip]

Contents: the two RESULT.md write-ups, every figure they cite, the generated tables, the per-cell JSON rows (small),
the scripts that produced them, and an INDEX.md that says what each file is and which job produced it.
"""
import argparse, glob, json, os, zipfile

FIG_DIRS = ["figures/record_strength", "figures/cifar_charts", "figures/cifar_newclass"]
FIG_FILES = ["experiments/cifar/k32/cifar_certificate.png", "experiments/cifar/k48/cifar_certificate.png",
             "experiments/cifar/k32_onchart/cifar_certificate_onchart.png", "experiments/cifar/k48_onchart/cifar_certificate_onchart.png",
             "figures/exact_inversion/letters_recovery_k32_760909.png", "figures/exact_inversion/certificate_recovery_k6_706721.png"]
DOCS = ["experiments/record_strength/RESULT.md", "experiments/cifar/RESULT.md"]
EXTRA_DOCS = [("notes/ntk_vs_certificate_comparison.md", "COMPARISON_ntk_vs_certificate.md")]
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

## 3. How this compares with the route we were on before

`COMPARISON_ntk_vs_certificate.md` puts the NTK-regime / linearized reconstruction results (Experiment B, the anchor
sweep, direct weight inversion, the gradient bridge) beside the certificate results, with each number labelled by
what the attacker had to know to get it. Short version: the earlier route's best numbers are oracle-coefficient or
full-recipe numbers, its recognisable recovery is an N=2 phenomenon, and its linearization is not valid in the
regime that carries signal; the certificate route is exact and self-checking but has its own open gap, stated there.

## 4. Scripts

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
