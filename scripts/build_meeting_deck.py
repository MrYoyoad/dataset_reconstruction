"""Assemble the consolidated Monday-meeting deck (2026-08-31) into one PDF.

Re-runnable: figures are embedded BY PATH, so re-running after any figure is
re-rendered (valley n=6 scale-up 695782, crux panels 390026/392821) refreshes
the deck automatically. Captions are drawn tight from notes/meeting_prep_2026-08-31.md
and hold the swarm posture: OBSERVE-don't-conclude, weakest-attacker-scoped,
positive discoveries + clean rejections-with-mechanism, NO pass/fail-baseline numbers.

Usage:  python scripts/build_meeting_deck.py
Output: notes/meeting_deck_2026-08-31.pdf   (+ prints present-vs-placeholder per figure)

Uses the project's only working PDF toolchain (fpdf2 + DejaVu via scripts/md_to_pdf.py).
No GPU / no bsub.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from md_to_pdf import build  # noqa: E402

OUT_PDF = os.path.join(ROOT, "notes", "meeting_deck_2026-08-31.pdf")
TMP_MD = os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "meeting_deck_2026-08-31.md"
)

# ---- summary page -----------------------------------------------------------
SUMMARY = r"""# Leakage characterization — meeting deck (2026-08-31)

*Early exploratory research: we OBSERVE, we don't conclude. Every number below bounds only the
WEAKEST attacker (prior-free, adapter-only, per-image); priors / known-recipe inversion / structural
leakage go beyond it. Small-n and caveats are stated, not buried.*

## The one-paragraph story
Using a validated instrument (whitened Mahalanobis sensitivity, proven unbiased) we characterize how
LoRA fine-tuning records its training images.

- **(1) WHO leaks is predictable from the PUBLIC base model.** Per-image exposure tracks the base
  gradient g0 — and that predictor *transfers* to full fine-tuning.
- **(2) The adapter records the CONCEPT, not the instance.** Near-duplicate images are interchangeable
  to it.
- **(3) New parameterization result (n=6).** Full fine-tuning records ~5× more signal per image than a
  rank-8 adapter (descriptive), but at ≈ the same per-image resolution: the dial is target-dependent
  (full narrower on 4/6, 2 flip; geomean 1.02 ≈ median 0.86 ≈ equal), the noise-free Jacobian reads
  equal, the SGD check diverges — more signal clearly, finer resolution NOT established (the n=2
  "narrower" hint reflected 2 targets; 2 others flip at n=6).
- **(4) That extra signal lives in the first (pixel-carrying) layer.**
- **(5) We already recover recognizable images** in favorable (full-gradient) settings; a robust
  adapter-only inversion is the immediate next step.

## Most interesting, ranked for the talk
- **F-0** positive reconstructions — open here: recognizable faces / rose; robust inversion = next weeks.
- **F-F** activation crux — supervisor's OWN axis + a robust positive: kinked activations leak most, does
  NOT flip under the realistic attack on MNIST (two-cluster; dataset-dependence open). Lead the science here.
- **F-A** margin scatter — the strongest per-image positive (attacker predicts exposure from the public model).
- **F-B** distance dial — the deepest privacy statement (concept, not instance).
- **F-C** valley ladder — the new characterization (full = more signal; finer-resolution UNSETTLED, n=6 pending).
- **F-D** removal + g0 transfer — WHO-leaks ties across parameterizations.
- **F-E** depth fan — the mechanistic "which layer" answer.
"""

# ---- figure pages: (heading, path, caption) ---------------------------------
FIGURES = [
    (
        "F-0. Positive reconstructions — the information IS recoverable (LEAD)",
        "figures/meeting/positive_reconstruction_gallery.png",
        "Full-gradient / TRUE-ΔW CEILING reconstructions across mnist / fashion / cifar10 / flowers × "
        "N∈{2,4,10} × {gelu, softplus}, plus the ViT-scale Phase-0 result (recognizable faces / rose "
        "structure). Across datasets, image-counts and activations the attack recovers recognizable "
        "images — the information IS in the weight change and IS recoverable. These are the CEILING "
        "(true ΔW), NOT the adapter-only attack; making the robust adapter-only inversion work across "
        "the board is the next-weeks work (F-C: full FT holds ~5× more signal to invert).",
    ),
    (
        "F-A. Which images leak — sensitivity vs the g0 predictor",
        "figures/margin_at_scale/f3_margin_who_leaks.png",
        "Per-image sensitivity vs the PUBLIC base-model gradient g0 (n=24, stratified). Which images "
        "leak is predictable from the public model: ρ=+0.777 (p=1e-4) — real and strong, but the verdict "
        "is INDETERMINATE (95% CI [0.53, 0.91], wider than the pre-registered ±0.15; a tercile sign-flip). "
        "Correct reading: STRONG at low g0 (+0.88), SATURATES / reverses at high g0 (−0.12) — why it "
        "saturates is OPEN. Survives a θ0-independent typicality control (+0.78), so it is not just image "
        "atypicality. Caveat: n=24; g0 needs the base model + candidate image, not the base alone.",
    ),
    (
        "F-B. Distance dial — the adapter records the concept, not the instance",
        "figures/similarity_ladder/f2_similarity_ladder.png",
        "Swap-sensitivity vs graded visual distance d; the d=0 self-swap control reads exactly 0. "
        "The parametric near-dup rungs (blur / rotate / brightness) sit ≈ at the floor and sensitivity "
        "climbs to the cross-digit anchor — the adapter records 'a kind of image', not the exact pixels "
        "(the retrieved nearest-neighbor rung is moderate, s≈0.24–0.39, not floor). Privacy statement: the "
        "attacker recovers the CONCEPT. Caveat: ~9 rungs per target, small-n; the axis is pixel-distance.",
    ),
    (
        "F-C. The valley ladder — full-FT vs LoRA (new positive characterization)",
        "figures/fullft_valley/fig_valley_ladder.png",
        "Normalized profile s(d) for LoRA / full-single-layer / full-all-layers, with d* bars. Full "
        "fine-tuning records ~5× more per-image signal than a rank-8 adapter (footprint, F-D, descriptive, "
        "n=6). Whether it ALSO resolves finer is ≈EQUAL / target-dependent — the n=6 scale-up STRENGTHENS "
        "the equal-resolution read: the finite-swap dial is TARGET-DEPENDENT (%%DSTAR%%) with no robust "
        "narrower direction; the noise-free Jacobian reads EQUAL (P7 full≈LoRA, gap not direction-robust); "
        "the SGD-noise check (B2) diverges. So — clearly more signal, but valley width ≈ EQUAL in central "
        "tendency, NOT a clean 'full narrower' (the earlier n=2 'modestly narrower' was a 2-target artifact: "
        "both n=2 targets ARE narrower, but 2 of the other 4 flip → target-dependent, no global narrowing). "
        "Cross-gauge caveat (full-FT Δθ incl εξ vs LoRA ΔW=BA, B2-divergent) is a further reason not to "
        "over-quote any precise ratio. Do NOT overstate either direction. Weakest-attacker scope.",
    ),
    (
        "F-D. Removal cross-regime + g0 transfer (the robust one)",
        "figures/fullft_valley/fig_removal_crossregime.png",
        "Left: full leave-one-out footprint vs LoRA footprint per image. Right: full footprint vs g0. The "
        "SAME images imprint most in both regimes (strong rank corr ρ≈+0.94), and the g0 predictor TRANSFERS "
        "to full fine-tuning (ρ≈+0.83). Absolute footprint ~5× bigger in full (target-median; per-target "
        "~3–6×) — a signal-MAGNITUDE result, not itself a resolution claim (whether full resolves finer is "
        "the open question examined in F-C). Feeds the 'extraction, not missing-information' reading. "
        "Caveat: n=6, exploratory; the absolute-magnitude comparison is descriptive (N→N−1 offset).",
    ),
    (
        "F-E. Depth fan — the extra signal lives in the first layer",
        "figures/fullft_valley/fig_valley_depth.png",
        "Per-layer numerator ‖Δμ_ℓ‖ vs distance for the full network. At the near-duplicate rung the first "
        "(pixel-carrying) layer reacts most (%%DEPTH%%), fading with depth — instance / pixel signal is "
        "concentrated early. Directly answers how the imprint distributes across layers. Caveat: read on the "
        "NUMERATOR (per-layer d* is denominator-confounded); values auto-read from the arm-D summary "
        "(refresh at n=6).",
    ),
    (
        "F-F. Activation crux — the supervisor's top ask (kinked activations leak most; robust on MNIST)",
        "figures/crux/freec_ladder_ranking.png",
        "The REALISTIC free-coefficient leakage ranking across activations at 4 weight-change rungs, oracle "
        "overlaid (authoritative, job 392821, full 52/52; owner yoado-ed). Ordering-INDEPENDENT hard facts "
        "first: (1) the KINKED activations (relu / leaky_relu / selu) leak MOST — kinked-mean ~5× the smooth "
        "family at matched weight-change (ctrl_margin_norm; ~2× on ssim_norm), no smoothness-ranking needed; "
        "(2) smoothness ANTI-correlates with leakage — Spearman(smoothness, leakage) is NEGATIVE at every "
        "weight-change rung and STRENGTHENS with wc (≈ −0.2 → −0.8; the sign and trend are robust, the exact "
        "value is smoothness-ordering-dependent); (3) it does NOT flip the oracle ranking on MNIST. So 'kink "
        "leaks most' survives BOTH the coefficient mode and the wc level. Honest caveats: (a) NOT a monotonic "
        "'smoother ⇒ less leakage' law — the sign still flips WITHIN the smooth-only subset, so it is a "
        "two-cluster (kinked-vs-smooth) effect, not a gradient; (b) MNIST / N=2 / T=1 / n=13 activations — "
        "exploratory; (c) the free-c 'flip' precedent was on FLOWERS, not MNIST — dataset-dependence is OPEN "
        "(the flowers band is still pending). Direction-count (eff_rank) is T-driven, not activation-driven "
        "(2 → 6.3 with T). See panel 2/2 (NTK-survival) for the linearization-fidelity story — which "
        "DISSOCIATES from leakage.",
    ),
    (
        "F-F (panel 2/2). NTK-survival — linearization fidelity dissociates from leakage",
        "figures/crux/feature_stability_vs_T.png",
        "Linearization fidelity (feature-stability = cosine of the gradient features at θ0 vs θ_T) vs training "
        "length T, per activation (job 390026, full 65/65; owner yoado-35; data rescored_tsweep). OBSERVED: at "
        "every T the SMOOTHEST activations (sigmoid / softplus) sustain the highest fidelity and the KINKED "
        "(relu / leaky_relu) the lowest — robust at the EXTREMES but NOT a clean monotone smoothness order "
        "(C∞ gelu/silu decay fast, C¹ elu/celu hold) — the same two-cluster shape as panel 1. The strict NTK "
        "regime (fs > 0.99) is reached only briefly and only by the smoothest. KEY CROSS-PANEL OBSERVATION: "
        "the kinked activations have the WORST linearization fidelity HERE yet LEAK MOST on the ladder — "
        "pooled Spearman(feature-stability, leakage) = +0.08 ≈ 0 (verified at source, n=426) → linearization "
        "fidelity / laziness does NOT drive leakage; the two DISSOCIATE. Leakage is a kink / geometry effect, "
        "not a laziness effect. (This supersedes the earlier n=6 'feature-stability → fidelity ρ≈0.94' hint, "
        "which does not survive the full activation set — the same small-n dissolution as smoothness→fidelity "
        "0.85→0.11.) Observe-framed: a mechanism observation, not a settled law; MNIST, small-n, exploratory.",
    ),
    (
        "(Optional) Data-latent Jacobian spectra — full vs LoRA",
        "figures/fullft_valley/fig_jacobian_spectra.png",
        "The noise-free method behind F-C's equal-resolution read: the data-latent Jacobian spectrum, full "
        "vs LoRA (P7: full ≈ LoRA, the gap not direction-robust). This is the SUPPORTING leg of F-C — the "
        "finite-swap dial (n=2, B2-divergent) is only qualitatively consistent, so treat the Jacobian as the "
        "primary evidence, not a clean two-method agreement. Caveat: early-T behaviour noted.",
    ),
]

NEXT_WEEKS = r"""## Next-weeks work (forward, not failure)

- **Robust adapter-only inversion.** We have recognizable recoveries in favorable settings (F-0); the
  valley result says the information IS present (~5× more signal than LoRA), so the task is *extracting*
  it (better decoder / prior-equipped inversion), not overcoming a missing-information wall.
- **Firm the under-powered readouts.** Margin at n=24 → more; valley dial n=2 → 6 (scale-up running).
- **Full validation gate at scale** (spot-check ρ=0.88 done) — the behavioral-memorization tie.
- **Activation crux** — finish the two running jobs and the fuller analysis (F-F panels c/d).
- **Shared-transform recovery** — attempted (known-recipe ΔW-matching): honest NULL at n=8 draws —
  rotation is suggestive (skill +0.44, mean error 12.5 vs blind 22.5, but CI [−0.15, +0.79] straddles 0),
  blur is null (−0.26). Mechanism: the shared-transform signal is swamped by proxy-vs-private content in
  ΔW; separating transform-from-content is the next step. (Honest-null backup slide exists if asked.)
"""


def load_dstar():
    """Read the four live d* valley widths so the F-C caption never goes stale.

    Written by the valley-ladder figure on every render (auto-refreshes at n=6).
    Falls back to a scale-neutral phrase if the JSON is absent so the deck still builds.
    """
    import json
    import statistics
    import math
    jp = os.path.join(ROOT, "results", "fullft_valley", "valley_headline_dstar.json")
    try:
        d = json.load(open(jp))
        lo, fu = d["lora_A_dstar"], d["full_D_dstar"]
        keys = [k for k in fu if k in lo]  # common targets
        ratios = [fu[k] / lo[k] for k in keys]  # D/A: <1 => full narrower
        n = len(keys)
        narrower = [r for r in ratios if r < 1.0]
        med = statistics.median(ratios)
        # geomean is the scale-invariant central tendency for RATIOS (arithmetic
        # mean is Jensen-biased upward and spuriously inverts the count majority)
        geo = math.exp(sum(math.log(r) for r in ratios) / n)
        if n <= 2:
            s = (f"d*: full-D {fu['t0']:.2f}/{fu['t1']:.2f} vs "
                 f"LoRA-A {lo['t0']:.2f}/{lo['t1']:.2f}")
        else:
            rng = f"{min(narrower):.2f}–{max(narrower):.2f}" if narrower else "—"
            s = (f"full narrower on {len(narrower)}/{n} (D/A {rng}), "
                 f"{n - len(narrower)} flip; geomean {geo:.2f} ≈ median {med:.2f} ≈ equal")
        print(f"[deck] live d* read (n={n}): {s}")
        return s
    except Exception as e:
        print(f"[deck] WARNING: could not read live d* ({e}); using generic phrase")
        return "d* full-D vs LoRA-A: see valley_headline_dstar.json"


def load_depth():
    """Read per-layer ‖Δμ‖ at the near-dup (p0_noise) rung so the F-E line self-refreshes at n=6.

    Prefers the n=6 arm-D summary when present; falls back to n=2, then to a generic phrase.
    """
    import json
    import statistics
    for fn in ("D_n6_summary.json", "D_summary.json"):
        jp = os.path.join(ROOT, "results", "fullft_valley", fn)
        try:
            d = json.load(open(jp))
            layers = ["L0", "L1", "L2"]
            acc = {L: [] for L in layers}
            for t in d["targets"]:
                for pr in t["per_rung"]:
                    if pr["rung"] == "p0_noise":
                        for L in layers:
                            acc[L].append(pr["readouts"][L]["dmu_norm"])
            m = {L: statistics.mean(acc[L]) for L in layers}
            s = f"‖Δμ‖ L0 {m['L0']:.3f} > L1 {m['L1']:.3f} > L2 {m['L2']:.3f}"
            n = len(d["targets"])
            print(f"[deck] live depth read ({fn}, n={n}): {s}")
            return s
        except Exception:
            continue
    print("[deck] WARNING: could not read live depth; using generic phrase")
    return "‖Δμ‖ L0 > L1 > L2 (first layer largest)"


def main():
    dstar_str = load_dstar()
    depth_str = load_depth()
    md = [SUMMARY, ""]
    present, placeholder = [], []
    for heading, path, caption in FIGURES:
        caption = caption.replace("%%DSTAR%%", dstar_str).replace("%%DEPTH%%", depth_str)
        md.append(r"\pagebreak")
        md.append(f"## {heading}")
        abspath = os.path.join(ROOT, path)
        if os.path.exists(abspath):
            md.append(f"![{caption}]({abspath})")
            present.append(path)
        else:
            md.append(f"*[figure pending — {path} not yet rendered]*")
            md.append("")
            md.append(caption)
            placeholder.append(path)
        md.append("")
    md.append(r"\pagebreak")
    md.append(NEXT_WEEKS)

    with open(TMP_MD, "w") as f:
        f.write("\n".join(md))
    build(TMP_MD, OUT_PDF, title="Meeting deck 2026-08-31")

    size = os.path.getsize(OUT_PDF)
    print(f"\n[deck] {OUT_PDF}  ({size/1024:.0f} KB)")
    print(f"[deck] figures PRESENT ({len(present)}): " + ", ".join(present))
    if placeholder:
        print(f"[deck] figures PLACEHOLDER ({len(placeholder)}): " + ", ".join(placeholder))
    else:
        print("[deck] figures PLACEHOLDER (0): none — all figures embedded")
    if size < 50_000:
        print("[deck] WARNING: PDF suspiciously small — check figures embedded")


if __name__ == "__main__":
    main()
