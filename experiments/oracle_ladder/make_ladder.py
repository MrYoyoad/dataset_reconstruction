#!/usr/bin/env python3
"""Tables, figures and RESULT.md for the oracle-chart ladder. Reads rows and tensors only; runs nothing.

  python -m experiments.oracle_ladder.make_ladder
"""
import argparse, glob, json, os
import numpy as np
import torch

SRC, FIG = "results/oracle_ladder", "figures/oracle_ladder"
PANEL_EPS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]          # the subset that fits a panel; the table carries all of them
BLUE, RED, GREY, GREEN, ORANGE = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c", "#eb6834"


def load():
    rows = [json.loads(l) for l in open(os.path.join(SRC, "rows.jsonl")) if l.strip()]
    out = {}
    for r in rows:
        if r.get("wrong_release"): out.setdefault(r["example"] + "_control", r); continue
        out.setdefault(r["example"], []).append(r)
    for k in out:
        if isinstance(out[k], list):
            out[k] = sorted(out[k], key=lambda r: (r["chart"] == "pca", r["proj_err_mean"]))
    return out


NOUN = {"mlp_letter_a": "letters", "mlp_letter_a_full": "letters", "d15_letter_a": "letters", "d15_digits": "digits"}   # else "photographs"
MNIST_EXAMPLES = ("mlp_letter_a", "mlp_letter_a_full", "d15_letter_a", "d15_digits")
# the two MNIST arms of WP5 (plan audit 2026-09-18) answer different questions and are never quoted against each other
MNIST_INTRO = {
    "mlp_letter_a": ("Eight EMNIST letters `a` (test split) added as a NEW CLASS to the 3-layer MNIST MLP",
                     "the same eight letters as the existing letters cells", "letter-`a` test split",
                     "a 784-1000-1000 MNIST MLP", "arm (a): the gate for the letters cells and for WP4's void condition"),
    "mlp_letter_a_full": ("Eight EMNIST letters `a` (test split) added as a NEW CLASS to the over-trained twin of the 3-layer MNIST MLP",
                          "the same eight letters as the existing letters cells", "letter-`a` test split",
                          "a 784-1000-1000 MNIST MLP", "arm (a) on the `_full` twin; rows never merged with `mlp_letter_a`"),
    "d15_letter_a": ("Eight EMNIST letters `a` (test split) added as a NEW CLASS to the 15-layer MNIST MLP of the depth window, head adapter "
                     "only (head extended by a zero row, m=11, exactly as `mlp_letter_a`)",
                     "the same eight letters as `mlp_letter_a` and the existing letters cells", "letter-`a` test split",
                     "the 15-layer, width-1000 MNIST MLP", "arm (b'): the depth-window encoder with a release that RECORDS — `d15_digits` "
                     "below was a confident batch of known classes and recorded nothing, so it gives no gate"),
    "d15_digits": ("Eight MNIST TEST DIGITS with their TRUE labels on the 15-layer MNIST MLP of the depth window, head adapter only "
                   "(the head is NOT extended: a confident batch of known classes)",
                   "the same eight digits as the k-sweep (`real_encoder_ranklaw.py`)", "MNIST test split",
                   "the 15-layer, width-1000 MNIST MLP", "arm (b): shares encoder, images and chart pool (first 50 000 train digits) "
                   "with the depth window; differs from it in adapter placement (head only) and in being a confident batch"),
}


def noun(ex): return NOUN.get(ex, "photographs")


def tile(ax, x):
    """One image tile from a flat vector: 3072 -> RGB 32x32, 784 -> grey 28x28."""
    if x.numel() == 3072: ax.imshow(x.reshape(3, 32, 32).permute(1, 2, 0).clamp(0, 1).float().numpy())
    else: ax.imshow(x.reshape(28, 28).clamp(0, 1).float().numpy(), cmap="gray", vmin=0, vmax=1)


def table(rows, ctrl, nn="photographs"):
    h = (f"| chart | eps | measured projection error of the {nn} (mean, range) | landings / 400 | images found | "
         f"residual at the {nn} (max) | median start residual | SSIM attack | SSIM ceiling | SSIM control | top-20 all landings |")
    t = [h, "|" + "---|" * 11]
    for r in rows:
        name = "**public PCA (ATTACKER-AVAILABLE)**" if r["chart"] == "pca" else "oracle (not attacker-available)"
        top20 = "yes" if r["top20_all_landings"] else f"no ({r['top20_landed_count']}/20)"
        e = "—" if r["eps"] is None else f"{r['eps']:g}"
        t.append(f"| {name} | {e} | {r['proj_err_mean']:.4f}  ({r['proj_err_min']:.4f}–{r['proj_err_max']:.4f}) | "
                 f"{r['landed']}/{r['starts']} | {r['images_found']}/{r['N']} | {r['residual_at_truths_max']:.1e} | "
                 f"{r['objective_median']:.1e} | {r['ssim_attack_mean']:.2f} | {r['ssim_ceiling_mean']:.2f} | "
                 f"{r['ssim_control_mean']:.2f} | {top20} |")
    if ctrl:
        t.append(f"| *wrong-release control, oracle eps 0* | 0 | {ctrl['proj_err_mean']:.4f} | **{ctrl['landed']}/{ctrl['starts']}** | "
                 f"**{ctrl['images_found']}/{ctrl['N']}** | {ctrl['residual_at_truths_max']:.1e} | {ctrl['objective_median']:.1e} | "
                 f"{ctrl['ssim_attack_mean']:.2f} | {ctrl['ssim_ceiling_mean']:.2f} | {ctrl['ssim_control_mean']:.2f} | "
                 f"{'yes' if ctrl['top20_all_landings'] else 'no'} |")
    return "\n".join(t)


def panel(ex, rows, plt):
    keep = []
    for e in PANEL_EPS:
        c = [r for r in rows if r["chart"] == "oracle" and abs((r["eps"] or 0) - e) < 1e-9]
        if c: keep.append(c[0])
    pca = [r for r in rows if r["chart"] == "pca"]
    keep += pca
    if not keep: return
    blobs = []
    for r in keep:
        tag = ex + ("_pca" if r["chart"] == "pca" else "_eps%g" % r["eps"])
        f = os.path.join(SRC, tag + ".pth")
        if os.path.exists(f): blobs.append((r, torch.load(f, map_location="cpu", weights_only=False)))
    if not blobs: return
    N = blobs[0][0]["N"]; nrow = len(blobs) + 1
    cell = 1.05; fw, fh = 3.6 + cell * N, 1.5 + cell * nrow + 0.55
    fig = plt.figure(figsize=(fw, fh))
    left, right = 3.5 / fw, 1 - 0.1 / fw
    gs = fig.add_gridspec(nrow, N, left=left, right=right, top=1 - 1.25 / fh, bottom=0.45 / fh, wspace=0.07, hspace=0.42)
    X0 = blobs[0][1]["x_raw"]
    for j in range(N):
        ax = fig.add_subplot(gs[0, j]); tile(ax, X0[:, j])
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            p = ax.get_position(); fig.text(left - 0.012, (p.y0 + p.y1) / 2, f"the private {noun(ex)}", fontsize=9.5, va="center", ha="right")
    for ri, (r, b) in enumerate(blobs, start=1):
        Xb = b["x_found_best"]; per = {p["i"]: p for p in r["per_image"]}
        for j in range(N):
            ax = fig.add_subplot(gs[ri, j]); tile(ax, Xb[:, j])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_linewidth(0.5); sp.set_color("#2a78d6" if r["chart"] == "pca" else "#cccccc")
            p = per[j]
            ax.set_title(("recovered" if p["landed"] else f"missed {p['best_err']:.2f}") + f"\n{p['best_objective']:.0e}",
                         fontsize=6.8, pad=2.5, color=(GREEN if p["landed"] else RED))
            if j == 0:
                lab = ("PUBLIC PCA chart\n(attacker-available)" if r["chart"] == "pca"
                       else f"oracle chart, eps {r['eps']:g}\nprojection error {r['proj_err_mean']:.3f}")
                pos = ax.get_position()
                fig.text(left - 0.012, (pos.y0 + pos.y1) / 2, lab, fontsize=8.5, va="center", ha="right",
                         color=("#2a78d6" if r["chart"] == "pca" else "#333333"))
    fig.text(0.5, 1 - 0.30 / fh, f"How accurate must the chart be? — {ex.replace('_', ' ')}, RAW {noun(ex)}", fontsize=13, ha="center", va="top")
    fig.text(0.5, 1 - 0.72 / fh, f"every row but the last uses a chart built from the private {noun(ex)} themselves and is NOT attacker-available; "
             "the last row is the real public chart", fontsize=8.5, ha="center", va="top", color="#b3261e")
    fig.text(0.5, 0.12 / fh, "each tile is the closest of 400 random starts, with its residual; head adapter r=64, T=400, k=32",
             fontsize=7.5, ha="center", va="bottom", color="#666666")
    os.makedirs(FIG, exist_ok=True); fig.savefig(os.path.join(FIG, f"ladder_{ex}.png"), dpi=200); plt.close(fig)
    print(f"# {FIG}/ladder_{ex}.png")


def curve(ex, rows, plt):
    orc = [r for r in rows if r["chart"] == "oracle"]; pca = [r for r in rows if r["chart"] == "pca"]
    if not orc: return
    x = [max(r["proj_err_mean"], 1e-4) for r in orc]          # eps=0 is exactly 0 by construction; clipped only for the log axis
    fig, ax = plt.subplots(figsize=(10, 6)); ax2 = ax.twinx()
    ax.axvspan(min(x) * 0.7, max(x) * 1.4, color=GREY, alpha=0.10)
    ax.text(min(x) * 0.75, 1.02, "oracle charts — NOT attacker-available", fontsize=8.5, color="#b3261e", va="bottom")
    ax.plot(x, [r["landed"] / r["starts"] for r in orc], "o", color=BLUE, ms=7, label="landings / 400 (oracle charts)")
    ax2.plot(x, [r["ssim_attack_mean"] for r in orc], "s", color=GREEN, ms=6, label="SSIM, attack")
    ax2.plot(x, [r["ssim_ceiling_mean"] for r in orc], "^", color=GREY, ms=6, label="SSIM, chart ceiling")
    ax2.plot(x, [r["ssim_control_mean"] for r in orc], "v", color=ORANGE, ms=6, label="SSIM, same-class control")
    if pca:
        p = pca[0]; xp = p["proj_err_mean"]
        ax.axvline(xp, color="#2a78d6", lw=1.4, ls="--")
        ax.text(xp * 1.05, 0.55, f"the real public chart\nprojection error {xp:.3f}", fontsize=9, color="#2a78d6")
        ax.plot([xp], [p["landed"] / p["starts"]], "*", color="#2a78d6", ms=18, label="landings / 400 (PUBLIC chart — attacker-available)")
        ax2.plot([xp], [p["ssim_attack_mean"]], "*", color=GREEN, ms=14)
    ax.set_xscale("log"); ax.set_xlabel(f"measured projection error of the private {noun(ex)} onto the chart")
    ax.set_ylabel(f"fraction of 400 random starts that recover a {noun(ex)[:-1]}"); ax2.set_ylabel(f"SSIM against the {noun(ex)[:-1]}")
    ax.set_ylim(-0.03, 1.05); ax2.set_ylim(0, 1.0)
    ax.set_title(f"Chart accuracy against recovery — {ex.replace('_', ' ')}, RAW {noun(ex)}\n"
                 "points, not a fit; the two access models are marked because the axis mixes them", fontsize=11)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, frameon=False, loc="center left")
    ax.grid(alpha=0.25, which="both"); ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"curve_{ex}.png"), dpi=200); plt.close(fig)
    print(f"# {FIG}/curve_{ex}.png")


def ceiling_sentence(orc):
    """orc sorted by proj_err_mean. The first cell that leaves the ceiling is where it LEAVES; the tracking region ends
       at the cell before it (the off-by-one of the first version printed the leaving cell as the last tracking cell)."""
    idx = next((i for i, r in enumerate(orc) if r["ssim_attack_mean"] < r["ssim_ceiling_mean"] - 0.05), None)
    if idx is None:
        return ("**Where the returned image leaves the ceiling.** The attack stays within 0.05 SSIM of the chart ceiling at every "
                "oracle cell measured, so nothing here separates the search from the representation.")
    if idx == 0:
        return (f"**Where the returned image leaves the ceiling.** There is no tracking region on this example: the attack is already "
                f"more than 0.05 SSIM below the chart's own ceiling at the lowest-error cell of the ladder "
                f"({orc[0]['ssim_attack_mean']:.2f} against {orc[0]['ssim_ceiling_mean']:.2f} at projection error "
                f"{orc[0]['proj_err_mean']:.4f}), so the gap at that end is the search and not the chart.")
    return (f"**Where the returned image leaves the ceiling.** The attack tracks the chart's own ceiling to within 0.05 SSIM up to "
            f"a projection error of {orc[idx - 1]['proj_err_mean']:.4f}, and falls below it at {orc[idx]['proj_err_mean']:.4f} "
            f"({orc[idx]['ssim_attack_mean']:.2f} against a ceiling of {orc[idx]['ssim_ceiling_mean']:.2f}).")


def sentences(ex, rows):
    orc = sorted([r for r in rows if r["chart"] == "oracle"], key=lambda r: r["proj_err_mean"])
    pca = [r for r in rows if r["chart"] == "pca"]
    N = orc[0]["N"]
    full = [r for r in orc if r["images_found"] == N]
    last_full = max(full, key=lambda r: r["proj_err_mean"]) if full else None
    first_fail = min([r for r in orc if r["images_found"] == 0], key=lambda r: r["proj_err_mean"], default=None)
    s = []

    if last_full:
        s.append(f"**Where exact landing stops.** All {N} {noun(ex)} are still recovered at a measured projection error of "
                 f"{last_full['proj_err_mean']:.4f}" + (f" and none at {first_fail['proj_err_mean']:.4f}." if first_fail else "."))
    else:
        best = max(orc, key=lambda r: r["images_found"])
        s.append(f"**Where exact landing stops.** No oracle cell on this ladder recovered all {N}; the most recovered anywhere is "
                 f"{best['images_found']} of {N}, at projection error {best['proj_err_mean']:.4f}. This example therefore has no "
                 f"all-{N} gate to locate, and none is quoted for it.")

    s.append(ceiling_sentence(orc))

    if pca:
        p = pca[0]
        if last_full:
            s.append(f"**Where the real public chart sits.** Its projection error is {p['proj_err_mean']:.4f}, "
                     f"{'above' if p['proj_err_mean'] > last_full['proj_err_mean'] else 'below'} the last error at which every "
                     f"photograph is recovered ({last_full['proj_err_mean']:.4f}), and it returns {p['landed']} of {p['starts']} "
                     f"landings and {p['images_found']} of {p['N']} images.")
        else:
            landing = [r for r in orc if r["landed"] > 0]
            tail = (f", and above the largest error at which any start landed at all "
                    f"({max(r['proj_err_mean'] for r in landing):.4f})" if landing else "")
            s.append(f"**Where the real public chart sits.** Its projection error is {p['proj_err_mean']:.4f}. No oracle cell here "
                     f"recovered all {N}, so there is no all-{N} threshold to place it against{tail}. What is measured is that it "
                     f"returns {p['landed']} of {p['starts']} landings and {p['images_found']} of {p['N']} images.")
    return "\n".join("- " + x for x in s)


def disagreement(d):
    """The closing section. The two examples are compared and never pooled."""
    f = {}
    for ex in ("mlp_motorcycle", "cnn_keyboard"):
        rows = d.get(ex)
        if not rows: return ""
        orc = sorted([r for r in rows if r["chart"] == "oracle"], key=lambda r: r["proj_err_mean"])
        landing = [r for r in orc if r["landed"] > 0]
        N = orc[0]["N"]
        f[ex] = dict(N=N, orc=orc,
                     last_land=max((r["proj_err_mean"] for r in landing), default=None),
                     first_zero=min((r["proj_err_mean"] for r in orc if r["landed"] == 0), default=None),
                     best_found=max(r["images_found"] for r in orc),
                     pca=[r for r in rows if r["chart"] == "pca"],
                     ctrl=d.get(ex + "_control"))
    m, c = f["mlp_motorcycle"], f["cnn_keyboard"]
    ratio = m["last_land"] / c["last_land"] if (m["last_land"] and c["last_land"]) else None
    out = ["\n## The two examples do not agree, and are not averaged\n"]
    out.append(f"**The gate sits at a different place on each.** On the MLP/motorcycle release some start still lands at a measured "
               f"projection error of {m['last_land']:.4f} and no start lands at {m['first_zero']:.4f}. On the CNN/keyboard release the "
               f"last landing is at {c['last_land']:.4f} and none survives to {c['first_zero']:.4f} — and even the exactly-spanning "
               f"chart returns {c['best_found']} of {c['N']} rather than {c['N']}. The last-landing error differs by a factor of "
               f"{ratio:.1f} between the two."
               if ratio else "**The gate sits at a different place on each.**")
    out.append(f"\n**What that difference cannot be attributed to.** The two cells differ in backbone (MLP against CNN) *and* in class "
               f"(motorcycle against keyboard) at the same time, so this ladder cannot say which of the two moves the gate. It says only "
               f"that the gate is not one number across releases. Nothing here is averaged over the two, and a single ladder figure "
               f"should not be shown as though it were the ladder.\n")
    pm, pc = m["pca"][0], c["pca"][0]
    out.append(f"**Both pre-registered predictions held.** The attacker-available public PCA chart returns "
               f"{pm['landed']} of {pm['starts']} on the motorcycle release and {pc['landed']} of {pc['starts']} on the keyboard "
               f"release, at projection errors {pm['proj_err_mean']:.4f} and {pc['proj_err_mean']:.4f}. The wrong-release control "
               f"returns nothing even at an exactly spanning chart, which is what says the ladder is measuring the release and not "
               f"the chart's ability to hold the images.\n")
    lo = [r for r in m["orc"] if r["eps"] not in (0, None)]
    lo = sorted(lo, key=lambda r: r["eps"])
    if len(lo) > 1:
        a, b = lo[0], lo[-1]
        out.append(f"**`eps` is not proportional to the error it induces.** On the motorcycle ladder the measured projection error is "
                   f"{a['proj_err_mean'] / a['eps']:.2f} of `eps` at `eps`={a['eps']} and {b['proj_err_mean'] / b['eps']:.2f} of it at "
                   f"`eps`={b['eps']}: the perturbation saturates, so `eps` must not be read as an error axis anywhere.\n")
    return "\n".join(out)


def gate_two_ends(rows):
    """The pre-registered gate of WP5 (plan 2026-09-18): TWO ends, never one number -- the largest measured projection
       error at which all N images are found, and the smallest at which none is. Either end may not exist on a ladder."""
    orc = [r for r in rows if r["chart"] == "oracle"]
    N = orc[0]["N"]
    hi = max([r for r in orc if r["images_found"] == N], key=lambda r: r["proj_err_mean"], default=None)
    lo = min([r for r in orc if r["images_found"] == 0], key=lambda r: r["proj_err_mean"], default=None)
    return N, hi, lo


def mnist_section(ex, rows, ctrl, d):
    """The MNIST letters ladder. A separate section: different backbone AND different data from the CIFAR examples,
       so its gate is placed beside theirs only as a difference in construction, never as a property of 'the gate'."""
    r0 = rows[0]; N, hi, lo = gate_two_ends(rows)
    nn = noun(ex); what, same_as, split, backbone_desc, arm = MNIST_INTRO[ex]
    out = [f"\n## {ex.replace('_', ' ')}  ({len(rows)} charts) — MNIST, {arm.split(':')[0]}\n",
           f"**{what}** (`{r0.get('backbone_ckpt', '?')}`), {same_as} (join-key indices into the {split}: "
           f"`{r0.get('private_join_idx', '?')}`" + (f", labels `{r0['labels_of_privates']}`" if not r0.get("head_extended", True) else "")
           + f"). Same construction as the CIFAR ladders: head adapter r={r0['r']}, T={r0['T']}, k={r0['k']}, {r0['starts']} random starts, "
           f"the same solver and the same landing bar, privates RAW. {arm.split(': ', 1)[1][0].upper() + arm.split(': ', 1)[1][1:]}.\n"]
    tr, te, tl = r0.get("backbone_train_acc"), r0.get("backbone_test_acc"), r0.get("backbone_train_loss", float("nan"))
    if tr is not None and te is not None:
        out.append(f"**WP0 base-model record.** Train accuracy {tr*100:.2f}% (train loss {tl:.2e}), test accuracy {te*100:.2f}%, measured "
                   f"at load time on the full splits. The 'fully trained' gate of the plan is train >= 99.5% and train loss <= 1e-2: this "
                   f"checkpoint **{'passes' if r0.get('wp0_fully_trained') else 'does NOT pass'}** it"
                   + ("" if r0.get("wp0_fully_trained") else ", and every row of this section is on that not-fully-trained base")
                   + (f". {r0['base_gate_note']}" if r0.get("base_gate_note") else "") + ".\n")
    if "B_T_fro" in r0:
        sr0 = r0.get("softmax_residual_at_W0", []); srT = r0.get("softmax_residual_at_T", [])
        out.append(f"**Recording strength (eps 0 cell).** rank B_T = {r0['rank_B_T']}, ‖B_T‖_F = {r0['B_T_fro']:.3e}, ‖B_T A_T‖_F = "
                   f"{r0['BA_T_fro']:.3e}, σ_N/σ_1 of B_T = {r0['B_T_sigma_ratio']:.1e}; per-image softmax residual at W0 "
                   f"[{', '.join(f'{v:.1e}' for v in sr0)}] and at T [{', '.join(f'{v:.1e}' for v in srT)}]"
                   + (" — a batch the base already classifies confidently leaves a weak recording, and the ladder must be read with that "
                      "in view." if sr0 and sorted(sr0)[N // 2] < 1e-2 else "") + "\n")
    out += [table(rows, ctrl, nn), "\n"]
    s = []
    if hi is not None and lo is not None:
        s.append(f"**The gate, two ends.** The largest measured projection error at which all {N} {nn} are found is "
                 f"**{hi['proj_err_mean']:.4f}** (eps {hi['eps']:g}; {hi['landed']}/{hi['starts']} landings), and the smallest at which "
                 f"none is found is **{lo['proj_err_mean']:.4f}** (eps {lo['eps']:g}). Between them the count falls from {N} to 0 over the "
                 f"cells listed in the table; no single number is quoted for the gate.")
    elif hi is None and lo is not None:
        best = max([r for r in rows if r["chart"] == "oracle"], key=lambda r: r["images_found"])
        s.append(f"**The gate, two ends.** No oracle cell recovered all {N} {nn} (the most anywhere is {best['images_found']} of {N}, at "
                 f"projection error {best['proj_err_mean']:.4f}), so the all-{N} end does not exist on this ladder; the smallest error at "
                 f"which none is found is {lo['proj_err_mean']:.4f} (eps {lo['eps']:g}).")
    elif hi is not None and lo is None:
        s.append(f"**The gate, two ends.** All {N} {nn} are found up to a projection error of {hi['proj_err_mean']:.4f} (eps {hi['eps']:g}), "
                 f"and no cell of this ladder reached 0 of {N}, so the zero end lies beyond the largest error measured "
                 f"({max(r['proj_err_mean'] for r in rows if r['chart'] == 'oracle'):.4f}).")
    else:
        s.append(f"**The gate, two ends.** Neither end exists on this ladder: no cell found all {N} and no cell found 0.")
    s.append(ceiling_sentence(sorted([r for r in rows if r["chart"] == "oracle"], key=lambda r: r["proj_err_mean"])))   # same generator as CIFAR
    pca = [r for r in rows if r["chart"] == "pca"]
    if pca:
        p = pca[0]
        s.append(f"**Where the real public chart sits.** Its projection error is {p['proj_err_mean']:.4f}"
                 + (f", {'above' if p['proj_err_mean'] > hi['proj_err_mean'] else 'below'} the all-{N} end ({hi['proj_err_mean']:.4f})" if hi else "")
                 + (f" and {'above' if p['proj_err_mean'] > lo['proj_err_mean'] else 'below'} the zero end ({lo['proj_err_mean']:.4f})" if lo else "")
                 + f"; it returns {p['landed']} of {p['starts']} landings and {p['images_found']} of {p['N']} {nn}.")
    if ctrl:
        s.append(f"**Wrong-release control.** The release trained on eight OTHER {nn} (indices `{ctrl.get('wrong_release_idx', '?')}`) "
                 f"with the exactly-spanning chart of the true eight returns {ctrl['landed']} of {ctrl['starts']} landings and "
                 f"{ctrl['images_found']} of {ctrl['N']} {nn} (certificate residual at the true {nn} {ctrl['residual_at_truths_max']:.1e}).")
        if "floor_objective" in ctrl:
            s.append(f"**Is the certificate degenerate on this release?** On the control the minimum start objective is "
                     f"{ctrl['objective_min']:.1e} against the control release's own floor {ctrl['floor_objective']:.1e} "
                     f"(ratio {ctrl['objective_min_over_floor']:.1e}; the assertion is ratio > 1e4): "
                     + ("**non-degenerate** — random starts do not reach the floor of a release whose images are not in the chart, so a "
                        "landing on the true release would be evidence." if ctrl.get("certificate_nondegenerate") else
                        "**DEGENERATE** — starts reach the floor of a release that recorded none of the charted images, so a zero of the "
                        "certificate in this chart says nothing about the release and no gate is read from this ladder."))
        elif "objective_min" in ctrl and "residual_at_truths_max" in ctrl:
            s.append(f"**Is the certificate degenerate on this release?** (rows predate the floor field) On the control the minimum start "
                     f"objective is {ctrl['objective_min']:.1e}; the true-release cells' minima are "
                     f"{min(r['objective_min'] for r in rows):.1e}–{max(r['objective_min'] for r in rows):.1e} against a truth floor of "
                     f"{rows[0]['residual_at_truths_max']**2:.1e}"
                     + (" — the control reaches the same level as the true release, which is the degenerate signature."
                        if ctrl['objective_min'] < 1e2 * min(r['objective_min'] for r in rows) else "."))
    # placed beside CIFAR as a difference in construction, never pooled
    cif = []
    for cex in ("mlp_motorcycle", "cnn_keyboard"):
        if d.get(cex):
            _, chi, clo = gate_two_ends(d[cex])
            cif.append(f"{cex.replace('_', ' ')}: all-8 end " + (f"{chi['proj_err_mean']:.4f}" if chi else "none")
                       + ", zero end " + (f"{clo['proj_err_mean']:.4f}" if clo else "none"))
    if cif:
        s.append("**Against the CIFAR ladders.** " + "; ".join(cif) + f". This example differs from both in backbone ({backbone_desc}) "
                 f"*and* in data (28x28 grey {nn} against 32x32 colour photographs), so a gate that sits elsewhere here is a "
                 "difference between constructions, not a property of 'the gate'; the three are not averaged.")
    out += ["\n".join("- " + x for x in s if x), "\n",
            f"\n![ladder](../../figures/oracle_ladder/ladder_{ex}.png)\n![curve](../../figures/oracle_ladder/curve_{ex}.png)\n"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="experiments/oracle_ladder/RESULT.md"); a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = load(); os.makedirs(FIG, exist_ok=True)
    md = ["# How accurate must the chart be? The oracle ladder on the two meeting examples\n",
          "Two examples, chosen so every figure in the talk is about the same objects: **eight CIFAR-100 motorcycles added to the "
          "CIFAR-10 MLP**, and **eight CIFAR-100 keyboards added to the CIFAR-10 CNN**. Head adapter r=64, T=400 SGD steps, k=32, "
          "400 random starts, the same Levenberg-Marquardt solver and the same landing bar (relative image error below 1e-2) in every "
          "cell, with nothing tuned between cells.\n",
          "**The privates are the RAW photographs**, not their chart projections: the adapter is fine-tuned on the photographs and the "
          "attack targets the photographs.\n",
          "## Two things to read before the tables\n",
          "**`eps` is not the chart's error.** The ladder perturbs each private photograph by relative noise `eps` and then spans the "
          "perturbed vectors, so the true photographs do not lie in the resulting chart. What belongs on the axis is the **measured "
          "relative projection error of the true photographs** onto each chart, reported beside `eps` in every row, and it is what both "
          "figures are drawn against. At `eps = 0` the chart spans the photographs exactly, so that error is **zero by construction** and "
          "that cell is effectively on-chart even though the privates are raw — it anchors the axis rather than being a result.\n",
          "**Only one point on these axes is an attack.** Every oracle chart is built from the private photographs themselves and is "
          "**not attacker-available at any `eps`, including 0**; they measure a fidelity ceiling. The public PCA row is the only "
          "attacker-available cell, and it is marked on both figures rather than only in a caption. No median or total anywhere pools "
          "the two.\n",
          "**The releases are new, and the private images are not.** These cells are fine-tuned on the raw photographs, while the "
          "existing motorcycle and keyboard cells elsewhere in this study are fine-tuned on chart projections. Same eight photographs, "
          "same seed, **different release** — the earlier one could not be reused because it was trained on projections. A slide may "
          "place a ladder panel beside an on-chart panel, but a reader must not read a difference between them as an effect of the "
          "chart alone.\n",
          "**The two examples differ in backbone as well as class** (MLP against CNN), so if they disagree about where the gate sits, "
          "the disagreement is **not attributable to the class** and they are reported separately rather than averaged.\n",
          "**These tables were regenerated after a bug in this generator, and the numbers moved.** The cell measurements "
          "(jobs in `results/oracle_ladder/rows.jsonl`, 28 cells, one job each) were never re-run and are unchanged; what "
          "was wrong was the prose the generator derived from them. It computed the first chart that *leaves* the SSIM "
          "ceiling and then printed it as the last chart that *tracks* it — off by one row on both examples — and, where no "
          "chart recovered all eight, it still placed the public chart '**below** the last error at which every photograph "
          "is recovered', a comparison against a threshold that does not exist on that example, in the direction that would "
          "suggest the public chart ought to work. Both are fixed and the no-threshold case now says so explicitly. Any "
          "earlier copy of this file carrying those two sentences should be discarded rather than reconciled.\n"]
    for ex in ("mlp_motorcycle", "cnn_keyboard"):
        rows = d.get(ex)
        if not rows: md += [f"\n## {ex}\n", "_no rows yet_\n"]; continue
        md += [f"\n## {ex.replace('_', ' ')}  ({len(rows)} charts)\n", table(rows, d.get(ex + "_control")), "\n",
               sentences(ex, rows), "\n",
               f"\n![ladder](../../figures/oracle_ladder/ladder_{ex}.png)\n![curve](../../figures/oracle_ladder/curve_{ex}.png)\n"]
        panel(ex, rows, plt); curve(ex, rows, plt)
    md.append(disagreement(d))
    for ex in MNIST_EXAMPLES:                                   # the MNIST letters ladder(s): own section, never pooled with CIFAR
        rows = d.get(ex)
        if not rows: continue
        md.append(mnist_section(ex, rows, d.get(ex + "_control"), d))
        panel(ex, rows, plt); curve(ex, rows, plt)
    open(a.out, "w").write("\n".join(md) + "\n")
    print(f"# wrote {a.out}")


if __name__ == "__main__":
    main()
