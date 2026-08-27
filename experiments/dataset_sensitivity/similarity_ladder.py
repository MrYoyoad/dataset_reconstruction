#!/usr/bin/env python
"""
SIMILARITY-GRADED SWAP ("similarity ladder") — does the LoRA adapter's sensitivity to
swapping ONE training image depend on how VISUALLY SIMILAR the replacement is?

Program: dataset-composition-sensitivity battery (notes/dataset_sensitivity_program_plan.md).
The arms so far swap a training image T for ONE arbitrary held-out same-class control T' and
ask "is the swap detectable?". This arm grades the swap: build a LADDER of replacements T'
at increasing visual distance from T and run the IDENTICAL swap measurement at every rung.

  sens(rung) rising with distance + near-duplicate rungs ~null
      => the adapter records the CONCEPT (an equivalence class of similar images), not the
         exact INSTANCE — swapping in a near-duplicate is invisible.
  sens(rung) flat-high (even the tiny-noise rung fires)
      => INSTANCE-level memorization: the adapter pins the exact pixels.

MEASUREMENT (mirrors arm_b_dilution / arm_d_context EXACTLY):
  * Fixed private set D (N=16, the arm-B/D construction: get_finetuning_data(seed=42) from the
    MNIST-TEST pool; honest pretrained θ0 via _honest_target; ds_mean FROZEN thereafter).
  * Fixed target T = a class-1 image of D at a fixed position (first class-1 slots; one
    ladder per target, --n_targets distinct T's).
  * Baseline reseed ensemble ΔW(D, seed_j) over K seeds — trained ONCE, shared by every rung
    (and every target: D never changes). The swap arm retrains per rung per seed.
  * Per rung: x_sw = D with T -> T'; paired per-seed diff (init CANCELS per pair)
        v_j = ΔW(D, seed_j) − ΔW(D_swap, seed_j)
    whitened_sensitivity([v.cpu()...], [r.cpu()...], n_folds=5, p_max=3, n_perm=500).
    ΔW = B·A only (gauge-invariant), float64, mnist / gelu MLP / binary / rank 8 / T=1000 / lr 0.5.

THE LADDER (two rung families, all deterministic — fixed seeds/magnitudes):
  (a) PARAMETRIC (same image perturbed, near-zero semantic distance):
        p0_noise   T + 0.02·N(0,1) (fixed gen seed), clip [0,1]  — the near-duplicate anchor
        p1_bright  clip(1.25·T + 0.05)                            — brightness/contrast shift
        p2_rot5    5°  rotation (affine_grid/grid_sample, bilinear, zero pad)
        p3_rot15   15° rotation
        p4_blur    gaussian blur σ=1.0 (5x5 kernel)
  (b) RETRIEVED (different images, ranked by ENCODER distance): embed a bank of --bank (~200)
      held-out same-DIGIT MNIST-test images (disjoint from D) + T with an image encoder;
      pick nearest-neighbour / median / farthest as rungs r_nn / r_med / r_far. Plus ONE
      cross-digit far anchor r_cross: a DIFFERENT digit of the SAME binary class (parity),
      so the label vector stays correct and NO label-noise confound enters (the far anchor
      is "as different as a valid same-label swap gets").
  ENCODER: lazy timm import (ViT-harness pattern) — try vit_small_patch16_224.dino (DINO);
  on create_model failure fall back to vit_tiny_patch16_224 (ImageNet, proven in the WEXAC
  rec env). embed = CLS of forward_features on _upscale(x28) (timm 0.9.12 ViT returns
  [B, tokens, dim] -> take [:, 0]); cosine distance. Embedding float32 on GPU; distances
  stored float64. The encoder actually used is PRINTED. Raw-pixel L2 ‖T−T'‖ is ALWAYS
  computed alongside as the simple check (both reported, both correlated).

PREDICTORS recorded per rung: d_pixel = ‖T−T'‖_2 (raw [0,1] pixels), d_encoder =
1 − cos(e_T, e_T'), and |Δg0| = |g0(T) − g0(T')| with g0 = per-image ‖∇_{W0} BCE‖_F at θ0
on the LoRA layer (margin_vs_sensitivity.layer0_grad_norms on ds_mean-centered input) —
the margin-lens predictor. Spearman(sens, ·) over rungs for each (n≈9 per target: SMALL-n).

KNOWN CONFOUNDS (flagged, not fixed):
  * p1_bright changes global image STATISTICS (mean brightness). The training input is
    ds_mean-CENTERED with no per-image renormalization, so a uniform brightness shift is a
    large L2 move along the mean-image direction — the net may amplify exactly that
    direction, making p1 loud out of proportion to its (tiny) semantic distance.
  * p2/p3 rotations + p4 blur (zero padding) slightly change edge statistics / total ink.
  * r_nn / r_med / r_far are SELECTED at the extremes of the encoder-distance distribution,
    so Spearman(sens, d_encoder) is partially selection-conditioned on those rungs.
  * DINO embeds 28x28 digits upscaled to 224 — distances of out-of-domain inputs may be
    compressed; d_pixel is the model-free cross-check.

bsub-only. Saves the FULL ladder images (T + every T') per target in .pth + a PNG grid so
the ladder can be SEEN later (CLAUDE.md experiment-output rules). float64 training.
"""
import os, json, math, argparse
import torch
import torch.nn.functional as F

from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.data_utils import get_finetuning_data, _load_dataset, _get_binary_label
from experiments.dataset_sensitivity.whitened_metric import whitened_sensitivity
from experiments.dataset_sensitivity.arm_b_dilution import draw_B0, train_adapter, subspace_cos
from experiments.dataset_sensitivity.margin_vs_sensitivity import layer0_grad_norms, spearman
from experiments.dataset_sensitivity.vit_lora_sensitivity import _upscale

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/similarity_ladder"
FIGURES = "/home/projects/galvardi/yoado/figures/similarity_ladder"
N_FOLDS = 5

STAGE0_RUNGS = ["p0_noise", "r_nn", "r_far"]      # near-duplicate + retrieved bracket only


# --------------------------------------------------------------------------- #
# parametric rungs (deterministic perturbations of T; all on CPU float64, [0,1])
# --------------------------------------------------------------------------- #
def _rotate(img, deg):
    """img [1,28,28] float64 in [0,1] -> rotated by `deg` (bilinear, zero padding)."""
    th = math.radians(deg)
    x = img.unsqueeze(0)                                    # [1,1,28,28]
    m = torch.tensor([[math.cos(th), -math.sin(th), 0.0],
                      [math.sin(th),  math.cos(th), 0.0]], dtype=x.dtype).unsqueeze(0)
    grid = F.affine_grid(m, list(x.shape), align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.squeeze(0).clamp(0.0, 1.0)


def _gauss_blur(img, sigma=1.0, k=5):
    """img [1,28,28] -> gaussian-blurred (k x k kernel, zero padding)."""
    ax = torch.arange(k, dtype=img.dtype) - (k - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    ker = (g[:, None] @ g[None, :]).view(1, 1, k, k)
    out = F.conv2d(img.unsqueeze(0), ker, padding=k // 2)
    return out.squeeze(0).clamp(0.0, 1.0)


def parametric_rungs(T_img):
    """Ordered [(name, T' [1,28,28])] — same image, perturbed at graded magnitude.
    All deterministic: fixed noise seed, fixed magnitudes."""
    g = torch.Generator().manual_seed(7)
    noise = torch.randn(T_img.shape, generator=g, dtype=T_img.dtype)
    return [
        ("p0_noise",  (T_img + 0.02 * noise).clamp(0.0, 1.0)),
        ("p1_bright", (1.25 * T_img + 0.05).clamp(0.0, 1.0)),
        ("p2_rot5",   _rotate(T_img, 5.0)),
        ("p3_rot15",  _rotate(T_img, 15.0)),
        ("p4_blur",   _gauss_blur(T_img, sigma=1.0)),
    ]


# --------------------------------------------------------------------------- #
# encoder (lazy timm import — ViT-harness pattern) + banks
# --------------------------------------------------------------------------- #
def load_encoder(device):
    """Try DINO ViT-S/16 first (the ask); fall back to ImageNet vit_tiny (proven in the
    WEXAC rec env, timm 0.9.12). Prints which encoder is actually used."""
    import timm
    last_err = None
    for name in ("vit_small_patch16_224.dino", "vit_tiny_patch16_224"):
        try:
            model = timm.create_model(name, pretrained=True, num_classes=0)
            model = model.eval().to(device).to(torch.float32)
            print(f"[encoder] using {name}", flush=True)
            return model, name
        except Exception as e:                              # noqa: BLE001 — fallback chain
            print(f"[encoder] {name} unavailable ({type(e).__name__}: {e}) — falling back",
                  flush=True)
            last_err = e
    raise RuntimeError(f"no image encoder available (both timm candidates failed): {last_err}")


@torch.no_grad()
def embed_images(model, x28, device, batch=32):
    """x28 [n,28,28] float in [0,1] -> [n, dim] float32 CPU embeddings (CLS of
    forward_features; timm 0.9.12 ViT returns [B, tokens, dim] -> take [:, 0])."""
    outs = []
    for i in range(0, x28.shape[0], batch):
        xb = _upscale(x28[i:i + batch].to(torch.float32)).to(device)
        f = model.forward_features(xb)
        if isinstance(f, dict):                             # some timm variants return dicts
            f = f.get("x_norm_clstoken", f.get("x"))
        if f.dim() == 3:                                    # [B, tokens, dim] -> CLS
            f = f[:, 0]
        outs.append(f.float().cpu())
    return torch.cat(outs, 0)


def cos_dist(e_ref, e_all):
    """1 − cosine(e_ref [dim], e_all [n,dim]) -> [n] float64."""
    a = e_ref.to(torch.float64)
    B = e_all.to(torch.float64)
    num = B @ a
    den = B.norm(dim=1) * a.norm() + 1e-30
    return 1.0 - num / den


def build_banks(digit_T, exclude_indices, bank_n, cross_n=60):
    """Held-out MNIST-TEST banks, disjoint from D (and from each other by construction):
      same_bank  [bank_n,1,28,28] — same DIGIT as T (the retrieval pool),
      cross_bank [cross_n,1,28,28] — DIFFERENT digit, SAME binary class (parity) — the
                 label-safe cross-digit far-anchor pool.
    Deterministic order (fixed permutation seed 777)."""
    ds = _load_dataset("mnist", train=False)
    rng = torch.Generator().manual_seed(777)
    perm = torch.randperm(len(ds), generator=rng)
    excl = set(int(i) for i in exclude_indices)
    parity_T = _get_binary_label(digit_T)
    same, cross = [], []
    for idx in perm.tolist():
        if idx in excl:
            continue
        if len(same) >= bank_n and len(cross) >= cross_n:
            break
        img, digit = ds[idx]
        digit = int(digit)
        if digit == digit_T and len(same) < bank_n:
            same.append(img.to(torch.float64))
        elif digit != digit_T and _get_binary_label(digit) == parity_T and len(cross) < cross_n:
            cross.append(img.to(torch.float64))
    assert len(same) >= min(bank_n, 20), f"same-digit bank starved ({len(same)}/{bank_n})"
    assert len(cross) >= 5, f"cross-digit bank starved ({len(cross)})"
    return torch.stack(same), torch.stack(cross)


def retrieved_rungs(encoder, T_img, same_bank, cross_bank, device):
    """Encoder-ranked retrieval rungs: [(name, T', d_encoder)] for r_nn / r_med / r_far
    (same digit) + r_cross (cross-digit same-parity, encoder-median of its pool)."""
    e_T = embed_images(encoder, T_img, device)[0]
    e_same = embed_images(encoder, same_bank.squeeze(1), device)
    e_cross = embed_images(encoder, cross_bank.squeeze(1), device)
    d_same = cos_dist(e_T, e_same)                          # [bank] float64
    d_cross = cos_dist(e_T, e_cross)
    order = torch.argsort(d_same)
    i_nn = int(order[0])
    i_med = int(order[len(order) // 2])
    i_far = int(order[-1])
    i_cross = int(torch.argsort(d_cross)[len(d_cross) // 2])   # median = representative, no outlier
    return [
        ("r_nn",    same_bank[i_nn],    float(d_same[i_nn])),
        ("r_med",   same_bank[i_med],   float(d_same[i_med])),
        ("r_far",   same_bank[i_far],   float(d_same[i_far])),
        ("r_cross", cross_bank[i_cross], float(d_cross[i_cross])),
    ], e_T


# --------------------------------------------------------------------------- #
# swap measurement for one rung (the arm-B/D pattern, baseline ensemble SHARED)
# --------------------------------------------------------------------------- #
def measure_rung(x_ft, y_ft, t_pos, T_prime, dW_base, frozen, b0, ds_mean,
                 lr, T, act, rank, out_f, device, subk, seed_tag):
    """Swap D[t_pos] -> T_prime and run the paired-per-seed whitened measurement against
    the SHARED baseline ensemble dW_base = {seed_j: ΔW(D, seed_j)} (trained once upstream).
    ds_mean FROZEN; label vector unchanged (every rung is a same-binary-class image)."""
    x_sw = x_ft.clone()
    x_sw[t_pos] = T_prime.to(x_ft.device, torch.float64)
    x0_sw = x_sw - ds_mean

    vs, vs_reseed, dropped = [], [], 0
    for s, dW_ref_s in dW_base.items():
        _, _, _, dW_sw_s = train_adapter(frozen, b0, draw_B0(s, out_f, rank, device),
                                         x0_sw, y_ft, lr, T, act, rank)
        v = dW_ref_s - dW_sw_s
        if torch.isfinite(v).all():
            vs.append(v); vs_reseed.append(dW_ref_s)        # keep v_j aligned with its reseed
        else:
            dropped += 1
    coherent = torch.stack(vs).mean(0).norm().item() if vs else float("nan")
    # known-init diagnostic on the FIRST finite pair: ΔW_swap = ΔW_ref − v (same seed, aligned)
    subcos = (subspace_cos(vs_reseed[0], vs_reseed[0] - vs[0], subk)
              if vs else float("nan"))

    sens = pval = d2obs = qeff = float("nan")
    if len(vs) >= 2 * N_FOLDS:
        ws = whitened_sensitivity([v.cpu() for v in vs], [r.cpu() for r in vs_reseed],
                                  n_folds=N_FOLDS, p_max=3, n_perm=500, seed=int(seed_tag))
        sens, pval, d2obs, qeff = ws["sensitivity"], ws["pvalue"], ws["d2_obs"], ws["qeff_count"]
        assert math.isfinite(d2obs), f"rung seed_tag={seed_tag}: d2_obs non-finite (metric broken)"
    return dict(sensitivity=sens, pvalue=pval, d2_obs=d2obs, qeff=qeff,
                coherent_signal=coherent, swap_subspace_cos=subcos,
                dropped=dropped, n_pairs=len(vs))


# --------------------------------------------------------------------------- #
# per-target ladder
# --------------------------------------------------------------------------- #
def run_target(tgt_id, t_pos, x_ft, y_ft, digits, indices, dW_base, frozen, b0, ds_mean,
               encoder, enc_name, lr, T, rank, bank_n, device, rung_filter=None, tag=""):
    act = make_activation("gelu")
    out_f = frozen[0].shape[0]
    subk = min(rank, 8)
    T_img = x_ft[t_pos].detach().cpu()                       # [1,28,28] float64, [0,1]
    digit_T = int(digits[t_pos])
    print(f"\n--- target {tgt_id}: D position {t_pos}, digit {digit_T} ---", flush=True)

    # ladder construction
    same_bank, cross_bank = build_banks(digit_T, indices, bank_n)
    ret, e_T = retrieved_rungs(encoder, T_img, same_bank, cross_bank, device)
    rungs = [(nm, tp, float("nan")) for nm, tp in parametric_rungs(T_img)] + ret
    # encoder distance for the parametric rungs too (they were built pixel-side)
    par_stack = torch.stack([tp for nm, tp, _ in rungs[:5]]).squeeze(1)   # [5,28,28]
    d_par = cos_dist(e_T, embed_images(encoder, par_stack, device))
    rungs = ([(nm, tp, float(d_par[i])) for i, (nm, tp, _) in enumerate(rungs[:5])]
             + rungs[5:])
    if rung_filter is not None:
        rungs = [r for r in rungs if r[0] in rung_filter]

    names = [r[0] for r in rungs]
    tp_stack = torch.stack([r[1] for r in rungs])            # [n_rungs,1,28,28] CPU float64
    d_enc = [r[2] for r in rungs]
    d_pix = [(T_img - r[1]).norm().item() for r in rungs]

    # margin-lens predictor |Δg0| (g0 at θ0, zero adapter, ds_mean-centered, label = T's label)
    y_T = y_ft[t_pos].detach().cpu()
    x0_all = torch.cat([T_img.unsqueeze(0), tp_stack], 0).to(device) - ds_mean
    y_all = y_T.repeat(x0_all.shape[0]).to(device)
    g0_all = layer0_grad_norms(x0_all, y_all, frozen, b0, act)
    g0_T = g0_all[0].item()
    dg0 = [abs(g0_all[1 + i].item() - g0_T) for i in range(len(rungs))]

    per_rung = []
    for i, (nm, tp, de) in enumerate(rungs):
        r = measure_rung(x_ft, y_ft, t_pos, tp, dW_base, frozen, b0, ds_mean,
                         lr, T, act, rank, out_f, device, subk,
                         seed_tag=100 * (tgt_id + 1) + i)
        r.update(rung=nm, d_pixel=d_pix[i], d_encoder=de, dg0=dg0[i])
        per_rung.append(r)
        print(f"[t{tgt_id} {nm:>9}] d_pix={d_pix[i]:.3f} d_enc={de:.4f} |Δg0|={dg0[i]:.3e} "
              f"sens={r['sensitivity']:.4g} p={r['pvalue']:.3f} qeff={r['qeff']} "
              f"(pairs={r['n_pairs']}, dropped={r['dropped']})", flush=True)

    sens_v = [r["sensitivity"] for r in per_rung]
    rho_enc, n_enc = spearman(sens_v, d_enc)
    rho_pix, n_pix = spearman(sens_v, d_pix)
    rho_g0, n_g0 = spearman(sens_v, dg0)
    print(f"[t{tgt_id}] Spearman(sens, d_encoder)={rho_enc:+.3f} (n={n_enc}, SMALL-n)  "
          f"(sens, d_pixel)={rho_pix:+.3f} (n={n_pix})  (sens, |Δg0|)={rho_g0:+.3f} (n={n_g0})",
          flush=True)

    res = dict(tgt_id=tgt_id, t_pos=int(t_pos), digit=digit_T, encoder=enc_name,
               g0_T=g0_T, rungs=names, per_rung=per_rung,
               spearman=dict(d_encoder=dict(rho=rho_enc, n=n_enc),
                             d_pixel=dict(rho=rho_pix, n=n_pix),
                             dg0=dict(rho=rho_g0, n=n_g0)))
    # save the VISIBLE ladder: T + every T' (CLAUDE.md: always save image tensors)
    os.makedirs(RESULTS, exist_ok=True)
    torch.save(dict(T_img=T_img, T_prime_stack=tp_stack, rung_names=names,
                    d_pixel=d_pix, d_encoder=d_enc, dg0=dg0,
                    sensitivity=sens_v, metrics=res, digit=digit_T, t_pos=int(t_pos)),
               os.path.join(RESULTS, f"ladder_t{tgt_id}{tag}.pth"))
    return res


def save_grid(all_targets, tag=""):
    """PNG grid: one row per target — [T | rung T's], annotated d_enc + sens.
    Best-effort (a plotting failure must not kill the metrics)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_rows = len(all_targets)
        n_cols = 1 + max(len(t["T_prime_stack"]) for t in all_targets)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 2.0 * n_rows),
                                 squeeze=False)
        for r, t in enumerate(all_targets):
            axes[r][0].imshow(t["T_img"].squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
            axes[r][0].set_title(f"T (digit {t['digit']})", fontsize=8)
            for c in range(len(t["T_prime_stack"])):
                ax = axes[r][c + 1]
                ax.imshow(t["T_prime_stack"][c].squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
                s = t["sensitivity"][c]
                ax.set_title(f"{t['rung_names'][c]}\nd={t['d_encoder'][c]:.3f} "
                             f"s={s:.2g}" if math.isfinite(s) else t["rung_names"][c],
                             fontsize=7)
            for c in range(n_cols):
                axes[r][c].axis("off")
        os.makedirs(FIGURES, exist_ok=True)
        out = os.path.join(FIGURES, f"similarity_ladder{tag}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[fig] saved {out}", flush=True)
    except Exception as e:                                   # noqa: BLE001 — best effort
        print(f"WARNING: ladder grid PNG failed ({type(e).__name__}: {e}) — "
              "tensors are still in the .pth files.", flush=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--n_targets", type=int, default=2, help="# distinct fixed T's (full ladder each)")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--bank", type=int, default=200, help="same-digit retrieval bank size")
    ap.add_argument("--stage0", action="store_true",
                    help="tiny sanity: N=12, K=12, 1 target, rungs = p0_noise + r_nn + r_far")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    if args.stage0:
        N, K, n_targets, rung_filter, tag = 12, 12, 1, STAGE0_RUNGS, "_stage0"
        print(f"=== STAGE-0 SANITY (N={N}, K={K}, 1 target, rungs={rung_filter}) ===", flush=True)
    else:
        N, K, n_targets, rung_filter, tag = args.N, args.K, args.n_targets, None, ""
    lr, T, rank, bank_n = args.lr, args.T, args.rank, args.bank

    # fixed private set D (arm-B/D construction) + honest θ0 + FROZEN ds_mean
    n_per_class = N // 2
    x_ft, y_ft, digits, indices = get_finetuning_data(n_per_class, seed=42, device=dev,
                                                      dataset="mnist")
    x_ft = x_ft.to(torch.float64); y_ft = y_ft.to(torch.float64)
    _, frozen, b0, _, ds_mean = _honest_target(x_ft, y_ft, T, rank, "gelu", lr, dev,
                                               "mnist", num_classes=2)
    x0 = x_ft - ds_mean
    out_f = frozen[0].shape[0]
    act = make_activation("gelu")

    # fixed target positions: first n_targets class-1 slots of D
    c1_pos = [i for i in range(N) if int(y_ft[i].item()) == 1]
    assert len(c1_pos) >= n_targets, f"need {n_targets} class-1 targets, have {len(c1_pos)}"
    target_pos = c1_pos[:n_targets]
    print(f"[setup] N={N} K={K} rank={rank} T={T} lr={lr}  targets at positions {target_pos} "
          f"(digits {[int(digits[p]) for p in target_pos]})", flush=True)

    # baseline reseed ensemble ΔW(D, seed_j) — trained ONCE, shared by ALL rungs and targets
    seeds = [1000 + j for j in range(K)]
    dW_base, mbce_ref = {}, None
    for s in seeds:
        _, _, mbce, dW = train_adapter(frozen, b0, draw_B0(s, out_f, rank, dev),
                                       x0, y_ft, lr, T, act, rank)
        if torch.isfinite(dW).all():
            dW_base[s] = dW
            if s == seeds[0]:
                mbce_ref = mbce
    base_dropped = K - len(dW_base)
    assert len(dW_base) >= 2 * N_FOLDS, \
        f"only {len(dW_base)} finite baseline draws (< 2*N_FOLDS={2 * N_FOLDS}); metric starved"
    stk = torch.stack(list(dW_base.values()))
    reseed_noise = ((stk - stk.mean(0)).flatten(1).norm(dim=1) ** 2).mean().sqrt().item()
    memorized = bool(mbce_ref is not None and mbce_ref < 1e-3)
    mbce_str = f"{mbce_ref:.2e}" if mbce_ref is not None else "n/a (seed0 dropped)"
    print(f"[baseline] {len(dW_base)}/{K} finite seeds (dropped {base_dropped}), "
          f"reseed_noise={reseed_noise:.4f}, max_bce(seed0)={mbce_str}, memorized={memorized}",
          flush=True)
    if not memorized:
        print("WARNING: baseline NOT memorized — off-convergence sensitivity is confounded.",
              flush=True)

    encoder, enc_name = load_encoder(dev)

    all_res, all_vis = [], []
    for tgt_id, t_pos in enumerate(target_pos):
        r = run_target(tgt_id, t_pos, x_ft, y_ft, digits, indices, dW_base, frozen, b0,
                       ds_mean, encoder, enc_name, lr, T, rank, bank_n, dev,
                       rung_filter=rung_filter, tag=tag)
        all_res.append(r)
        all_vis.append(torch.load(os.path.join(RESULTS, f"ladder_t{tgt_id}{tag}.pth"),
                                  weights_only=False))

    # pooled Spearman across all (target, rung) points — still small-n, labelled as such
    pool_s = [pr["sensitivity"] for r in all_res for pr in r["per_rung"]]
    pool_e = [pr["d_encoder"] for r in all_res for pr in r["per_rung"]]
    pool_p = [pr["d_pixel"] for r in all_res for pr in r["per_rung"]]
    pool_g = [pr["dg0"] for r in all_res for pr in r["per_rung"]]
    rho_e, n_e = spearman(pool_s, pool_e)
    rho_p, n_p = spearman(pool_s, pool_p)
    rho_g, n_g = spearman(pool_s, pool_g)

    save_grid(all_vis, tag=tag)
    os.makedirs(RESULTS, exist_ok=True)
    summary = dict(N=N, K=K, T=T, lr=lr, rank=rank, n_targets=n_targets, bank=bank_n,
                   encoder=enc_name, base_dropped=base_dropped, reseed_noise=reseed_noise,
                   ref_max_bce=mbce_ref, memorized=memorized,
                   target_pos=[int(p) for p in target_pos],
                   results=all_res,
                   pooled_spearman=dict(d_encoder=dict(rho=rho_e, n=n_e),
                                        d_pixel=dict(rho=rho_p, n=n_p),
                                        dg0=dict(rho=rho_g, n=n_g)))
    with open(os.path.join(RESULTS, f"similarity_ladder_summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {RESULTS}/similarity_ladder_summary{tag}.json", flush=True)

    if args.stage0:
        for r in all_res:
            for pr in r["per_rung"]:
                assert math.isfinite(pr["sensitivity"]), \
                    f"stage0 rung {pr['rung']}: sensitivity NaN (metric integration broken)"
                assert math.isfinite(pr["d_pixel"]) and math.isfinite(pr["d_encoder"]), \
                    f"stage0 rung {pr['rung']}: distance non-finite (encoder/pixel path broken)"
        assert reseed_noise > 0, "reseed_noise degenerate"
        print("STAGE-0 OK", flush=True)
        return

    # -------- READ block --------
    print("\n=== SUMMARY (per target: rung | d_enc | d_pix | |Δg0| | sens | p) ===", flush=True)
    for r in all_res:
        print(f"target {r['tgt_id']} (digit {r['digit']}):", flush=True)
        for pr in r["per_rung"]:
            print(f"  {pr['rung']:>9}: d_enc={pr['d_encoder']:.4f} d_pix={pr['d_pixel']:.3f} "
                  f"|Δg0|={pr['dg0']:.3e}  sens={pr['sensitivity']:.4g}  p={pr['pvalue']:.3f}",
                  flush=True)
        sp = r["spearman"]
        print(f"  Spearman: d_enc {sp['d_encoder']['rho']:+.3f}  d_pix {sp['d_pixel']['rho']:+.3f}  "
              f"|Δg0| {sp['dg0']['rho']:+.3f}  (n={sp['d_encoder']['n']}, SMALL-n)", flush=True)
    print(f"\nPOOLED Spearman (n={n_e}, SMALL-n): sens~d_encoder {rho_e:+.3f}  "
          f"sens~d_pixel {rho_p:+.3f}  sens~|Δg0| {rho_g:+.3f}", flush=True)
    print("READ: sens RISING with distance + near-duplicate rungs (p0_noise) ~null ⇒ the adapter "
          "records the CONCEPT, not the instance (a near-duplicate swap is invisible). FLAT-HIGH "
          "sens (even p0_noise fires) ⇒ INSTANCE-level memorization of the exact pixels. Compare "
          "the three pooled rhos to see which predictor (encoder distance / raw-pixel L2 / the "
          "margin-lens |Δg0|) best explains sensitivity. CAVEATS: n~9 rungs per target (small-n); "
          "p1_bright shifts global brightness, which ds_mean-centered inputs may amplify along the "
          "mean-image direction — read that rung with care.", flush=True)


if __name__ == "__main__":
    main()
