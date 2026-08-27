#!/usr/bin/env python
"""
ViT LoRA dataset-sensitivity MVP — is a SINGLE private image detectable in a real
ViT's LoRA adapter, the same way it is in the toy MLP?

This ports the dataset-sensitivity measurement (arm_b_dilution / arm_e_duplication)
from the D-1000-1000-1 MLP onto a real pretrained ViT (timm) fine-tuned with a
peft LoRA adapter. The metric (whitened_sensitivity) is architecture-agnostic: it
only ever sees a list of flattened ΔW=B·A tensors for ONE fixed target module plus
a reseed ensemble, so nothing about it changes between MLP and ViT — only the
producer of the ΔW vectors changes.

MEASUREMENT PATTERN (mirrors arm_b / arm_e exactly):
  * Frozen pretrained ViT (backbone + classification head frozen); ONLY the LoRA
    A/B factors on the attention qkv projection(s) are trainable.
  * ΔW is read from ONE fixed target module (default blocks.0.attn.qkv) as
    scaling·(lora_B @ lora_A). This is the gauge-invariant B·A product — never the
    raw A or B (those carry an arbitrary GL(r) gauge that cancels only in B·A).
  * SEEDED LoRA init → reseed ensemble. A "reseed" re-initializes the LoRA factors
    under a controlled torch RNG seed (kaiming_uniform on A, zeros on B — peft's
    own default init, but seeded) and refits. This is the ViT analogue of the MLP's
    draw_B0(seed) init ensemble: the seed drives the ONLY randomness in the init, so
    ΔW(seed) forms the seed-noise ensemble Σ whose covariance the metric whitens by.
  * Paired-per-seed diff, init CANCELS per pair:
        v_j = ΔW(D, seed_j) − ΔW(D_swap, seed_j)
    where D_swap replaces ONE private image by a held-out SAME-CLASS control (the
    ViT analogue of get_control_images_in_distribution). Because both members of the
    pair start from the identical seed_j init, the init contribution cancels and v_j
    isolates "swap one image → how does the adapter move".
  * reseed_list = ΔW(D, seed_j) ensemble.
  * whitened_sensitivity(v_list, reseed_list) → sensitivity, pvalue, qeff.
    sensitivity > 0 and pvalue < 0.05 ⇒ the single swapped image is DETECTABLE in
    the ViT's LoRA adapter above the training-seed noise floor.

PRECISION: the ViT forward/train runs in float32 (float64 on 86M/6M-param ViTs is
prohibitive); the ΔW vectors are cast to float64 before whitened_sensitivity (the
metric assumes double and does tiny float64 linear algebra internally).

bsub-only (WEXAC `rec` env: timm 0.9.12 + peft 0.7.1). CUDA. Binary task.
"""
import os
import math
import json
import argparse

import torch
import torch.nn.functional as F

from experiments.dataset_sensitivity.whitened_metric import whitened_sensitivity

RESULTS = "/home/projects/galvardi/yoado/results/vit_lora_sensitivity"
N_FOLDS = 5

# ImageNet normalization (matches phase0_vit_inversion.py convention)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# lazy imports (timm / peft live only in the WEXAC `rec` env, not necessarily
# in the shell that syntax-checks this file)
# --------------------------------------------------------------------------- #
def _lazy_imports():
    import timm
    import torchvision
    from peft import get_peft_model, LoraConfig
    return timm, torchvision, get_peft_model, LoraConfig


# --------------------------------------------------------------------------- #
# model + LoRA
# --------------------------------------------------------------------------- #
def build_vit_lora(model_name, lora_blocks, rank, num_classes, device):
    """Frozen pretrained ViT (timm) + peft LoRA on the qkv of `lora_blocks`.

    LoRA is attached to a small set of blocks for training realism, but the
    metric only ever reads ONE fixed module's ΔW (see get_delta_w). The backbone
    AND the classification head stay frozen — only the LoRA factors move, exactly
    like the single-LoRA-layer MLP measurement (target_layers=(0,)).
    """
    timm, _, get_peft_model, LoraConfig = _lazy_imports()
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device).to(torch.float32)
    # target one module per requested block: 'blocks.{i}.attn.qkv' (peft suffix-matches)
    target_modules = [f"blocks.{i}.attn.qkv" for i in lora_blocks]
    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,          # scaling = alpha/r = 1 (mirrors phase0)
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, cfg)
    # peft freezes everything except LoRA; the fresh num_classes head is frozen too.
    return model


def _lora_modules(model):
    """Yield (name, module) for every peft LoRA layer that carries a 'default' adapter."""
    for name, module in model.named_modules():
        la = getattr(module, "lora_A", None)
        if la is not None and "default" in la:
            yield name, module


def reinit_lora(model, seed):
    """Re-initialize ALL LoRA factors under a controlled torch RNG seed.

    peft's default init = kaiming_uniform(A), zeros(B). We reproduce it but seed it
    so the init is reproducible and varies ONLY with `seed` — the ViT analogue of
    the MLP's draw_B0(seed). B=0 at init ⇒ ΔW(init)=0; the seed drives the A init,
    hence the fine-tuning trajectory, hence ΔW after training. Pairing D and D_swap
    on the SAME seed makes the init cancel in v_j.
    """
    torch.manual_seed(int(seed))
    for _, m in _lora_modules(model):
        torch.nn.init.kaiming_uniform_(m.lora_A["default"].weight, a=math.sqrt(5))
        torch.nn.init.zeros_(m.lora_B["default"].weight)


def get_delta_w(model, target_module):
    """ΔW = scaling · (lora_B @ lora_A) for the ONE fixed target module, [out, in].

    Gauge-invariant B·A product only (never raw A/B). Detached, on-device float32.
    """
    for name, m in _lora_modules(model):
        if name.endswith(target_module):
            A = m.lora_A["default"].weight          # [r, in]
            B = m.lora_B["default"].weight          # [out, r]
            scaling = m.scaling["default"]
            return (scaling * (B @ A)).detach()
    raise ValueError(
        f"target_module {target_module!r} not found among LoRA modules "
        f"({[n for n, _ in _lora_modules(model)]}). "
        "Is it in --lora_blocks?")


# --------------------------------------------------------------------------- #
# data — MNIST upscaled to 3x224x224, binary two-class task
# --------------------------------------------------------------------------- #
def _upscale(x28):
    """[n,28,28] float in [0,1] -> [n,3,224,224] ImageNet-normalized float32."""
    x = x28.unsqueeze(1)                                     # [n,1,28,28]
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)                                 # [n,3,224,224]
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return ((x - mean) / std).to(torch.float32)


def build_data(N, classes, device):
    """Return (x_priv [N,3,224,224], y_priv [N], x_ctrl [N,...], y_ctrl [N]).

    x_priv[i] and x_ctrl[i] are DISTINCT images of the SAME class, so D_swap that
    replaces private image i by control i is an in-distribution same-class swap
    (the ViT analogue of get_control_images_in_distribution). per_class = N//2 of
    each of the two classes; controls are a disjoint held-out per_class of each.
    """
    _, torchvision, _, _ = _lazy_imports()
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    data = ds.data.to(torch.float32) / 255.0                # [60000,28,28]
    targets = ds.targets                                    # [60000]
    per_class = N // 2
    priv_x, priv_y, ctrl_x, ctrl_y = [], [], [], []
    for lbl, cls in enumerate(classes):                     # binary label lbl in {0,1}
        idx = (targets == cls).nonzero(as_tuple=True)[0]
        assert len(idx) >= 2 * per_class, \
            f"class {cls}: need {2*per_class} imgs, have {len(idx)}"
        priv_idx = idx[:per_class]
        ctrl_idx = idx[per_class:2 * per_class]             # disjoint, same class
        priv_x.append(data[priv_idx]); priv_y += [lbl] * per_class
        ctrl_x.append(data[ctrl_idx]); ctrl_y += [lbl] * per_class
    x_priv = _upscale(torch.cat(priv_x, 0)).to(device)
    x_ctrl = _upscale(torch.cat(ctrl_x, 0)).to(device)
    y_priv = torch.tensor(priv_y, dtype=torch.long, device=device)
    y_ctrl = torch.tensor(ctrl_y, dtype=torch.long, device=device)
    return x_priv, y_priv, x_ctrl, y_ctrl


# --------------------------------------------------------------------------- #
# fit + ΔW extraction for one (dataset, seed)
# --------------------------------------------------------------------------- #
def train_delta_w(model, x, y, seed, steps, lr, target_module):
    """Reinit LoRA at `seed`, fine-tune on (x,y), return (ΔW_target, final_loss).

    Full-batch AdamW on the LoRA factors, CrossEntropy over the 2 logits.
    """
    reinit_lora(model, seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    loss_val = float("nan")
    for _ in range(steps):
        opt.zero_grad()
        logits = model(x)                                   # [n, num_classes]
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        loss_val = loss.item()
    dW = get_delta_w(model, target_module)
    return dW, loss_val


# --------------------------------------------------------------------------- #
# main measurement
# --------------------------------------------------------------------------- #
def run(model_name, target_module, lora_blocks, N, K, steps, lr, rank,
        classes, n_targets, device, tag=""):
    print(f"[build] {model_name}  LoRA rank={rank} on blocks {lora_blocks} "
          f"(qkv)  read ΔW from {target_module}", flush=True)
    model = build_vit_lora(model_name, lora_blocks, rank, len(classes), device)
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[build] trainable LoRA params: {n_lora}", flush=True)

    x_priv, y_priv, x_ctrl, y_ctrl = build_data(N, classes, device)
    print(f"[data] N={N} private (classes {classes}), {N} same-class controls, "
          f"3x224x224", flush=True)

    seeds = [1000 + j for j in range(K)]

    # --- reseed ensemble: ΔW(D, seed_j) — the seed-noise ensemble Σ --------- #
    # Keep it seed-keyed so the paired diff below reuses ΔW(D, seed_j) exactly
    # (D is trained ONCE per seed here; the swap arm retrains only D_swap).
    reseed_by_seed = {}
    reseed_losses = []
    for s in seeds:
        dW, lo = train_delta_w(model, x_priv, y_priv, s, steps, lr, target_module)
        if torch.isfinite(dW).all():
            reseed_by_seed[s] = dW
            reseed_losses.append(lo)
    reseed_list = list(reseed_by_seed.values())
    reseed_dropped = K - len(reseed_list)
    fit_loss_mean = (sum(reseed_losses) / len(reseed_losses)) if reseed_losses else float("nan")
    print(f"[reseed] fitted D on {len(reseed_list)}/{K} seeds  "
          f"mean_final_loss={fit_loss_mean:.4e}  (dropped {reseed_dropped})", flush=True)

    mean = lambda a: (sum(a) / len(a)) if a else float("nan")
    all_sens, all_pval, all_qeff, all_d2obs = [], [], [], []
    per_target = []

    targets = list(range(min(n_targets, N)))
    for t in targets:
        # D_swap: replace private image t by its held-out SAME-CLASS control
        x_swap = x_priv.clone()
        x_swap[t] = x_ctrl[t]
        y_swap = y_priv  # label unchanged (same class)
        v_list, r_list = [], []
        for s, dW_D in reseed_by_seed.items():
            dW_swap, _ = train_delta_w(model, x_swap, y_swap, s, steps, lr, target_module)
            if not torch.isfinite(dW_swap).all():
                continue
            v = (dW_D - dW_swap)                            # init cancels (same seed)
            if torch.isfinite(v).all():
                v_list.append(v)
                r_list.append(dW_D)                         # seed-aligned reseed
        if len(v_list) < 2 * N_FOLDS:
            print(f"[target {t}] only {len(v_list)} finite pairs (< {2*N_FOLDS}); "
                  "skipping metric", flush=True)
            continue
        # metric wants float64 on CPU (matches its CPU self-test; ΔW cast to double)
        ws = whitened_sensitivity(
            [v.double().cpu() for v in v_list],
            [r.double().cpu() for r in r_list],
            n_folds=N_FOLDS, p_max=3, n_perm=500, seed=t)
        all_sens.append(ws["sensitivity"]); all_pval.append(ws["pvalue"])
        all_qeff.append(ws["qeff_count"]); all_d2obs.append(ws["d2_obs"])
        per_target.append(dict(target=t, sensitivity=ws["sensitivity"], pvalue=ws["pvalue"],
                               d2_obs=ws["d2_obs"], qeff_count=ws["qeff_count"],
                               n_pairs=len(v_list)))
        print(f"[target {t}] sensitivity={ws['sensitivity']:.4g}  "
              f"pvalue={ws['pvalue']:.4g}  qeff={ws['qeff_count']}  "
              f"d2_obs={ws['d2_obs']:.4g}  (n_pairs={len(v_list)})", flush=True)

    detectable = (math.isfinite(mean(all_sens)) and mean(all_sens) > 0
                  and mean(all_pval) < 0.05)
    result = dict(
        model=model_name, target_module=target_module, lora_blocks=lora_blocks,
        rank=rank, N=N, K=K, steps=steps, lr=lr, classes=list(classes),
        n_targets=len(targets), n_lora_params=n_lora,
        reseed_dropped=reseed_dropped, fit_loss_mean=fit_loss_mean,
        delta_w_shape=list(reseed_list[0].shape) if reseed_list else None,
        sensitivity=mean(all_sens), pvalue=mean(all_pval),
        qeff_count=mean(all_qeff), d2_obs=mean(all_d2obs),
        detectable=bool(detectable), per_target=per_target,
    )

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"vit_lora_{model_name}_r{rank}_N{N}{tag}.pth")
    torch.save(dict(
        metrics=result,
        reseed_mean=(torch.stack(reseed_list).mean(0).cpu() if reseed_list else None),
        reseed_stack=torch.stack(reseed_list).cpu() if reseed_list else None,
    ), out)
    print(f"[save] {out}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_tiny_patch16_224")
    ap.add_argument("--target_module", default="blocks.0.attn.qkv",
                    help="the ONE fixed module whose ΔW=B·A is the leakage vector")
    ap.add_argument("--lora_blocks", type=int, nargs="+", default=[0, 1, 2],
                    help="blocks whose attn.qkv get a LoRA adapter (training realism)")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--classes", type=int, nargs=2, default=[3, 8])
    ap.add_argument("--n_targets", type=int, default=1)
    ap.add_argument("--stage0", action="store_true",
                    help="tiny sanity: N=6, K=10, steps=100, 1 target")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # keep the metric's tiny float64 ops from oversubscribing the fat node
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    if args.stage0:
        print("=== STAGE-0 SANITY (N=6, K=10, steps=100, 1 target) ===", flush=True)
        r = run(args.model, args.target_module, args.lora_blocks,
                N=6, K=10, steps=100, lr=args.lr, rank=args.rank,
                classes=tuple(args.classes), n_targets=1, device=args.device,
                tag="_stage0")
        print(json.dumps(r, indent=2), flush=True)
        assert r["delta_w_shape"] is not None, "no finite ΔW produced (fit/extract broken)"
        assert math.isfinite(r["fit_loss_mean"]), "fit loss NaN (training broken)"
        assert math.isfinite(r["sensitivity"]), \
            "sensitivity NaN (metric integration broken / too many dropped pairs)"
        print("STAGE-0 OK", flush=True)
        return

    r = run(args.model, args.target_module, args.lora_blocks,
            args.N, args.K, args.steps, args.lr, args.rank,
            tuple(args.classes), args.n_targets, args.device)
    print("\n=== RESULT ===", flush=True)
    print(f"sensitivity={r['sensitivity']:.4g}  pvalue={r['pvalue']:.4g}  "
          f"qeff={r['qeff_count']:.2f}  fit_loss={r['fit_loss_mean']:.3e}", flush=True)
    print(f"single private image DETECTABLE in the ViT LoRA adapter: {r['detectable']}",
          flush=True)


if __name__ == "__main__":
    main()
