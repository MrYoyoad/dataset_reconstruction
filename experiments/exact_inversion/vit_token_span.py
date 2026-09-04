#!/usr/bin/env python3
"""Does the recipe-free certificate exist on a REAL transformer? The token-span saturation check.

Today's convolutional result said weight sharing decides whether a certificate exists at all: a shared kernel is
applied at every spatial position, so one image contributes P vectors to the recorded span instead of one, and the
certificate is vacuous wherever those vectors fill the layer's input dimension. A transformer shares every linear
inside a block across TOKENS in exactly the same way. With N images and T tokens the recorded span is fed N.T
vectors into a d-dimensional input, and for ViT-B/16 at N = 8 that is 8 x 197 = 1576 against d = 768.

So the prediction that matters for any real deployment is immediate, and it is measured here on the pretrained
weights rather than argued: **wherever the token span fills d, the certificate is vacuous at every adapter rank,
and the recipe-free channel does not exist on that module.** The counter-case is equally concrete — token
activations in a trained transformer are famously redundant, so the span may sit far below both N.T and d, in
which case the margin is set by the number of DISTINCT token directions and the channel survives.

Measured, per candidate adapted module (attention qkv, attention proj, MLP fc1, MLP fc2) and per block:
  * d, tokens, N.T, and the measured rank of the stacked token matrix at real images;
  * whether that rank reaches d (vacuous at every rank) or falls short (margin = min(r, d) - span for each r);
  * the same against N, to locate the crossover -- the largest private batch for which a certificate still exists.
Everything is measured on the FROZEN pretrained model, before any adapter is trained, so the answer depends on
neither the recipe nor the release. Rows are algebraic checks on real weights and real inputs, not attacks.

  python -u -m experiments.exact_inversion.vit_token_span --model vit_base_patch16_224.augreg2_in21k_ft_in1k
"""
import argparse, glob, json, os, socket, sys, time
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash

torch.set_default_dtype(torch.float64)


def load_images(root, n, size, dev):
    """Real photographs at native resolution (flowers-102), centre-cropped -- NOT upsampled CIFAR, which would
       manufacture token redundancy and bias every rank below toward the optimistic answer."""
    from PIL import Image
    import torchvision.transforms as T
    files = sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True))[:2000]
    if len(files) < n:
        raise SystemExit(f"only {len(files)} images under {root}; need {n}")
    step = max(1, len(files) // n)
    tf = T.Compose([T.Resize(size + 32), T.CenterCrop(size), T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    xs = [tf(Image.open(f).convert("RGB")) for f in files[::step][:n]]
    return torch.stack(xs).to(dev).double(), files[::step][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    ap.add_argument("--data-root", default="dataset_reconstruction/data/flowers-102")
    ap.add_argument("--Ns", nargs="*", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--ranks", nargs="*", type=int, default=[8, 16, 64, 256])
    ap.add_argument("--blocks", nargs="*", type=int, default=None, help="default: first, middle, last")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    import timm
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = timm.create_model(a.model, pretrained=True).to(dev).double().eval()
    nblocks = len(model.blocks)
    blocks = a.blocks or sorted({0, 1, nblocks // 2, nblocks - 2, nblocks - 1})

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    Nmax = max(a.Ns)
    X, files = load_images(a.data_root, Nmax, a.size, dev)
    print(f"# {a.model}: {nblocks} blocks, blocks measured {blocks}, {Nmax} real images from {a.data_root} "
          f"git={git_hash()} host={socket.gethostname()} dev={dev}", flush=True)

    # capture the INPUT to each candidate adapted linear -- that is the vector the certificate condition acts on
    caps = {}
    hooks = []
    def mk(name):
        def h(mod, inp, out): caps[name] = inp[0].detach()
        return h
    for b in blocks:
        blk = model.blocks[b]
        for name, mod in (("attn.qkv", blk.attn.qkv), ("attn.proj", blk.attn.proj),
                          ("mlp.fc1", blk.mlp.fc1), ("mlp.fc2", blk.mlp.fc2)):
            hooks.append(mod.register_forward_hook(mk(f"block{b}.{name}")))
    with torch.no_grad():
        model(X)
    for h in hooks: h.remove()

    for name in sorted(caps, key=lambda s: (int(s.split(".")[0][5:]), s)):
        H = caps[name]                                        # (N, tokens, d)
        Nfull, tokens, d = H.shape
        for N in a.Ns:
            if N > Nfull: continue
            M = H[:N].reshape(N * tokens, d).T.contiguous()   # (d, N*tokens)
            sv = torch.linalg.svdvals(M)
            rk = int((sv > 1e-10 * sv[0]).sum()) if float(sv[0]) > 0 else 0
            # an "effective" rank too: directions carrying more than 1e-6 of the top singular value, because a
            # numerically-present direction the release cannot resolve is not a direction the defender loses.
            rk_eff = int((sv > 1e-6 * sv[0]).sum()) if float(sv[0]) > 0 else 0
            margins = {str(r): max(0, min(r, d) - rk) for r in a.ranks}
            emit(dict(part="TOKEN_SPAN", module=name, model=a.model, d=d, tokens=tokens, N=N,
                      token_vectors=N * tokens, span_rank=rk, span_rank_eff_1e6=rk_eff,
                      spans_input_dim=bool(rk >= d), margin_by_rank=margins,
                      vacuous_at_every_tested_rank=bool(all(v == 0 for v in margins.values())),
                      note="a shared linear records one vector per TOKEN per image; where the span fills d the "
                           "certificate is the zero matrix at every adapter rank",
                      start_model="n/a (frozen pretrained weights, real inputs)", claim_class="algebraic check",
                      git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))
        H8 = H[:min(8, Nfull)]
        M = H8.reshape(H8.shape[0] * tokens, d).T
        sv = torch.linalg.svdvals(M)
        rk = int((sv > 1e-10 * sv[0]).sum()) if float(sv[0]) > 0 else 0
        print(f"  {name:22s} d={d:5d} tokens={tokens} N=8 -> {8*tokens:6d} vectors  span rank {rk:5d}"
              f"  {'SPANS d (vacuous at every rank)' if rk >= d else f'deficient by {d-rk}'}", flush=True)


if __name__ == "__main__":
    main()
