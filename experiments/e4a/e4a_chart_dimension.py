#!/usr/bin/env python3
"""E4a — how many dimensions does a public chart of FEATURES need? Three regimes, two backbones.

Gal's realism objection operationalised on the feature side. In pixel space private images sit 0.25 off a
32-dimensional public PCA chart while blends of privates sit within 0.01 (Lemma 15's signature). Feature space
should need far fewer dimensions; this measures how many.

  backbones   DINO ViT-B/16 and CLIP ViT-L/14 (timm), frozen, embeddings only
  regimes     universal        -- chart fitted on RANDOM public images
              target           -- chart fitted on public images of the PRIVATE CATEGORY, subjects disjoint
              shared_concept   -- concept means (k_shared) + within-concept PCA (k_nuisance), reported separately
  charts      PCA; a linear autoencoder ASSERTED against PCA; a nonlinear autoencoder
  k           8, 16, 32, 64, 128
  measured    projection residual of PRIVATE features, and of BLENDS of privates, and the k at which the privates
              come within ~2 percent

HOLD-OUT IS BY SUBJECT, never by adapter (rule 12): the public pool and the private set never share a class.

  python -m experiments.e4a.e4a_chart_dimension --backbone dino
"""
import argparse, json, math, os, pickle, socket, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_default_dtype(torch.float32)
KS = [8, 16, 32, 64, 128]


def log(s): print(s, flush=True)


def load_cifar100(root="data"):
    out = {}
    for split in ("train", "test"):
        b = pickle.load(open(os.path.join(root, "cifar-100-python", split), "rb"), encoding="bytes")
        out[split] = (b[b"data"].astype(np.float32) / 255.0, np.array(b[b"fine_labels"]))
    meta = pickle.load(open(os.path.join(root, "cifar-100-python", "meta"), "rb"), encoding="bytes")
    return out, [n.decode() for n in meta[b"fine_label_names"]]


def embed(net, X, dev, bs=64, size=224):
    """X: (n, 3072) in [0,1] -> embeddings. Upsampled to the backbone's input size; this is a FEATURE experiment."""
    out = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs], device=dev).reshape(-1, 3, 32, 32)
            xb = F.interpolate(xb, size=(size, size), mode="bilinear", align_corners=False)
            out.append(net((xb - mean) / std).float().cpu())
    return torch.cat(out)


def pca_chart(Z, k):
    mu = Z.mean(0); U, S, Vh = torch.linalg.svd(Z - mu, full_matrices=False)
    V = Vh[:k].T.contiguous()
    return lambda X: mu + (X - mu) @ V @ V.T


def linear_ae(Z, k, epochs=200, lr=1e-3):
    """A linear autoencoder. ASSERTED against PCA: with a linear decoder the optimum spans the top-k subspace."""
    d = Z.shape[1]; mu = Z.mean(0); Zc = (Z - mu)
    enc = nn.Linear(d, k, bias=False); dec = nn.Linear(k, d, bias=False)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr)
    for _ in range(epochs):
        for i in torch.randperm(len(Zc)).split(256):
            xb = Zc[i]; opt.zero_grad(); F.mse_loss(dec(enc(xb)), xb).backward(); opt.step()
    return lambda X: mu + dec(enc(X - mu))


def nonlinear_ae(Z, k, epochs=300, lr=1e-3, h=1024):
    d = Z.shape[1]; mu = Z.mean(0); sd = Z.std(0).clamp(min=1e-6); Zc = (Z - mu) / sd
    enc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, k))
    dec = nn.Sequential(nn.Linear(k, h), nn.GELU(), nn.Linear(h, d))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr)
    for _ in range(epochs):
        for i in torch.randperm(len(Zc)).split(256):
            xb = Zc[i]; opt.zero_grad(); F.mse_loss(dec(enc(xb)), xb).backward(); opt.step()
    return lambda X: mu + dec(enc((X - mu) / sd)) * sd


def resid(f, X):
    with torch.no_grad(): R = f(X)
    return (torch.linalg.norm(R - X, dim=1) / torch.linalg.norm(X, dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["dino", "clip"], required=True)
    ap.add_argument("--private-class", default="motorcycle"); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--n-public", type=int, default=2000); ap.add_argument("--n-concepts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="results/e4a")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(a.out_dir, exist_ok=True)
    import timm
    name = "vit_base_patch16_224.dino" if a.backbone == "dino" else "vit_large_patch14_clip_224.openai"
    net = timm.create_model(name, pretrained=True, num_classes=0).to(dev).eval()
    for p in net.parameters(): p.requires_grad_(False)
    size = 224
    log(f"# E4a  backbone={name}  device={dev}")

    data, names = load_cifar100()
    ci = names.index(a.private_class)
    Xtr, ytr = data["train"]; Xte, yte = data["test"]
    g = np.random.RandomState(a.seed)
    priv_idx = g.permutation(np.where(yte == ci)[0])[: a.N]
    X_priv = Xte[priv_idx]
    # HOLD-OUT BY SUBJECT: the universal pool excludes the private class entirely
    other = np.where(ytr != ci)[0]; uni_idx = g.permutation(other)[: a.n_public]
    tgt_idx = g.permutation(np.where(ytr == ci)[0])[: a.n_public]
    log(f"# private class '{a.private_class}' ({a.N} held-out test images); universal pool {len(uni_idx)} images "
        f"EXCLUDING that class; target pool {len(tgt_idx)} public images OF that class (train split, disjoint from privates)")

    t0 = time.time()
    Zp = embed(net, X_priv, dev, size=size)
    Zu = embed(net, Xtr[uni_idx], dev, size=size)
    Zt = embed(net, Xtr[tgt_idx], dev, size=size)
    log(f"# embeddings: private {tuple(Zp.shape)}, universal {tuple(Zu.shape)}, target {tuple(Zt.shape)}  [{time.time()-t0:.0f}s]")

    gg = torch.Generator().manual_seed(a.seed)
    # 64 blends of the N privates, one per COLUMN, each column of coefficients summing to 1
    Wb = torch.rand(a.N, 64, generator=gg); Wb = Wb / Wb.sum(0, keepdim=True)
    Zb = (Zp.T @ Wb).T                                                            # (64, d)
    assert Zb.shape == (64, Zp.shape[1]), Zb.shape
    assert torch.allclose(Wb.sum(0), torch.ones(64), atol=1e-6)

    rows = []
    for regime, Zfit in (("universal", Zu), ("target", Zt)):
        for cname, ctor in (("pca", pca_chart), ("linear_ae", linear_ae), ("nonlinear_ae", nonlinear_ae)):
            for k in KS:
                if k >= min(Zfit.shape): continue
                f = ctor(Zfit, k)
                rp, rb = resid(f, Zp), resid(f, Zb)
                r = dict(part="E4a", backbone=a.backbone, model=name, regime=regime, chart=cname, k=k,
                         private_resid_mean=float(rp.mean()), private_resid_max=float(rp.max()),
                         blend_resid_mean=float(rb.mean()), N=a.N, private_class=a.private_class,
                         n_fit=int(len(Zfit)), d=int(Zp.shape[1]))
                rows.append(r); log(f"   {regime:10s} {cname:12s} k={k:<4d} private {r['private_resid_mean']:.4f}  blend {r['blend_resid_mean']:.4f}")
    # shared-concept: concept means (k_shared) + within-concept PCA (k_nuisance), reported separately
    cls = [c for c in np.unique(ytr) if c != ci][: a.n_concepts]
    means = torch.stack([embed(net, Xtr[g.permutation(np.where(ytr == c)[0])[:64]], dev, size=size).mean(0) for c in cls])
    U, S, Vh = torch.linalg.svd(means - means.mean(0), full_matrices=False)
    ev = (S ** 2).cumsum(0) / (S ** 2).sum()
    k_shared = int((ev < 0.95).sum()) + 1
    Zt_c = Zt - Zt.mean(0)
    Uw, Sw, _ = torch.linalg.svd(Zt_c, full_matrices=False)
    evw = (Sw ** 2).cumsum(0) / (Sw ** 2).sum()
    k_nuis = int((evw < 0.95).sum()) + 1
    rows.append(dict(part="E4a", backbone=a.backbone, model=name, regime="shared_concept", chart="concept_means+within_pca",
                     k_shared=k_shared, k_nuisance=k_nuis, n_concepts=len(cls),
                     note="k_shared = concept-mean directions to 95% variance; k_nuisance = within-target-class directions to 95%",
                     d=int(Zp.shape[1])))
    log(f"   shared_concept: k_shared={k_shared} (over {len(cls)} concepts), k_nuisance={k_nuis}")
    with open(os.path.join(a.out_dir, "rows.jsonl"), "a") as f:
        for r in rows: f.write(json.dumps(dict(r, seed=a.seed, host=socket.gethostname(), cmd=" ".join(sys.argv))) + "\n")
    torch.save(dict(Zp=Zp, Zu=Zu, Zt=Zt, Zb=Zb, rows=rows), os.path.join(a.out_dir, f"e4a_{a.backbone}.pth"))
    log(f"# wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
