#!/usr/bin/env python3
"""A REAL membership baseline for the head claim: shadow-model / LiRA, not a loss threshold.

A loss threshold is the weakest comparator in the membership-inference literature, so "the certificate beats the
baseline" would read as "beats a straw man" and a referee would say so immediately. This measures the certificate
against a proper likelihood-ratio attack (Carlini et al.'s LiRA in its offline/online form) on the same releases.

**Why this is cheap here, which is the only reason it is affordable at all.** The encoder is FROZEN: only the head
adapter trains. So the pool's CLS features are computed ONCE with a single forward pass, and every shadow release
is head-only training on cached features -- linear algebra on an (N x 768) matrix rather than a transformer
forward. 128 shadows cost less than one of today's Jacobian runs.

Protocol, standard: for each shadow s, sample half the pool IN and train a head LoRA on it; record for every pool
image the logit-scaled confidence phi = log p_y - log(1 - p_y). Per image this gives an IN distribution and an OUT
distribution across shadows; fit Gaussians to each. For the TARGET release, score each image by the likelihood
ratio of its observed phi under IN vs OUT. That is LiRA. Compare, on the same target release and the same
members/non-members:
    * the certificate residual  ||C h|| / ||A_T h||   -- recipe-free, one forward pass, no shadows;
    * LiRA                                             -- requires the recipe, the data distribution and S shadows;
    * the loss threshold                               -- the straw man, reported for continuity with earlier rows.
The comparison that matters is not only AUC but COST: the certificate needs the release and the public model; LiRA
needs a shadow-training budget and a distributional assumption the certificate never makes. Report both.

  python -u -m experiments.exact_inversion.head_lira --shadows 128 --T 100
"""
import argparse, json, math, os, socket, sys, time
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.vit_token_span import load_images
from experiments.exact_inversion.graded_imprint import auc

torch.set_default_dtype(torch.float64)


def train_head_lora(F, y, Whead, r, T, lr, seed, classes):
    """Head-only LoRA on CACHED features F (n, d). Identical recipe to graded_imprint's head arm."""
    d = F.shape[1]
    g = torch.Generator().manual_seed(seed)
    A = ((torch.randn(r, d, generator=g) / math.sqrt(d)).to(F.device)).requires_grad_(True)
    B = torch.zeros(classes, r, device=F.device, requires_grad=True)
    for _ in range(T):
        z = F @ Whead.T + torch.nn.functional.linear(F, A) @ B.T
        loss = torch.nn.functional.cross_entropy(z, y)
        gA, gB = torch.autograd.grad(loss, [A, B])
        A = (A - lr * gA).detach().requires_grad_(True)
        B = (B - lr * gB).detach().requires_grad_(True)
    return A.detach(), B.detach()


def phi_of(F, y_all, Whead, A, B):
    """Logit-scaled confidence log p_y - log(1-p_y), the LiRA statistic."""
    with torch.no_grad():
        z = F @ Whead.T + torch.nn.functional.linear(F, A) @ B.T
        lp = torch.log_softmax(z, dim=1)
        py = lp.gather(1, y_all[:, None]).squeeze(1).exp().clamp(1e-12, 1 - 1e-12)
        return (py.log() - (1 - py).log())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    ap.add_argument("--data-root", default="dataset_reconstruction/data/flowers-102/jpg")
    ap.add_argument("--pool", type=int, default=256, help="images the shadow attacker draws from")
    ap.add_argument("--N", type=int, default=32); ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--shadows", type=int, default=128)
    ap.add_argument("--Ts", nargs="*", type=int, default=[5, 20, 50, 100, 200, 400])
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--classes", type=int, default=102); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    import timm
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = timm.create_model(a.model, pretrained=True, num_classes=0).to(dev).double().eval()
    for p in model.parameters(): p.requires_grad_(False)

    X, _ = load_images(a.data_root, a.pool, 224, dev)
    with torch.no_grad():                                              # ONE forward pass; the encoder is frozen
        F = torch.cat([model(X[i:i + 16]) for i in range(0, X.shape[0], 16)])
    d = F.shape[1]
    gh = torch.Generator().manual_seed(a.seed)
    Whead = (torch.randn(a.classes, d, generator=gh) / math.sqrt(d)).to(dev)
    y_all = (torch.arange(a.pool, device=dev) % a.classes)
    print(f"# head LiRA: pool {a.pool} cached features ({d}-d), {a.shadows} shadows, N={a.N}, r={a.r}  "
          f"git={git_hash()} host={socket.gethostname()}", flush=True)

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    gsel = torch.Generator().manual_seed(a.seed + 5)
    for T in a.Ts:
        t0 = time.time()
        phis, masks = [], []
        for s in range(a.shadows):
            idx = torch.randperm(a.pool, generator=gsel)[:a.N].to(dev)
            m = torch.zeros(a.pool, dtype=torch.bool, device=dev); m[idx] = True
            A, B = train_head_lora(F[idx], y_all[idx], Whead, a.r, T, a.lr, a.seed + 1000 + s, a.classes)
            phis.append(phi_of(F, y_all, Whead, A, B)); masks.append(m)
        P = torch.stack(phis); M = torch.stack(masks)                  # (S, pool)

        # the TARGET release: a fresh draw, disjoint seed from every shadow
        idx_t = torch.randperm(a.pool, generator=torch.Generator().manual_seed(a.seed + 99))[:a.N].to(dev)
        mem = torch.zeros(a.pool, dtype=torch.bool, device=dev); mem[idx_t] = True
        A_T, B_T = train_head_lora(F[idx_t], y_all[idx_t], Whead, a.r, T, a.lr, a.seed + 7, a.classes)
        phi_t = phi_of(F, y_all, Whead, A_T, B_T)

        # LiRA: per-image Gaussians over shadows, likelihood ratio at the target's observed statistic
        lira = torch.zeros(a.pool, device=dev)
        for i in range(a.pool):
            pin, pout = P[M[:, i], i], P[~M[:, i], i]
            if pin.numel() < 2 or pout.numel() < 2: lira[i] = 0.0; continue
            mi, si = pin.mean(), pin.std().clamp_min(1e-6)
            mo, so = pout.mean(), pout.std().clamp_min(1e-6)
            lira[i] = (torch.distributions.Normal(mi, si).log_prob(phi_t[i])
                       - torch.distributions.Normal(mo, so).log_prob(phi_t[i]))

        # the certificate on the same target release, and the loss threshold for continuity
        U, S, Vh = torch.linalg.svd(B_T, full_matrices=False)
        rank_B = int((S > 1e-12 * S[0]).sum()) if float(S[0]) > 0 else 0
        Q = Vh[:rank_B].T
        C = A_T - Q @ (Q.T @ A_T)
        rank_C = int((torch.linalg.svdvals(C) > 1e-10 * float(torch.linalg.svdvals(A_T)[0])).sum())
        with torch.no_grad():
            cert = (torch.linalg.norm(F @ C.T, dim=1) / (torch.linalg.norm(F @ A_T.T, dim=1) + 1e-300))
            z = F @ Whead.T + torch.nn.functional.linear(F, A_T) @ B_T.T
            lossv = -torch.nn.functional.cross_entropy(z, y_all, reduction="none")
        mi = mem.tolist()
        pos = lambda v: [float(v[i]) for i in range(a.pool) if mi[i]]
        neg = lambda v: [float(v[i]) for i in range(a.pool) if not mi[i]]
        a_cert = auc([-x for x in pos(cert)], [-x for x in neg(cert)]) if rank_C else float("nan")
        a_lira = auc(pos(lira), neg(lira)); a_loss = auc(pos(lossv), neg(lossv))
        emit(dict(part="LIRA", T=T, N=a.N, r=a.r, pool=a.pool, shadows=a.shadows,
                  rank_B_T=rank_B, rank_C=rank_C, certificate_vacuous=bool(rank_C == 0),
                  auc_certificate=a_cert, auc_lira=a_lira, auc_loss_threshold=a_loss,
                  cert_beats_lira=bool(a_cert == a_cert and a_cert > a_lira),
                  cost_certificate="the release + the public model; one forward pass; no shadows, no recipe, "
                                   "no distributional assumption",
                  cost_lira=f"{a.shadows} shadow releases + the recipe + a sample from the data distribution",
                  seconds=time.time() - t0, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))
        print(f"  T={T:4d}  rank C={rank_C}  AUC cert {a_cert:.3f}   AUC LiRA {a_lira:.3f}   "
              f"AUC loss {a_loss:.3f}   ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
