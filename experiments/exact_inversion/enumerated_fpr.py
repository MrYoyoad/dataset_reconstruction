#!/usr/bin/env python3
"""Measure the ENUMERATED attack directly instead of union-bounding it (yoado-cd).

An attacker facing an augmentation they cannot invert tests every element of the group and takes the best score.
Bounding that by `1 - (1-p)^|G|` assumes the candidates fail INDEPENDENTLY, and they do not: an image and its
mirror, or two nearby crops, have correlated activations and therefore correlated scores. The bound over-counts,
possibly by a lot.

The direct measurement avoids the question. For a NON-MEMBER image, generate the attacker's full candidate set,
score all of them, and count how often the MINIMUM crosses the bar. That is the actual attacker procedure and its
actual error rate -- no bound, no independence assumption.

PRE-REGISTERED: the measured enumerated rate comes in BELOW the union bound at every group size, and the gap
widens with the correlation between candidates -- which is itself worth reporting, since it says how much
structure the augmentation group shares.

Also reports the single-candidate rate on the LARGEST available non-member population, because the previous bound
(3e-3) was set by 1,000 non-members and "a realistic stack defeats the test" must not stand on a sample size.

  python -u -m experiments.exact_inversion.enumerated_fpr --nonmembers 8000 --enum-sample 1500
"""
import argparse, glob, json, math, os, socket, sys, time
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18"); ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--data-root", default="dataset_reconstruction/data/flowers-102/jpg")
    ap.add_argument("--r", type=int, default=64); ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--classes", type=int, default=102)
    ap.add_argument("--nonmembers", type=int, default=8000,
                    help="single-candidate population; the rule-of-three bound is 3/n, so this sets how tight the "
                         "per-candidate rate can be stated at all")
    ap.add_argument("--enum-sample", type=int, default=1500,
                    help="non-members carried through the FULL candidate set (|G| forwards each)")
    ap.add_argument("--bar", type=float, default=1e-2); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    import torchvision.models as tvm, torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from PIL import Image
    w = tvm.ResNet18_Weights.IMAGENET1K_V1
    net = tvm.resnet18(weights=w).to(dev).double().eval()
    for p in net.parameters(): p.requires_grad_(False)
    conv = getattr(net, f"layer{a.stage}")[0].conv1
    k_, st_, pd_ = conv.kernel_size, conv.stride, conv.padding
    d_in, d_out = conv.in_channels * k_[0] * k_[1], conv.out_channels
    feat = net.fc.in_features; net.fc = torch.nn.Identity()
    gh = torch.Generator().manual_seed(a.seed)
    Whead = (torch.randn(a.classes, feat, generator=gh) / math.sqrt(feat)).to(dev)

    files = sorted(glob.glob(os.path.join(a.data_root, "*.jpg")))
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    n_load = min(a.nonmembers + 1, len(files))
    print(f"# enumerated FPR: {len(files)} images available, loading {n_load}  (the population is capped by the "
          f"DATASET, not by choice: the rule-of-three bound is 3/n)  git={git_hash()} host={socket.gethostname()}",
          flush=True)
    imgs = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[:n_load]]).to(dev).double()
    member, nonmembers = imgs[:1], imgs[1:]

    caps = {}
    hk = conv.register_forward_hook(lambda m, i, o: caps.__setitem__("p", torch.nn.functional.unfold(
        i[0].detach(), k_, dilation=m.dilation, padding=pd_, stride=st_)))

    def patches(X, bs=25):
        out = []
        for i in range(0, X.shape[0], bs):
            with torch.no_grad():
                net(X[i:i + bs]); out.append(caps["p"])
        return torch.cat(out)

    gA = torch.Generator().manual_seed(a.seed + 11)
    st = {"A": ((torch.randn(a.r, d_in, generator=gA) / math.sqrt(d_in)).to(dev)).requires_grad_(True),
          "B": torch.zeros(d_out, a.r, device=dev, requires_grad=True)}
    h = conv.register_forward_hook(lambda m, i, o: o + (st["B"] @ (st["A"] @ torch.nn.functional.unfold(
        i[0], k_, dilation=m.dilation, padding=pd_, stride=st_))).reshape(o.shape))
    yv = torch.zeros(1, dtype=torch.long, device=dev)
    for _ in range(a.T):
        loss = torch.nn.functional.cross_entropy(net(member) @ Whead.T, yv)
        gA_, gB_ = torch.autograd.grad(loss, [st["A"], st["B"]])
        st["A"] = (st["A"] - a.lr * gA_).detach().requires_grad_(True)
        st["B"] = (st["B"] - a.lr * gB_).detach().requires_grad_(True)
    h.remove()
    A_T, B_T = st["A"].detach(), st["B"].detach()
    U_, S_, Vh = torch.linalg.svd(B_T, full_matrices=False)
    Np = int((S_ > 1e-12 * S_[0]).sum()); Q = Vh[:Np].T
    C = A_T - Q @ (Q.T @ A_T)

    def q(X):
        P = patches(X)
        cn = torch.linalg.norm(torch.einsum("ndp,kd->nkp", P, C), dim=1)
        an = torch.linalg.norm(torch.einsum("ndp,kd->nkp", P, A_T), dim=1) + 1e-300
        return (cn / an).mean(-1)

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    t0 = time.time()
    q1 = q(nonmembers)
    n = int(q1.shape[0]); k = int((q1 < a.bar).sum())
    from scipy.stats import beta
    p_hi = float(beta.ppf(0.95, k + 1, n - k))
    print(f"# single candidate: {k}/{n} below the bar; 95% upper bound on the per-candidate rate {p_hi:.3e} "
          f"(was 2.99e-3 on 1000)   [{time.time()-t0:.0f}s]", flush=True)
    emit(dict(part="SINGLE", n_nonmembers=n, k_below_bar=k, fpr=k / n, upper_95=p_hi, bar=a.bar,
              n_prime=Np, rank_C=int((torch.linalg.svdvals(C) > 1e-10 * float(S_[0])).sum()),
              note="the population is capped by the DATASET size, not by choice", git=git_hash()))

    # the attacker's actual procedure: build the candidate set, score all, take the MINIMUM
    def group(X, size):
        out = [X]
        if size >= 2: out.append(TF.hflip(X))
        if size >= 10:
            for s in (0.5, 1.0, 1.5, 2.0):
                out += [TF.gaussian_blur(X, 5, [s]), TF.gaussian_blur(TF.hflip(X), 5, [s])]
        if size >= 25:
            for c in (196, 202, 208, 214):
                for fl in (False, True):
                    z = TF.hflip(X) if fl else X
                    out.append(TF.resize(TF.center_crop(z, c), [224, 224], antialias=True))
        return out[:size]

    sub = nonmembers[:min(a.enum_sample, nonmembers.shape[0])]
    for gsz in (1, 2, 10, 25):
        t1 = time.time()
        # CHUNKED: building the whole candidate set at once is 1500 x |G| images at 224^2 in FP64 and OOMs the
        # card at |G| = 25 (job 318567 died there). Candidates are generated per chunk instead.
        parts = []
        for i in range(0, sub.shape[0], 250):
            blk = sub[i:i + 250]; bbest = None
            for cand in group(blk, gsz):
                v = q(cand)
                bbest = v if bbest is None else torch.minimum(bbest, v)
                del cand
            parts.append(bbest); torch.cuda.empty_cache()
        best = torch.cat(parts)
        kk = int((best < a.bar).sum()); nn = int(best.shape[0])
        union = 1 - (1 - p_hi) ** gsz
        # THE DISTRIBUTION, not just the count (yoado-cd). With zero false positives at every group size, every
        # number quotable from a COUNT is a bound set by the non-member population, and reaching larger groups by
        # counting failures would need impossibly many negatives. The minimum non-member score shifts toward the
        # bar measurably as the group grows, so its trend extrapolates the crossing group size WITHOUT ever
        # observing a false positive.
        srt = torch.sort(best).values
        qs = {f"q{p}": float(srt[max(0, min(nn - 1, int(p / 100 * nn)))]) for p in (0, 1, 5, 10, 50)}
        emit(dict(part="ENUM", group_size=gsz, n=nn, k_below_bar=kk, measured_fpr=kk / nn,
                  best_score_quantiles=qs, best_min=float(srt[0]), margin_to_bar_orders=math.log10(
                      float(srt[0]) / a.bar) if float(srt[0]) > 0 else float("nan"),
                  union_bound=union, bound_over_measured=(union / (kk / nn) if kk else float("inf")),
                  bar=a.bar, seconds=time.time() - t1,
                  note="the MEASURED attacker error rate: full candidate set per non-member, minimum taken. No "
                       "independence assumption -- candidates are correlated (an image and its mirror), so the "
                       "union bound over-counts.", git=git_hash(), cmd=" ".join(sys.argv)))
        print(f"  |G|={gsz:3d}: min {float(srt[0]):.4f} (q1 {qs['q1']:.4f}, q5 {qs['q5']:.4f}, med {qs['q50']:.4f})  "
              f"margin to bar {math.log10(float(srt[0])/a.bar):.2f} orders   FPR {kk}/{nn} = {kk/nn:.5f}   union {union:.5f}   "
              f"{'bound is ' + format(union/(kk/nn), '.1f') + 'x loose' if kk else 'bound is INFINITELY loose (0 measured)'}"
              f"   [{time.time()-t1:.0f}s]", flush=True)
    hk.remove()


if __name__ == "__main__":
    main()
