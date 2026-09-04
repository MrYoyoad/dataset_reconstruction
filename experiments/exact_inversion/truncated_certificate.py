#!/usr/bin/env python3
"""Does a TRUNCATED certificate exist where the exact one is identically zero? (yoado-cd's proposal.)

The exact certificate projects out the whole row space of `B_T`, which is why a full span kills it: at
`rank B_T = r` the projector annihilates `A_T` and `C = 0`. But the span was measured **full rank, not flat** --
and those are different facts. Project out only the top-`k` directions instead:

    C_k = P_{top-k(B_T)}^perp A_T,   giving  r - k  conditions,  with  ||C_k h_i|| ~ the DISCARDED TAIL's energy

The trade is exactness for existence: `C_k` exists at every `k < r`, including on modules where `C` is zero, and
it separates whenever the discarded tail is smaller than the member/non-member gap it has to resolve. **This is
NOT the graded-energy statistic that failed earlier today**: that was a score with no error control, this is a
projector whose error is a computable quantity (the tail), so a threshold on it means something.

It is also the honest generalisation of the counting rule's first condition: `r >` span is the condition for an
*exact* certificate, and truncation replaces it with a condition on the SPECTRUM'S DECAY -- weaker, and
measurable. If truncation works, the rule must be restated as a condition on exactness rather than on existence.

MEASURE FIRST, ATTACK SECOND. This reports, on saturated real modules:
  * `B_T`'s normalised singular spectrum and the tail energy beyond every `k`;
  * the member and non-member residuals `||C_k h|| / ||A_T h||` at every `k`, and the AUC between them.
PRE-REGISTERED: if the tail at useful `k` sits orders below the non-member residual (O(0.1-1)), a truncated
certificate separates and the channel is not closed on transformers -- only its exact version is. If the spectrum
is flat, truncation buys nothing and the closure stands exactly as reported. Neither answer can come back void.

  python -u -m experiments.exact_inversion.truncated_certificate --module qkv --r 16 64
"""
import argparse, json, math, os, socket, sys, time
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.vit_token_span import load_images
from experiments.exact_inversion.graded_imprint import auc

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    ap.add_argument("--data-root", default="dataset_reconstruction/data/flowers-102/jpg")
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--n-nonmember", type=int, default=64)
    ap.add_argument("--r", nargs="*", type=int, default=[16, 64])
    ap.add_argument("--blocks", nargs="*", type=int, default=[0, 6, 11])
    ap.add_argument("--T", type=int, default=100); ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--classes", type=int, default=102); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    import timm
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = timm.create_model(a.model, pretrained=True, num_classes=0).to(dev).double().eval()
    for p in model.parameters(): p.requires_grad_(False)
    feat = model.num_features
    gh = torch.Generator().manual_seed(a.seed)
    Whead = (torch.randn(a.classes, feat, generator=gh) / math.sqrt(feat)).to(dev)
    X, _ = load_images(a.data_root, a.N + a.n_nonmember, 224, dev)
    Xm, Xn = X[:a.N], X[a.N:]
    y = torch.arange(a.N, device=dev) % a.classes

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    for blk in a.blocks:
        tgt = model.blocks[blk].attn.qkv
        d_in, d_out = tgt.in_features, tgt.out_features
        for r in a.r:
            gA = torch.Generator().manual_seed(a.seed + 11)
            st = {"A": ((torch.randn(r, d_in, generator=gA) / math.sqrt(d_in)).to(dev)).requires_grad_(True),
                  "B": torch.zeros(d_out, r, device=dev, requires_grad=True)}
            h = tgt.register_forward_hook(
                lambda m, i, o: o + torch.nn.functional.linear(i[0], st["A"]) @ st["B"].T)
            t0 = time.time()
            for _ in range(a.T):
                loss = torch.nn.functional.cross_entropy(model(Xm) @ Whead.T, y)
                gA_, gB_ = torch.autograd.grad(loss, [st["A"], st["B"]])
                st["A"] = (st["A"] - a.lr * gA_).detach().requires_grad_(True)
                st["B"] = (st["B"] - a.lr * gB_).detach().requires_grad_(True)
            A_T, B_T = st["A"].detach(), st["B"].detach()
            caps = {}
            h2 = tgt.register_forward_hook(lambda m, i, o: caps.__setitem__("h", i[0].detach()))
            with torch.no_grad():
                model(Xm); Hm = caps["h"]
                Hn = torch.cat([(model(Xn[i:i + 8]), caps["h"])[1] for i in range(0, Xn.shape[0], 8)])
            h.remove(); h2.remove()
            U, S, Vh = torch.linalg.svd(B_T, full_matrices=False)
            rank_B = int((S > 1e-12 * S[0]).sum()) if float(S[0]) > 0 else 0
            spec = [float(v / S[0]) for v in S]
            tail = [float((S[k:] ** 2).sum().sqrt() / (S ** 2).sum().sqrt()) for k in range(len(S))]

            def resid(H, k):
                Q = Vh[:k].T
                Ck = A_T - Q @ (Q.T @ A_T)
                cn = torch.linalg.norm(torch.einsum("btd,kd->btk", H, Ck), dim=-1)
                an = torch.linalg.norm(torch.einsum("btd,kd->btk", H, A_T), dim=-1) + 1e-300
                return (cn / an).mean(-1)
            rows = []
            for k in range(1, r):
                rm, rn = resid(Hm, k), resid(Hn, k)
                rows.append(dict(k=k, conditions=r - k, tail_energy=tail[k],
                                 member_median=float(rm.median()), nonmember_median=float(rn.median()),
                                 ratio=float(rn.median() / (rm.median() + 1e-300)),
                                 auc=auc([-float(v) for v in rm], [-float(v) for v in rn])))
            best = max(rows, key=lambda z: z["auc"])
            emit(dict(part="TRUNCATED", block=blk, r=r, N=a.N, d_in=d_in, rank_B_T=rank_B,
                      exact_certificate_vacuous=bool(rank_B >= min(r, d_in)),
                      spectrum_rel=spec, per_k=rows, best_k=best["k"], best_auc=best["auc"],
                      best_ratio=best["ratio"], spectrum_flat=bool(spec[min(len(spec) - 1, r // 2)] > 0.5),
                      reading=("SEPARATES" if best["auc"] > 0.9 else "PARTIAL" if best["auc"] > 0.7 else "FLAT"),
                      note="C_k is a PROJECTOR with computable error (the discarded tail), not the graded-energy "
                           "SCORE that failed earlier; the trade is exactness for existence",
                      start_model="n/a (scoring at the truth)", claim_class="membership signal",
                      seconds=time.time() - t0, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))
            print(f"  block {blk:2d} r={r:3d}  rank B_T={rank_B} (exact cert {'VACUOUS' if rank_B >= min(r,d_in) else 'alive'})"
                  f"  sigma_r/sigma_1={spec[-1]:.2e}  best k={best['k']} AUC {best['auc']:.3f} "
                  f"ratio {best['ratio']:.2e} tail {best['tail_energy']:.2e}  -> "
                  f"{'SEPARATES' if best['auc']>0.9 else 'PARTIAL' if best['auc']>0.7 else 'FLAT'}", flush=True)


if __name__ == "__main__":
    main()
