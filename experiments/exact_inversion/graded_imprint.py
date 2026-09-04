#!/usr/bin/env python3
"""When the recorded span SATURATES, the certificate dies but the SPECTRUM does not. A graded test.

Job 273322 killed the recipe-free certificate on transformers: a block linear records one direction per token, so
`rank B_T = r` at any deployed rank and `C = P_{row(B_T)^perp} A_T` is the zero matrix. The certificate is a
NULL-SPACE test, and a saturated release has no null space. That is the whole of the negative.

But the null space is not the only thing the release carries. With `B_T = sum_i sum_p q_ip (A h_ip)^T`, the
singular SPECTRUM of `B_T` is not flat: the directions the private data actually drove are the ones with large
singular values, and directions nothing drove keep small ones. So replace the binary annihilation test by a graded
one, on the SAME released numbers and with the same attacker knowledge:

    s(x)  =  sum_p  ( sum_k sigma_k <v_k, A_T phi_p(x)>^2 )  /  ( sum_k <v_k, A_T phi_p(x)>^2 )

with `B_T = U Sigma V^T` and `v_k` the right singular vectors, i.e. the singular-value-weighted share of a
candidate's adapter-space energy. **This reduces to the certificate exactly when the release is unsaturated**: if
`sigma_k = 0` above `N'`, then `s(x) = 0` iff `A_T phi(x)` lies wholly in the null space, which is `C phi(x) = 0`.
So it is the certificate's continuation into the regime where the certificate is vacuous, not a different idea.

PRE-REGISTERED, and the negative is the likely outcome:
  DEAD      member and non-member scores separate at AUC ~ 0.5. Then saturation is total, the recipe-free channel
            on transformers is finished in graded form too, and that is the honest end of this line.
  GRADED    members score above non-members. The channel then survives as a MEMBERSHIP signal, not reconstruction,
            and the number to report is AUC at deployed rank on a real model.
MANDATORY CONTROL, without which a positive means nothing: the same AUC from a plain loss-threshold membership
attack on the adapted model. Our score has to BEAT that baseline to be worth anything at all -- otherwise it is a
worse way of doing something already standard, and it should be reported as such.

Setting: frozen pretrained ViT-B/16, a fresh public-seeded 102-way head, LoRA on ONE block's qkv, FP64 throughout,
private batch of N real flowers, non-members drawn from the same pool and never trained on.

  python -u -m experiments.exact_inversion.graded_imprint --r 16 --block 6
"""
import argparse, glob, json, math, os, socket, sys, time
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.vit_token_span import load_images

torch.set_default_dtype(torch.float64)


def auc(pos, neg):
    """Rank-based AUC, ties at 0.5; no sklearn dependency."""
    xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    n1 = len(pos); n0 = len(neg)
    r = 0.0; i = 0
    while i < len(xs):
        j = i
        while j < len(xs) and xs[j][0] == xs[i][0]: j += 1
        avg = (i + j + 1) / 2.0
        for k in range(i, j):
            if xs[k][1] == 1: r += avg
        i = j
    return (r - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    ap.add_argument("--data-root", default="dataset_reconstruction/data/flowers-102/jpg")
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--n-nonmember", type=int, default=64)
    ap.add_argument("--r", nargs="*", type=int, default=[8, 16, 64])
    ap.add_argument("--block", type=int, default=6)
    ap.add_argument("--module", default="qkv", choices=["qkv", "head"],
                    help="qkv: a weight-shared block linear, where the span saturates and the certificate is the "
                         "zero matrix. head: the pooled classification head, the ONE non-shared module, where the "
                         "recorded count is N and the certificate is non-vacuous whenever r > N. The head arm is "
                         "the positive control for the graded statistic AND the deployment-relevant attack cell.")
    ap.add_argument("--T", type=int, default=100); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--classes", type=int, default=102); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--baseline-band", nargs=2, type=float, default=[0.6, 0.9],
                    help="the band of TRIVIAL-BASELINE performance in which this comparison is meaningful, fixed "
                         "in advance (yoado-cd/81). Above it the baseline is saturated and nothing can beat it, "
                         "only tie -- which is what job 280255 measured. Below it the release is too weak for "
                         "anything to be detectable and a tie means nothing either. Cells outside the band are "
                         "reported VOID and are NOT scored.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    import timm
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = timm.create_model(a.model, pretrained=True, num_classes=0).to(dev).double().eval()
    for p in model.parameters(): p.requires_grad_(False)
    gh = torch.Generator().manual_seed(a.seed)                       # PUBLIC head init: attacker knows the seed
    feat = model.num_features
    Whead = (torch.randn(a.classes, feat, generator=gh) / math.sqrt(feat)).to(dev)

    X, files = load_images(a.data_root, a.N + a.n_nonmember, 224, dev)
    Xm, Xn = X[:a.N], X[a.N:]
    y = torch.arange(a.N, device=dev) % a.classes                    # distinct labels, the personalisation regime

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    if a.module == "qkv":
        target = model.blocks[a.block].attn.qkv
        d_in, d_out = target.in_features, target.out_features
    else:
        target = None                                   # the head is applied by hand below, so hook it directly
        d_in, d_out = feat, a.classes
    print(f"# graded imprint: {a.model} block {a.block} qkv {d_in}->{d_out}, N={a.N} members, "
          f"{a.n_nonmember} non-members, FP64  git={git_hash()} host={socket.gethostname()}", flush=True)

    for r in a.r:
        gA = torch.Generator().manual_seed(a.seed + 11)
        A = ((torch.randn(r, d_in, generator=gA) / math.sqrt(d_in)).to(dev)).requires_grad_(True)
        B = torch.zeros(d_out, r, device=dev, requires_grad=True)
        state = {"A": A, "B": B}
        if a.module == "qkv":
            def hook(mod, inp, out): return out + torch.nn.functional.linear(inp[0], state["A"]) @ state["B"].T
            h = target.register_forward_hook(hook)
        else:
            h = None
        def fwd(Z):
            f = model(Z)
            z = f @ Whead.T
            if a.module == "head":
                z = z + torch.nn.functional.linear(f, state["A"]) @ state["B"].T
            return z, f
        t0 = time.time()
        for t in range(a.T):
            logits, _ = fwd(Xm)
            loss = torch.nn.functional.cross_entropy(logits, y)
            gA_, gB_ = torch.autograd.grad(loss, [state["A"], state["B"]])
            state["A"] = (state["A"] - a.lr * gA_).detach().requires_grad_(True)
            state["B"] = (state["B"] - a.lr * gB_).detach().requires_grad_(True)
        A_T, B_T = state["A"].detach(), state["B"].detach()
        with torch.no_grad():
            final_loss = float(torch.nn.functional.cross_entropy(fwd(Xm)[0], y))
        U, S, Vh = torch.linalg.svd(B_T, full_matrices=False)
        rank_B = int((S > 1e-12 * S[0]).sum()) if float(S[0]) > 0 else 0
        V = Vh.T                                                      # (r, r): adapter-space directions

        # the recipe-free CERTIFICATE itself, which is non-vacuous exactly when rank B_T < min(r, d_in)
        Q = Vh[:rank_B].T
        C = A_T - Q @ (Q.T @ A_T)
        svC = torch.linalg.svdvals(C)
        rank_C = int((svC > 1e-10 * float(torch.linalg.svdvals(A_T)[0])).sum())
        caps = {}
        def cap(mod, inp, out): caps["h"] = inp[0].detach()
        h2 = target.register_forward_hook(cap) if a.module == "qkv" else None

        def scores(Z, bs=8):
            graded, losses, cert = [], [], []
            for i in range(0, Z.shape[0], bs):
                xb = Z[i:i + bs]
                with torch.no_grad():
                    logits, f = fwd(xb)
                    hb = caps["h"] if a.module == "qkv" else f[:, None, :]   # (b, tokens, d_in)
                    if rank_C:                                        # ||C h|| / ||A_T h||: 0 for a recorded item
                        cn = torch.linalg.norm(torch.einsum("btd,kd->btk", hb, C), dim=-1)
                        an = torch.linalg.norm(torch.einsum("btd,kd->btk", hb, A_T), dim=-1) + 1e-300
                        cert += (cn / an).mean(-1).tolist()
                    else:
                        cert += [float("nan")] * xb.shape[0]
                    proj = torch.einsum("btd,kd->btk", hb, A_T) @ V   # (b, tokens, r)
                    e = proj ** 2
                    num = (e * S[None, None, :]).sum(-1)
                    den = e.sum(-1) + 1e-300
                    graded += (num / den).mean(-1).tolist()
                    losses += (-torch.logsumexp(logits, -1) + logits.max(-1).values).tolist()
            return graded, losses, cert
        gm, lm, cm = scores(Xm); gn, ln, cn_ = scores(Xn)
        if h is not None: h.remove()
        if h2 is not None: h2.remove()
        a_graded = auc(gm, gn); a_loss = auc(lm, ln)
        # the certificate is SMALL for members, so members are the low tail: flip the direction
        a_cert = (auc([-v for v in cm], [-v for v in cn_]) if rank_C else float("nan"))
        emit(dict(part="GRADED", model=a.model, module=a.module, block=a.block, r=r, N=a.N,
                  n_nonmember=a.n_nonmember, rank_B_T=rank_B, rank_C=rank_C,
                  certificate_vacuous=bool(rank_C == 0),
                  saturated=bool(rank_B >= r), final_loss=final_loss,
                  auc_certificate=a_cert, auc_graded_imprint=a_graded, auc_loss_baseline=a_loss,
                  beats_baseline=bool(max(a_graded, a_cert if a_cert == a_cert else 0) > a_loss),
                  cert_median_member=(float(sorted(cm)[len(cm) // 2]) if rank_C else None),
                  cert_median_nonmember=(float(sorted(cn_)[len(cn_) // 2]) if rank_C else None),
                  median_member=float(sorted(gm)[len(gm) // 2]), median_nonmember=float(sorted(gn)[len(gn) // 2]),
                  baseline_band=a.baseline_band,
                  cell_valid=bool(a.baseline_band[0] <= a_loss <= a.baseline_band[1]),
                  reading=("VOID (baseline outside the pre-registered band; measures the baseline's ceiling or "
                           "floor, not this statistic)" if not (a.baseline_band[0] <= a_loss <= a.baseline_band[1])
                          else "BEATS BASELINE" if max(a_graded, a_cert if a_cert == a_cert else 0) > a_loss
                          else "LOSES TO BASELINE"),
                  comparator="membership baselines (loss threshold here); this is a MEMBERSHIP statistic, not a "
                             "reconstruction metric, and must be judged against that literature",
                  note="the certificate is the sigma_k = 0 limit of this score; a saturated release has no null "
                       "space, so this is the only form in which the recipe-free test can survive",
                  start_model="n/a (scoring, no solve)", claim_class="membership signal",
                  seconds=time.time() - t0, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))
        print(f"  [{a.module}] r={r:3d}  rank B_T={rank_B}/{r} rank C={rank_C} "
              f"{'SATURATED' if rank_B >= r else ''}  final loss {final_loss:.3e}"
              f"   AUC cert {a_cert:.3f}   AUC graded {a_graded:.3f}   AUC loss-baseline {a_loss:.3f}"
              f"   {'VOID (baseline outside band)' if not (a.baseline_band[0] <= a_loss <= a.baseline_band[1]) else ('BEATS baseline' if max(a_graded, a_cert if a_cert == a_cert else 0) > a_loss else 'LOSES to baseline')}",
              flush=True)


if __name__ == "__main__":
    main()
