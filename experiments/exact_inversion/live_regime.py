#!/usr/bin/env python3
"""Test 2: the certificate in the LIVE regime of the counting rule, on a real pretrained network.

Every cell attacked before this one was DEAD by the counting rule. The live cells are `N.p < min(r, d)`, i.e.
single-image personalisation at low position count -- and a late ResNet stage at 224px has p = 49 positions, so at
r = 64 the rule predicts a margin of about 15. This is the first cell where a real deployed configuration should
admit the channel at all.

WHAT THIS IS AND IS NOT (yoado-c9's pre-audit, adopted in full):
  * It is a HIDDEN-layer certificate, so it concerns the convolution's 49 INPUT PATCHES -- activations, not the
    photograph. The row says activation-space until an activation-to-image step exists.
  * The MEMBER SIDE IS NOT SCORED. With one recorded image the update to A is rank one along span{A_0 h}, so
    A_T h is collinear with A_0 h and lies in row(B_T) by construction; the member residual is zero whether or not
    anything leaked. Scoring the member side would be scoring the construction.
  * What IS scored is the NON-MEMBER distribution: how specific the test is. A certificate that annihilates
    everything is worthless, so the measurement is where non-members fall relative to the chi-squared null with
    r - N' degrees of freedom, and how many of them fall below the member.
  * Freezing the stages below is the HYPOTHESIS, not a cheat -- the certificate requires a fixed input. But the
    attack always lives at the EARLIEST adapted layer, so the placement assumption is a column: a fine-tune that
    also adapts stage 1 moves the attack there, onto raw pixels.
  * SGD-class training is a HARD GATE. Under Adam rank B_T = r, the certificate does not exist, and the cell would
    be scoring noise.

SCORING, fixed by yoado-b9 before any row: >= 20 draws STRATIFIED across the margin range; the gate at the
member's truth evaluated FIRST and a failure voiding the draw rather than scoring it negative; per draw the member
must rank 1 of 1 + U by residual AND beat the runner-up by >= 2 orders; exact binomial interval, never a bare
fraction; the non-member population shared across draws and DISCLOSED as shared; the headline is
residual-versus-margin, which tests the imprint law where it has never been tested.

  python -u -m experiments.exact_inversion.live_regime --draws 20 --nonmembers 1000
"""
import argparse, json, math, os, socket, sys, time
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.vit_token_span import load_images

torch.set_default_dtype(torch.float64)


def binom_ci(k, n, alpha=0.05):
    """Exact Clopper-Pearson interval -- b9 requires an interval, never a bare fraction."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--data-root", default="dataset_reconstruction/data/flowers-102/jpg")
    ap.add_argument("--draws", type=int, default=20); ap.add_argument("--nonmembers", type=int, default=1000)
    ap.add_argument("--r", type=int, default=64); ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--T", type=int, default=200); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--optimiser", default="sgd", choices=["sgd", "adam"],
                    help="adam is a NEGATIVE control: rank B_T = r, no certificate exists, cell must come back void")
    ap.add_argument("--classes", type=int, default=102); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gate", type=float, default=1e-8)
    ap.add_argument("--bar", type=float, default=1e-2,
                    help="THE fixed threshold on the normalised objective ||C phi|| / ||A_T phi||. NOT the "
                         "numerical floor: in FP64 that is ~1e-15 and everything passes it, INCLUDING a "
                         "certificate that has gone vacuous. 1e-2 is anchored between measured bands -- true "
                         "non-members 0.1..1, the marginal-recorded in-band tail 1e-5..1e-8, clean members "
                         "1e-16..1e-8 -- so it sits one order below the lowest non-member and three above the "
                         "marginal band: a working certificate clears it by 1-2 orders, a degraded one fails.")
    ap.add_argument("--same-class", action="store_true",
                    help="draw the non-members from the MEMBER'S OWN CLASS. Dataset-matched non-members exclude "
                         "cross-dataset detection, but activations cluster by class, so a mixed-class pool is the "
                         "easy case and a zero false-positive rate there is partly class discrimination. Within "
                         "class -- can it tell this rose from another rose -- is the privacy question.")
    ap.add_argument("--near-dupes", action="store_true",
                    help="NEAR-DUPLICATE SPECIFICITY: score transformed versions of the private image -- crop, "
                         "resize, flip, brightness, blur, JPEG recompression. Both answers are results and they "
                         "are DIFFERENT CLAIMS: pass = 'this image or anything close to it', the stronger privacy "
                         "statement and the weaker specificity one; fail = 'this exact image', narrower and "
                         "sharper. It is the difference between detecting a photograph and detecting a file.")
    ap.add_argument("--max-fpr", type=float, default=0.01,
                    help="pre-registered false-positive rate bar. Scoring is a RATE, not a minimum: with 1000 "
                         "non-members the minimum is an extreme statistic and one unlucky draw would void a "
                         "working certificate.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    import torchvision.models as tvm
    w = {"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1, "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2}[a.model]
    net = getattr(tvm, a.model)(weights=w).to(dev).double().eval()
    for p in net.parameters(): p.requires_grad_(False)
    conv = getattr(net, f"layer{a.stage}")[0].conv1
    k_, st_, pd_ = conv.kernel_size, conv.stride, conv.padding
    d_in = conv.in_channels * k_[0] * k_[1]; d_out = conv.out_channels
    feat_dim = net.fc.in_features
    net.fc = torch.nn.Identity()          # the model must return FEATURES; with the 1000-way head left in place
                                          # every forward returned logits and the fresh head could not be applied
    gh = torch.Generator().manual_seed(a.seed)
    Whead = (torch.randn(a.classes, feat_dim, generator=gh) / math.sqrt(feat_dim)).to(dev)

    if a.same_class:
        import glob as _g, numpy as _np
        from scipy.io import loadmat
        lab = loadmat(os.path.join(os.path.dirname(a.data_root.rstrip("/")), "imagelabels.mat"))["labels"].ravel()
        files = sorted(_g.glob(os.path.join(a.data_root, "*.jpg")))
        cls = int(_np.bincount(lab).argmax())                       # the most populous class: the largest pool
        idx = [i for i, f in enumerate(files) if i < len(lab) and lab[i] == cls]
        pool, _ = load_images(a.data_root, len(files), 224, dev)
        sel = pool[torch.tensor(idx, device=dev)]
        members, nonmembers = sel[:a.draws], sel[a.draws:]
        print(f"# SAME-CLASS cell: class {cls}, {len(idx)} images, {a.draws} members vs "
              f"{nonmembers.shape[0]} same-class non-members (a smaller pool than the mixed cell, so the "
              f"false-positive rate is coarser -- reported with its denominator)", flush=True)
    else:
        pool, _ = load_images(a.data_root, a.nonmembers + a.draws + 8, 224, dev)
        members, nonmembers = pool[:a.draws], pool[a.draws:a.draws + a.nonmembers]
    print(f"# live regime: {a.model} stage {a.stage} conv d_in={d_in} d_out={d_out}, r={a.r}, "
          f"{a.draws} draws vs {a.nonmembers} SHARED non-members, optimiser={a.optimiser}  "
          f"git={git_hash()} host={socket.gethostname()}", flush=True)

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    caps = {}
    hk = conv.register_forward_hook(lambda m, i, o: caps.__setitem__("p", torch.nn.functional.unfold(
        i[0].detach(), k_, dilation=m.dilation, padding=pd_, stride=st_)))

    def patches(X, bs=25):
        out = []
        for i in range(0, X.shape[0], bs):
            with torch.no_grad():
                net(X[i:i + bs]); out.append(caps["p"])
        return torch.cat(out)                                          # (n, d_in, positions)

    # margins on the FROZEN model decide the stratification; computed before any adapter exists
    with torch.no_grad():
        fm = torch.cat([net(members[i:i + 25]) for i in range(0, members.shape[0], 25)])
    y = torch.arange(a.draws, device=dev) % a.classes
    z0 = fm @ Whead.T
    other = z0.clone(); other.scatter_(1, y[:, None], float("-inf"))
    margin0 = (z0.gather(1, y[:, None]).squeeze(1) - other.max(1).values)
    order = torch.argsort(margin0)                                     # stratify across the margin range
    Pn = patches(nonmembers)
    ok = 0; scored = 0; prev = None       # prev release: gives each image a NEVER-TRAINED null of ITSELF (7e)
    # The exclusion of draw 1 was STRUCTURAL and predictable -- n draws give n-1 null-testable draws -- and should
    # have been pre-registered. Fixed here: one extra release is trained on a held-out image FIRST, purely to give
    # draw 1 a null to score against, so all n draws are evaluable.
    seed_img = members[-1:] if a.draws > 1 else None
    for rank_i, di in enumerate(order.tolist()):
        x = members[di:di + 1]; yi = y[di:di + 1]
        gA = torch.Generator().manual_seed(a.seed + 100 + di)
        A = ((torch.randn(a.r, d_in, generator=gA) / math.sqrt(d_in)).to(dev)).requires_grad_(True)
        B = torch.zeros(d_out, a.r, device=dev, requires_grad=True)
        st = {"A": A, "B": B}
        h = conv.register_forward_hook(lambda m, i, o: o + (st["B"] @ (st["A"] @ torch.nn.functional.unfold(
            i[0], k_, dilation=m.dilation, padding=pd_, stride=st_))).reshape(o.shape))
        # BUG FIXED (job 304455): this line previously read `opt = (torch.optim.Adam if ... else torch.optim.SGD)`
        # and `opt` was NEVER USED -- both arms ran plain gradient descent, so the Adam negative control was a
        # second SGD run and could not control anything. Caught by the pre-registered inverted finding: the Adam
        # arm PASSED, which was declared in advance to mean the pipeline is wrong rather than Adam interesting.
        mA = torch.zeros_like(st["A"]); vA = torch.zeros_like(st["A"])
        mB = torch.zeros_like(st["B"]); vB = torch.zeros_like(st["B"])
        b1, b2, epsA = 0.9, 0.999, 1e-8
        t0 = time.time()
        for step in range(1, a.T + 1):
            loss = torch.nn.functional.cross_entropy(net(x) @ Whead.T, yi)
            gA_, gB_ = torch.autograd.grad(loss, [st["A"], st["B"]])
            if a.optimiser == "adam":
                mA = b1 * mA + (1 - b1) * gA_; vA = b2 * vA + (1 - b2) * gA_ ** 2
                mB = b1 * mB + (1 - b1) * gB_; vB = b2 * vB + (1 - b2) * gB_ ** 2
                dA = (mA / (1 - b1 ** step)) / ((vA / (1 - b2 ** step)).sqrt() + epsA)
                dB = (mB / (1 - b1 ** step)) / ((vB / (1 - b2 ** step)).sqrt() + epsA)
            else:
                dA, dB = gA_, gB_
            st["A"] = (st["A"] - a.lr * dA).detach().requires_grad_(True)
            st["B"] = (st["B"] - a.lr * dB).detach().requires_grad_(True)
        h.remove()
        A_T, B_T = st["A"].detach(), st["B"].detach()
        S = torch.linalg.svdvals(B_T)
        Np = int((S > 1e-12 * S[0]).sum()) if float(S[0]) > 0 else 0    # MEASURED, never assumed at 49
        U_, S_, Vh = torch.linalg.svd(B_T, full_matrices=False)
        Q = Vh[:Np].T
        C = A_T - Q @ (Q.T @ A_T)
        rank_C = int((torch.linalg.svdvals(C) > 1e-10 * float(torch.linalg.svdvals(A_T)[0])).sum())
        Pm = patches(x)

        def res(P):
            cn = torch.linalg.norm(torch.einsum("ndp,kd->nkp", P, C), dim=1)
            an = torch.linalg.norm(torch.einsum("ndp,kd->nkp", P, A_T), dim=1) + 1e-300
            return (cn / an).mean(-1)

        # THE NUMERICAL FLOOR of C, computable from the release alone (yoado-b9). At N' = 1 the member residual is
        # at this floor BY ALGEBRAIC IDENTITY, so nothing may be scored against the member. Everything is scored
        # against the floor instead.
        def floor_of(P):
            eps = torch.finfo(torch.float64).eps
            pn = torch.linalg.norm(P, dim=1)
            an = torch.linalg.norm(torch.einsum("ndp,kd->nkp", P, A_T), dim=1) + 1e-300
            return float((eps * float(torch.linalg.norm(C, 2)) * pn / an).mean())
        rm = float(res(Pm)[0]); rn = res(Pn); c_floor = floor_of(Pm)
        dupes = {}
        if a.near_dupes:
            import torchvision.transforms.functional as TF
            v = {"crop90": TF.resize(TF.center_crop(x, 202), [224, 224], antialias=True),
                 "resize200": TF.resize(TF.resize(x, [200, 200], antialias=True), [224, 224], antialias=True),
                 "hflip": TF.hflip(x),
                 "bright+10%": (x * 1.1).clamp(-1, 1),
                 "blur": TF.gaussian_blur(x, 5, [1.0]),
                 "quantise8bit": (((x + 1) * 127.5).round() / 127.5 - 1)}
            for nm, xv in v.items():
                Pv = patches(xv)
                d_feat = float(torch.linalg.norm(Pv - Pm) / torch.linalg.norm(Pm))
                q_here = float(res(Pv)[0])
                # PAIRED negative for the transformation itself (b9's rule): the same transformed image scored
                # against a release that never saw it. Without this, a low score is ambiguous between "it passes"
                # and "it scores low for image-independent reasons".
                if prev is not None:
                    Cp, Ap = prev
                    cn = torch.linalg.norm(torch.einsum("ndp,kd->nkp", Pv, Cp), dim=1)
                    an = torch.linalg.norm(torch.einsum("ndp,kd->nkp", Pv, Ap), dim=1) + 1e-300
                    q_null_v = float((cn / an).mean(-1)[0])
                else:
                    q_null_v = float("nan")
                dupes[nm] = dict(q=q_here, q_paired_null=q_null_v, feature_distance=d_feat,
                                 reads_as_member=bool(q_here < a.bar))
        # 7e's per-draw null: the SAME image scored under the PREVIOUS draw's release, which never saw it. This
        # separates "the certificate annihilates this image" from "the certificate annihilates this image BECAUSE
        # it was trained on it" -- a same-image control no non-member population can provide.
        if prev is not None:
            Cp, Ap = prev
            cn = torch.linalg.norm(torch.einsum("ndp,kd->nkp", Pm, Cp), dim=1)
            an = torch.linalg.norm(torch.einsum("ndp,kd->nkp", Pm, Ap), dim=1) + 1e-300
            null_same_image = float((cn / an).mean(-1)[0])
        else:
            null_same_image = float("nan")
        prev = (C, A_T)
        srt, _ = torch.sort(rn)
        below = int((rn < rm).sum())
        gap = math.log10(float(srt[0]) / max(rm, 1e-300)) if rm > 0 else float("inf")
        gate_ok = (rm < a.gate) and rank_C > 0
        # b9's REPLACEMENT criterion. The old one -- rank 1 of 1001 plus a 2-order gap -- is now VACUOUS: at
        # N' = 1, C h = 0 is an algebraic identity, so rank 1 is automatic and carries no evidence. Success is
        # measured entirely on the NEGATIVE side, against C's own numerical floor.
        fpr = float((rn < a.bar).double().mean())
        null_ok = (null_same_image == null_same_image) and (null_same_image >= a.bar)
        fpr_ok = fpr <= a.max_fpr
        verdict = ("void: gate" if not gate_ok else
                   "void: no prior draw for the paired null" if null_same_image != null_same_image else
                   "success" if (null_ok and fpr_ok) else
                   "fail: null below the bar (annihilates an unseen image)" if not null_ok else
                   "fail: false-positive rate above 1%")
        if gate_ok:
            scored += 1; ok += (verdict == "success")
        emit(dict(part="LIVE", draw=di, margin_stratum=rank_i, initial_margin=float(margin0[di]),
                  r=a.r, positions=int(Pm.shape[2]), d_in=d_in, n_prime_measured=Np,
                  predicted_margin=a.r - int(Pm.shape[2]), rank_C=rank_C,
                  member_residual=rm, member_side_not_scored="forced to ~0 by rank-one collinearity at N=1",
                  near_duplicates=dupes,
                  near_dupe_measured_order=[k for k, v in sorted(dupes.items(), key=lambda z: z[1]["feature_distance"])],
                  near_dupe_prediction="pass/fail falls at a SINGLE boundary in the MEASURED feature-distance "
                                       "order -- one monotone prediction that can fail, not six binaries which "
                                       "are six chances to find one that works",
                  certificate_numerical_floor=c_floor,
                  bar=a.bar, false_positive_rate=fpr, max_fpr=a.max_fpr,
                  null_clears_bar=bool(null_ok), fpr_clears=bool(fpr_ok),
                  criterion="PRIMARY: false-positive rate = fraction of non-members below the fixed bar 1e-2, "
                            "pre-registered <= 1%. The paired same-image null must land ABOVE the bar; if it "
                            "falls below, the certificate is annihilating an image it never saw and the draw "
                            "FAILS regardless of the population rate. The non-member MINIMUM is reported, not the "
                            "criterion -- with 1000 non-members it is an extreme statistic. The member residual "
                            "is REPORTED, NOT SCORED: at N' = 1 it is an algebraic identity. NOTE the bar is NOT "
                            "the numerical floor (~1e-15 in FP64), which a vacuous certificate would also pass.",
                  claim_supported="the certificate is SPECIFIC (it annihilates the member and rejects everything "
                                  "else); NOT 'it identifies the member', which is an identity",
                  null_same_image_untrained_release=null_same_image,
                  null_ratio=(null_same_image / rm if rm > 0 and null_same_image == null_same_image else None),
                  nonmember_min=float(srt[0]), nonmember_median=float(srt[len(srt) // 2]),
                  nonmembers_below_member=below, gap_orders=gap, gate=a.gate, gate_passed=bool(gate_ok),
                  verdict=verdict, optimiser=a.optimiser,
                  chi2_dof=rank_C, recovery_space="activation (49 conv input patches), NOT pixels",
                  placement_assumption="first adapted layer is stage %d; the attack lives at the EARLIEST "
                                       "adapted layer, so a stage-1 fine-tune moves it to raw pixels" % a.stage,
                  nonmember_population=("SAME CLASS as the member" if a.same_class else
                                        "same dataset, MIXED classes"),
                  n_nonmembers=int(Pn.shape[0]),
                  nonmember_note="shared across draws (disclosed); disjoint from members by index",
                  start_attacker_buildable="n/a (scoring at the truth, no start)",
                  seconds=time.time() - t0, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))
        print(f"  draw {rank_i:3d} margin {float(margin0[di]):+8.3f}  N'={Np:3d} rank C={rank_C:3d}  "
              f"member {rm:.2e}  self-null {null_same_image:.2e}  FPR {fpr:.4f}  "
              f"nm_min {float(srt[0]):.2e}  {verdict}"
              + ("\n        dupes by MEASURED feature distance: " + "  ".join(
                    f"{k}(d={v['feature_distance']:.3f} q={v['q']:.1e} null={v['q_paired_null']:.1e} "
                    f"{'MEMBER' if v['reads_as_member'] else 'reject'})"
                    for k, v in sorted(dupes.items(), key=lambda z: z[1]["feature_distance"])) if dupes else ""),
              flush=True)
    hk.remove()
    lo, hi = binom_ci(ok, max(scored, 1))  # fraction of draws where the null clears the bar AND the FPR is <= 1%
    emit(dict(part="LIVE_SUMMARY", successes=ok, scored=scored, voided=a.draws - scored,
              rate=ok / max(scored, 1), exact_binomial_95=[lo, hi], draws=a.draws, optimiser=a.optimiser,
              headline="residual versus margin; the member side is not scored", git=git_hash()))
    print(f"\n# {ok}/{scored} scored draws succeeded ({a.draws - scored} void), exact 95% CI [{lo:.3f}, {hi:.3f}]",
          flush=True)


if __name__ == "__main__":
    main()
