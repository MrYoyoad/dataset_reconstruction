#!/usr/bin/env python3
"""The distributional-access question on the surface where it is still ALIVE: reconstruction.

Job 289251 killed the argument for membership inference, and for a reason specific to it: **the attacker holds the
candidate image by definition**, so it enters half the shadows whatever else they own and any co-training pool
substitutes. In RECONSTRUCTION they do not hold the image -- recovering it is the task -- so the public pool is the
only source of a chart or prior, and a mismatched pool may genuinely cost fidelity. That is unmeasured, and it is
what this measures.

The certificate route's own framing makes the question sharp: the prior's job on this route is the CHART, not the
start. So the chart is the whole of the attacker's distributional assumption, and varying the pool it is fitted on
varies exactly the thing an attacker may not be able to obtain -- while the release, the recipe, the private batch
and the solver are all held fixed.

Chart pools, all 28x28 greyscale so the chart has identical shape and only its CONTENT differs:
  matched  MNIST train      -- the private data's own distribution (the assumption every earlier cell made)
  near     EMNIST letters   -- handwritten, same modality and stroke statistics, different classes
  far      FashionMNIST     -- greyscale objects, same size, different content entirely
  gross    uniform noise    -- no structure at all; the floor, and the control that says what a chart is worth

PRE-REGISTERED, before any row:
  * chart FIDELITY (the private images' projection error onto the chart) degrades monotonically matched -> gross;
  * RECOVERY degrades with it, and the reported quantity is the pool at which recovery stops working;
  * if recovery survives on 'far' or 'gross', the attacker needs no distributional access on this surface either
    and the argument is dead everywhere, which would be a clean negative and should be stated as one.
Recovery is from RANDOM public-scale starts and is scored against the ground truth, never against the release.

  python -u -m experiments.exact_inversion.chart_mismatch --pools matched near far gross
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.certificate import certificate, lm_cert
from experiments.exact_inversion.subset_and_ood import pick_batch, release_and_imprints

torch.set_default_dtype(torch.float64)


def read_idx_file(path, n=None):
    with open(path, "rb") as f:
        f.read(16); a = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)
    a = a.astype(np.float64) / 255.0
    return a[:n] if n else a


def chart_pool(name, root, n, seed):
    """The attacker's public pool. Only its CONTENT varies; shape and scaling are identical throughout."""
    if name == "matched":
        return read_idx(root, "train")[0][:n]
    if name == "near":
        return read_idx_file(os.path.join(root, "EMNIST", "raw", "emnist-letters-train-images-idx3-ubyte"), n)
    if name == "far":
        return read_idx_file(os.path.join(root, "FashionMNIST", "raw", "train-images-idx3-ubyte"), n)
    if name == "gross":
        g = np.random.default_rng(seed)
        return g.random((n, 784))
    raise SystemExit(f"unknown pool {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--pools", nargs="*", default=["matched", "near", "far", "gross"])
    ap.add_argument("--ks", nargs="*", type=int, default=[12, 14, 16])
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", default="mnist_control")
    ap.add_argument("--starts", type=int, default=200); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(1000)
    bb = TrainedBackbone(a.model, dev, "gelu")
    Xte, yte = read_idx(a.data_root, "test")
    Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    perm = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7)).tolist()
    if a.set == "mnist_control":                                   # one image per digit, the standard cell here
        sel, seen = [], set()                                      # (certificate.py defines this inline; matched)
        for i in perm:
            if int(yte[i]) not in seen: sel.append(i); seen.add(int(yte[i]))
            if len(sel) == a.N: break
    else:
        sel = pick_batch(a.set, bb, Xte_t, yte_t, a.N, perm)
    idx = torch.tensor(sel, device=dev)
    X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    print(f"# chart mismatch: model={os.path.basename(a.model)} set={a.set} N={a.N} r={a.r} T={a.T} "
          f"pools={a.pools} ks={a.ks} git={git_hash()} host={socket.gethostname()}", flush=True)

    for pool in a.pools:
        P = torch.tensor(chart_pool(pool, a.data_root, a.n_fit, a.seed), device=dev)
        for k in a.ks:
            chart = PCAChart(P, k, dev)
            X_on = chart.psi(chart.coords_of(X_real))
            chart_err = (torch.linalg.norm(X_on - X_real, dim=0) / torch.linalg.norm(X_real, dim=0))
            coord_std = chart.coords_of(P[:10000].T).std(dim=1, keepdim=True)
            # the RELEASE is trained on the private images as the chart sees them, exactly as in every earlier
            # cell -- so the only thing varying across pools is the attacker's chart.
            g = torch.Generator().manual_seed(a.seed + 7)
            A0 = (a.sigma0 * torch.randn(a.r, bb.n, generator=g)).to(dev)
            aa = argparse.Namespace(**{**vars(a), "k": k})
            A_T, B_T, imp, sB, _ = release_and_imprints(bb, X_on, y, A0, aa)
            C, Np, S = certificate(A_T, B_T)
            H = bb.phi(X_on)
            cert_res = (torch.linalg.norm(C @ H, dim=0) / torch.linalg.norm(A_T @ H, dim=0))
            rec = [i for i in range(a.N) if float(imp[i] / imp.max()) > 1e-12]

            def obj(w):
                h = bb.phi(chart.psi(w.reshape(k, 1)))
                return (C @ h).reshape(-1) / torch.linalg.norm(A_T @ h)
            gs = torch.Generator().manual_seed(a.seed + 31); errs = []; errs_raw = []; t0 = time.time()
            for _ in range(a.starts):
                w0 = (torch.randn(k, 1, generator=gs).to(dev) * coord_std).reshape(-1)
                w, o, _ = lm_cert(obj, w0, a.iters)
                xh = chart.psi(w.reshape(k, 1))[:, 0]
                # TWO errors, and conflating them makes the measurement impossible. The release saw the images AS
                # THE CHART REPRESENTS THEM, and no candidate inside the chart can be closer to the raw image than
                # the chart's own error (0.52 at k=16 here). So a 1e-2 threshold on the RAW error is unreachable by
                # construction and fails even the matched control -- which is what the first run of this job did.
                #   err_on   did the solver find the right point IN the chart?   <- what the attack controls
                #   err_raw  distance to the true image                          <- err_on PLUS the chart's ceiling
                errs.append(min(float(torch.linalg.norm(xh - X_on[:, i]) / torch.linalg.norm(X_on[:, i]))
                                for i in rec))
                errs_raw.append(min(float(torch.linalg.norm(xh - X_real[:, i]) / torch.linalg.norm(X_real[:, i]))
                                    for i in rec))
            se = sorted(errs); se_raw = sorted(errs_raw)
            emit(dict(part="CHART_MISMATCH", pool=pool, k=k, N=a.N, r=a.r, rank_B_T=Np, cert_line=a.r - Np,
                      chart_err_median=float(chart_err.median()), chart_err_max=float(chart_err.max()),
                      explained_var=float(chart.explained), n_recorded=len(rec),
                      cert_residual_recorded_max=float(max(cert_res[i] for i in rec)),
                      starts=a.starts,
                      err_on_min=se[0], err_on_p10=se[len(se) // 10], err_on_median=se[len(se) // 2],
                      err_raw_min=se_raw[0], err_raw_median=se_raw[len(se_raw) // 2],
                      frac_landed_on_1e2=float(sum(1 for e in errs if e < 1e-2) / len(errs)),
                      frac_landed_on_1e1=float(sum(1 for e in errs if e < 1e-1) / len(errs)),
                      raw_error_floor_is_chart_error=True,
                      metric_note="err_on is what the ATTACK controls; err_raw is bounded below by the chart's own "
                                  "error, so a raw-error threshold cannot discriminate between chart pools.",
                      start_model="random public-scale (attacker-buildable)", claim_class="attack",
                      note="only the attacker's CHART POOL varies; release, recipe, private batch and solver fixed",
                      seconds=time.time() - t0, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))
            print(f"  [{pool:7s}] k={k:3d}  chart err {float(chart_err.median()):.3f}  expl {chart.explained:.3f}  "
                  f"N'={Np}  landed_on<1e-2 {sum(1 for e in errs if e < 1e-2)}/{a.starts}  "
                  f"<1e-1 {sum(1 for e in errs if e < 1e-1)}/{a.starts}  median err_on {se[len(se)//2]:.3e}  "
                  f"median err_raw {se_raw[len(se_raw)//2]:.3e}", flush=True)


if __name__ == "__main__":
    main()
