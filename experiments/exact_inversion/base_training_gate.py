#!/usr/bin/env python3
"""WP0 gate: is every frozen base in the 2026-09-18 plan FULLY TRAINED?  Measured from the checkpoint, not a note.

A base passes when its train loss has collapsed (Haim et al.'s regime): train accuracy >= 0.995 AND mean train
cross-entropy <= 1e-2.  Every checkpoint in `notes/plan_2026-09-18_cnn_ranklaw_newclass_charts.md` WP0 is loaded
with the loader that WROTE it (so a format drift shows up here, not inside a package), evaluated on the FULL train
and test splits in float64, and one JSON row per checkpoint is printed and appended to `--out`.  Nothing is trained
here and no checkpoint is modified.

Families (loader -> checkpoint keys):
  mnist_mlp     train_strong_backbone.save_ckpt       {"state_dict": layers.{0,1,2}.*, epoch, batch, test_acc}
  cifar_mlp     train_cifar_backbone (same keys, 3072-in)
  mnist_deep    deep_stack.save_deep                  {"Ws": [...], "b1", test_acc, depth, width, ...}
  Both MLP families read an optional "act" field (P5 activation twins, default gelu).  deep_stack's forward
  hard-codes GELU, so a relu/tanh deep checkpoint is evaluated with a gate-local copy of forward_deep.
  mnist_conv    conv_certificate (spec shallow/deep)  {"Wms", "bs", "Whead", "bhead", test_acc, git}
  cifar_newclass cifar_newclass.MLP / CNN              {"state_dict", test_acc, train_acc, train_loss, margin_*, overtrain, epochs}

  python -u -m experiments.exact_inversion.base_training_gate --out results/base_training_gate.jsonl
  python -u -m experiments.exact_inversion.base_training_gate --ckpts models/exact_inversion/mnist_mlp_strong_full.pth
"""
import argparse, json, os, socket, sys, time
import numpy as np
import torch, torch.nn.functional as F

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import read_idx
from experiments.exact_inversion import deep_stack, conv_certificate
from experiments.exact_inversion.train_strong_backbone import ACTS
from experiments.cifar import cifar_newclass

torch.set_default_dtype(torch.float64)

GATE_TRAIN_ACC = 0.995
GATE_TRAIN_LOSS = 1e-2

# path stem -> (family, extra).  The WP0 table plus the optional `_full` twins (gated with --ckpts).
FAMILY = {
    "mnist_mlp_strong":                  ("mnist_mlp", {}),
    "mnist_mlp_strong_full":             ("mnist_mlp", {}),
    "mnist_mlp_m26_strong":              ("mnist_mlp", {}),
    "mnist_mlp_d15w1000":                ("mnist_deep", {}),
    "mnist_conv_deep":                   ("mnist_conv", {"spec": "deep"}),
    "mnist_conv_deep_full":              ("mnist_conv", {"spec": "deep"}),
    "mnist_conv":                        ("mnist_conv", {"spec": "shallow"}),
    "mnist_conv_bottleneck":             ("mnist_conv", {"spec": "bottleneck"}),
    "mnist_mlp_d15w1000_full":           ("mnist_deep", {}),
    "mnist_mlp_strong_relu":             ("mnist_mlp", {}),      # P5 activation twins (act read from the dict)
    "mnist_mlp_strong_tanh":             ("mnist_mlp", {}),
    "mnist_mlp_d15w1000_relu_full":      ("mnist_deep", {}),
    "mnist_mlp_d15w1000_tanh_full":      ("mnist_deep", {}),
    "cifar10_cnn_newclass":              ("cifar_newclass", {"arch": "cnn"}),
    "cifar10_mlp_overtrained_newclass":  ("cifar_newclass", {"arch": "mlp"}),
    "cifar10_mlp_newclass":              ("cifar_newclass", {"arch": "mlp"}),
    "cifar10_mlp":                       ("cifar_mlp", {}),
}
WP0_ORDER = ["mnist_mlp_strong", "mnist_mlp_m26_strong", "mnist_mlp_d15w1000", "mnist_conv_deep", "mnist_conv",
             "cifar10_cnn_newclass", "cifar10_mlp_overtrained_newclass", "cifar10_mlp"]


def recorded_of(ck):
    """Everything the checkpoint dict itself recorded that is not a tensor / list of tensors / state_dict."""
    out = {}
    for k, v in ck.items():
        if k in ("state_dict", "Ws", "Wms", "bs", "b1", "Whead", "bhead", "Wd", "bd", "__path__"): continue
        if torch.is_tensor(v): continue
        if isinstance(v, (list, tuple)) and any(torch.is_tensor(t) for t in v): continue
        if isinstance(v, (int, float, str, bool)) or v is None: out[k] = v
        else: out[k] = str(v)
    return out


# ------------------------------------------------------------------------------------------- forward maps
def build_forward(family, extra, ck, dev):
    """Returns (logits_fn over a (B, d_in) float64 batch, description string, kind 'mnist'|'cifar')."""
    if family in ("mnist_mlp", "cifar_mlp"):
        sd = ck["state_dict"]
        W1, b1 = sd["layers.0.weight"].to(dev).double(), sd["layers.0.bias"].to(dev).double()
        W2, W3 = sd["layers.1.weight"].to(dev).double(), sd["layers.2.weight"].to(dev).double()
        act = ck.get("act", "gelu"); phi = ACTS[act]
        def f(x): return phi(phi(x @ W1.T + b1) @ W2.T) @ W3.T
        return f, f"{W1.shape[1]}-{W1.shape[0]}-{W2.shape[0]}-{W3.shape[0]} {act.upper()} (state_dict)", \
            ("mnist" if family == "mnist_mlp" else "cifar")
    if family == "mnist_deep":
        Ws, b1, _ = deep_stack.load_deep(str(ck["__path__"]), dev)
        act = ck.get("act", "gelu")
        if act == "gelu":
            def f(x): return deep_stack.forward_deep(x.T, Ws, b1).T
            return f, f"784-{Ws[0].shape[0]}x{len(Ws)-1}-{Ws[-1].shape[0]} GELU (deep_stack, depth {len(Ws)})", "mnist"
        phi = ACTS[act]                                       # deep_stack.forward_deep hard-codes GELU: same map, this act
        def f(x):
            h = x.T
            for l in range(len(Ws) - 1):
                h = phi(Ws[l] @ h + (b1[:, None] if l == 0 else 0))
            return (Ws[-1] @ h).T
        return f, f"784-{Ws[0].shape[0]}x{len(Ws)-1}-{Ws[-1].shape[0]} {act.upper()} (gate-local forward_deep, depth {len(Ws)})", "mnist"
    if family == "mnist_conv":
        conv_certificate.SPEC = conv_certificate.SPECS[extra["spec"]]
        Wms = [w.to(dev).double() for w in ck["Wms"]]; bs_ = [b.to(dev).double() for b in ck["bs"]]
        Wh, bh = ck["Whead"].to(dev).double(), ck["bhead"].to(dev).double()
        Wd = ck["Wd"].to(dev).double() if ck.get("Wd") is not None else None
        bd = ck["bd"].to(dev).double() if ck.get("bd") is not None else None
        def f(x): return conv_certificate.conv_forward(x.reshape(-1, 1, 28, 28), Wms, bs_, Wh, bh, Wd=Wd, bd=bd)
        return f, (f"conv spec '{extra['spec']}' {conv_certificate.SPEC}"
                   + (f" + dense {Wd.shape[1]}->{Wd.shape[0]} GELU" if Wd is not None else "")
                   + " + linear head (conv_certificate)"), "mnist"
    if family == "cifar_newclass":
        Net = cifar_newclass.MLP if extra["arch"] == "mlp" else cifar_newclass.CNN
        net = Net().to(dev); net.load_state_dict(ck["state_dict"]); net.eval(); net.double()
        for p in net.parameters(): p.requires_grad_(False)
        def f(x): return net(x)
        return f, f"cifar_newclass.{Net.__name__} (state_dict, eval mode)", "cifar"
    raise ValueError(family)


@torch.no_grad()
def evaluate(f, X, y, bs=1000):
    """Accuracy, mean cross-entropy, and the margin z_y - max_{j!=y} z_j over the whole split."""
    n = X.shape[0]; correct = 0; loss = 0.0; margins = []
    for i in range(0, n, bs):
        z = f(X[i:i + bs]); yb = y[i:i + bs]
        loss += float(F.cross_entropy(z, yb, reduction="sum"))
        correct += int((z.argmax(1) == yb).sum())
        zy = z.gather(1, yb[:, None])[:, 0]; zo = z.clone(); zo.scatter_(1, yb[:, None], -float("inf"))
        margins.append((zy - zo.max(1).values).cpu())
    mar = torch.cat(margins)
    return dict(acc=correct / n, loss=loss / n, margin_pos_frac=float((mar > 0).double().mean()),
                margin_median=float(mar.median()), margin_min=float(mar.min()), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=None,
                    help="checkpoint paths; default = the WP0 table in models/exact_inversion")
    ap.add_argument("--model-dir", default="models/exact_inversion")
    ap.add_argument("--mnist-root", default="dataset_reconstruction/data")
    ap.add_argument("--cifar-root", default="data")
    ap.add_argument("--out", default="results/base_training_gate.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device)
    paths = a.ckpts or [os.path.join(a.model_dir, s + ".pth") for s in WP0_ORDER]
    if a.out: os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    data = {}
    def split(kind):
        if kind not in data:
            if kind == "mnist":
                Xtr, ytr = read_idx(a.mnist_root, "train"); Xte, yte = read_idx(a.mnist_root, "test")
            else:
                Xtr, ytr, Xte, yte = cifar_newclass.load_cifar10(a.cifar_root)
            data[kind] = (torch.tensor(Xtr, device=dev).double(), torch.tensor(ytr, device=dev),
                          torch.tensor(Xte, device=dev).double(), torch.tensor(yte, device=dev))
        return data[kind]

    print(f"# base_training_gate  gate = train_acc >= {GATE_TRAIN_ACC} AND train_loss <= {GATE_TRAIN_LOSS}  "
          f"host={socket.gethostname()} git={git_hash()} device={dev}", flush=True)
    print(f"# {'checkpoint':<42} {'train_acc':>9} {'test_acc':>9} {'train_CE':>10} {'margin>0':>9}  gate", flush=True)
    rows = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem not in FAMILY:
            print(f"# SKIP {path}: no loader family registered for stem '{stem}'", flush=True); continue
        if not os.path.exists(path):
            print(f"# MISSING {path}", flush=True)
            rows.append(dict(part="BASE_GATE", ckpt=path, stem=stem, missing=True, fully_trained=False)); continue
        family, extra = FAMILY[stem]
        t0 = time.time()
        ck = torch.load(path, map_location="cpu", weights_only=False); ck["__path__"] = path
        f, desc, kind = build_forward(family, extra, ck, dev)
        Xtr, ytr, Xte, yte = split(kind)
        tr = evaluate(f, Xtr, ytr); te = evaluate(f, Xte, yte)
        rec = recorded_of(ck)
        passed = bool(tr["acc"] >= GATE_TRAIN_ACC and tr["loss"] <= GATE_TRAIN_LOSS)
        row = dict(part="BASE_GATE", ckpt=path, stem=stem, family=family, loader_extra=extra, arch=desc,
                   dataset=kind, n_train=tr["n"], n_test=te["n"],
                   train_acc=tr["acc"], test_acc=te["acc"], train_loss=tr["loss"], test_loss=te["loss"],
                   train_margin_pos_frac=tr["margin_pos_frac"], train_margin_median=tr["margin_median"],
                   train_margin_min=tr["margin_min"],
                   ckpt_recorded=rec,
                   recorded_test_acc_matches=(abs(float(rec["test_acc"]) - te["acc"]) < 5e-4
                                              if isinstance(rec.get("test_acc"), (int, float)) else None),
                   gate_train_acc=GATE_TRAIN_ACC, gate_train_loss=GATE_TRAIN_LOSS, fully_trained=passed,
                   file_mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))),
                   seconds=time.time() - t0, dtype="float64", device=str(dev), git=git_hash(),
                   host=socket.gethostname(), cmd=" ".join(sys.argv),
                   timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        rows.append(row)
        print(f"  {stem:<42} {tr['acc']*100:8.3f}% {te['acc']*100:8.2f}% {tr['loss']:10.3e} "
              f"{tr['margin_pos_frac']*100:8.3f}%  {'PASS' if passed else 'FAIL'}"
              f"   recorded: {json.dumps(rec)}", flush=True)
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as fh: fh.write(json.dumps(row) + "\n")
    n_pass = sum(1 for r in rows if r.get("fully_trained"))
    print(f"# {n_pass}/{len(rows)} checkpoints pass the gate; failing: "
          f"{[r['stem'] for r in rows if not r.get('fully_trained')]}", flush=True)


if __name__ == "__main__":
    main()
