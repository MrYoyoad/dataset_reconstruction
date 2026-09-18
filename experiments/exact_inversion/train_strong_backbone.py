#!/usr/bin/env python3
"""Train MNIST backbones of the SAME architecture to different accuracies, to isolate encoder QUALITY.

The trained-backbone result currently rests on the repo's checkpoint, which reaches 78.45% because it was
trained on a small balanced subset for the reconstruction setting.  On it the conditioning penalty against
a random encoder grew from 7x at k=6 to 1100x at k=17.  The open question -- and it is the load-bearing one,
since every realistic release sits on a strong backbone -- is whether a GOOD encoder makes that worse,
better, or leaves it unchanged.

Same architecture as the repo's model so the only variable is quality: 784 -> 1000 -> 1000 -> 10, GELU,
bias on the first layer only (CreateModel.NeuralNetwork with use_bias=False).  Trained on the FULL train
split.  Checkpoints are written in the repo's own format (a dict with a "state_dict" key) so
trained_backbone.py loads them unchanged.

Saves a mid-accuracy checkpoint (first to cross --mid-acc) and a final one, giving three points on the
quality axis together with the existing 78.45% model: weak / mid / strong.

  python -m experiments.exact_inversion.train_strong_backbone --out-dir models/exact_inversion

WP0 (2026-09-18) "fully trained" mode -- a `_full` twin trained to the base-training gate:
  --target-train-acc A / --min-train-loss L   stop at the first epoch whose FULL-train-split accuracy >= A and
                                              mean cross-entropy <= L (either alone if only one is given);
  --max-epochs E                              cap when the rule is on (default: --epochs);
  --init-from CKPT                            warm-start from an existing checkpoint of this format (the twin is
                                              then a CONTINUATION of the original's training, same architecture);
  --out PATH                                  write ONLY the final checkpoint to PATH (no _mid/_strong files).
  When the rule is on, Adam's lr is halved after 5 epochs without train-loss improvement (floor 1e-5) so the loss
  can actually collapse; the schedule is recorded in the checkpoint.  Train acc / loss / margin fraction and the
  test acc are recorded in the checkpoint dict; the tensor keys are unchanged, so every existing loader works.
  No augmentation, no weight decay, in every mode.

  python -u -m experiments.exact_inversion.train_strong_backbone --init-from models/exact_inversion/mnist_mlp_strong.pth \
      --out models/exact_inversion/mnist_mlp_strong_full.pth --target-train-acc 0.995 --min-train-loss 1e-2 --max-epochs 300

P5 (2026-09-18) activation twins:  --act {gelu,relu,tanh}  (default gelu = the original code path, byte-identical)
  selects the hidden activation and is recorded in the checkpoint dict as "act".  Loaders that do not read the field
  (trained_backbone, multilayer_lora, deep_stack) will apply GELU to a relu/tanh checkpoint and be WRONG.
  Warm-starting from a checkpoint that records a different act is refused.
"""
import argparse, json, os, socket, sys, time
import numpy as np
import torch, torch.nn as nn

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import read_idx

# hidden activation by name; "gelu" is the original hard-coded path (P5 activation twins read the others)
ACTS = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "tanh": torch.tanh}


def build(n_out=10):
    """n_out > 10 pads the head with logits that are never targets (softmax still runs over all of them): the
       certificate cap N' <= m - 1 is then moved with the data, encoder, batch and training all held fixed."""
    return nn.ModuleList([nn.Linear(784, 1000), nn.Linear(1000, 1000, bias=False),
                          nn.Linear(1000, n_out, bias=False)])


def fwd(layers, x, act="gelu"):
    phi = ACTS[act]
    h = phi(layers[0](x))
    h = phi(layers[1](h))
    return layers[2](h)


def save_ckpt(layers, path, acc, epoch, extra=None):
    sd = {"layers.0.weight": layers[0].weight.detach().double().cpu(),
          "layers.0.bias":   layers[0].bias.detach().double().cpu(),
          "layers.1.weight": layers[1].weight.detach().double().cpu(),
          "layers.2.weight": layers[2].weight.detach().double().cpu()}
    torch.save({"state_dict": sd, "epoch": epoch, "batch": None, "test_acc": acc, **(extra or {})}, path)
    print(f"    saved {path}  (test acc {acc*100:.2f}%)", flush=True)


def load_into(layers, path, act="gelu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("act", "gelu") != act:
        raise SystemExit(f"{path} records act={ck.get('act', 'gelu')!r} but --act {act!r} was requested")
    sd = ck["state_dict"]
    with torch.no_grad():
        layers[0].weight.copy_(sd["layers.0.weight"]); layers[0].bias.copy_(sd["layers.0.bias"])
        layers[1].weight.copy_(sd["layers.1.weight"]); layers[2].weight.copy_(sd["layers.2.weight"])


@torch.no_grad()
def split_stats(layers, X, y, bs=5000, act="gelu"):
    """Accuracy, mean cross-entropy and the fraction of positive margins over a whole split."""
    correct = 0; loss = 0.0; pos = 0
    for i in range(0, X.shape[0], bs):
        z = fwd(layers, X[i:i + bs], act); yb = y[i:i + bs]
        loss += float(nn.functional.cross_entropy(z, yb, reduction="sum")); correct += int((z.argmax(1) == yb).sum())
        zy = z.gather(1, yb[:, None])[:, 0]; zo = z.clone(); zo.scatter_(1, yb[:, None], -float("inf"))
        pos += int((zy - zo.max(1).values > 0).sum())
    n = X.shape[0]
    return dict(acc=correct / n, loss=loss / n, margin_pos_frac=pos / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mid-acc", type=float, default=0.90)
    ap.add_argument("--target-acc", type=float, default=0.97)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--out-dir", default="models/exact_inversion")
    ap.add_argument("--n-out", type=int, default=10, help="head width; > 10 pads with never-target logits")
    ap.add_argument("--name", default=None, help="checkpoint stem (default mnist_mlp); padded heads get _m{n_out}")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # ---- WP0 "fully trained" mode (see the docstring) ----
    ap.add_argument("--target-train-acc", type=float, default=None, help="stop when full-train accuracy >= this")
    ap.add_argument("--min-train-loss", type=float, default=None, help="stop when mean train cross-entropy <= this")
    ap.add_argument("--max-epochs", type=int, default=None, help="epoch cap when a stopping rule is on (default --epochs)")
    ap.add_argument("--init-from", default=None, help="warm-start from a checkpoint of this format")
    ap.add_argument("--out", default=None, help="write ONLY the final checkpoint here (no _mid/_strong files)")
    ap.add_argument("--plateau-patience", type=int, default=5, help="rule mode: halve lr after this many epochs without train-loss improvement")
    # ---- P5 activation twins ----
    ap.add_argument("--act", default="gelu", choices=sorted(ACTS), help="hidden activation; recorded in the checkpoint as 'act'")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    Xtr, ytr = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr, device=dev).float(); ytr_t = torch.tensor(ytr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev).float(); yte_t = torch.tensor(yte, device=dev)
    layers = build(a.n_out).to(dev).float()
    if a.init_from: load_into(layers, a.init_from, a.act)
    stem = a.name or ("mnist_mlp" if a.n_out == 10 else f"mnist_mlp_m{a.n_out}")
    opt = torch.optim.Adam(layers.parameters(), lr=a.lr)
    lossf = nn.CrossEntropyLoss()
    rule = a.target_train_acc is not None or a.min_train_loss is not None
    n_epochs = (a.max_epochs or a.epochs) if rule else a.epochs
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=a.plateau_patience, min_lr=1e-5)
             if rule else None)
    mid_done = False; t0 = time.time(); tr = None; stopped_at = None; lr_hist = []
    print(f"# training 784-1000-1000-{a.n_out} {a.act.upper()} on the FULL train split ({Xtr_t.shape[0]} images)  "
          f"git={git_hash()} host={socket.gethostname()}" + (f"  warm-start {a.init_from}" if a.init_from else "") +
          (f"  RULE train_acc>={a.target_train_acc} & train_loss<={a.min_train_loss}, max {n_epochs} epochs, "
           f"no augmentation, no weight decay" if rule else ""), flush=True)
    if rule:
        tr = split_stats(layers, Xtr_t, ytr_t, act=a.act); te0 = split_stats(layers, Xte_t, yte_t, act=a.act)
        print(f"  epoch  0  train acc {tr['acc']*100:.3f}%  train loss {tr['loss']:.3e}  test acc {te0['acc']*100:.2f}%", flush=True)
    for ep in range(n_epochs):
        perm = torch.randperm(Xtr_t.shape[0], device=dev)
        for i in range(0, Xtr_t.shape[0], a.bs):
            j = perm[i:i + a.bs]
            opt.zero_grad(); lossf(fwd(layers, Xtr_t[j], a.act), ytr_t[j]).backward(); opt.step()
        with torch.no_grad():
            acc = float((fwd(layers, Xte_t, a.act).argmax(1) == yte_t).float().mean())
        if rule:
            tr = split_stats(layers, Xtr_t, ytr_t, act=a.act); sched.step(tr["loss"]); lr_hist.append(opt.param_groups[0]["lr"])
            print(f"  epoch {ep+1:>3}  train acc {tr['acc']*100:.3f}%  train loss {tr['loss']:.3e}  "
                  f"test acc {acc*100:.2f}%  lr {opt.param_groups[0]['lr']:.1e}  {time.time()-t0:.0f}s", flush=True)
            ok_acc = a.target_train_acc is None or tr["acc"] >= a.target_train_acc
            ok_loss = a.min_train_loss is None or tr["loss"] <= a.min_train_loss
            if ok_acc and ok_loss:
                stopped_at = ep + 1; print(f"  stopping rule met at epoch {stopped_at}", flush=True); break
            continue
        print(f"  epoch {ep+1:>2}  test acc {acc*100:.2f}%", flush=True)
        if a.out: continue
        if not mid_done and acc >= a.mid_acc:
            save_ckpt(layers, os.path.join(a.out_dir, f"{stem}_mid.pth"), acc, ep + 1); mid_done = True
        if acc >= a.target_acc and ep >= 4:
            save_ckpt(layers, os.path.join(a.out_dir, f"{stem}_strong.pth"), acc, ep + 1)
    epochs_run = stopped_at or n_epochs
    tr = split_stats(layers, Xtr_t, ytr_t, act=a.act); te = split_stats(layers, Xte_t, yte_t, act=a.act); acc = te["acc"]
    extra = dict(act=a.act, train_acc=tr["acc"], train_loss=tr["loss"], train_margin_pos_frac=tr["margin_pos_frac"],
                 test_loss=te["loss"], epochs_run=epochs_run, init_from=a.init_from, seed=a.seed, lr=a.lr, bs=a.bs,
                 optimizer="Adam", augmentation=False, weight_decay=0.0,
                 stopping_rule=(dict(target_train_acc=a.target_train_acc, min_train_loss=a.min_train_loss,
                                     max_epochs=n_epochs, met=stopped_at is not None, plateau_patience=a.plateau_patience,
                                     lr_final=opt.param_groups[0]["lr"]) if rule else None),
                 git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
    path = a.out or os.path.join(a.out_dir, f"{stem}_strong.pth")
    if a.out: os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_ckpt(layers, path, acc, epochs_run, extra)
    print(json.dumps(dict(final_test_acc=acc, final_train_acc=tr["acc"], final_train_loss=tr["loss"],
                          rule_met=stopped_at is not None if rule else None, epochs=epochs_run, out=path,
                          git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))), flush=True)


if __name__ == "__main__":
    main()
