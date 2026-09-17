#!/usr/bin/env python3
"""C7 recomputed on the gate's images (ledger Q7): pixel-PCA projection error in two_walls.py's EXACT convention, on
BOTH index sets -- the oracle ladder's (torch.Generator().manual_seed(seed+7), randperm over the class TEST split; the
images the gate brackets were measured on) and two_walls' (np.random.RandomState(seed).permutation; the images C7
reported) -- for motorcycle (MLP release) and keyboard (CNN release). Closed form, FP64, CPU, no decoder.

  python -u -m experiments.decoder_chart.c7_index_sets
"""
import argparse, json, os, socket, sys
import numpy as np, torch
from experiments.cifar.cifar_newclass import load_cifar100_class
from experiments.exact_inversion.lora_exact_inversion import git_hash

torch.set_default_dtype(torch.float64)
GATES = {"mlp_motorcycle": ("motorcycle", (0.0124, 0.0186)), "cnn_keyboard": ("keyboard", (0.0045, 0.0090))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 32, 66, 128, 384]); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="results/decoder_chart")
    a = ap.parse_args(); jobid = os.environ.get("LSB_JOBID", "local"); os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"c7_index_sets_{jobid}.jsonl")
    for name, (cls, gate) in GATES.items():
        pool, cname = load_cifar100_class(a.data_root, cls)
        Pub = torch.tensor(pool["train"], dtype=torch.float64); Pri = torch.tensor(pool["test"], dtype=torch.float64)
        g = torch.Generator().manual_seed(a.seed + 7); idx_ladder = torch.randperm(Pri.shape[0], generator=g)[: a.N].tolist()
        idx_c7 = np.random.RandomState(a.seed).permutation(len(Pri))[: a.N].tolist()
        mu = Pub.mean(0); _, _, Vh = torch.linalg.svd(Pub - mu, full_matrices=False)           # two_walls.py verbatim
        print(f"\n## {name} ('{cname}')  gate bracket {gate}  ladder idx {idx_ladder}  two_walls idx {idx_c7}")
        print(f"{'k':>4} | {'set':>9} | {'mean':>7} {'median':>7} {'min':>7} {'max':>7} | x low  x high")
        for k in a.ks:
            V = Vh[:k].T
            for label, idx in (("ladder", idx_ladder), ("two_walls", idx_c7)):
                X = Pri[idx]; R = (X - mu) - (X - mu) @ V @ V.T
                err = torch.linalg.norm(R, dim=1) / torch.linalg.norm(X, dim=1)
                row = dict(part="c7_index_sets", example=name, class_name=cname, index_set=label, indices=idx,
                           selection=("ladder_cell.py: torch.randperm(seed+7)" if label == "ladder" else "two_walls.py: np.random.RandomState(seed)"),
                           k=k, N=a.N, seed=a.seed, precision="fp64", chart="public PCA of the added class (train split)",
                           err_mean=float(err.mean()), err_median=float(err.median()), err_min=float(err.min()), err_max=float(err.max()),
                           err_per_image=[float(v) for v in err], gate_low=gate[0], gate_high=gate[1],
                           ratio_mean_to_gate_low=float(err.mean()) / gate[0], ratio_mean_to_gate_high=float(err.mean()) / gate[1],
                           ratio_median_to_gate_low=float(err.median()) / gate[0], ratio_median_to_gate_high=float(err.median()) / gate[1],
                           git=git_hash(), host=socket.gethostname(), jobid=jobid, cmd=" ".join(sys.argv))
                with open(out, "a") as f: f.write(json.dumps(row) + "\n")
                print(f"{k:>4} | {label:>9} | {row['err_mean']:7.4f} {row['err_median']:7.4f} {row['err_min']:7.4f} {row['err_max']:7.4f} | "
                      f"{row['ratio_mean_to_gate_low']:5.1f}  {row['ratio_mean_to_gate_high']:5.1f}   (mean vs bracket)")
    print(f"# rows -> {out}")


if __name__ == "__main__":
    main()
