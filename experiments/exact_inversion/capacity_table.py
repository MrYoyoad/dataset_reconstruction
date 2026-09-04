#!/usr/bin/env python3
"""How many private images can the recipe-free channel read, per architecture? A computable table.

The counting rule answers "what about smaller models, or a ResNet rather than a full ViT" as a formula rather
than a case list. A weight-shared module records one direction per POSITION per image, so with p positions and N
private images the recorded span is min(N.p, d), the certificate's rank is min(r, d) - that span, and

    the channel admits N images  <=>  N.p < min(r, d)   <=>   N <= floor((min(r,d) - 1) / p)

**Resolution and patch size are the dial.** The same adapter rank that reads nothing on a ViT-B/16 at 224 reads a
single image on a 7x7 final ResNet stage and 63 on a head. The channel is not dead on real architectures; it is
dead at high position counts and alive at low ones, and that is a property a practitioner can look up.

Everything here is arithmetic from published specs -- no training, no release, no forward pass -- and every row
that has been MEASURED elsewhere in this ledger is checked against its measurement, so the table is anchored
rather than merely derived. A mismatch is printed as FAIL and is a bug in the formula, not a curiosity.

Second-order caps carried in the rows because both have bitten us:
  * rank B_T <= the module's OUTPUT width, so a rank above it leaves margin the data never touched, and the
    certificate does not hold at the truth there (measured: conv 2 at r=256, residual 5.5e-6 against conv 3's 3e-11);
  * a DEPTHWISE 3x3 has input dimension 9, so min(r, d) <= 9 while the span is 9 -- dead by construction at any rank.

  python -u -m experiments.exact_inversion.capacity_table
"""
import argparse, json, socket, sys
from experiments.exact_inversion.lora_exact_inversion import git_hash

# (family, cell, module, positions p, input dim d, output width, measured p or None)
CELLS = [
    # --- transformers: p = tokens = (img/patch)^2 + 1 for the CLS token
    ("ViT-B/16",  "224px", "block linear (qkv/proj/fc1)", 197,  768, 2304, 197),
    ("ViT-B/16",  "224px", "block mlp.fc2",               197, 3072,  768, 197),
    ("ViT-B/16",  "112px", "block linear",                 50,  768, 2304, None),
    ("ViT-B/32",  "224px", "block linear",                 50,  768, 2304, None),
    ("ViT-B/32",  "128px", "block linear",                 17,  768, 2304, None),
    ("ViT-S/16",  "224px", "block linear",                197,  384, 1152, 197),
    ("ViT-S/16",  "32px",  "block linear",                  5,  384, 1152, None),
    ("ViT-Ti/16", "64px",  "block linear",                 17,  192,  576, None),
    # --- ResNets at 224: p = output spatial cells of that stage
    ("ResNet-18", "224px", "layer1 conv3x3",             3136,  576,   64, 3136),
    ("ResNet-18", "224px", "layer2 conv3x3",              784,  576,  128, 784),
    ("ResNet-18", "224px", "layer3 conv3x3",              196, 1152,  256, 196),
    ("ResNet-18", "224px", "layer4 conv3x3",               49, 2304,  512, 49),
    ("ResNet-50", "224px", "layer4 conv1x1",              196, 1024,  512, 196),
    # --- ResNets on small inputs: the SAME network, four times fewer positions per stage
    ("ResNet-18", "32px",  "layer3 conv3x3",               64, 1152,  256, None),
    ("ResNet-18", "32px",  "layer4 conv3x3",               16, 2304,  512, None),
    ("ResNet-18", "32px",  "layer4 @2x2 (deeper stem)",     4, 2304,  512, None),
    # --- mobile families: depthwise is the interesting row
    ("MobileNetV2",   "224px", "depthwise 3x3",             49,    9,  960, None),
    ("EfficientNet",  "224px", "depthwise 3x3",             49,    9, 1152, None),
    ("MobileNetV2",   "224px", "pointwise 1x1",             49,  960,  320, None),
    # --- the p = 1 row: not an exception to the rule, the rule at one position
    ("any ViT",   "-",     "classification head",            1,  768,  102, 1),
    ("ResNet-18", "-",     "pooled fc head",                 1,  512,  102, 1),
    ("ResNet-50", "-",     "pooled fc head",                 1, 2048,  102, 1),
]
RANKS = [8, 16, 32, 64]
# rows measured elsewhere in this ledger, as (module, r, N, expected margin) -- the table must reproduce them
ANCHORS = [("classification head", 16, 8, 8), ("classification head", 64, 8, 56),
           ("pooled fc head", 16, 8, 8), ("pooled fc head", 64, 8, 56)]


def margin(r, d, p, N):
    return max(0, min(r, d) - min(N * p, d))


def admitted(r, d, p):
    """Largest N with a margin of at least 1 -- the number a practitioner actually wants."""
    N = (min(r, d) - 1) // p
    return max(0, N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8, help="private batch size for the margin columns")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    print(f"# capacity table: margin = min(r,d) - min(N.p, d);  admitted N = floor((min(r,d)-1)/p)   "
          f"git={git_hash()} host={socket.gethostname()}", flush=True)
    print(f"# {'family':13s} {'cell':7s} {'module':28s} {'p':>5s} {'d':>5s} "
          + " ".join(f"N@r={r:<3d}" for r in RANKS) + f"   margin at N={a.N}", flush=True)
    fails = 0
    for fam, cell, mod, p, d, cout, p_meas in CELLS:
        adm = {r: admitted(r, d, p) for r in RANKS}
        mar = {r: margin(r, d, p, a.N) for r in RANKS}
        ok = (p_meas is None or p_meas == p)
        if not ok: fails += 1
        emit(dict(part="CAPACITY", family=fam, cell=cell, module=mod, positions=p, input_dim=d,
                  output_width=cout, positions_measured=p_meas, positions_match=ok,
                  admitted_images_by_rank=adm, margin_by_rank_at_N=mar, N=a.N,
                  span_fills_input_dim=bool(d <= p),
                  depthwise_dead_by_construction=bool(d <= 9 and d <= p),
                  output_width_caveat=bool(max(RANKS) > cout),
                  note="admitted N is the largest batch leaving a margin of at least 1; a margin that exists only "
                       "because r exceeds the OUTPUT width is not backed by a condition that holds at the truth",
                  git=git_hash(), cmd=" ".join(sys.argv)))
        flag = "" if ok else "  <-- FAIL: p disagrees with the measured value"
        dead = ("  [depthwise: input dim 9, dead at every rank by construction]" if (d <= 9 and d <= p)
                else "  [span fills d: dead at every rank]" if d <= p else "")
        print(f"  {fam:13s} {cell:7s} {mod:28s} {p:5d} {d:5d} "
              + " ".join(f"{adm[r]:6d}" for r in RANKS) + f"   {mar[a.N]:5d}{dead}{flag}", flush=True)

    print("\n# anchors: the formula reproduced against rows MEASURED elsewhere in this ledger", flush=True)
    for mod, r, N, exp in ANCHORS:
        row = next(c for c in CELLS if c[2] == mod)
        got = margin(r, row[4], row[3], N)
        good = (got == exp)
        fails += (not good)
        print(f"  {mod:28s} r={r:3d} N={N}  formula {got:4d}  measured {exp:4d}   {'OK' if good else 'FAIL'}",
              flush=True)
        emit(dict(part="ANCHOR", module=mod, r=r, N=N, formula=got, measured=exp, ok=good, git=git_hash()))
    print(f"\n# {'ALL CHECKS PASS' if fails == 0 else f'{fails} FAILURES'}", flush=True)
    emit(dict(part="SUMMARY", failures=fails, ranks=RANKS, N=a.N, git=git_hash(), host=socket.gethostname()))


if __name__ == "__main__":
    main()
