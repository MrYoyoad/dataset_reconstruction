"""Depth-generic frozen MLP + LoRA on every layer, shared by the deep-backbone trainer and the layer curve.

`multilayer_lora.py` hard-wires three weight matrices (784-1000-1000-10).  The extended-layer question -- does
the pixel-space constraint count keep growing as more layers are adapted, or flatten -- needs arbitrary depth,
so the forward map, the unrolled SGD release and the checkpoint format live here once.

Convention: activations are COLUMN-major, x is (d_in, N).  With D weight matrices there are D adapted layers;
`inputs_of` returns the input to each of them, h^0 = x through h^{D-1}, which is what the per-layer certificate
condition C_l h^l = 0 is stated on.  Bias on the first layer only, GELU between layers, none after the head --
the same family as the repo's 784-1000-1000-10 backbone, so the 3-layer numbers stay comparable.
"""
import torch

GELU = torch.nn.functional.gelu


def inputs_of(x, Ws, b1, As=None, Bs=None):
    """Input to every weight matrix: [h^0 = x, h^1, ..., h^{D-1}].  As/Bs None = the frozen backbone."""
    hs = [x]
    h = x
    for l in range(len(Ws) - 1):
        z = Ws[l] @ h + (b1[:, None] if l == 0 else 0)
        if As is not None and As[l] is not None:            # None = this layer is FROZEN (part of the encoder)
            z = z + Bs[l] @ (As[l] @ h)
        h = GELU(z)
        hs.append(h)
    return hs


def forward_deep(x, Ws, b1, As=None, Bs=None):
    """Logits (m, N).  The head is Ws[-1] with no activation after it."""
    hs = inputs_of(x, Ws, b1, As, Bs)
    h = hs[-1]
    z = Ws[-1] @ h
    if As is not None and As[-1] is not None:
        z = z + Bs[-1] @ (As[-1] @ h)
    return z


def run_training_deep(x, Ws, b1, A0s, y, m, T, lr, create_graph=False):
    """Plain unrolled SGD on all LoRA factors of every layer; B starts at zero (so the certificate exists)."""
    As = [None if a is None else (a if a.requires_grad else a.detach().requires_grad_(True)) for a in A0s]
    Bs = [None if a is None else torch.zeros(w.shape[0], a.shape[0], dtype=x.dtype, device=x.device,
                                             requires_grad=True) for w, a in zip(Ws, A0s)]
    live = [i for i, a in enumerate(As) if a is not None]
    Y = torch.eye(m, device=x.device)[y].T
    N = x.shape[1]
    with torch.enable_grad():
        for _ in range(T):
            z = forward_deep(x, Ws, b1, As, Bs)
            zs = z - z.max(dim=0, keepdim=True).values
            p = torch.exp(zs); p = p / p.sum(dim=0, keepdim=True)
            loss = -(Y * torch.log(p + 1e-300)).sum() / N
            params = [As[i] for i in live] + [Bs[i] for i in live]
            gs = torch.autograd.grad(loss, params, create_graph=create_graph)
            n = len(live)
            for j, i in enumerate(live):
                As[i] = As[i] - lr * gs[j]
                Bs[i] = Bs[i] - lr * gs[n + j]
                if not create_graph:
                    As[i] = As[i].detach().requires_grad_(True)
                    Bs[i] = Bs[i].detach().requires_grad_(True)
    if not create_graph:
        As = [None if a is None else a.detach() for a in As]
        Bs = [None if b is None else b.detach() for b in Bs]
    return As, Bs


def save_deep(path, Ws, b1, acc, meta):
    """Explicit ORDERED list, not a state_dict: 'layers.10.weight' sorts before 'layers.2.weight'."""
    torch.save(dict(Ws=[w.detach().double().cpu() for w in Ws], b1=b1.detach().double().cpu(),
                    test_acc=acc, **meta), path)


def load_deep(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    return [w.to(dev).double() for w in ck["Ws"]], ck["b1"].to(dev).double(), ck
