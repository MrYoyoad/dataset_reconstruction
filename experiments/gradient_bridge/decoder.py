"""GB-Phase 1 gradient decoder f_φ: (A,B) -> ∇_W L, plus honest-metric helpers.

The decoder is a small MLP over the flattened adapter (A,B). Two output heads:
  - 'dense'   : predict the full [out,in] gradient directly. Feasible for the
                output layer (out·in = 1000) — the pipeline-validation smoke.
  - 'lowrank' : predict rank-k factors U[out,k], V[k,in]; grad = U@V. Keeps the
                hidden layer (out·in = 1e6) feasible without a 1e9-param head.

analytic_projection() gives the honest ceiling: the best recovery of ∇_W L that
is possible from the measurement alone (the projection onto col(B0)). Comparing
the decoder's full-gradient cosine against this projected cosine is the
anti-self-deception check (experiment_plan.md GB-Phase 1): a single-step adapter
only observes col(B0), so beating the projection means genuinely hallucinating
the out-of-subspace component from the proxy prior.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientDecoder(nn.Module):
    def __init__(self, in_dim, out_features, in_features, hidden=1024, depth=2,
                 out_mode='dense', out_rank=16):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.out_mode = out_mode
        self.out_rank = out_rank

        body, d = [], in_dim
        for _ in range(depth):
            body += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        self.body = nn.Sequential(*body)

        if out_mode == 'dense':
            self.head = nn.Linear(d, out_features * in_features)
        elif out_mode == 'lowrank':
            self.head_U = nn.Linear(d, out_features * out_rank)
            self.head_V = nn.Linear(d, out_rank * in_features)
        else:
            raise ValueError(f"out_mode must be 'dense' or 'lowrank', got {out_mode}")

    def forward(self, x):
        """x: [batch, in_dim] flattened (A,B). Returns [batch, out*in] predicted grad."""
        z = self.body(x)
        if self.out_mode == 'dense':
            return self.head(z)
        b = x.shape[0]
        U = self.head_U(z).view(b, self.out_features, self.out_rank)
        V = self.head_V(z).view(b, self.out_rank, self.in_features)
        return (U @ V).reshape(b, -1)


def cosine_loss(pred, target, eps=1e-12):
    """1 − mean cosine similarity (direction is what downstream inversion needs)."""
    return (1.0 - F.cosine_similarity(pred, target, dim=1, eps=eps)).mean()


def mean_cosine(pred, target, eps=1e-12):
    return F.cosine_similarity(pred, target, dim=1, eps=eps).mean().item()


def project_onto_colspace(grad, B0):
    """Projection of ∇_W L onto the measurement subspace col(B0). Returns [b,out,in].

    This is the honest measurement-only ceiling: a single-step adapter observes
    ∇_W L only through col(B0), so cos(P∇_W L, ∇_W L) = the fraction of gradient
    energy inside col(B0) (≈ √(r/out) for a generic direction). Computed with an
    orthonormal basis Q (reduced QR of B0) — stable even when out < r (the output
    layer, where B0ᵀB0 is rank-deficient and a gram pseudo-inverse blows up).

    Args:
        grad: [b, out, in] the true per-sample gradients.
        B0:   [b, out, r]  the LoRA measurement bases.
    """
    Q, _ = torch.linalg.qr(B0, mode='reduced')       # Q: [b, out, min(out,r)]
    return Q @ (Q.transpose(1, 2) @ grad)            # P·grad = Q Qᵀ grad
