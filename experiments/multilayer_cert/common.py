"""Shared pieces for the multilayer-certificate track (theory/T1..T6).

FP64 everywhere.  The network is a GELU MLP whose every layer carries a LoRA adapter; layer 1's input is the
frozen data, so layer 1 is the classical single-layer (frozen-input) case and layers >= 2 are the objects the
track is about.  Conventions match experiments/cifar/cifar_newclass.py exactly:

    B, A = B - lr * (D @ (A @ H).T),  A - lr * (B.T @ D @ H.T)      (simultaneous, both from time-t values)

with A_0 Gaussian / sqrt(n), B_0 = 0.
"""
import math
import torch

torch.set_default_dtype(torch.float64)


def gelu(x):
    return torch.nn.functional.gelu(x)


class MultiLoRANet:
    """L layers, adapters on all of them.  Everything the theory needs is recorded per step."""

    def __init__(self, dims, r, N, seed=0, dev="cpu", s=1.0):
        g = torch.Generator().manual_seed(seed)
        self.dims, self.L, self.r, self.N, self.s, self.dev = dims, len(dims) - 1, r, N, s, dev
        # frozen base weights, unit-ish operator norm so kappa ~ 1 (T1's normalised regime)
        self.W0 = [(torch.randn(dims[i + 1], dims[i], generator=g) / math.sqrt(dims[i])).to(dev)
                   for i in range(self.L)]
        self.A0 = [(torch.randn(r, dims[i], generator=g) / math.sqrt(dims[i])).to(dev) for i in range(self.L)]
        self.B0 = [torch.zeros(dims[i + 1], r, dtype=torch.float64, device=dev) for i in range(self.L)]

    def base_reps(self, x):
        """H_l^0 for every layer: the base network's representations (all adapters at 0)."""
        H, out = x, [x]
        for i in range(self.L):
            H = self.W0[i] @ H
            if i < self.L - 1:
                H = gelu(H)
            out.append(H)
        return out                                    # out[l] = input to layer l (0-indexed), out[L] = logits

    def forward_reps(self, x, A, B):
        H, out = x, [x]
        for i in range(self.L):
            H = (self.W0[i] + self.s * B[i] @ A[i]) @ H
            if i < self.L - 1:
                H = gelu(H)
            out.append(H)
        return out

    def train(self, x, y, T, lr, record=True):
        """Full-batch SGD on cross-entropy.  Returns (A, B, reps_per_step).

        reps[t][l] is the input to layer l at step t.  reps[0][l] == H_l^0 exactly, because B_0 = 0.
        """
        A = [a.clone() for a in self.A0]
        B = [b.clone() for b in self.B0]
        Y = torch.eye(self.dims[-1], device=self.dev)[y].T
        reps = []
        for t in range(T):
            H = self.forward_reps(x, A, B)
            if record:
                reps.append([h.detach().clone() for h in H])
            # backprop by hand so D_l (dL/dZ_l) is available explicitly -- the theory is stated in terms of it
            D = [None] * self.L
            d = (torch.softmax(H[-1], 0) - Y) / self.N                       # dL/dZ_{L-1}
            for i in range(self.L - 1, -1, -1):
                D[i] = d
                if i > 0:
                    Wi = self.W0[i] + self.s * B[i] @ A[i]
                    pre = self.W0[i - 1] @ H[i - 1] + self.s * (B[i - 1] @ (A[i - 1] @ H[i - 1]))
                    g = _gelu_prime(pre)                             # explicit GELU derivative
                    d = (Wi.T @ d) * g
            nA, nB = [], []
            for i in range(self.L):
                Hi = H[i]
                nB.append(B[i] - lr * (D[i] @ (A[i] @ Hi).T) * self.s)
                nA.append(A[i] - lr * (B[i].T @ D[i] @ Hi.T) * self.s)
            A, B = nA, nB
        return A, B, reps


def _gelu_prime(z):
    """d/dz of the exact (erf) GELU used by torch's default."""
    c = 1.0 / math.sqrt(2.0)
    Phi = 0.5 * (1.0 + torch.erf(z * c))
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return Phi + z * phi


def certificate(A_T, B_T, keep=None, tol=1e-10):
    """C = P_{row(B_T)^perp} A_T.

    keep=None -> FULL certificate: project out every numerically nonzero right-singular direction of B_T
                 (this is C_full of theory/T2, exact but of rank r - N').
    keep=q    -> TRUNCATED certificate Ctil: keep only the top-q directions (rank r - q).
    Returns (C, q_used, singular values of B_T).
    """
    sB = torch.linalg.svdvals(B_T)
    _, _, Vh = torch.linalg.svd(B_T, full_matrices=False)
    q = int((sB > tol * sB[0]).sum()) if keep is None else int(keep)
    Q = Vh[:q].T
    return A_T - Q @ (Q.T @ A_T), q, sB


def rel_annihilation(C, H):
    """The scale-free residual rho = ||C H|| / (||C|| ||H||)."""
    return float((C @ H).norm() / (C.norm() * H.norm()))


def numrank(M, rtol=1e-10, ref=None):
    """Numerical rank.

    `ref` is the natural scale of the matrix M was built FROM (e.g. ||A_T|| for C = (I-QQ^T)A_T).  Without it a
    purely relative threshold `s > rtol*s[0]` calls a numerically ZERO matrix FULL rank, because its own s[0] is
    already at rounding level -- which is exactly how a certificate that has been annihilated (rank 0) reports
    rank r.  See LESSONS_LEARNED 2026-09-07.
    """
    s = torch.linalg.svdvals(M)
    scale = float(s[0]) if ref is None else max(float(s[0]), float(ref))
    return int((s > rtol * scale).sum()), s


def span_of(mats, rtol=1e-10):
    """dim and orthonormal basis of the span of the columns of a list of matrices."""
    Mcat = torch.cat(mats, dim=1)
    U, s, _ = torch.linalg.svd(Mcat, full_matrices=False)
    d = int((s > rtol * s[0]).sum())
    return d, U[:, :d]
