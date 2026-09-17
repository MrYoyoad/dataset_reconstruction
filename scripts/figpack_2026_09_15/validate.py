"""Bounded numerical sanity checks for the new theoretical illustrations.

These are not reconstruction experiments and do not establish empirical claims.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import fitz

work = Path(__file__).resolve().parent
rng = np.random.default_rng(713)
n, r, m, count = 9, 6, 4, 2
H = rng.normal(size=(n, count))
A0 = rng.normal(size=(r, n))
A = A0.copy()
B = np.zeros((m, r))
W = rng.normal(size=(m, n))
Y = rng.normal(size=(m, count))
u, singular, _ = np.linalg.svd(A0 @ H, full_matrices=True)
q = np.linalg.matrix_rank(H)
P = np.eye(r) - u[:, :q] @ u[:, :q].T
eta = .0005
for step in range(100):
    D = ((W + B @ A) @ H - Y) / count
    grad_B = D @ (A @ H).T
    grad_A = B.T @ D @ H.T
    A, B = A - eta * grad_A, B - eta * grad_B
    assert np.linalg.norm(B @ P) < 1e-10
    assert np.linalg.norm(P @ (A - A0)) < 1e-10
    if step == 0:
        assert np.array_equal(A, A0)
        assert np.allclose(B, -eta * D @ (A0 @ H).T)
        assert np.allclose(B @ A, -eta * D @ H.T @ A0.T @ A0)

assert np.linalg.matrix_rank(B) == q
assert np.linalg.matrix_rank(P @ A) == r - q
assert np.linalg.norm(P @ A @ H) < 1e-10

V = A0 @ H
Lambda = rng.normal(size=(m, count))
M = np.array([[2., .4], [.3, 1.1]])
assert np.allclose(Lambda @ V.T,
                   (Lambda @ np.linalg.inv(M).T) @ (V @ M).T)

normal = np.array([-1., 0., 1.]) / np.sqrt(2)
C = np.outer(normal, normal)
t = np.linspace(-2., 2., 40001)
psi = np.stack([np.ones_like(t), t, t*t])
score = np.abs(t*t - 1) / (np.sqrt(2)*np.sqrt(1 + t*t + t**4))
assert np.allclose(np.linalg.norm(C @ psi, axis=0) / np.linalg.norm(psi, axis=0), score)
epsilon = .25
gamma = (1.25**2 - 1) / (np.sqrt(2)*np.sqrt(1+1.25**2+1.25**4))
distance = np.minimum(abs(t+1), abs(t-1))
assert np.min(score[distance >= epsilon]) >= gamma - 1e-12
assert np.all(distance[score <= .09] < epsilon)

sources = sorted(work.glob('[0-9][0-9]_*.tex'))
assert len(sources) == 8
for source in sources:
    doc = fitz.open(source.with_suffix('.pdf'))
    assert len(doc) == 1
    page = doc[0]
    assert abs(page.rect.width / page.rect.height - 1.6) < 1e-5
    for word in page.get_text('words'):
        assert word[0] >= 8 and word[1] >= 8
        assert word[2] <= page.rect.width-8 and word[3] <= page.rect.height-8
    log = source.with_suffix('.log').read_text()
    assert 'Overfull' not in log
    assert 'Underfull' not in log
    assert source.with_suffix('.png').is_file()

with fitz.open(work/'LoRA_figures_only.pdf') as doc:
    assert len(doc) == 8
    assert len(doc.get_toc()) == 8

provenance = {}
originals = work.parents[2] / 'output' / 'meeting_handoff' / 'figures'
if originals.is_dir():
    for asset in sorted((work/'assets').glob('*.pdf')):
        original = originals/asset.name
        assert original.is_file()
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        assert digest == hashlib.sha256(original.read_bytes()).hexdigest()
        provenance[asset.name] = digest

print(json.dumps({'figures':8, 'finite_step_invariants':'pass',
                  'one_step_identity':'pass', 'basis_change_identity':'pass',
                  'toy_score_and_threshold':'pass', 'layout_checks':'pass',
                  'source_asset_hashes':provenance}, indent=2))
