"""Equations as PNG via matplotlib mathtext (no LaTeX toolchain on WEXAC).

Copied from notes/make_identifiability_pdf.py::render_math (md5-cached, tight bbox),
with a white background so PowerPoint shows crisp black ink.
"""
import hashlib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config as C

INK = "#000000"
plt.rcParams["mathtext.fontset"] = "cm"       # Computer Modern look, like the thesis


def render_math(latex, name, *, fontsize=26, dpi=300, color=INK, out_dir=None):
    """Render `$latex$` to <out_dir>/<name>.png (cached by content hash). Returns the path."""
    out_dir = out_dir or C.EQ_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    key = hashlib.md5((latex + str(fontsize) + color).encode()).hexdigest()[:12]
    stamp = os.path.join(out_dir, f".{name}.{key}")
    if os.path.exists(path) and os.path.exists(stamp):
        return path
    for f in os.listdir(out_dir):
        if f.startswith(f".{name}."):
            os.remove(os.path.join(out_dir, f))
    fig = plt.figure(figsize=(0.1, 0.1))
    t = fig.text(0, 0, f"${latex}$", fontsize=fontsize, color=color)
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    w_in, h_in = bb.width / fig.dpi, bb.height / fig.dpi
    plt.close(fig)
    fig = plt.figure(figsize=(w_in + 0.1, h_in + 0.1))
    fig.text(0.5, 0.5, f"${latex}$", fontsize=fontsize, ha="center", va="center", color=color)
    fig.savefig(path, dpi=dpi, transparent=False, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    open(stamp, "w").close()
    return path


def render_lines(lines, name, *, fontsize=24, dpi=300, color=INK, out_dir=None, gap=0.55):
    """Several equations stacked (one PNG). `lines` = list of latex strings (no $)."""
    out_dir = out_dir or C.EQ_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    key = hashlib.md5(("|".join(lines) + str(fontsize) + str(gap)).encode()).hexdigest()[:12]
    stamp = os.path.join(out_dir, f".{name}.{key}")
    if os.path.exists(path) and os.path.exists(stamp):
        return path
    for f in os.listdir(out_dir):
        if f.startswith(f".{name}."):
            os.remove(os.path.join(out_dir, f))
    n = len(lines)
    fig = plt.figure(figsize=(8, gap * n + 0.2))
    for i, l in enumerate(lines):
        fig.text(0.02, 1 - (i + 0.5) / n, f"${l}$", fontsize=fontsize, ha="left", va="center", color=color)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    open(stamp, "w").close()
    return path
