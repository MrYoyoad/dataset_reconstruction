"""Supervisor-meeting deck (2026-08-31) — modular python-pptx generator.

Layout per style_guide/pptx.md §Modular Generator Architecture:
    config.py     paths, palette, layout constants, figure dictionary
    helpers.py    slide primitives (text, runs, images, rects, equations, notes)
    eq_render.py  matplotlib-mathtext equation PNGs (no LaTeX on WEXAC)
    slides_*.py   one function per slide, grouped by deck part
"""
