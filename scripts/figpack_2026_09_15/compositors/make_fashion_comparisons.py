"""Recompose verified original/output pairs from figures_for_gal, page 3.

All image content comes from the supplied source panel. No image synthesis,
interpolation-based enhancement, substitution or polarity change is applied.
Coordinates below are in the embedded 1754 x 1240 raster's pixel system.
"""
from pathlib import Path
import fitz

LABEL_FONT = '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf'
LABEL_METRICS = fitz.Font(fontfile=LABEL_FONT)

FIG = Path(__file__).resolve().parents[1] / "figures"
SOURCE = fitz.open(FIG / "fashion_original.pdf")
SCALE = SOURCE[0].rect.width / 1754
# Inclusive black-tile extents in source raster; use 133-pixel square clips.
LEFT = {1: 306.5, 2: 465.5, 3: 622.5, 4: 781.5,
        5: 939.5, 6: 1097.5, 7: 1256.5, 8: 1413.5}
TOP = {"original": 410.5, "output": 791.5}


def compose(name, columns, per, status=None):
    tile, gap, pitch, row_pitch = 64, 4, 146, 75
    if status:
        row_pitch += 16
    rows = (len(columns) + per - 1) // per
    doc = fitz.open()
    page = doc.new_page(width=per * pitch - 10, height=rows * row_pitch - 5)
    page.insert_font(fontname='LabelRoman', fontfile=LABEL_FONT)
    for i, column in enumerate(columns):
        x, y = (i % per) * pitch, (i // per) * row_pitch
        for j, row in enumerate(("original", "output")):
            sx, sy = LEFT[column], TOP[row]
            clip = fitz.Rect(sx * SCALE, sy * SCALE,
                             (sx + 133) * SCALE, (sy + 133) * SCALE)
            target = fitz.Rect(x + j * (tile + gap), y,
                               x + j * (tile + gap) + tile, y + tile)
            page.show_pdf_page(target, SOURCE, 0, clip=clip)
            page.draw_rect(target, color=(.65, .65, .65), width=.2)
        if status:
            text = status[i]
            tx = x + (132 - LABEL_METRICS.text_length(text, fontsize=10.5)) / 2
            page.insert_text((tx, y + tile + 14), text, fontname="LabelRoman", fontsize=10.5)
    doc.save(FIG / name, garbage=4, deflate=True)


# All four successes, identified as "landed" in the supplied panel.
compose("fashion_mnist_recovered.pdf", [2, 3, 6, 7], 4)
compose("fashion_mnist_recovered_grid.pdf", [2, 3, 6, 7], 2)
# Two genuine closest outputs that did not recover their projected targets.
compose("fashion_mnist_remaining.pdf", [4, 5], 2)
# Positive overview: all target matches plus two recognizable approximations.
compose("fashion_mnist_overview.pdf", [2, 3, 6, 7, 4, 5], 3,
        ["Target match"] * 4 + ["Approximate"] * 2)
