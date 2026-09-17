"""Original, actual PCA training target, NTK and certificate: four matched A's.

The underlying scientific source crops remain unchanged. The raw originals were
matched across supplied panels; see ground_truth_provenance.md for that limit.
"""
from pathlib import Path
import fitz

LABEL_FONT = '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf'

FIG = Path(__file__).resolve().parents[1] / 'figures'
rows = [
    ('Original', 'letters_a_raw_gt.pdf'),
    ('PCA training target', 'letters_a_pca.pdf'),
    ('NTK output', 'letters_a_ntk.pdf'),
    ('Certificate output', 'letters_a_certificate.pdf'),
]
doc = fitz.open()
tile, step, label_width, row_height = 52, 64, 113, 60
page = doc.new_page(width=label_width + 3 * step + tile,
                    height=3 * row_height + tile)
page.insert_font(fontname='LabelRoman', fontfile=LABEL_FONT)
for row, (label, name) in enumerate(rows):
    source = fitz.open(FIG / name)
    y = row * row_height
    page.insert_text((0, y + tile / 2 + 3.5), label,
                     fontname='LabelRoman', fontsize=11.5)
    for col, sx in enumerate([0, 177, 354, 531]):
        x = label_width + col * step
        target = fitz.Rect(x, y, x + tile, y + tile)
        page.show_pdf_page(target, source, 0,
                           clip=fitz.Rect(sx, 0, sx + 147, 147))
        page.draw_rect(target, color=(.72, .72, .72), width=.2)
doc.save(FIG / 'letters_four_rows.pdf', garbage=4, deflate=True)
