from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
SOURCE = OUT / 'letters_raw_source_panel.png'

# Crop existing scientific figures only. Do not synthesize or enhance any glyph.
# The first four PCA glyphs agree across the A-only and mixed A/T source panels.
# Use the corresponding raw row from the former, and all fitted rows from the latter.
source_picture = fitz.open(SOURCE)
source = fitz.open('pdf', source_picture.convert_to_pdf())
source_page = source[0]
image_box = fitz.Rect(source_page.get_image_info()[0]['bbox'])
sx, sy = image_box.width / 672, image_box.height / 243
raw = fitz.open()
page = raw.new_page(width=679, height=147)
for j in range(4):
    x0 = 235 + 46.5 * j
    clip = fitz.Rect(image_box.x0 + sx*x0,
                     image_box.y0 + sy*50.5,
                     image_box.x0 + sx*(x0+46.5),
                     image_box.y0 + sy*97)
    target = fitz.Rect(177*j, 0, 177*j+147, 147)
    page.show_pdf_page(target, source, 0, clip=clip)
raw.save(OUT / 'letters_a_raw_gt.pdf', garbage=4, deflate=True)

for source_name, output_name in [
    ('letters_row_1.png', 'letters_a_pca.pdf'),
    ('letters_ntk_row.png', 'letters_a_ntk.pdf'),
    ('letters_row_2.png', 'letters_a_certificate.pdf'),
]:
    picture = fitz.open(OUT / source_name)
    pdf = fitz.open('pdf', picture.convert_to_pdf())
    box = pdf[0].rect
    # Existing image rows are aligned on the 177-pixel column stride.
    clip = fitz.Rect(0, 0, box.width * 679/1387, box.height)
    row = fitz.open()
    p = row.new_page(width=679, height=147)
    p.show_pdf_page(p.rect, pdf, 0, clip=clip, keep_proportion=False)
    row.save(OUT / output_name, garbage=4, deflate=True)

print('Created four source-derived A-column rows; original display polarity retained.')
