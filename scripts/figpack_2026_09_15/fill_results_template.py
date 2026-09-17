"""Eight matched columns: actual original, PCA training input, reconstruction.

All image tiles are placed directly from the supplied result PDFs. No image
content is generated, enhanced, resynthesized, or substituted.
"""
from pathlib import Path
import subprocess
import fitz

WORK = Path(__file__).resolve().parent
ASSETS = WORK / 'assets'
FONT = subprocess.check_output(['kpsewhich', 'cmr10.pfb'], text=True).strip()
BOLD = subprocess.check_output(['kpsewhich', 'cmbx12.pfb'], text=True).strip()


def fill_template():
    document = fitz.open()
    page = document.new_page(width=956.41, height=597.75625)
    page.insert_font(fontname='ResultRoman', fontfile=FONT)
    page.insert_font(fontname='ResultBold', fontfile=BOLD)
    fonts = {'ResultRoman': fitz.Font(fontfile=FONT),
             'ResultBold': fitz.Font(fontfile=BOLD)}
    ink = (.10, .11, .12)
    muted = (.38, .40, .42)

    def label(x, y, text, size=14, bold=False, color=ink, align='left'):
        name = 'ResultBold' if bold else 'ResultRoman'
        width = fonts[name].text_length(text, fontsize=size)
        if align == 'right':
            x -= width
        elif align == 'center':
            x -= width / 2
        page.insert_text((x, y), text, fontsize=size, fontname=name, color=color)

    label(40, 59, 'Motorcycles', size=28, bold=True)
    label(40, 84, 'CIFAR-100', size=14, color=muted)
    label(916, 72, 'r = 64    k = 32', size=15, color=muted, align='right')

    rows = [
        ('motorcycles_row_1.pdf', 122, 'Ground truth', 'original image'),
        ('motorcycles_row_2.pdf', 236, 'PCA input', 'used for training'),
        ('motorcycles_row_3.pdf', 350, 'Reconstruction', ''),
    ]
    size, left, stride = 88, 156, 96
    for filename, top, title, subtitle in rows:
        label(139, top + 44 if subtitle else top + 49,
              title, size=13, bold=filename != 'motorcycles_row_2.pdf', align='right')
        if subtitle:
            label(139, top + 61, subtitle, size=11, color=muted, align='right')
        with fitz.open(ASSETS / filename) as source:
            for i in range(8):
                # Same column coordinates in all three independently sourced rows.
                clip = fitz.Rect(i * 232, 0, i * 232 + 197, 197)
                target = fitz.Rect(left + i * stride, top,
                                   left + i * stride + size, top + size)
                page.show_pdf_page(target, source, 0, clip=clip)
                page.draw_rect(target, color=(.85, .86, .87), width=.4)

    # Reported in figures_for_gal(1).pdf, page 2. Counts concern PCA inputs.
    label(156, 502, '8/8 training inputs recovered', size=17)
    label(916, 502, '135/200 successful starts', size=17, align='right')

    output = WORK / 'motorcycle_results_template.pdf'
    document.set_metadata({
        'title': 'Motorcycles: ground truth, PCA input, reconstruction',
        'author': 'Yoad Oxman',
        'subject': 'All eight actual originals with their PCA training inputs and matching certificate reconstructions.',
    })
    document.save(output, garbage=4, deflate=True)
    with fitz.open(output) as final:
        final[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(
            WORK / 'motorcycle_results_template.png')
    return output


if __name__ == '__main__':
    print(fill_template())
