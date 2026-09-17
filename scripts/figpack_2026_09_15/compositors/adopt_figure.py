from pathlib import Path
import shutil
import fitz

ROOT = Path('/workspace/scratch/2ebbd02e0944')
WORK = ROOT / 'tmp/pdfs/figure_update'
OUT = ROOT / 'output/pdf/figure_update'
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

source = fitz.open(ROOT / 'upload/lora_brief_v2.pdf')
page = source[0]
# Preserve the supplied vector illustration; change only two scope labels.
page.add_redact_annot(fitz.Rect(281, 171.0, 426, 181.2), fill=(1, 1, 1))
page.add_redact_annot(fitz.Rect(210, 203.1, 265, 212.3), fill=(1, 1, 1))
page.apply_redactions(images=0, graphics=0)
font = '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf'
page.insert_font(fontname='ScopeRoman', fontfile=font)
orange = (0.81, 0.36, 0.0)
assert page.insert_textbox(
    fitz.Rect(256, 170.8, 451, 184.7),
    'exact equations, under the certificate conditions',
    fontname='ScopeRoman', fontsize=8.2, align=1, color=orange
) >= 0
assert page.insert_textbox(
    fitz.Rect(204, 202.8, 271, 217.1),
    'on adapted layers',
    fontname='ScopeRoman', fontsize=8.0, align=1, color=orange
) >= 0
clip = fitz.Rect(66, 156, 530, 347)
figure = fitz.open()
new_page = figure.new_page(width=clip.width, height=clip.height)
new_page.show_pdf_page(new_page.rect, source, 0, clip=clip)
figure.set_metadata({'title': 'Small image controls before the full network', 'subject': 'Diagram adapted from the supplied lora_brief_v2.pdf; certificate conditions clarified.'})
figure.save(WORK / 'image_family_architecture.pdf', garbage=4, deflate=True)

for stem, folder in [('LoRA_reconstruction_brief', 'brief_redesign'),
                     ('LoRA_meeting_personal_companion', 'personal_companion')]:
    path = ROOT / 'tmp/pdfs' / folder / (stem + '.tex')
    text = path.read_text()
    (WORK / (stem + '.before_figure.tex')).write_text(text)
    old_graphics = r'\graphicspath{{/workspace/scratch/2ebbd02e0944/output/meeting_handoff/figures/}{/workspace/scratch/2ebbd02e0944/tmp/pdfs/strategy_story/}}'
    new_graphics = r'\graphicspath{{/workspace/scratch/2ebbd02e0944/tmp/pdfs/figure_update/}{/workspace/scratch/2ebbd02e0944/output/meeting_handoff/figures/}}'
    assert old_graphics in text
    text = text.replace(old_graphics, new_graphics)
    old_include = '\\begin{center}\n\\includegraphics[width=\\linewidth]{image_family_architecture.pdf}\n\\end{center}'
    new_include = '{\\centering\\includegraphics[width=\\linewidth]{image_family_architecture.pdf}\\par}'
    assert old_include in text
    text = text.replace(old_include, new_include, 1)
    if stem == 'LoRA_reconstruction_brief':
        old = "The network still receives a full image. I search only the $k$ inputs to the decoder, which ties the pixels together. In these experiments, the decoder is PCA: a public mean image plus a weighted combination of public image directions. The predicted label is the original task; reconstruction is guided by the adapter equations."
        new = "Only the green controls are searched; the full image and its features follow from them. Here the public decoder is PCA. The multiple branches illustrate the extension; the image experiments train only the head. Reconstruction uses the adapter equations, not the predicted label alone."
    else:
        old = r"The small input comes \emph{before} the image and the full network. Each setting of $z$ produces an entire candidate image; the model computes its features as usual. During reconstruction the released model stays fixed. The objective tests compatibility with the adapter, rather than merely seeking a desired predicted label."
        new = "Only the green controls are searched; images and features follow from them. The released network stays fixed, and reconstruction uses the adapter equations. The multiple branches illustrate the extension; the image experiments train only the head. Note E gives the deeper-layer conditions."
    assert old in text
    text = text.replace(old, new, 1)
    (WORK / (stem + '.tex')).write_text(text)

print('Extracted the vector figure and updated the two first-page sources.')
