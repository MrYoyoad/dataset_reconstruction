from pathlib import Path
import subprocess
import concurrent.futures
import json
import fitz
from fill_results_template import fill_template

WORK = Path(__file__).resolve().parent

def build(path):
    logfile = path.with_suffix('.build.log')
    with logfile.open('w') as stream:
        result = subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',path.name],
                                cwd=WORK,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        return {'name':path.name,'error':logfile.read_text()[-2500:]}
    with fitz.open(path.with_suffix('.pdf')) as document:
        assert len(document) == 1
        page=document[0]
        page.get_pixmap(matrix=fitz.Matrix(1.8,1.8),alpha=False).save(path.with_suffix('.png'))
        return {'name':path.stem,'pages':1,'size':[round(page.rect.width,2),round(page.rect.height,2)]}

sources = sorted(WORK.glob('[0-9][0-9]_*.tex'))
assert len(sources)==7
filled_results = fill_template()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(build,sources))
for result in results:
    print(json.dumps(result))
assert not any('error' in r for r in results)
document=fitz.open()
toc=[]
for source in sources:
    if source.stem=='04_reconstructions':
        # Keep all eight matches, including actual originals, on one results page.
        toc.append([1,'Motorcycles: ground truth and reconstruction',len(document)+1])
        with fitz.open(filled_results) as result:
            document.insert_pdf(result)
    toc.append([1,source.stem.replace('_',' '),len(document)+1])
    with fitz.open(source.with_suffix('.pdf')) as part:
        document.insert_pdf(part)
document.set_metadata({'title':'LoRA reconstruction: figure pack','author':'Yoad Oxman',
                       'subject':'Corrected certificate, image-family and NTK illustrations; real reconstructions; proposed local families.'})
document.set_toc(toc)
document.save(WORK/'figure_pack.pdf',garbage=4,deflate=True)
print('Saved corrected figure pack:',len(document),'pages')
