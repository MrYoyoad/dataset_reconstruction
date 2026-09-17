from pathlib import Path
import re
import fitz
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'

def source(name):
    d=fitz.open(FIG/name)
    if name.endswith('.pdf'): return d
    w,h=Image.open(FIG/name).size
    q=fitz.open('pdf',d.convert_to_pdf()); out=fitz.open()
    p=out.new_page(width=w,height=h);p.show_pdf_page(p.rect,q,0)
    return out

def pairs(name,raw,rec,starts,w,h,per=4,status=None):
    a,b=source(raw),source(rec);d=fitz.open()
    tw=64; th=tw*h/w; pitch=146
    rh=th+11+(16 if status else 0)
    rows=(len(starts)+per-1)//per
    p=d.new_page(width=per*pitch-10,height=rows*rh-5)
    for i,x in enumerate(starts):
        xx=(i%per)*pitch; yy=(i//per)*rh
        for j,src in enumerate([a,b]):
            dx=xx+j*68
            box=fitz.Rect(dx,yy,dx+tw,yy+th)
            p.show_pdf_page(box,src,0,clip=fitz.Rect(x,0,x+w,h))
            p.draw_rect(box,color=(.65,.65,.65),width=.2)
        if status:
            label=status[i]; tx=xx+(132-fitz.get_text_length(label,fontsize=10.5))/2; p.insert_text((tx,yy+th+14),label,fontsize=10.5)
    d.save(FIG/(name+'.pdf'),garbage=4,deflate=True)

pairs('letters_gt_certificate','letters_a_raw_gt.pdf','letters_a_certificate.pdf',[0,177,354,531],147,147)
pairs('letters_gt_ntk','letters_a_raw_gt.pdf','letters_a_ntk.pdf',[0,177,354,531],147,147)
pairs('motorcycles_gt_reconstruction','motorcycles_row_1.pdf','motorcycles_row_3.pdf',[232*i for i in range(8)],197,197)
for j in [2,3,4]:
    pairs('apple_gt_'+str(j),'apple_row_1.png',f'apple_row_{j}.png',[133*i for i in range(8)],121,151)
    pairs('apple_gt_'+str(j)+'_four','apple_row_1.png',f'apple_row_{j}.png',[133*i for i in range(4)],121,151)
for j in [2,3,4]:
    pairs('mnist_gt_'+str(j),'three_charts_row_1.png',f'three_charts_row_{j}.png',[0,163,326,488,651,814],146,146,6)
pairs('fashion_mnist_gt','bags_row_1.pdf','bags_row_3.pdf',[0,76.4,152.8,229.2],64.5,65.7,status=['Found','Found','Missed','Missed'])
pairs('keyboards_gt','keyboard_row_1.png','keyboard_row_3.png',[232*i for i in range(8)],192,192,status=['Found','Found','Found','Found','Missed','Found','Missed','Found'])

