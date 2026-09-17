from pathlib import Path
import json, shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/workspace/scratch/2ebbd02e0944')
OUT=ROOT/'output/meeting_handoff/figures'
def crop(name, src, box):
    Image.open(OUT/src).crop(box).save(OUT/name)

# These are exact pixel crops. No image content is generated or reconstructed.
for j,(y0,y1) in enumerate([(42,189),(428,575)],1):
    crop(f'letters_row_{j}.png','letters_original.png',(396,y0,1783,y1))
for j,(y0,y1) in enumerate([(7,158),(170,321),(496,647),(985,1136)],1):
    crop(f'apple_row_{j}.png','apple_original.png',(560,y0,1612,y1))
for j,(y0,y1) in enumerate([(93,239),(402,548),(710,856),(1019,1165)],1):
    crop(f'three_charts_row_{j}.png','fig_chart_dependence.png',(228,y0,1188,y1))
for domain in ['motorcycles','bags']:
    for j in range(1,4):
        shutil.copy2(ROOT/f'tmp/pdfs/brief_redesign/{domain}_row_{j}.pdf',OUT/f'{domain}_row_{j}.pdf')

d=json.loads((ROOT/'tmp/lora_thesis_bundle/03_scripts/trajspan_fig.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                    'axes.spines.right':False,'axes.labelsize':9,'legend.fontsize':7.5})
fig,ax=plt.subplots(1,3,figsize=(10.9,3.0),layout='constrained')
a=np.array(d['vsT'])
ax[0].plot(a[:,0],a[:,1],'o-',label='Input trajectory')
ax[0].plot(a[:,0],a[:,2],'s-',label='Recorded in B')
ax[0].axhline(16,color='.4',ls='--',lw=.8,label='Adapter width 16')
ax[0].set(xscale='log',xlabel='Training steps',ylabel='Numerical dimension',title='Span grows; excitation can lag')
ax[0].set_xticks([1,2,4,10,40],[1,2,4,10,40]);ax[0].legend(loc='upper left',frameon=False)
ax[1].semilogy(a[:,0],a[:,3],'o-',label='Members: maximum')
ax[1].semilogy(a[:,0],a[:,4],'^-',label='Fresh inputs: minimum')
ax[1].set(xscale='log',xlabel='Training steps',ylabel='Normalized residual',title='Base-feature test under light drift')
ax[1].set_xticks([1,2,4,10,40],[1,2,4,10,40]);ax[1].legend(loc='lower right',frameon=False)
t=np.array([r[1:] for r in d['trunc']])
for j,(vals,marker,label) in enumerate([(t[:,0],'o','Members: maximum'),(t[:,1],'s','10% perturbations: minimum'),(t[:,3],'^','Fresh inputs: minimum')]):
    ax[2].semilogy(np.arange(3),vals,marker+'-',label=label)
ax[2].set(xticks=[0,1,2],xticklabels=['Light\n40 steps','Heavy\n200 steps','Heavy\n1000 steps'],ylabel='Normalized residual',title='Top-8 test: separation weakens')
ax[2].legend(loc='lower right',frameon=False)
for a in ax:a.grid(axis='y',alpha=.15)
fig.savefig(OUT/'deep_certificate_audited.pdf')
fig.savefig(OUT/'deep_certificate_audited.png',dpi=180)
plt.close(fig)

# Replot only the capacity cells explicitly listed in the Rev 11.2 record.
fig,ax=plt.subplots(figsize=(7.2,3.4),layout='constrained')
N=np.linspace(2,15,100)
ax.plot(N,16-N,color='#3e6f8e',label='Certificate row count: r - N')
ax.plot(N,36-N,'--',color='#a86231',label='Replay onset of excess coordinates: m + r - N')
ax.scatter([4,8,12,14],[30,27,22,21],marker='o',s=50,color='#3b7254',label='Reported local recovery')
ax.scatter([4,8,12,14],[34,28,26,22],marker='x',s=55,color='#a54343',label='Reported different-image fit at floor')
ax.set(xlabel='Number of private inputs N',ylabel='Chart coordinates per image k',xlim=(2,15),ylim=(0,37))
ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8,loc='lower left')
fig.savefig(OUT/'two_boundaries_audited.pdf');fig.savefig(OUT/'two_boundaries_audited.png',dpi=180)
plt.close(fig)
print('Extracted source rows and generated two plots from documented measurements.')
