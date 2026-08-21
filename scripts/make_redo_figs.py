"""Figures for the honest re-do: N-collapse curve, matched-wc optimizer bars, and N=2 vs N=8
example grids. Run on WEXAC (bsub), not the login node. Data are the log-extracted final numbers;
example grids load the saved .pth tensors."""
import os, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.plotting import plot_reconstruction_grid

OUT='figures/pdf_examples'; os.makedirs(OUT, exist_ok=True)

# ---- 1. N-collapse curve (3-seed margin mean + range) ----
N=[2,4,8,16]; mean=[0.260,0.175,0.021,0.030]; lo=[0.16,0.17,0.00,-0.00]; hi=[0.43,0.18,0.04,0.08]
fig,ax=plt.subplots(figsize=(6,4))
yerr=[[m-l for m,l in zip(mean,lo)],[h-m for m,h in zip(mean,hi)]]
ax.errorbar(N,mean,yerr=yerr,marker='o',capsize=4,lw=2,color='#1f77b4')
ax.axhline(0,color='gray',ls='--',lw=1)
ax.set_xscale('log',base=2); ax.set_xticks(N); ax.set_xticklabels(N)
ax.set_xlabel('N (fine-tune images)'); ax.set_ylabel('control margin (ssim_norm sample - control)')
ax.set_title('Honest N-collapse (flowers32, free-c, r=8, 3 seeds)\nbox on joint N<=4; peel N>=8 hard to score (+-0.05)')
ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(f'{OUT}/REDO_N_collapse.png',dpi=130); plt.close(fig)
print('wrote REDO_N_collapse.png')

# ---- 2. optimizer matched-wc bars ----
labels=['sgd\nl2','sgd\ncosine','adamw\nl2','adamw\ncosine']
snorm=[0.668,0.790,0.540,0.559]; marg=[0.195,0.277,0.188,0.176]
x=range(len(labels)); w=0.38
fig,ax=plt.subplots(figsize=(6.5,4))
ax.bar([i-w/2 for i in x],snorm,w,label='ssim_norm',color='#1f77b4')
ax.bar([i+w/2 for i in x],marg,w,label='control margin',color='#ff7f0e')
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel('score'); ax.set_title('Optimizer axis at matched wc~0.03 (clip-controlled)\ncosine>>l2 for sgd only; sgd > adamw')
ax.legend(); ax.grid(axis='y',alpha=0.3); fig.tight_layout(); fig.savefig(f'{OUT}/REDO_optimizer_bars.png',dpi=130); plt.close(fig)
print('wrote REDO_optimizer_bars.png')

# ---- 3. example grids: N=2 (npc=1, boxed) vs N=8 (npc=4, unboxed peel), seed 42 ----
for tag,path in [('N2', 'results/exp_b_T1_flowers32_r8_free_s42_a10000_vw5_pbox.pth'),
                 ('N8', 'results/exp_b_T1_flowers32_r8_free_s42_a10000_npc4_vw5.pth')]:
    if not os.path.exists(path):
        print(f'skip {tag}: {path} missing'); continue
    d=torch.load(path,map_location='cpu',weights_only=False)
    plot_reconstruction_grid(x_train=d['x_train'], x_recon_lora=d['x_recon_lora'], x_ctrl=d['x_ctrl'],
                             ds_mean=d.get('ds_mean'), rank=8, class_label='Species-parity',
                             save_path=f'{OUT}/REDO_grid_{tag}.png',
                             title=f'{tag}: flowers32 free-c r=8 (honest re-do)')
    print(f'wrote REDO_grid_{tag}.png')
print('DONE')
