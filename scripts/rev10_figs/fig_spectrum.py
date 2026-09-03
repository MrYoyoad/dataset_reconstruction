"""Figure 3 — the singular spectrum of DF at the TRUTH: one Jacobian, three readings.
rank = identifiability; spread = conditioning; the spectrum IS the local resolution ellipsoid's axes.
Per-block spectra (B-block, A-block) overlaid with their caps — the B-block cap is N(m-1+r-N) under SGD (softmax simplex constraint 1^T B_T = 0) and the plain N(m+r-N) under Adam, which breaks it. SGD n96 vs Adam n96 vs SGD n32.
CPU-only from results/exact_inversion/spectrum_*.pth (job 479684). style_guide/plots.md conventions."""
import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R,M=16,20
cells=[("spectrum_sgd_n96_k12_N8.pth","SGD  n=96, k=12, N=8   (unknowns Nk+rN = 224)", dict(n=96,k=12,N=8,nuis="rN")),
       ("spectrum_adam_n96_k12_N8.pth","Adam  n=96, k=12, N=8   (unknowns Nk+rn = 1632)", dict(n=96,k=12,N=8,nuis="rn")),
       ("spectrum_sgd_n32_k6_N4.pth","SGD  n=32, k=6, N=4   (unknowns Nk+rN = 88)", dict(n=32,k=6,N=4,nuis="rN"))]
blue,orange,green,red="#1f77b4","#ff7f0e","#2ca02c","#d62728"
def rank(s,tol=1e-10): s=np.asarray(s); return int((s>tol*s.max()).sum())
fig,axes=plt.subplots(1,3,figsize=(16,5.6),dpi=200)
for ax,(f,title,c) in zip(axes,cells):
    d=torch.load(os.path.join("results/exact_inversion",f),map_location="cpu",weights_only=False)
    s=np.sort(np.asarray(d["sigma"]))[::-1]; sB=np.sort(np.asarray(d["sigma_B"]))[::-1]; sA=np.sort(np.asarray(d["sigma_A"]))[::-1]
    n,k,N=c["n"],c["k"],c["N"]; sgd=(c["nuis"]=="rN"); capB=(N*((M-1)+R-N) if sgd else N*(M+R-N)); capBlab=(f"N(m−1+r−N)={capB} (simplex)" if sgd else f"N(m+r−N)={capB} (plain)"); capA=(R*N if sgd else R*n)
    ax.semilogy(np.arange(1,len(s)+1),s,color="k",lw=2.2,label=f"full DF  (rank {rank(s)} / {len(s)} cols)")
    ax.semilogy(np.arange(1,len(sB)+1),sB,color=blue,lw=1.6,ls="--",label=f"B-block  (rank {rank(sB)}, cap {capBlab})")
    ax.semilogy(np.arange(1,len(sA)+1),sA,color=orange,lw=1.6,ls=":",label=f"A-block  (rank {rank(sA)}, cap {c['nuis']}={capA})")
    ax.axhline(s.max()*1e-10,color="gray",lw=0.8,ls=":"); ax.text(1.5,s.max()*1.6e-10,"rank tolerance 1e-10·σ_max",fontsize=8.5,color="gray")
    cond=s.max()/s[s>s.max()*1e-10].min()
    ax.annotate(f"σ_min = {s[s>s.max()*1e-10].min():.1e}\ncond = {cond:.1e}",xy=(len(s),s[s>s.max()*1e-10].min()),xytext=(-6,14),textcoords="offset points",ha="right",fontsize=9.5,
                bbox=dict(boxstyle="round",fc="#fffbe6",ec="#ccc"))
    ax.set_title(title,fontsize=12,fontweight="bold"); ax.set_xlabel("singular-value index",fontsize=12)
    ax.grid(alpha=0.3,which="major"); ax.tick_params(labelsize=10); ax.legend(fontsize=8.6,loc="lower left",framealpha=0.95)
axes[0].set_ylabel("singular value of DF at the truth  (log)",fontsize=12)
fig.suptitle("One Jacobian, three readings — rank = identifiability · spread = conditioning · spectrum = the resolution ellipsoid",fontsize=13,fontweight="bold",y=1.01)
out=os.path.join("figures/rev10","fig_spectrum.png")
fig.tight_layout(); fig.savefig(out,bbox_inches="tight"); print("saved",out)
