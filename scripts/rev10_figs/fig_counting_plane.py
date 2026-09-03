"""(N,k) plane, both boundaries, with the MEASURED capacity brackets overlaid.
Three-way cell classification: RECOVERED (floor residual, err<1e-8) / ALIAS (floor residual, a different release-consistent point, err 1e-3..1e-2, sigma_min collapsed) / SEARCH-FAIL (residual not at floor).
NOTE: the line is a boundary of EXACT identifiability, not (on this evidence) of leakage — past-line image errors are sub-percent.
CPU-only from committed jsonl. style_guide/plots.md: DPI 200, matplotlib defaults, T5 fonts."""
import json, glob, os, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
R, M = 16, 20
def load(pat):
    rows=[]
    for f in sorted(set(glob.glob(pat))):
        for ln in open(f):
            ln=ln.strip()
            if ln: rows.append(json.loads(ln))
    return rows
def cls(r):
    res=r.get("residual",9); err=r.get("final_err_max",9); sm=r.get("jac_sigma_min") or r.get("jac_sigma_min_truth") or 1.0
    if res>1e-25: return "search"
    return "alias" if (sm<1e-12 or err>1e-6) else "recovered"
grid = load("results/exact_inversion/step4_sweep_*.jsonl")+load("results/exact_inversion/step7_disambig_*.jsonl")
cap  = load("results/exact_inversion/step11_capacity_*.jsonl")+load("results/exact_inversion/step13_capacity_law_*.jsonl")+load("results/exact_inversion/step16_softmax_*.jsonl")+load("results/exact_inversion/step18_bracket_*.jsonl")+load("results/exact_inversion/step23_n14k21_*.jsonl")
gridrec={(r["N"],r["k"]) for r in grid if cls(r)=="recovered"}
# per-cell outcome = best run at that cell (a later, larger-budget rerun supersedes a search failure)
rank={"recovered":0,"alias":1,"search":2}; best={}
for r in cap:
    c=(r["N"],r["k"]); best[c]=min(best.get(c,"search"),cls(r),key=lambda z:rank[z])
caprec={c for c,v in best.items() if v=="recovered"}; capali={c for c,v in best.items() if v=="alias"}; capsrch={c for c,v in best.items() if v=="search"}
blue,orange,green,red="#1f77b4","#ff7f0e","#2ca02c","#d62728"
fig, ax = plt.subplots(figsize=(10, 7.6), dpi=200)
Nl=np.linspace(1,15,200); cert=R-Nl; inv=M+R-Nl
ax.fill_between(Nl,0,np.clip(cert,0,None),color=blue,alpha=0.08)
ax.fill_between(Nl,np.clip(cert,0,None),inv,color=orange,alpha=0.10)
ax.fill_between(Nl,inv,50,color=red,alpha=0.08)
ax.plot(Nl,cert,color=blue,lw=2.2,label="certificate boundary  k = r − N   (Rev 9 Thm 3, on the certificate)")
ax.plot(Nl,inv,color=orange,lw=2.4,ls="--",label="capacity boundary  k = m + r − N   (Theorem 6; measured sharp)")
ax.scatter([c[0] for c in gridrec],[c[1] for c in gridrec],s=60,color=green,edgecolor="k",lw=0.5,zorder=5,label="recovered — 49-cell grid")
ax.scatter([c[0] for c in caprec],[c[1] for c in caprec],s=95,color=green,marker="D",edgecolor="k",lw=0.7,zorder=6,label="recovered — capacity sweep")
ax.scatter([c[0] for c in capali],[c[1] for c in capali],s=120,color=red,marker="X",edgecolor="k",lw=0.6,zorder=6,label="ALIAS — floor residual, a DIFFERENT point (image error 0.2–1.7%)")
if capsrch: ax.scatter([c[0] for c in capsrch],[c[1] for c in capsrch],s=110,facecolor="none",edgecolor=blue,marker="s",lw=2,zorder=7,label="identifiable (σ_min not collapsed) but the run search-failed")
# bracket text per N
for N,dx,dy in ((4,10,6),(8,14,-30),(12,10,6)):
    ok=max(k for (n,k) in caprec if n==N); al=min(k for (n,k) in capali if n==N); kstar=M+R-N
    ax.annotate(f"N={N}: ok@{ok} · alias@{al}\nline k*={kstar}",(N,al),textcoords="offset points",xytext=(dx,dy),fontsize=9.5,ha="left")
ok=max(k for (n,k) in caprec if n==14); al=min(k for (n,k) in capali if n==14)
ax.annotate(f"N=14: ok@{ok} · alias@{al}\nline k*={M+R-14}",xy=(14,al),xytext=(9.4,17.2),textcoords="data",fontsize=9.5,ha="left",
            arrowprops=dict(arrowstyle="-",color="gray",lw=0.8,shrinkB=6))
ax.annotate("",xy=(8,M+R-8),xytext=(8,R-8),arrowprops=dict(arrowstyle="<->",color="k",lw=1.3))
ax.text(6.1,17.5,"+ (m − 1)  per image:\nthe coefficients of P_T\nthe certificate projects away",fontsize=10.5,va="center",ha="right")
ax.set_xlim(1,15); ax.set_ylim(0,46)
ax.set_xlabel("N  (private images / distinct representations)",fontsize=13)
ax.set_ylabel("k  (dimension of the attacker's search chart, per image)",fontsize=13)
ax.set_title("Two channels, two boundaries — k is the dimension of the attacker's search chart\nthe exact channel's capacity  k < m + r − N:  sharp to one unit of k at N = 8, 14; bracketed to two at N = 4, 12",fontsize=12.5,fontweight="bold")
ax.tick_params(labelsize=11); ax.grid(alpha=0.3)
ax.legend(fontsize=9.3,loc="upper center",bbox_to_anchor=(0.5,-0.12),ncol=2,framealpha=0.95)
out=os.path.join("figures/rev10","fig_counting_plane.png")
fig.tight_layout(); fig.savefig(out); print("saved",out,"| grid",len(gridrec),"cap-rec",len(caprec),"alias",len(capali))
