"""Residual vs image-error scatter, coloured by which side of k = m+r-N the cell sits on.
The 'alias quadrant' (floor residual, image error 1e-3..1e-2 — sub-percent, NOT 'wrong') is EMPTY below the capacity line and POPULATED above it:
identifiability made visual, with the boundary. Search failures (nonzero residual) shown for contrast."""
import json, glob, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
R, M = 16, 20
def load(pat):
    rows=[]
    for f in sorted(set(glob.glob(pat))):
        for ln in open(f):
            ln=ln.strip()
            if ln: rows.append(json.loads(ln))
    return rows
grid = load("results/exact_inversion/step4_sweep_*.jsonl")+load("results/exact_inversion/step7_disambig_*.jsonl")
cap  = load("results/exact_inversion/step11_capacity_*.jsonl")+load("results/exact_inversion/step13_capacity_law_*.jsonl")+load("results/exact_inversion/step16_softmax_*.jsonl")+load("results/exact_inversion/step18_bracket_*.jsonl")+load("results/exact_inversion/step23_n14k21_*.jsonl")
inits= load("results/exact_inversion/step2_basin_init_*.jsonl")   # attacker-realizable starts: search failures
adam = [r for r in load("results/exact_inversion/step3_adam_*.jsonl") if r.get("release")=="adam"]
def side(r): return "above" if r["k"]>=M+R-r["N"] else "below"
def cls(r):
    res=r.get("residual",9); err=r.get("final_err_max",9); sm=r.get("jac_sigma_min") or r.get("jac_sigma_min_truth") or 1.0
    if res>1e-25: return "search"
    return "alias" if (sm<1e-12 or err>1e-6) else "recovered"
pts=[]
for r in grid+cap: pts.append((r.get("residual",9),r.get("final_err_max",9),side(r),cls(r)))
for r in inits+adam: pts.append((r.get("residual",9),r.get("final_err_max",9),side(r),"search"))
# verification of the claim the figure makes
assert all(sd=="above" for x,y,sd,c in pts if c=="alias"), "an alias below the line would falsify the figure"
assert all(sd=="below" for x,y,sd,c in pts if c=="recovered"), "recovered above the line?"
print("check: all aliases above the line, all recoveries below — OK")
blue,orange,green,red="#1f77b4","#ff7f0e","#2ca02c","#d62728"
fig, ax = plt.subplots(figsize=(10,6.8),dpi=200)
for c,(col,mk,lab) in {
  "recovered":(green,"o","RECOVERED — floor residual, err at machine precision (all below the line)"),
  "alias":(red,"X","ALIAS — floor residual, a DIFFERENT point at 0.2–1.2% image error, σ_min collapsed (all above the line)"),
  "search":(blue,"s","SEARCH FAILURE — residual not at floor (attacker-start / Adam runs; (14,21) at 1 restart — recovered at 6)"),
}.items():
    q=[(x,y) for x,y,sd,cc in pts if cc==c]
    if q: ax.scatter([max(x,1e-33) for x,_ in q],[max(y,1e-17) for _,y in q],s=75,color=col,marker=mk,edgecolor="k",lw=0.5,label=lab,zorder=4)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-33,10); ax.set_ylim(1e-17,10)
ax.axvline(1e-25,color="gray",lw=1,ls=":"); ax.axhline(1e-8,color="gray",lw=1,ls=":")
ax.text(3e-27,3e-15,"RECOVERED\n(release reproduced,\ntruth found)",fontsize=10.5,color=green,fontweight="bold")
ax.text(3e-33,2e-2,"ALIAS quadrant\n(release reproduced by a\nDIFFERENT point — image\nerror 0.2–1.2%, sub-percent)\nempty below the line\n(cells tested), populated above",fontsize=10.5,color=red,fontweight="bold")
ax.text(1e-12,2e-2,"SEARCH FAILURE\n(release NOT reproduced)",fontsize=10.5,color=blue,fontweight="bold")
ax.set_xlabel("final residual  ‖Recipe_T(ŵ, X̂) − release‖²   (floor ≈ 1e-30)",fontsize=12.5)
ax.set_ylabel("max relative image error",fontsize=12.5)
ax.set_title("The residual separates the regimes\nno alias observed below  k = m + r − N  in the cells tested; every alias observed lies above it",fontsize=12.5,fontweight="bold")
ax.tick_params(labelsize=10.5); ax.grid(alpha=0.3,which="major")
ax.legend(fontsize=9.5,loc="lower right",framealpha=0.95)
out=os.path.join("figures/rev10","fig_alias_quadrant.png")
fig.tight_layout(); fig.savefig(out); print("saved",out,"| pts",len(pts),"| search rows",len(inits)+len(adam))
