#!/usr/bin/env python
# Detailed, self-contained proof PDF.
# Prose: fpdf2 + DejaVu.  Display math: matplotlib mathtext (no external LaTeX).
import os, io, re, tempfile, hashlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from fpdf import FPDF

# Portable font dir: use whatever matplotlib install ships (DejaVu is always bundled).
FD = os.path.join(matplotlib.get_data_path(), "fonts", "ttf") + os.sep
SANS, BOLD, ITAL, MONO = FD+"DejaVuSans.ttf", FD+"DejaVuSans-Bold.ttf", FD+"DejaVuSans-Oblique.ttf", FD+"DejaVuSansMono.ttf"

INK=(20,20,20); MUTE=(95,95,95); RULE=(214,214,214); ACC=(28,60,120)
AMB=(150,96,12); AMBBAR=(214,158,54); GRY=(247,248,250)
EQINK=(0.08,0.10,0.16)

# --- turn ASCII math in prose (w_k, c_i, M_{ki}, x_i^T) into real Unicode sub/superscripts ---
_SUB={'0':'₀','1':'₁','2':'₂','3':'₃','4':'₄','5':'₅','6':'₆','7':'₇','8':'₈','9':'₉',
      'i':'ᵢ','j':'ⱼ','k':'ₖ','l':'ₗ','m':'ₘ','n':'ₙ','x':'ₓ'}
_SUP={'T':'ᵀ','d':'ᵈ','n':'ⁿ','m':'ᵐ','-':'⁻','0':'⁰','1':'¹','2':'²'}
def mathify(t):
    t=t.replace("^{\\top}","ᵀ").replace("^\\top","ᵀ").replace("^T","ᵀ").replace("^{-1}","⁻¹")
    t=re.sub(r'_\{([^}]+)\}', lambda m:''.join(_SUB.get(c,c) for c in m.group(1)), t)
    t=re.sub(r'\^\{([^}]+)\}', lambda m:''.join(_SUP.get(c,c) for c in m.group(1)), t)
    t=re.sub(r'_([0-9A-Za-z])', lambda m:_SUB.get(m.group(1), m.group(0)), t)
    t=re.sub(r'\^([0-9A-Za-z])', lambda m:_SUP.get(m.group(1), m.group(0)), t)
    return t

TMP=tempfile.mkdtemp(prefix="ident_proof_eq_")   # scratch dir for rendered equation images

def render_math(latex, fontsize=17, dpi=340):
    key=hashlib.md5((latex+str(fontsize)).encode()).hexdigest()[:12]; path=f"{TMP}/{key}.png"
    fig=plt.figure(figsize=(0.1,0.1)); t=fig.text(0,0,f"${latex}$",fontsize=fontsize,color=EQINK)
    fig.canvas.draw(); bb=t.get_window_extent(fig.canvas.get_renderer())
    w_in,h_in=bb.width/fig.dpi,bb.height/fig.dpi; plt.close(fig)
    fig=plt.figure(figsize=(w_in+0.06,h_in+0.06))
    fig.text(0.5,0.5,f"${latex}$",fontsize=fontsize,ha="center",va="center",color=EQINK)
    fig.savefig(path,dpi=dpi,transparent=True,bbox_inches="tight",pad_inches=0.02); plt.close(fig)
    im=Image.open(path); return path,im.width,im.height

def math_img(latex, fontsize=16):
    path,_,_=render_math(latex, fontsize=fontsize); return Image.open(path).convert("RGBA")

def matrix_img(rows, fontsize=16, dpi=340):
    nrow=len(rows); ncol=len(rows[0]); cellw=0.44; cellh=0.44
    Wf=cellw*ncol+0.30; Hf=cellh*nrow+0.14
    fig=plt.figure(figsize=(Wf,Hf)); ax=fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.set_xlim(0,Wf); ax.set_ylim(0,Hf)
    for r,row in enumerate(rows):
        for c,val in enumerate(row):
            ax.text(0.16+cellw*(c+0.5), Hf-0.07-cellh*(r+0.5), f"{val}",
                    ha="center", va="center", fontsize=fontsize, color=EQINK)
    bl,br,top,bot,tick,lw=0.07,Wf-0.07,Hf-0.05,0.05,0.09,1.3
    ax.plot([bl,bl],[bot,top],color=EQINK,lw=lw)
    ax.plot([bl,bl+tick],[top,top],color=EQINK,lw=lw); ax.plot([bl,bl+tick],[bot,bot],color=EQINK,lw=lw)
    ax.plot([br,br],[bot,top],color=EQINK,lw=lw)
    ax.plot([br-tick,br],[top,top],color=EQINK,lw=lw); ax.plot([br-tick,br],[bot,bot],color=EQINK,lw=lw)
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=dpi,transparent=True,bbox_inches="tight",pad_inches=0.02)
    plt.close(fig); buf.seek(0); return Image.open(buf).convert("RGBA")

def render_composite(items, fontsize=16, gap_px=16):
    imgs=[math_img(v,fontsize) if k=="t" else matrix_img(v,fontsize) for k,v in items]
    H=max(im.height for im in imgs); Wt=sum(im.width for im in imgs)+gap_px*(len(imgs)-1)
    canvas=Image.new("RGBA",(Wt,H),(0,0,0,0)); x=0
    for im in imgs:
        canvas.paste(im,(x,(H-im.height)//2),im); x+=im.width+gap_px
    key=hashlib.md5(str(items).encode()).hexdigest()[:12]; p=f"{TMP}/comp_{key}.png"
    canvas.save(p); return p,canvas.width,canvas.height

# ---------- schematic figure: the rank bottleneck ----------
def make_schematic():
    p=f"{TMP}/schematic.png"
    fig,ax=plt.subplots(figsize=(8.4,2.9)); ax.axis("off")
    ax.set_xlim(0,10.5); ax.set_ylim(0,3.4)
    def box(x,y,w,h,fc,ec,lw=1.4):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=fc,edgecolor=ec,lw=lw))
    ink="#141414"; blu="#1c3c78"; amb="#b8801f"
    # Omega  (m x d)
    box(0.2,0.7,1.9,2.1,"#eef2fb",blu); ax.text(1.15,1.75,r"$\Omega$",ha="center",va="center",fontsize=20,color=ink)
    ax.text(1.15,0.42,"m × d",ha="center",fontsize=10,color=MUTEc); ax.text(1.15,3.02,"what you observe",ha="center",fontsize=9.5,color=blu)
    ax.text(2.55,1.75,"=",ha="center",va="center",fontsize=22,color=ink)
    # G (m x N) rank k -- the bottleneck
    box(3.0,0.7,1.15,2.1,"#fbf1dd",amb,1.8)
    # shade only k of the N columns to hint rank deficiency
    box(3.0,0.7,0.38,2.1,"#f1d79a",amb,0.8)
    ax.text(3.57,1.75,r"$G$",ha="center",va="center",fontsize=20,color=ink)
    ax.text(3.57,0.42,"m × N",ha="center",fontsize=10,color=MUTEc)
    ax.text(3.57,3.02,"the gates",ha="center",fontsize=9.5,color=amb)
    ax.annotate("rank = k < N\n(bottleneck)", xy=(3.2,2.5), xytext=(4.5,3.15),
                fontsize=9.5,color=amb,ha="left",
                arrowprops=dict(arrowstyle="->",color=amb,lw=1.2))
    ax.text(4.55,1.75,"×",ha="center",va="center",fontsize=18,color=ink)
    # X^T (N x d)
    box(4.9,1.35,2.0,0.8,"#e9f7ee","#2e8b57",1.6)
    ax.text(5.9,1.75,r"$X^{\top}$",ha="center",va="center",fontsize=18,color=ink)
    ax.text(5.9,1.07,"N × d",ha="center",fontsize=10,color=MUTEc)
    ax.text(5.9,2.35,"the data (unknown)",ha="center",fontsize=9.5,color="#2e8b57")
    # right: the consequence
    ax.text(7.35,1.75,"⇒",ha="center",va="center",fontsize=20,color=ink)
    ax.text(9.0,1.95,"the N samples are\nsqueezed through a\nrank-k channel",ha="center",va="center",
            fontsize=10,color=ink)
    ax.text(9.0,1.05,"k < N  ⟹  they blend\nand cannot be un-mixed",ha="center",va="center",
            fontsize=10,color="#a02020")
    fig.savefig(p,dpi=300,bbox_inches="tight",pad_inches=0.08,facecolor="white"); plt.close(fig)
    im=Image.open(p); return p,im.width,im.height
MUTEc=(0.37,0.37,0.37)

class Doc(FPDF):
    def footer(self):
        self.set_y(-15); self.set_font("mono","",8); self.set_text_color(*MUTE)
        self.cell(0,8,f"{self.page_no()}",align="C")

pdf=Doc(format="A4"); pdf.set_auto_page_break(auto=True,margin=18)
for st,fp in [("",SANS),("B",BOLD),("I",ITAL)]: pdf.add_font("sans",st,fp)
pdf.add_font("mono","",MONO); pdf.add_font("mono","B",FD+"DejaVuSansMono-Bold.ttf")
pdf.set_margins(20,18,20); pdf.add_page(); W=pdf.w-40

def h1(t,size=15):
    pdf.ln(2.5); pdf.set_font("sans","B",size); pdf.set_text_color(*INK)
    pdf.multi_cell(W,7,mathify(t)); pdf.ln(1)
def h2(t):
    pdf.ln(1.5); pdf.set_font("sans","B",11.8); pdf.set_text_color(*ACC)
    pdf.multi_cell(W,6,mathify(t)); pdf.ln(0.6); pdf.set_text_color(*INK)
def para(t,gap=1.8,font=("sans",""),size=10.6,lh=5.7):
    pdf.set_font(*font,size); pdf.set_text_color(*INK); pdf.multi_cell(W,lh,mathify(t)); pdf.ln(gap)
DPI=340
def place_png(path,wpx,hpx,gap=2.6,align="C",indent=6):
    w_mm=wpx/DPI*25.4; maxw=W-4
    if w_mm>maxw: w_mm=maxw
    h_mm=hpx/DPI*25.4*(w_mm/(wpx/DPI*25.4))
    if pdf.get_y()+h_mm+gap>pdf.h-18: pdf.add_page()
    pdf.ln(1.4); x=20+(W-w_mm)/2 if align=="C" else 20+indent
    pdf.image(path,x=x,y=pdf.get_y(),w=w_mm); pdf.set_y(pdf.get_y()+h_mm); pdf.ln(gap)
def eqn(latex,fontsize=17,gap=2.6,align="C",indent=6):
    path,wpx,hpx=render_math(latex,fontsize=fontsize); place_png(path,wpx,hpx,gap,align,indent)
def eqn_composite(items,fontsize=16,gap=2.8):
    p,w,h=render_composite(items,fontsize=fontsize); place_png(p,w,h,gap)
def barbox(title,lines,barcol,titcol,eqs=None,fill=None):
    pdf.ln(1); top=pdf.get_y()
    if title:
        pdf.set_text_color(*titcol); pdf.set_font("sans","B",10.8); pdf.set_x(24)
        pdf.multi_cell(W-6,5.5,mathify(title))
    pdf.set_text_color(*INK); pdf.set_font("sans","",10.4)
    for ln in lines: pdf.set_x(24); pdf.multi_cell(W-6,5.5,mathify(ln))
    if eqs:
        for e in eqs: eqn(e,fontsize=15,gap=1.5)
    bot=pdf.get_y(); pdf.set_draw_color(*barcol); pdf.set_line_width(0.9); pdf.line(20.9,top,20.9,bot); pdf.ln(2.6)
def thm(title,lines,eqs=None): barbox(title,lines,ACC,ACC,eqs)
def intuition(lines):
    barbox("Intuition.",lines,AMBBAR,AMB)
def rule():
    pdf.ln(1); pdf.set_draw_color(*RULE); pdf.set_line_width(0.2); y=pdf.get_y()
    pdf.line(20,y,pdf.w-20,y); pdf.ln(2.5)
def notation_row(sym,desc):
    if pdf.get_y()+12 > pdf.h-18: pdf.add_page()   # keep the whole row on one page
    y0=pdf.get_y()
    pdf.set_font("sans","B",10.0); pdf.set_text_color(*ACC)
    pdf.set_xy(22,y0); pdf.multi_cell(42,5.6,mathify(sym))
    y_sym=pdf.get_y()
    pdf.set_font("sans","",10.0); pdf.set_text_color(*INK)
    pdf.set_xy(66,y0); pdf.multi_cell(W-46,5.6,mathify(desc))
    pdf.set_y(max(y_sym, pdf.get_y(), y0+5.6))

# ===================== TITLE =====================
pdf.set_font("sans","B",18); pdf.set_text_color(*INK)
pdf.set_x(20); pdf.multi_cell(W,8.6,"When Can Training Data Be Recovered?")
pdf.set_font("sans","B",13); pdf.set_text_color(*ACC)
pdf.set_x(20); pdf.multi_cell(W,6.6,"A rank condition for non-identifiability from the first-layer weight signal")
pdf.ln(1); pdf.set_font("sans","I",10.4); pdf.set_text_color(*MUTE)
pdf.set_x(20); pdf.multi_cell(W,5.3,"A detailed, self-contained proof: if the gate matrix has rank below the "
              "number of training samples, the inputs cannot be recovered from the first-layer "
              "weight signal — no matter the algorithm.")
pdf.ln(1); pdf.set_font("mono","",8.6); pdf.set_text_color(*MUTE)
pdf.cell(0,5,"Yoad Oxman  ·  thesis note  ·  2026-08-20"); pdf.ln(7); rule()

# ===================== PLAIN SUMMARY =====================
h2("In plain terms — read this first")
para("The reconstruction attack tries to read the training images back out of a trained "
     "network's weights. This note pins down a hard limit on when that is possible even in "
     "principle.")
para("Every training sample leaves a fingerprint in the first layer's weights. But the "
     "fingerprint is filtered through how strongly each neuron reacts to that sample — a number "
     "we call the GATE. Collect all the gates into a matrix M: one row per neuron, one column "
     "per sample. The claim is:")
barbox("",
       ["The samples can be separated (recovered) only if M has rank at least N, the number of "
        "samples. If rank(M) < N, several samples share essentially the same gating pattern, "
        "their fingerprints add together into a blend, and no algorithm can un-mix them."],
       ACC,ACC)
intuition([
    "It is the same as being told a + b = 5 and asked for a and b. You cannot: (1,4), (2,3), "
    "(0,5) all fit — a whole line of answers, because there are two unknowns but only one "
    "equation. A low-rank gate matrix does exactly this: it gives you fewer independent "
    "measurements than samples, so several images collapse into one sum you can never split "
    "back apart. 'rank(M)' is just the number of genuinely independent measurements the neurons "
    "provide; when it drops below N, you are under-determined and recovery is impossible."])
para("We (1) derive exactly what the attacker observes, (2) show it is a matrix product "
     "'gates × data', (3) prove the rank condition step by step, (4) count exactly how many "
     "indistinguishable alternative datasets exist, and (5) walk a tiny numeric example through "
     "every step by hand (§6).")

# ===================== NOTATION =====================
h2("Notation")
for s,d in [
    ("d","input dimension (e.g. number of pixels)."),
    ("N","number of training samples in the batch — the thing we count against."),
    ("m","number of hidden neurons (the width of the layer)."),
    ("x_i ∈ ℝ^d","the i-th training input; the columns of the data matrix X. This is what we recover."),
    ("w_k ∈ ℝ^d","weight vector of neuron k; the rows of the weight matrix W."),
    ("v_k ∈ ℝ","output weight of neuron k (second layer)."),
    ("σ, σ'","the activation function and its derivative."),
    ("⟨w_k, x_i⟩","the pre-activation of neuron k on sample i (a scalar)."),
    ("M_{ki}","the GATE MATRIX entry: M_{ki} = σ'(⟨w_k, x_i⟩)."),
    ("c_i ∈ ℝ","a per-sample scalar coefficient coming from the loss (defined in §1)."),
    ("Ω","the observed first-layer signal — an m×d matrix (a gradient or a weight change)."),
]:
    notation_row(s,d)
pdf.ln(1)

# ===================== 1. SETUP =====================
h1("1.  What the attacker sees, derived from scratch")
para("We use the standard two-layer network of the reconstruction literature. With m hidden "
     "units and activation σ,")
eqn(r"f(x;\theta)=\sum_{k=1}^{m} v_k\,\sigma(\langle w_k,\,x\rangle),\qquad "
    r"w_k\in\mathbb{R}^{d},\ v_k\in\mathbb{R}.")
para("Given the training set {(x_i, y_i)} and a per-sample loss ℓ, the empirical loss is "
     "L(θ) = Σ_i ℓ(f(x_i; θ), y_i). The attacker observes the FIRST-LAYER weight signal — "
     "either the gradient of the loss with respect to the first-layer weights (which, at a "
     "stationary or KKT point of training, equals a known function of the final weights), or the "
     "first-layer part of a fine-tuning weight change ΔW (one gradient step on those weights). "
     "Both have the same structure, which we now derive.")
para("Differentiate L with respect to a single neuron's weights w_k. Writing "
     "c_i := ℓ'(f(x_i), y_i) for the loss slope at sample i, the chain rule gives (only the "
     "k-th term of the sum defining f depends on w_k)")
eqn(r"\frac{\partial L}{\partial w_k}=\sum_{i=1}^{N} c_i\,"
    r"\frac{\partial f(x_i)}{\partial w_k},\qquad "
    r"\frac{\partial f(x_i)}{\partial w_k}=v_k\,\sigma'(\langle w_k,x_i\rangle)\,x_i.",
    fontsize=15)
para("Substituting, the (k, l) entry of the observed matrix Ω := ∂L/∂W — row k (neuron), "
     "column l (input coordinate) — is")
eqn(r"\Omega_{k,l}=\sum_{i=1}^{N} c_i\,v_k\,\sigma'(\langle w_k,\,x_i\rangle)\,x_{i,l},"
    r"\quad\text{equivalently}\quad "
    r"\Omega_{k,:}=\sum_{i=1}^{N} c_i\,v_k\,\sigma'(\langle w_k,\,x_i\rangle)\,x_i^{\top}"
    r"\ \ (\text{row } k).", fontsize=13.5)
para("Two facts about c_i that matter: it is a single scalar per sample, and it is the SAME for "
     "every neuron k (it does not carry a k index). For the max-margin / KKT form of the attack "
     "the identical structure appears with c_i = λ_i y_i (a Lagrange multiplier times the label). "
     "Everything below only uses that c_i is a nonzero scalar shared across neurons.")
intuition([
    "Each neuron reports a weighted sum of the training images. The weight it puts on image "
    "x_i is its gate σ'(⟨w_k, x_i⟩) — how sensitive that neuron is to that image. The image "
    "only ever enters the observation through these gated sums. So the question 'can we recover "
    "the images?' becomes 'can we undo these gated sums?' — a linear-algebra question."])

# ===================== 2. FACTORIZATION =====================
h1("2.  The observation is 'gates × data'")
para("Collect the pieces into matrices. Let")
eqn(r"X=[x_1,\dots,x_N]\in\mathbb{R}^{d\times N}\ \ (\text{columns are the samples}),\qquad "
    r"M_{ki}=\sigma'(\langle w_k,x_i\rangle)\in\mathbb{R}^{m\times N},")
eqn(r"D_v=\mathrm{diag}(v_1,\dots,v_m),\quad D_c=\mathrm{diag}(c_1,\dots,c_N),\quad "
    r"G:=D_v\,M\,D_c\in\mathbb{R}^{m\times N}.")
para("The two diagonal matrices merely rescale the rows and columns by the (nonzero) output "
     "weights and loss coefficients; the informative content is the gate matrix M inside G.")
thm("Lemma 1  (the observation factorizes).",
    ["The whole observation is a single matrix product of the gates and the data:"],
    eqs=[r"\Omega \;=\; G\,X^{\top}."])
para("Proof.  Compare entry (k, l) of both sides. On the right,",gap=0.8)
eqn(r"(G X^{\top})_{k,l}=\sum_{i=1}^{N} G_{ki}\,(X^{\top})_{i,l}"
    r"=\sum_{i=1}^{N} v_k\,M_{ki}\,c_i\,x_{i,l}"
    r"=v_k\sum_{i=1}^{N} c_i\,\sigma'(\langle w_k,x_i\rangle)\,x_{i,l},",fontsize=14.5)
para("which is exactly the entry Ω_{k,l} from §1. Since this holds for every k and l, "
     "Ω = G Xᵀ.  ∎")
barbox("Dimension check.",
       ["Every term has a clear shape, and they compose consistently (m = neurons, "
        "N = samples, d = input dimension):",
        "•  each sample x_i ∈ ℝ^d is a d-vector;  the data matrix X is d×N;  so Xᵀ is N×d.",
        "•  each w_k ∈ ℝ^d, and the first-layer weight matrix W is m×d.",
        "•  the gate σ'(⟨w_k, x_i⟩) is a scalar, so the gate matrix M is m×N.",
        "•  the two diagonal scalings are m×m and N×N, so G stays m×N.",
        "•  Ω = G Xᵀ multiplies an m×N by an N×d — the inner N's cancel — giving an m×d "
        "matrix, the same shape as W (as it must be, since Ω is the gradient with respect to W)."],
       AMBBAR, AMB,
       eqs=[r"\Omega\ (m\times d)\ =\ G\ (m\times N)\ \cdot\ X^{\top}\ (N\times d)"])
thm("Lemma 2  (rescaling does not change rank).",
    ["Assume (H3): every v_k ≠ 0 and every c_i ≠ 0. Then the two diagonal scalings are "
     "invertible, and multiplying by an invertible matrix never changes rank, so"],
    eqs=[r"\mathrm{rank}(G)=\mathrm{rank}(D_v\,M\,D_c)=\mathrm{rank}(M)."])
# figure
fp,fw,fh=make_schematic(); fw_mm=W; fh_mm=fh/fw*fw_mm
if pdf.get_y()+fh_mm+8>pdf.h-18: pdf.add_page()
pdf.ln(2); pdf.image(fp,x=20,y=pdf.get_y(),w=fw_mm); pdf.set_y(pdf.get_y()+fh_mm); pdf.ln(1)
pdf.set_font("sans","I",9.2); pdf.set_text_color(*MUTE)
pdf.multi_cell(W,4.8,mathify("Figure 1.  The observation Ω is the product of the gate matrix G "
              "(m×N, but only rank k) and the data Xᵀ (N×d). The N samples must pass through a "
              "channel of rank k. When k < N the channel is too narrow to carry them "
              "separately — they arrive blended."))
pdf.set_text_color(*INK); pdf.ln(2)

# ===================== 3. PRIMER =====================
h1("3.  Three linear-algebra facts we will use")
para("The proof needs only these standard facts; we state them plainly so the argument is "
     "self-contained.")
para("(i)  Solution sets of linear equations.  If a linear map Φ satisfies Φ(X) = Ω and also "
     "Φ(X′) = Ω, then Φ(X′ − X) = 0. So every solution X′ equals the true X plus something the "
     "map sends to zero. In symbols, the full solution set is X + ker Φ, where "
     "ker Φ = {H : Φ(H) = 0} is the KERNEL (null space).")
para("(ii)  Rank–nullity.  For a linear map given by a matrix A with n columns, "
     "dim(ker A) = n − rank(A). The rank counts the independent directions the map keeps; the "
     "nullity n − rank(A) counts the directions it destroys.")
para("(iii)  Positive dimension means infinitely many.  A subspace of dimension ≥ 1 contains a "
     "whole line's worth of points — uncountably many. So if the solution set X + ker Φ has "
     "dimension ≥ 1, there are infinitely many datasets consistent with the same observation.")
intuition([
    "Whatever the map 'destroys' (its kernel) is information you can never recover from the "
    "output. If the data can be changed along any kernel direction without changing Ω, then Ω "
    "simply does not determine the data. The size of the kernel is the size of your ignorance."])

# ===================== 4. IDENTIFIABILITY =====================
h1("4.  What 'recoverable' means, precisely")
para("Fix the gate values M and the scalars v, c — i.e. treat the gates as GIVEN (assumption "
     "H2; §8 explains why this is the right and strongest way to phrase the limit). Under H2 the "
     "map from data to observation is linear:")
eqn(r"\Phi:\mathbb{R}^{d\times N}\to\mathbb{R}^{m\times d},\qquad \Phi(X)=G\,X^{\top}.")
para("We say the data X is IDENTIFIABLE from Ω if the only datasets X′ with Φ(X′) = Ω are X "
     "itself and its column permutations. Column permutation is a benign symmetry — it only "
     "relabels which image we call 'the first' — so it does not count as a failure. "
     "NON-identifiability means some genuinely different dataset (not a relabeling) produces the "
     "identical observation; then no procedure, however powerful, can tell which one was the "
     "true training set.")

# ===================== 5. THEOREM =====================
h1("5.  The theorem and its proof")
thm("Theorem  (rank obstruction).",
    ["Assume (H1) Ω = G Xᵀ, (H2) the gates G are fixed, (H3) v_k, c_i ≠ 0. "
     "If rank(M) = k < N, then the set of datasets consistent with the observation is an "
     "affine subspace"],
    eqs=[r"\Phi^{-1}(\Omega)=X+K,\qquad K=\{H\in\mathbb{R}^{d\times N}:\ "
         r"\mathrm{every\ row\ of\ }H\ \mathrm{lies\ in}\ \ker G\},",
         r"\dim K \;=\; d\,(N-k)\ \geq\ d\ \geq\ 1."])
para("In words: there is a whole d(N−k)-dimensional family of different datasets that all "
     "produce the same Ω. Hence X is not identifiable.")
para("Proof, in five steps.",gap=1.0,font=("sans","B"))
para("Step 1 (the solution set is X + kernel).  Φ is linear (Lemma 1: Φ(X) = G Xᵀ). By fact "
     "(i), the datasets consistent with Ω are exactly Φ⁻¹(Ω) = X + ker Φ, since the true X is "
     "one solution.")
para("Step 2 (what is in the kernel?).  We find every H with Φ(H) = G Hᵀ = 0. Fix a column "
     "index l and look at column l of G Hᵀ:")
eqn(r"(G H^{\top})_{k,l}=\sum_{i=1}^{N} G_{ki}\,H_{l,i}"
    r"=\left(G\cdot(\mathrm{row}\ l\ \mathrm{of}\ H)^{\top}\right)_k.",fontsize=14.5)
para("This is zero for all neuron indices k exactly when G times (row l of H)ᵀ is the zero "
     "vector — that is, when row l of H lies in ker G. This must hold for each of the d rows of "
     "H, and the rows are unconstrained otherwise, so")
eqn(r"\ker\Phi=\{H:\ \mathrm{each\ of\ the}\ d\ \mathrm{rows\ of}\ H\in\ker G\}.")
para("Step 3 (count the dimensions).  Each of the d rows ranges independently over the space "
     "ker G, so the dimension of the kernel is d copies of dim(ker G). By rank–nullity (fact ii) "
     "applied to G, which has N columns,")
eqn(r"\dim\ker\Phi=d\cdot\dim(\ker G)=d\,(N-\mathrm{rank}\,G)=d\,(N-k),")
para("using rank G = rank M = k from Lemma 2.")
para("Step 4 (the family is nontrivial).  Because k < N we have N − k ≥ 1, so "
     "dim ker Φ = d(N−k) ≥ d ≥ 1. By fact (iii) this is a positive-dimensional space: it "
     "contains infinitely many distinct H, hence Φ⁻¹(Ω) = X + ker Φ contains infinitely many "
     "distinct datasets, all producing the identical Ω.")
para("Step 5 (they are genuinely different, not relabelings).  The benign symmetries are the "
     "N! column permutations (plus any finite sign/scale symmetries) — a FINITE set. A finite "
     "set cannot fill a positive-dimensional continuum, so all but a measure-zero portion of the "
     "consistent datasets are unrelated to X by any symmetry. Therefore no function of Ω can "
     "single out the true X.  ∎")

# ===================== 6. NUMERIC EXAMPLE =====================
h1("6.  A worked example, walked through step by step")
para("Smallest interesting world: 2 training images, each with d = 2 pixels, and m = 2 neurons.",
     font=("sans","I"))

para("Step 0 — the setup.", gap=0.5, font=("sans","B"))
para("The two secret images the attacker wants are x₁ = (1, 0) and x₂ = (0, 1). The gate matrix "
     "G has entry G[k, i] = how strongly neuron k reacts to image i (times the loss weight); "
     "take")
eqn_composite([("t",r"G="),("m",[[1,2],[2,4]]),
               ("t",r"\Rightarrow\ \mathrm{row\ 2}=2\times\mathrm{row\ 1}:\ "
                    r"\mathrm{rank}(G)=1<2=N.")])
para("Notice the two neurons react to the images in the SAME proportion (1 : 2) — they are not "
     "two independent viewpoints, really just one. That is what rank(G) = 1 means, and it is the "
     "whole problem.")

para("Step 1 — what the attacker sees.", gap=0.5, font=("sans","B"))
para("The observed weights are Ω = G Xᵀ, which is the same as one fingerprint per image — each "
     "image imprinted by how the neurons react to it (g_i = column i of G, so g₁ = (1,2), "
     "g₂ = (2,4)):")
eqn_composite([("t",r"\Omega=g_1 x_1^{\top}+g_2 x_2^{\top}="),
               ("m",[[1,0],[2,0]]),("t",r"+"),("m",[[0,2],[0,4]]),
               ("t",r"="),("m",[[1,2],[2,4]])])

para("Step 2 — the collapse (why it breaks).", gap=0.5, font=("sans","B"))
para("Because g₂ = 2·g₁, factor g₁ out of the sum:")
eqn(r"\Omega=g_1 x_1^{\top}+(2g_1)\,x_2^{\top}=g_1\,(x_1+2x_2)^{\top}.", fontsize=15)
para("The two images have merged: Ω depends on them ONLY through the single combination "
     "s := x₁ + 2x₂. Reading it off Ω gives s = (1, 2) — and that is ALL the attacker can ever "
     "learn. One equation, two unknown images. (This is the a + b = 5 situation, exactly.)")

para("Step 3 — infinitely many datasets fit.", gap=0.5, font=("sans","B"))
para("Any (x₁′, x₂′) with x₁′ + 2x₂′ = (1, 2) gives the identical Ω. Choose x₂′ freely; x₁′ is "
     "then forced. For example x₂′ = (−1, 0) forces x₁′ = (1,2) − 2(−1,0) = (3, 2). Check that "
     "this completely different dataset reproduces the weights:")
eqn_composite([("t",r"G\,X'^{\top}="),("m",[[1,2],[2,4]]),("m",[[3,2],[-1,0]]),
               ("t",r"="),("m",[[1,2],[2,4]]),("t",r"=\Omega")])
barbox("",
       ["{(1,0), (0,1)} and {(3,2), (−1,0)} — two datasets with nothing in common — produce the "
        "IDENTICAL Ω. The attacker cannot tell which was the training set. That is "
        "non-recoverability, concretely."],
       ACC,ACC)

para("Step 4 — this family IS the kernel.", gap=0.5, font=("sans","B"))
para("The difference between the impostor and the truth is (x₁′−x₁, x₂′−x₂) = ((2,2), (−1,−1)). "
     "Its effect on the only thing Ω sees: Δs = (2,2) + 2(−1,−1) = (0, 0) — invisible. The set of "
     "all such invisible differences (h₁, h₂) with h₁ + 2h₂ = 0 is the KERNEL: h₂ is free in ℝ² "
     "(2 numbers), h₁ = −2h₂ is forced, so its dimension is 2 — matching the formula "
     "d(N − rank) = 2(2 − 1) = 2. The 'maneuver' is just: count the directions you can move the "
     "data without changing the measurement.")

para("Step 5 — the contrast that shows rank is everything.", gap=0.5, font=("sans","B"))
para("Had the neurons given INDEPENDENT views — say")
eqn_composite([("t",r"G="),("m",[[1,0],[0,1]]),("t",r"\ (\mathrm{rank}\ 2=N)")])
para("— then Ω = g₁x₁ᵀ + g₂x₂ᵀ with g₁ = (1,0), g₂ = (0,1) hands you the images directly: row 1 "
     "of Ω is x₁ and row 2 is x₂. Unique recovery. Nothing collapsed because g₂ is not a "
     "multiple of g₁. rank = N ⟹ recoverable; rank < N ⟹ not. That is the theorem.")

# ===================== 7. WHY LOW RANK BLURS =====================
h1("7.  Why low rank blurs the samples (the mechanism)")
para("When two samples have proportional gate columns (the rank-1 case), every neuron reacts to "
     "them in lockstep — whatever neuron k does to x₁, it does a fixed multiple of to x₂. The "
     "observation then depends on the two images only through one fixed linear combination of "
     "them, say s = c₁x₁ + γc₂x₂. The individual images are free to slide along the line that "
     "keeps s fixed, and every point on that line fits the data equally well. The reconstruction "
     "can at best return the blend s — never the two images apart. This is exactly the "
     "'superposition' failure seen empirically for N ≥ 2 once the per-sample gradients become "
     "collinear, and it is why raising the gate rank (through the activation, the anchor, or the "
     "LoRA rank) is what buys separability.")

# ===================== 8. COROLLARY + SCOPE =====================
h1("8.  Corollary, scope, and honest caveats")
thm("Corollary  (a necessary width).",
    ["Recovering X from the first-layer signal requires rank(M) ≥ N. Since rank(M) ≤ min(m, N), "
     "it in particular requires the hidden width m ≥ N: a layer narrower than the batch cannot "
     "expose its samples through this signal, whatever the activation, optimizer, or algorithm."])
para("(a)  Why 'fixed gates' is the right framing.  In reality the gates depend on the data "
     "(M_{ki} = σ'(⟨w_k, x_i⟩)), so the true map X ↦ Ω(X) = G(X) Xᵀ is bilinear, and the "
     "alternative datasets X′ generally have different gates. The theorem therefore describes the "
     "STRONGEST possible attacker — one additionally handed the exact gate values of the true "
     "data. A real attacker knows less, so cannot do better on this count. What is rigorously "
     "proved: the first-order (gate-factorization / NTK-linearized) information in Ω is "
     "INSUFFICIENT when rank(M) < N, and that first-order regime is exactly where the attack "
     "operates. The extra self-consistency the true gates must satisfy can in special cases "
     "shrink the family, so this is a necessary condition, not a claim that no nonlinear trick "
     "could ever help.")
para("(b)  Necessary vs. sufficient.  rank(M) ≥ N is necessary. In the fixed-gate model it is "
     "also generically sufficient: if rank(G) = N then X = (G⁺Ω)ᵀ is the unique solution "
     "(G⁺ = Moore–Penrose pseudoinverse). Turning this into a clean sufficiency theorem for the "
     "full bilinear problem — exactly when ΔW determines {x_i} uniquely — is the open "
     "identifiability question the thesis targets.")
para("(c)  One layer.  We used the first-layer factorization, where the data appears explicitly "
     "(∂L/∂W₁ carries x_iᵀ). Deeper layers may add constraints; the bound is a lower bound on "
     "difficulty, tight in the linearized regime.")

# ===================== 9. CONSEQUENCE =====================
h1("9.  Why this matters for the thesis")
para("rank(M) is a hard ceiling on how many samples can be separated from the first-layer "
     "signal, and the three levers of the study act on it directly. The ACTIVATION determines "
     "σ' and hence the entries of M. The ANCHOR — linearizing partway between the initial "
     "weights θ₀ and the fine-tuned weights (the α-sweep below) — shifts the pre-activations "
     "⟨w_k, x_i⟩ and so reshapes M. A LoRA adapter replaces Ω by a rank-r projection of it, "
     "tightening the ceiling further to")
eqn(r"\mathrm{leakage}\ \leq\ \min\left(\mathrm{rank}(M),\ r,\ N\right),\qquad "
    r"\theta_{\mathrm{anchor}}=(1-\alpha)\,\theta_0+\alpha\,\theta_T.")
para("The theorem above is the first, information-theoretic, of these ceilings — the reason "
     "'how many samples, and through what activation and rank' is the right axis for the whole "
     "study.")

out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"identifiability_rank_bound.pdf")
pdf.output(out); print("wrote",out)
