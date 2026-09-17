import torch
torch.set_default_dtype(torch.float64)
# STRESS TEST: does Pi A_T = Pi A_0 survive when I + M_t G is far from identity, or singular?
# If H3 were load-bearing, a large step size should break the identity.
def run(lr, T=80, seed=0, d=40, r=12, m=7, N=5):
    torch.manual_seed(seed)
    H=torch.randn(d,N); A0=torch.randn(r,d)/d**.5; W0=torch.randn(m,d)/d**.5
    y=torch.arange(N)%m; Y=torch.eye(m)[y].T
    A,B=A0.clone(),torch.zeros(m,r)
    G=H.T@H; M=torch.zeros(N,N); P=torch.zeros(m,N)
    worst_cond=0.0
    for t in range(T):
        Z=W0@H+B@(A@H); D=(torch.softmax(Z,0)-Y)/N
        # track I + M G conditioning via the coefficient recurrence
        IMG=torch.eye(N)+M@G
        sv=torch.linalg.svdvals(IMG); worst_cond=max(worst_cond, float(sv[0]/max(sv[-1],1e-300)))
        gB=D@(A@H).T; gA=B.T@D@H.T
        Pn=P-lr*D@(torch.eye(N)+G@M.T); Mn=M-lr*P.T@D
        B,A=B-lr*gB,A-lr*gA; P,M=Pn,Mn
    S=torch.linalg.svdvals(B); q=int(torch.linalg.matrix_rank(H)); rk=int((S>1e-12*S[0]).sum())
    U,_,Vh=torch.linalg.svd(B,full_matrices=False); Q=Vh[:rk].T; Pi=torch.eye(r)-Q@Q.T
    ident=float((Pi@A-Pi@A0).norm()/(Pi@A0).norm())
    return lr, worst_cond, q, rk, ident, float((Pi@A@H).norm()/((Pi@A).norm()*H.norm()))
print("%-8s %-13s %-4s %-9s %-12s %s"%("lr","worst cond(I+MG)","q","rank B_T","||PiA_T-PiA_0||","||C H|| rel"))
for lr in (0.01,0.05,0.5,5.0,50.0,500.0):
    l,c,q,rk,i,ch=run(lr)
    flag="  <-- H4 FAILS" if rk!=q else ""
    print("%-8g %-13.3e %-4d %-9d %-12.2e %.2e%s"%(l,c,q,rk,i,ch,flag))
