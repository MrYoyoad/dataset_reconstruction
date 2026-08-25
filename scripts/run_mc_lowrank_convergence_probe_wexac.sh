#!/bin/bash
#BSUB -q short-gpu
#BSUB -R "rusage[mem=16384] select[ngpus>0 && hname!='lgn28' && hname!='hgn46']"
#BSUB -gpu "num=1"
#BSUB -o scripts/wexac_logs/mc_lowrank_conv_%J.out
#BSUB -e scripts/wexac_logs/mc_lowrank_conv_%J.err
#BSUB -J mc_lrconv

# LOW-RANK CONVERGENCE PROBE — is low-rank 10-class NON-memorization (r=2:6.1e-3, r=4:1.86e-3
# at lr=0.5/T=1000) a REAL capacity limit or under-training? Train HARDER, several ways, read
# max_bce. VALUE-ONLY unroll (create_graph=False) -> no Jacobian, cheap, dodges the AD wall.
# Never <1e-3 across aggressive recipes = genuine capacity limit; some recipe <1e-3 = under-training.
source /apps/easybd/programs/miniconda/24.11_environmentally/etc/profile.d/conda.sh
conda activate /home/projects/galvardi/yoado/.conda/envs/rec
cd /home/projects/galvardi/yoado
export PYTHONPATH="/home/projects/galvardi/yoado/dataset_reconstruction:$PYTHONPATH"
python -c "import torch; print(f'CUDA={torch.cuda.is_available()}')"
echo "=== START $(date) on $(hostname) ==="
python -u -c "
import torch, experiments.jacobian_spectrum as j
torch.set_default_dtype(torch.float64)
dev='cuda'
recipes=[(0.5,2000),(0.5,4000),(0.5,8000),(0.7,2000),(1.0,1000),(1.0,2000),(1.0,4000)]
print('  base  rank  lr    T      mean_bce   max_bce   memorized')
def probe(nc,r):
    ctx,_,_,_=j._mnist_ctx(N=10,k=8,T=1,rank=r,activation='gelu',seed=42,device=dev,
                           tangent_method='qr',dataset='mnist',num_classes=nc,classes_present=None,lr=0.5)
    a0=torch.zeros(ctx.U.shape[0],ctx.U.shape[2],dtype=torch.float64,device=dev)
    x_priv=j.make_images(ctx.x0_centered,ctx.U,a0)
    best=1e9
    for lr,T in recipes:
        A,B=j.unrolled_lora_AB(ctx.frozen,ctx.b0,ctx.B0,x_priv,ctx.y,lr,T,ctx.scaling,ctx.act,
                               ctx.target_layers,num_classes=ctx.num_classes,create_graph=False)
        A={l:A[l].detach() for l in A}; B={l:B[l].detach() for l in B}
        m=j.finetune_metrics(ctx.frozen,ctx.b0,A,B,ctx.x0_centered,ctx.y,ctx.scaling,ctx.act,
                             ctx.target_layers,num_classes=ctx.num_classes)
        mb=m['max_bce']; best=min(best,mb); tag='BIN' if nc<=2 else '10C'
        print('  %-4s  r=%-3d %-4.1f  %-5d  %.3e  %.3e  %s'%(tag,r,lr,T,m['mean_bce'],mb,'YES' if mb<1e-3 else 'no'))
    print('  -> %s r=%d: BEST max_bce = %.3e  (%s)'%('BIN' if nc<=2 else '10C',r,best,'MEMORIZABLE' if best<1e-3 else 'CAPACITY-LIMITED (never memorizes)'))
    print()
for r in [1,2,4,8]:
    probe(10,r)
probe(2,2)
"
echo ""; echo "=== DONE $(date) ==="
