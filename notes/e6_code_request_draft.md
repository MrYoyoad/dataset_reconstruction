# E6 — code request to the co-authors: DRAFT, **NOT SENT**

**Status: drafted and held. Nothing here has been sent to anyone.** Written per §12's day-one instruction ("draft
(do not send) a two-paragraph request to the co-authors for the stage-2 inversion code paths (DIP + cosine for
DINO/ViT; Karlo UnCLIP for CLIP) as they ran them, and list what you can rebuild locally if the code is not
shared"). Gal offered to arrange the meeting; sending is his call and the user's, not this session's.

**To:** Niv Haim, Michal Irani — co-authors of Oz, Yehudai, Vardi, Antebi, Irani, Haim, *Reconstructing Training
Data From Real-World Models Trained with Transfer Learning* (arXiv:2407.15845).

---

## The two paragraphs

Dear Niv, dear Michal,

We are working on reconstruction from released LoRA adapters, and we have reached the point where the private
**representations** come back but the **images** do not, which is exactly the boundary your stage 2 crosses. Where
a LoRA adapter is trained on a frozen backbone, the released factors determine a linear certificate that the
private representations satisfy exactly. That certificate **constrains** them to a subspace but by construction
cannot separate them from it — every blend of the private representations satisfies it too, so on its own it
returns blends and nothing else. **Replaying the training dynamics** then resolves which points inside that set
are the training representations: on a synthetic fp64 release, replay recovers all eight from random starts (19 of
60 starts at a 1e-2 image-error bar, 18 of those with worst-image error between 1.4e-15 and 2.2e-15, no aliases),
while the certificate alone lands none of them. It is the constrain-then-resolve division of labour that does the
work, not either half. That leaves us holding embeddings and needing your machinery to see pictures. One thing we can offer in
return, and it bears directly on something your paper flags as unresolved: you obtain the scale of a recovered
embedding by rescaling to the norm of the nearest *training* embedding, and note the reliance on the training set.
Our replay route appears not to need that. Rescaling the representation by α while rescaling the adapter seed by
1/α leaves the adapter path unchanged but sends the frozen base path to αW₀h, which moves the logits and therefore
the released factors, so the scale should be pinned by the dynamics themselves. We measured it rather than argued
it: the replay residual has a sharp machine-precision minimum at the true scale and rises **linearly** on both
sides with slope 0.414, so the scale is first-order identifiable, not merely identifiable in principle. The honest
scope is one synthetic fp64 release with an identity feature map and a single cell; whether it survives a real
backbone is untested, and that test is precisely what we are writing to you about.

What we would like, if you are willing to share it, is your **stage-2 inversion code paths as you actually ran
them**: the DIP + cosine path for DINO and ViT (the Tumanyan-style model inversion, where we understand the cosine
objective is load-bearing because recovered embeddings come back with the wrong norm and MSE fails), and the Karlo
UnCLIP decoder path for CLIP together with the scaling convention you used with it. We are asking for the code
rather than rebuilding from the paper for one reason: we need an **oracle ceiling** we can trust before we decode
anything we recover — the decoded *true* representation, per backbone — because without it we cannot separate a
decoder limit from an attack failure, and a reimplementation that is merely close would make that ceiling
meaningless and could let us report a decoder's weakness as a privacy result. If sharing is awkward for any
reason, a pointer to the exact configuration would still help a great deal, and we would be glad to show you what
we have first.

---

## What we can rebuild locally if the code is not shared

| path | rebuildable? | what we would be guessing |
|---|---|---|
| DIP U-Net + **cosine** loss against a frozen DINO / ViT | **yes** — standard components; the paper states the objective and why MSE fails | U-Net width/depth, noise schedule, iteration count, LR, early stopping. These move the oracle ceiling, so our ceiling would not be comparable to theirs |
| **Karlo UnCLIP** decoder for CLIP | **weakest link** — the checkpoint is public, the *configuration* is not | guidance scale, prior steps, decoder steps, and above all the **norm convention** for the input embedding. This is where their exact setup matters most and where a reimplementation is most likely to be quietly wrong |
| Candidate **clustering / selection** (their stage 3) | yes in principle | their selection criterion. Lower priority: it is a cost-reduction stage, not a fidelity stage |
| **Oracle ceiling** D(h\*) per backbone | yes — it needs only the decoder above | nothing extra; it inherits every uncertainty of the decoder path |

**Order of work regardless of the answer:** oracle ceiling **first**, per §9 and §12's nevers. No recovered
representation gets decoded for a backbone until D(h\*) exists for that backbone, and representation recovery
(distance and cosine to the true representation) is reported **separately** from image recovery (decoded image
against the photograph), never merged into one score. A poor oracle ceiling is a decoder limit, not an attack
failure. A generic decoder is not called "UnCLIP".

## Questions for the meeting (§13)

1. **Scale.** Does the norm-of-nearest-training-embedding step remain your practice, and would a dynamics-based
   scale recovery be useful to you? (This is the offer above; it is measured on one synthetic cell only.)
2. **Interface.** Can your inversion machinery accept representations recovered by a certificate or by replay,
   as-is, or does it assume properties of stage-1 candidates that ours would not satisfy?
3. **Cost.** The paper reports roughly thirty minutes per embedding on a V100. What is it on current hardware, and
   what dominates today — candidate generation, inversion, or selection?
4. **The linear regime.** Your KKT scheme performs poorly on linear probes, while LoRA on a head is a low-rank
   *linear* map where our finite-T certificate is exact. Have you seen anything that gives signal in that regime?
5. **Charts.** Did you try restricting stage-1 candidates to a low-dimensional or target-conditioned chart of
   embeddings, rather than searching the full embedding space?

## Provenance

Spec supplied by the approver lane (yoado-8b) as §11 and §12 verbatim; §3.2/§9 machinery description from the same
source. The slope 0.414 and the machine-precision minimum are measured in this repo — jobs **670990 / 670993**,
`scale_precheck` in `results/e1b/rows.jsonl`, fp64, nine α values, derivable from the stored artefact. The
recovery figure is job **331384**, stated in the draft in its precise form and credited to **replay**, not to the
certificate: 19 of 60 starts return all eight at the 1e-2 bar, 18 of those with worst-image error between 1.4e-15
and 2.2e-15, zero aliases, against **0 of 60** for the certificate route on the same release and the same starts.
The count that matters reads `worst_image_err`; `replay_best_err_min` is the best single image in the best start
and must never be quoted as a per-start worst case.
