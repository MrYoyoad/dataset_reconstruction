# Extending LoRA certificates and charts to text

Research plan for Yoad Oxman · 15 September 2026

**Recommendation.** Develop a text extension around exact, finite candidate separation from a released adapter. Begin with frozen contextual features and short private spans. Keep a second route for known adapter initialization: sparse recovery from the change in the input factor can remain informative after the original nullspace certificate becomes empty. Use discrete charts to state what can be verified; use language models and continuous latents to search those charts.

The strongest prospective contribution combines an exact endpoint identity, a justified text family, and an explicit uniqueness or exclusion margin. The literature already contains text subspace attacks, LoRA-gradient attacks, language priors, and sparse gradient reconstruction. The new contribution must be established against those specific results.

This note distinguishes consequences proved below, established external results, and proposed experiments. The only computation performed here is a small Gaussian-dictionary linear-regression example, documented in section 10. No transformer reconstruction experiments have been run.

## 1. Fix the observation model before choosing an algorithm

The primary target is recovery of private training token strings from a benign released LoRA adapter and its public base model. Treat these as separate observation regimes:

| Regime | Attacker receives | Available approach |
|---|---|---|
| Endpoint factors, initialization unknown | Base model, tokenizer, final A and B, adaptation locations and scaling | Our certificate, conditional on its training assumptions |
| Endpoint factors, initialization known | The preceding information and exact A at initialization | Certificate plus sparse recovery from the change in A |
| Training recipe known | Initialization, optimizer, steps, schedule, masks, minibatches and randomness, or a specified nuisance family | Forward-training replay and candidate exclusion |
| Gradients observed | A gradient at a known parameter state | DAGER and related gradient attacks |
| Only merged weights | The product BA, without the original factors | A different inverse problem; do not assume the factor certificate can be computed |

Knowing the base model does not reveal the random initialization of its subsequently attached adapter. Knowing a nominal seed is sufficient only if the initialization procedure and random-number consumption can actually be reproduced.

First use vanilla simultaneous SGD, B initialized to zero, frozen upstream features, no input or LoRA dropout, no weight decay, and deterministic computation. These are the setting of the first exact results, not a description of every public adapter. Gaussian initialization gives the clean probability calculation below. PEFT's documented default uses Kaiming-uniform A and zero B; Gaussian is an explicit alternative. The algebraic invariant does not require Gaussianity, but the stated chi-square distribution does. [PEFT documentation](https://huggingface.co/docs/peft/v0.20.0/package_reference/lora)

Report token support, ordered tokens, whole-record exact match, and whole-dataset recovery separately. For dataset recovery, compare modulo record permutation only when that is an actual symmetry of the declared training observation model. Minibatch order can itself affect the release.

## 2. The relevant literature and the remaining question

| Source | Established connection | Implication for this project |
|---|---|---|
| [DAGER, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/9ff1577a1f8308df1ccea6b4f64a103f-Paper-Conference.pdf) | Discrete text reconstruction through gradient subspace tests. Appendices B.3 and B.4 discuss local multistep updates and LoRA. The LoRA treatment explicitly uses a token-rank condition and accounts for zero initialization delaying the relevant gradient. | “Subspace tests work on LoRA text” is insufficient novelty. Compare exact endpoint access with unknown initialization against these appendices. |
| [LAMP, NeurIPS 2022](https://arxiv.org/abs/2202.08827) | Text inversion using a language prior and alternating continuous and discrete optimization. | A fluent reconstruction or adding an LM prior is not the theoretical contribution. |
| [TIGER, June 2026 preprint](https://arxiv.org/html/2606.18312v1) | Optimizes representations toward gradient-derived subspaces; considers numerical perturbations and encoder/decoder differences. Its threat model observes gradients at a known model state. | Subspace-distance optimization is also prior work. Reuse it as a search baseline. |
| [SOMP, March 2026 preprint](https://arxiv.org/html/2603.16761v1) | Uses geometric token filtering, decoding, and sparse fitting of candidate sentence gradients under FedSGD. Its authors distinguish the method from a worst-case recovery guarantee. | Compare the vocabulary-factor identity below to its candidate-gradient dictionary and observation model. |
| [UTR, January 2026 preprint](https://arxiv.org/html/2601.17533v1) | Reconstructs text from adapter gradients using word bags and subspaces. Section 3.1 specifies nonlinear downsampling/activation/upsampling adapters and excludes other PEFT types. | Its title should not be treated as evidence of an attack on an ordinary released BA LoRA endpoint. |
| [Text Embeddings Reveal (Almost) As Much As Text, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.765/) | Iterative decoding and re-embedding can recover text from some sentence embeddings. | Supports a decoder component; does not prove that our certificate has identified the target embedding. |
| [Optimus, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.378/) and [Bowman et al., CoNLL 2016](https://aclanthology.org/K16-1002/) | Explicit continuous latent representations for sentence generation. | Possible proposal mechanisms; their latent dimension is not an exact-recovery guarantee. |
| [Rank Awareness in Joint Sparse Recovery, Davies and Eldar](https://arxiv.org/abs/1004.4529) | Uniqueness results for joint sparse linear inverse problems. | Supplies an established theorem to compose with the endpoint identity, not a theorem to claim as new. |
| [Activated LoRA](https://arxiv.org/html/2504.12397v3) | Applies the adapter only from an invocation point onward, including during training. | Suggests a concrete architecture where the number of adapted input features can stay small despite a long context. |

A further lead is [MineGrad, August 2026](https://arxiv.org/abs/2608.01521). The searchable abstract describes a poisoned model and malicious training intervention. Full text could not be inspected through the available access route, so no stronger comparison is made here.

This is a targeted research audit, not a claim that novelty has been exhaustively established. The endpoint theorem, sparse identity, and their combination with finite verification require a line-by-line comparison against the closest papers before a priority claim.

## 3. Exact transfer of the certificate

Write the adapted linear map in column-vector convention as

\[
W_t=W_0+B_tA_t,\qquad
A_t\in\mathbb R^{r\times d},\quad B_t\in\mathbb R^{m\times r}.
\]

Absorb the fixed LoRA scaling into the gradient factors. Let H in R^{d by K} collect the frozen input features that can contribute to the chosen adapter across the dataset. K counts feature occurrences; q = rank(H) counts independent directions. Pad minibatch gradients with zero columns so D_t has size m by K. The simultaneous SGD updates are

\[
B_{t+1}=B_t-\eta_tD_t(A_tH)^\top,\qquad
A_{t+1}=A_t-\eta_tB_t^\top D_tH^\top.
\]

Define U = col(H), S = A_0 U, p = dim(S), and let P be the orthogonal projector onto S-perp.

**Proposition 1 — frozen-input endpoint certificate.** Suppose B_0 = 0 and the terminal excitation condition rank(B_T) = p holds. Then

\[
C=P_{\operatorname{row}(B_T)^\perp}A_T=P A_0,
\qquad CH=0.
\]

If A_0 has full row rank and is injective on U, then p = q and rank(C) = r-q.

**Proof.** Initially B_0P = 0 and PA_0 = PA_0. Suppose B_tP = 0 and PA_t = PA_0. Then PA_tH = PA_0H = 0, so the B update gives B_{t+1}P = 0. The A update gives PA_{t+1} = PA_t because PB_t-transpose = 0. Thus row(B_T) is contained in S. Equality follows from rank(B_T) = dim(S). Substituting the resulting projector proves the identity. Finally, a full-row-rank A_0 is onto R^r, so composing with a projector of rank r-q has rank r-q. ∎

The proof does not require a linear downstream model, a constant gradient, one optimization step, or knowledge of the initialization. A transformer downstream of the adapter is allowed. All losses and later adaptations are absorbed into D_t.

The condition rank(B_T) = p is essential. A low terminal rank alone is not evidence that the projector annihilates every training feature. Cancellation or structurally unexcited features can invalidate that inference. In experiments, q is available to the evaluator, not automatically to the attacker.

**The main text obstruction.** For contextual token features, one document can supply many independent columns. If q reaches r and B_T has rank r, C is identically zero. A small-dimensional semantic chart does not restore this lost nullspace.

At a position-independent first-block input, repeated occurrences of a vocabulary token share an embedding, so q is bounded by the number of distinct contributing token types. At later contextual inputs, repeated tokens can produce different feature directions.

The output dimension also matters: m must be at least p. A binary softmax classification head can have a severe output-gradient rank restriction even when the number of sentences is small. Do not replace a language-model head by a binary head and silently retain the same excitation claim.

**Deep layers.** An exact trajectory formulation remains possible. Replace U by the span of all actual layer inputs across all training steps. The same induction gives a certificate for that span under terminal excitation. Its rank may grow rapidly. With full-batch training, initial features are in this trajectory span; with minibatches, not every example is necessarily seen at the initial state. This distinction prevents automatically interpreting a deep terminal certificate on every example's base-model feature.

A useful approximate deep-layer theorem must separately bound feature drift and terminal subspace rotation, with a spectral gap. Tiny nonzero drift can increase exact algebraic rank. Summing r-minus-q counts from invalid deep certificates is not justified.

**A concrete excitation theorem for a language head.** There is a useful restricted family where excitation can be proved. Let a frozen transformer provide K linearly independent contextual features, with one supervised prediction per feature. Suppose the K target vocabulary IDs are distinct, K < V, and K < r. Write the positive softmax probability columns as p_j and the output gradients as d_j = p_j - e_{y_j}, ignoring a common positive loss-normalization factor.

Restrict the gradient matrix D_0 to the K rows corresponding to its distinct targets. The resulting matrix is P_S - I_K, where P_S is nonnegative. Every column sum of P_S is strictly below one because softmax assigns positive mass to the V-K unselected vocabulary items. Thus its induced matrix 1-norm is below one, I_K-P_S is invertible, and rank(D_0) = K.

Gaussian initialization makes A_0H full column rank almost surely. Therefore

\[
B_1=-\eta D_0(A_0H)^\top,\qquad
\operatorname{rank}(B_1)=K.
\]

For any fixed number T of full-batch SGD steps with a common step size eta, smoothness and finite-step continuity give

\[
\lim_{\eta\to0^+}\frac{B_T}{\eta}
=-T D_0(A_0H)^\top.
\]

The K-th singular value of this limit is positive. Consequently, for all sufficiently small positive eta, rank(B_T) is at least K, while the invariant bounds it above by K. This proves terminal excitation in this family.

This statement allows an arbitrarily nonlinear frozen transformer. It requires independent contextual features and distinct supervised targets; it does not cover arbitrary repeated-target language-model training. The small-step neighborhood depends on T and the particular instance, so this is not a uniform long-training or finite-precision guarantee. Extremely concentrated softmax probabilities can make the conditioning poor even though the exact rank statement holds.

## 4. Text offers a finite separation theorem

Images prompted us to intersect a kernel with a continuous chart. Text offers a second possibility: intersect the kernel with a finite public codebook.

Let F be a finite set of candidate feature vectors fixed independently of A_0. For example, F may be the vocabulary embeddings at the first input, or the contextual representations of a declared finite set of strings.

**Proposition 2 — finite codebook separation.** Suppose q < r <= d, A_0 has independent Gaussian entries, and Proposition 1's excitation assumption holds. Almost surely,

\[
\{h\in F:Ch=0\}=F\cap U.
\]

Consequently, if F intersect U consists exactly of the true feature vectors, scanning F recovers them exactly.

**Proof.** True vectors in U are annihilated. For a fixed h outside U, write h = h_parallel + h_perp with h_perp nonzero and orthogonal to U. Conditional on A_0 restricted to U, the Gaussian vector A_0 h_perp is independent of that restriction. Its projection onto S-perp has a nondegenerate Gaussian law in r-q dimensions, so it is zero with probability zero. A finite union over candidates proves the claim. Equivalently, this statement can be made for the ideal projector P_{A_0 U-perp} for every draw; whenever excitation holds, it equals the observed certificate. ∎

The no-extra-codeword condition F intersect U is substantive. If two different strings have identical features, or an unrelated candidate feature lies in U, the certificate cannot distinguish them. Random initialization does not remove that structural collision.

This result explains a useful difference from continuous charts: one exact real-valued constraint can separate a finite set of wrong candidates in principle. There is no general rule that c constraints recover at most c discrete token identities. Precision and separation determine robustness.

**A quantitative version.** Set A_0 entries to N(0,1/r), c = r-q, and assume all M false candidate vectors satisfy dist(h,U) >= Delta > 0. For the ideal certificate,

\[
\frac{r\|Ch\|_2^2}{\operatorname{dist}(h,U)^2}
\sim\chi^2_c.
\]

For 0 < a < 1, a chi-square lower-tail bound and a union bound imply

\[
\Pr\!\left[
\min_{h\in F\setminus U}\|Ch\|_2
\le \Delta\sqrt{ac/r}
\right]
\le M\bigl(ae^{1-a}\bigr)^{c/2}.
\]

On excited runs the ideal and observable certificates coincide. The displayed probability is an unconditional bound on the ideal failure event, hence also bounds failure together with excitation; it is not an unqualified conditional-probability claim after selecting successful runs.

This gives a precise role for a chart: decrease the number of false candidates and remove candidates close to the training span while retaining the truth. Gaussianity gives the explicit distribution. Extending the quantitative bound to default uniform initialization is a separate task.

If the exact certificate is perturbed by operator norm at most epsilon_C and candidate features have norm at most R, score error is at most epsilon_C R. True/false separation is stable when the exact false-candidate margin exceeds twice that error. Additional feature-computation or model errors must be added to the bound. Selecting an SVD cutoff without a justified perturbation bound yields an empirical filter, not this robustness theorem.

## 5. Where to obtain useful frozen features

| Architecture or setting | Rank budget | What the certificate can constrain |
|---|---|---|
| Only an LM-head adapter; transformer frozen | Rank of features at loss-bearing prediction positions | Contextual prefixes, including earlier private content |
| First adapted attention block after a frozen prefix | Rank of inputs at contributing positions | Contextual text features, with no upstream adapter drift |
| First-block value adapter, fixed embeddings and normalization | Rank of contributing vocabulary embeddings | Token support; positions require another source of evidence |
| Activated LoRA at an early adapted block | Rank of adapted/contributing suffix features | Can avoid consuming rank on every token of a long context |
| Ordinary LoRA across many blocks | First-block input remains fixed, subject to the stated conditions; later inputs move | Exact first-block result; deep results need extra analysis |

**A practical candidate: Activated LoRA.** Its training rule gates the adapter off before an invocation point. In the chain rule this removes those inputs from the adapter's feature matrix. This suggests that a short activated suffix can preserve q < r even with a long preceding context. The privacy conclusion is our proposed deduction and experiment; it is not a claim established by the aLoRA paper. Across many records, q still counts the union of their active feature directions. [Activated LoRA](https://arxiv.org/html/2504.12397v3)

**Loss masking is different.** In ordinary LoRA, masking prompt tokens out of the language-model loss does not generally remove their contributions to early adapter gradients: later predictions attend to prompt keys and values. Loss masking reduces the relevant positions for a frozen LM-head adapter, but does not have the same effect throughout the transformer.

**Order and terminal targets.** A first-block RoPE-style token codebook contains token identity without position. The certificate therefore gives support, not ordering or multiplicity. Also, a final target token that never enters any contributing input feature is invisible to CH. For completion experiments, supervise the EOS prediction after the complete answer so the last private answer token appears in a contributing prefix. Otherwise its identity must be supplied by full gradient/endpoint fitting.

If the only supervised position precedes a one-token private answer and its input prefix is entirely public, the head certificate contains no answer-specific feature information under the theorem. This is an explicit negative control.

## 6. A second endpoint identity can work when C is empty

Now add the assumption that A_0 is known. At an input where features are determined by token identity, let

\[
E=[e_1,\ldots,e_V]\in\mathbb R^{d\times V}
\]

be the public vocabulary dictionary, including the actual frozen preprocessing at that input. For example, use the normalized embeddings that enter the first value projection, not raw embeddings when the two differ.

At step t let M_t select token columns, so H_t = E M_t. Transposing and summing the A updates gives

\[
Y:=(A_T-A_0)^\top=EX,\qquad
X=-\sum_t\eta_t M_tD_t^\top B_t.
\]

Every nonzero row of X belongs to a token type that contributed to training at this module.

**Proposition 3 — sparse support identifiability.** Suppose X has at most s nonzero rows and every set of at most 2s columns of E is linearly independent. Then X is the unique solution of EX = Y with at most s nonzero rows.

**Proof.** For another such solution X-prime, E(X-X-prime) = 0. The difference has at most 2s nonzero rows. Linear independence of the corresponding dictionary columns forces every column of X-X-prime to vanish. ∎

Recovering every contributing token requires an additional noncancellation assumption: each such token has a nonzero accumulated row in X. Token occurrence counts are not those coefficient rows.

In spark notation, the sufficient condition is 2s < spark(E). It depends on dictionary geometry, not on s < r. The observed matrix has d rows and r columns; a low number of columns does not prevent sparse support identification in a known dictionary. Stronger rank-sensitive MMV conditions are available in the classical literature, including

\[
s<\frac{\operatorname{spark}(E)-1+\operatorname{rank}(Y)}2.
\]

Use the sufficient direction with its assumptions; failure of the inequality does not prove ambiguity of a particular observed instance. [Davies and Eldar](https://arxiv.org/abs/1004.4529)

The identity is exact for multistep SGD at the stated frozen dictionary input. The support theorem is an application of sparse recovery, not new sparse-recovery theory. Its potential novelty is what it establishes about this released LoRA observation.

**Public-context elimination.** If the set P of public prompt token types is known, let Q_P project orthogonally to span(E_P). Then

\[
Q_PY=Q_PE_{P^c}X_{P^c}.
\]

This eliminates arbitrary unknown coefficients on the public tokens. It does not require subtracting a fabricated “prompt gradient.” Apply sparse recovery in the projected dictionary and recheck its conditioning. Secret tokens already present in P disappear under this projection, so their extra occurrences remain unresolved.

There are concrete limitations:

- A_0 is additional information. This route is unavailable from arbitrary unknown initialization without a further argument.
- For standard B_0 = 0 simultaneous SGD, A_1 = A_0. The channel starts after the first update and is often small at short times.
- Finite-precision storage can erase that small difference; evaluate actual saved and reloaded factors.
- Position-dependent embeddings require a token-position dictionary. Deep contextual embeddings require a contextual dictionary or a chart. The simple vocabulary identity does not silently transfer to them.
- Learned dictionaries can be highly coherent. Full spark is an ideal uniqueness condition, not a practical conditioning guarantee.
- Sparse optimization can fail even when a unique sparse solution exists. A dual certificate for a convex objective only certifies that objective unless a recovery theorem links it to the true generating coefficients.
- With decay, subtract the correctly attenuated initialization; the required attenuation must be known. With general Adam updates, the displayed linear identity is not preserved.

## 7. Define the text chart as a candidate family

The closest text counterpart of the shared-identity image chart is

\[
D(z,u)=\{(p_i,T_{a_i}(z,u_i))\}_{i=1}^N.
\]

Here p_i are public or partially known contexts; z is shared private content; a_i chooses a public template; and u_i describes per-record wording or other nuisance choices. A chart must state which of these objects are known and which are searched.

Begin with finite slots: a fabricated name, an identifier, a date, a location, a short factual answer, or a small structured record. Specify allowed values before using the private target. Include combinations not seen when the chart was built. This tests reconstruction beyond retrieval from a catalogue of whole known sentences.

For broader text, use a hierarchy of candidate sets: length and format, template or topic, lexical alternatives, prefixes, and complete strings. For a continuous helper, a sentence VAE or soft embedding parameterization can propose strings. A decoder can refine a proposed embedding by generating and re-embedding text.

**Do not reuse the smooth image-chart theorem on hard strings.** A continuous map from a connected latent region to a discrete set is constant. Hard token decoding therefore has discontinuities or regions with constant output. A full-rank Jacobian of a soft embedding proxy proves something about the proxy, not exact token recovery. Decode, recompute the real token features, and run the actual certificate and training checks.

For a fixed family X, define a certificate score

\[
s_C(x)=\max_{j\in J(x)}
\frac{\|C h_j(x)\|_2}{\|h_j(x)\|_2},
\]

using nonzero features and an explicitly defined set of contributing positions J(x). The maximum prevents a longer candidate from improving its score just by diluting a mean.

Coverage comes first: if the true string is outside X, the family cannot recover it. A public-corpus candidate family supports candidate identification; it does not establish reconstruction of arbitrary unseen strings. A latent chart that preserves meaning but removes digits or names is unsuitable for exact private-span recovery.

**Causal prefix pruning.** At a frozen causal feature map, a prefix feature is unchanged by appending future tokens. A prefix that violates a valid necessary certificate condition can therefore be pruned with all its continuations. This supplies a rigorous search tree when the relevant position is covered by the certificate. It does not hold in this form for a bidirectional encoder or a feature affected by an unknown future continuation.

Language-model probabilities may order this search. A finite beam or top-k cutoff can discard the true branch and loses completeness. A certified finite search must exhaust the remaining declared family, or use valid lower bounds to eliminate every unexplored branch.

## 8. What “verified reconstruction” would mean

Use three distinct statements:

1. **Feasible feature candidate:** its actual discrete features pass a valid certificate.
2. **Feasible training candidate:** replay with an allowed training recipe reproduces the released adapter within the stated error.
3. **Unique candidate in the declared family:** every competing data/recipe explanation in that family has been excluded.

Only the third is an identification statement. The first two can hold for aliases.

Let R(D;omega) be the deterministic training-to-release map, including the actual factor representation and serialization. Omega is a declared family of allowed initialization, schedule, optimizer and other nuisance choices. Define

\[
e(D)=\inf_{\omega\in\Omega}
\|R(D;\omega)-Y_{\rm obs}\|.
\]

If observation error is at most epsilon, one candidate D-hat has e(D-hat) <= epsilon, and every alternative has e(D) > epsilon, D-hat is the unique explanation inside the family. For an a priori robust theorem, define M_D = {R(D;omega): omega in Omega}; pairwise distances greater than 2 epsilon separate the corresponding error balls.

Exhaustive evaluation works for a small finite chart and finite nuisance family. Unknown continuous optimizer states or an unrestricted A_0 make this a substantially larger problem. Optimizing an unconstrained initialization until a candidate fits is not a uniqueness proof.

For multiple records, include assignment, duplication and ordering explicitly. For next-token training, labels are generated by shifting each candidate sequence with the true masking rule; providing the private shifted labels to the attacker would reveal much of the answer.

The certificate is computed from the same release as replay. It can reduce the search cost and expose interpretable constraints, but it is not an independent measurement to count twice.

## 9. Adam and dropout are real theorem boundaries

The first exact experiment should isolate SGD. Test AdamW and dropout as separate extensions because they can violate the invariant, not merely add a small generic perturbation.

For a concrete Adam counterexample, take r = 2, one fixed feature h with A_0h = (1,2)-transpose, B_0 = 0, and a scalar backpropagated gradient equal to one. The initial B gradient is (1,2). With zero Adam denominator epsilon, the first Adam update is proportional to (1,1). A is unchanged because its initial gradient is zero. Therefore

\[
P_{\operatorname{row}(B_1)^\perp}A_1h
=(-1/2,1/2)^\top\ne0.
\]

Here rank(B_1) equals q = 1 but CH fails. A positive Adam epsilon changes the numbers without generically restoring alignment. Making the learning rate small does not fix this direction mismatch.

Input dropout similarly replaces one feature by multiple masked versions. Analyze the span of the actual inputs, rather than calling them the frozen unmasked dictionary.

For small drift at deeper layers, measure both the certificate error and the singular gap; a Wedin-style perturbation route needs both. An empirical truncated SVD is useful but must not be called an exact certificate for an unrestricted AdamW-trained adapter.

## 10. The computation performed during this analysis

A deterministic NumPy experiment used a random Gaussian vocabulary of 18 normalized vectors in dimension d = 10, a frozen-feature linear regression loss with output dimension 8, and eight simultaneous SGD updates. The input used four distinct dictionary atoms, with repeats. This is an algebraic model, not pretrained text.

| Quantity | Observed value |
|---|---:|
| LoRA rank r | 2 |
| Rank of H | 4 |
| Rank of B_T | 2 |
| Frobenius norm of C | 5.33 × 10^-16 |
| Frobenius norm of A_T − A_0 | 0.131653 |
| Error in (A_T − A_0)-transpose = EX | 3.65 × 10^-16 |
| Candidate supports exhaustively evaluated | 3,060 |
| True support | {1, 4, 9, 13} |
| Best recovered support | {1, 4, 9, 13} |
| Best least-squares residual | 2.56 × 10^-16 |
| Best wrong-support residual | 0.006853 |

A separate r = 6 run on the same feature set had rank(B_T) = 4, norm(CH) = 1.15 × 10^-15, and minimum false-vocabulary score 0.02496.

The first result demonstrates that an empty original certificate does not imply absence of recoverable dictionary support in a known-initialization endpoint. It does not establish performance with learned embeddings, natural text, realistic optimizers, or finite-precision adapter storage.

## 11. An ordered experimental programme

Treat these as effort estimates for implementation and analysis, not measured runtime. Start with a small inspectable causal transformer and a fixed tokenizer. Run algebra checks in float64 on a tiny instance, then test a pretrained decoder with both absolute-position and RoPE-style inputs as separate cases.

| Stage | Concrete work | Gate for continuing |
|---|---|---|
| Days 1–2: theorem instrumentation | Implement one frozen-input adapter; log H, rank(B_T), singular gaps, CH, contributing masks, and A drift. Sweep r = 4, 8, 16, 32 and active feature counts on both sides of r. | CH reaches the expected numerical floor only in valid, excited cases; rank saturation and Adam negative controls behave as predicted. |
| Days 3–4: finite text family | Recover 1–4 short records with independently chosen private slots. Start with 1,000–10,000 possible complete records and known masks. Use a frozen contextual input or frozen LM head. Include EOS supervision where needed. | Exhaustive candidate testing gives an isolated true candidate with a measured error margin and no excluded true branch. |
| Days 5–6: sparse endpoint route | Use the actual preprocessed first-block vocabulary; reveal exact initialization only in this arm. Sweep contributing support above r. Run exact support search on tiny dictionaries and sparse solvers on larger ones. | Demonstrate a case with q >= r and C empty where known-initialization support recovery succeeds on learned embeddings. Report cancellations and conditioning. |
| Week 2: practical architecture | Compare ordinary LoRA with Activated LoRA at matched suffix lengths and ranks. Add long public contexts and shared secrets across prompts. | The proposed gate changes the measured active-feature rank as predicted and exact private-span recovery improves beyond prior-only baselines. |
| Week 2: search and verification | Add hierarchical candidate generation, causal prefix pruning, and replay of final discrete candidates. | Search savings are measurable at matched coverage; small finite problems retain exhaustive uniqueness checks. |
| Later: realism | Repeat with saved/reloaded fp32, fp16/bfloat16 factors, AdamW, dropout, unknown seed, longer training, and upstream adapters. Change one factor at a time. | State which exact theorem still applies; treat other successes as empirical extensions until proved. |

The first-week milestone is deliberately narrow: one exact text result with a uniqueness margin, or one learned-dictionary example beyond the nullspace rank limit. If neither appears, do not invest first in a large sentence generator.

Use synthetic private values excluded from the public chart-training corpus, plus natural-text records chosen independently of the attack. Sweep record count, active suffix length, distinct token count, and contextual rank separately; these are different variables.

For excitation, begin with the first value projection rather than assuming that every query/key gradient has full token rank. Causal masking and attention symmetries can make some positions structurally uninformative. Trace which inputs actually contribute and retain excitation as a measured condition until it is proved for a stated architecture/loss family.

## 12. Baselines, metrics and figures

Run every optimization comparison on the same candidate family and with the same initialization/recipe knowledge:

- Public LM prior alone, and adapted-model likelihood or generation.
- Certificate alone.
- Full adapter replay alone.
- Certificate-pruned replay.
- Correct LoRA-coordinate linearization with the same chart.
- Known-initialization sparse endpoint recovery.
- Wrong-adapter and shuffled-secret controls.
- DAGER/TIGER-style gradient observations as explicitly stronger-information reference arms, or adaptations restricted to the same observed endpoint when justified.

Do not claim that parameter linearization removes all nonlinear dependence on tokens. A fair NTK comparison must preserve candidate-dependent tangent features and match the information available to each attacker.

Measure exact private-token match, ordered whole-record match, dataset match, token-support precision/recall, true-candidate survival, search coverage, number of feasible aliases, false-candidate margin, computation and memory. ROUGE and semantic similarity are secondary. Calibrate thresholds on separate development cases; do not tune them using the private test target.

The most useful figures are:

1. **Architecture and feature accounting.** Show public context, private span, prediction mask, adapter gate, frozen prefix, and the exact positions entering H.
2. **Rank and separation.** Plot q against r with certificate dimension and empirical true/false score distributions. Include the q >= r failure region.
3. **Ablation with actual ground truth.** Show the private text next to each recovered text, marking every incorrect token. Separate support recovery from ordering.
4. **Chart refinement.** Show candidate count, coverage and minimum competing residual at each refinement level.
5. **Beyond the nullspace limit.** Show a learned-embedding case where C is empty but the known-initialization sparse endpoint identifies support.
6. **Verification.** Plot the best and next-best complete-candidate replay residuals against the justified uncertainty threshold.

## 13. The theorem package to take to Gal

The first coherent package should contain:

1. The frozen-input endpoint invariant, stated with excitation and correct ranks.
2. Finite codebook separation, including a no-extra-codeword condition.
3. A finite-precision margin result and a complete finite search example.
4. The exact sparse endpoint identity and its composition with an established uniqueness theorem.
5. Explicit obstructions: rank saturation, lack of order, unseen chart branches, label-only information, unknown initialization, and the Adam counterexample.

The distinct-target frozen-language-head argument in section 3 already supplies one explicit excitation family. Extend that result toward repeated labels, several prediction positions and first-block value adapters. A possible additional route is analytic genericity: exhibit a configuration with a nonzero appropriate minor, then invoke analyticity on a fixed computation graph. This would establish a generic algebraic property in that parameter family, not a quantitative lower singular-value bound for an arbitrary pretrained model. ReLU, masks, normalization and weight sharing require their own treatment. The conditioning result remains a separate obligation.

My preferred first application is a few short private records represented through frozen contextual features, with exact recovery inside a declared discrete family. Activated LoRA offers a subsequent architecture test. Known-initialization sparse recovery is the alternative if the original certificate saturates. Full recovery of unrestricted documents from ordinary unknown-seed AdamW adapters is a further problem, not a consequence of this plan.

The two choices that most change implementation are whether the private target is a short completion or the preceding prompt/document, and whether the intended released-adapter setting reveals exact initialization. The plan supports both targets, but these choices select the first feature map and which recovery channel is available.
