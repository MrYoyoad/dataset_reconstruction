# Thesis meeting summary 15 September 2026

Based on Yoad's account of the meeting with Gal Vardi and research colleagues. The reported discussion and agreements are recorded first; interpretation and suggested experiments follow separately.

**Main outcome.** The meeting went very well. The participants were impressed by the figures, the certificate equation and the chart idea. They supported pursuing the combination as a practical reconstruction method, with agreed work on multiple layers, different charts, improved reconstruction and an initial text attempt. Yoad additionally reports that they explicitly described successful extensions to text or new types of images as potentially groundbreaking and capable of leading to a very good paper. This was their assessment of the potential if those extensions succeed.

**Reaction to the presentation.** The first two figures, and the first three or four more broadly, already introduced substantial material. Much of the discussion focused on understanding those ideas. The figures prompted detailed questions about the mechanism and its derivation. The participants knew LoRA, but this particular reconstruction approach appeared unfamiliar to them.

**The certificate and existing reconstruction methods.** They asked why the certificate equation CH = 0 holds, what its geometric intuition is and how Yoad arrived at it. They regarded it as a good equation. They initially asked about the difficulties encountered with existing LoRA reconstruction and NTK-based approaches. Yoad subsequently clarified that, by the end of the discussion, they were convinced that the NTK reconstruction approach under discussion would not work for the LoRA setting being considered. This was a conclusion they accepted during the meeting, rather than an unresolved objection. The record scopes that conclusion to the formulation and setting discussed; it does not assert an impossibility theorem for every possible NTK-based method.

They suggested looking for additional equations involving perturbed inputs. Gal's main extension question was whether the certificate remains useful when a layer's inputs change during training, as they can when earlier layers are adapted. The discussion included measuring feature or subspace changes using the frozen base model and final adapter.

**The chart idea.** Low-dimensional PCA was the participants' intuitive frame for understanding the search restriction. Yoad explained that this was already implemented. They also understood and liked the prospect of nonlinear charts and increasing the search dimension. They were interested in combining chart constraints with different reconstruction equations.

They explicitly distinguished this approach from their earlier deep image prior attempt, which had not worked for them. Their response supports recording the chart approach as conceptually different from that particular attempt; the meeting itself does not establish literature-wide novelty.

**Charts beyond the certificate.** They discussed whether PCA, potentially adapted iteratively during the search, could also reduce the search dimension for KKT reconstruction and other objectives. They raised memory limitations in KKT reconstruction with larger or more demanding images. This creates a separate experimental question: how much does changing the search parameterization help reconstruction quality, optimization and actual memory use?

**Properties they found appealing.** The method supports searching for one image at a time, without requiring a joint reconstruction of the entire dataset. The discussion also emphasized a potential connection between reconstructability and how strongly an example contributes during training, described informally as how wrong the model gets that example. This offers a different perspective from focusing only on margin examples in a KKT formulation. They found the demonstrated reconstruction encouraging despite the limited number of adapter parameters.

These observations motivate experiments. In particular, a monotone relationship between prediction error and reconstruction success was not established in the meeting.

**Possible scope.** The participants viewed the premise as different from the papers they were comparing it with. They saw several promising directions: better and more complex images, additional training settings, public adapters under suitable training conditions such as SGD, and text. Public-adapter reconstruction was an ambition discussed, rather than a result already demonstrated or a commitment to solve every setting.

**Agreed next steps.**

1. Test the multilayer extension and determine how changing layer inputs affects the certificate and reconstruction.
2. Try different charts, including extensions beyond a single fixed PCA parameterization.
3. Improve reconstruction in concrete experimental settings and explore useful combinations of equations and charts.
4. Make an initial attempt in text.

The emphasis was practical: develop the method and establish where it works. Theory remains useful for explaining the equations and identifying the conditions the experiments should test.

**Interpretation of the meeting.** The participants explicitly recognised the possibility of a major research contribution, conditional on successful extensions. Their detailed engagement supports taking that assessment seriously: they asked about the derivation, connected the chart to their own intuitions, distinguished it from a prior approach and proposed extensions. Relative to the earlier concern about finding a concrete solution path, this meeting supplied a shared experimental direction and an explicit assessment of its potential significance. A convincing result in a new setting is therefore a particularly valuable next milestone.

**Technical precision for the next experiments.** For a candidate image, the frozen base model and released adapter allow comparison of features at the start and endpoint, and evaluation of certificate residuals. They do not by themselves reveal the unknown intermediate feature trajectory. Controlled experiments can record that trajectory and test how well endpoint diagnostics predict certificate error and reconstruction success.

The connection to prediction error also needs a careful interpretation. Training residuals and gradient contributions are relevant quantities to investigate, but a large loss or an incorrect prediction alone is not a guarantee that an image is recoverable.

**Suggested experimental order**

The following operationalizes the agreement; this order was not reported as a meeting decision.

| Experiment | Question it should answer |
| --- | --- |
| Preserve a reproducible single-layer result | Can the certificate and chart result be shown clearly with ground truth, reconstructions and certificate residuals? |
| Adapt an earlier layer while keeping the dataset and chart fixed | How do feature drift, certificate residuals and reconstruction success change when the target layer's inputs evolve? |
| Compare chart choices under the same reconstruction information and objective | Which gains come from the chart, and which charts support more detailed images without making the search unreliable? |
| Use the same chart for certificate and KKT or NTK comparisons | How much does the parameterization help each objective, and what happens to measured runtime and peak memory? |
| Run one controlled text pilot | Is there a simple setting in which the proposed equations and a restricted text search provide useful reconstruction information? |

Public adapters are a natural subsequent target once a controlled setting identifies the required information and training assumptions.
