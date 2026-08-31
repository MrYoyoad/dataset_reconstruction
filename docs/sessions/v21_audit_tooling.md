# Code-correctness audit — pptx round-trip tooling (`import_pptx.py` / `build_from_spec.py`)

Audited 2026-08-31. Read-only on decks/spec. Tools were **not** edited.

- ORIGINAL (hand-finished, 37 slides): `supervisor_meeting_2026_08_31_v20.pptx` (repo root; this is the file the spec's `source` field points at, and what `import_pptx.py` consumed to make `figures/deck_v20_spec/`).
- REBUILT (shipped): `figures/supervisor_meeting_2026_08_31_v21.pptx`.
- Coarse check (37 slides / 863 shapes / 54 pics / 5695 words / 82926 notes-chars, 0 text/notes mismatch) passes — problems are all in **attributes**, not counts.

Method: parallel walk of v20 vs v21 (same slide index, same shape index, recursive through groups) diffing every geometry / z-order / fill / line / shape-type / adjustment / run-font / paragraph / text-frame / table / crop attribute. Repro scripts in `/tmp/.../scratchpad/audit_diff.py`, `audit_detail.py`.

---

## SEVERITY 1 — CORRUPTION (broken / data-lost content)

### C1. `--fix-page-numbers` overwrote **real data values** in the shipped v21 (ACTUAL data loss, present now)
- **Behaviour.** The predicate is `re.fullmatch(r"\s*\d+\s*/\s*\d+\s*", flat)` where `flat` is the concatenation of *all* runs across *all* paragraphs of any `textbox`/`shape`. Any box whose entire text is `digits / digits` is rewritten to `"{slide_idx} / {total}"`.
- **Hit count.** 38 boxes rewritten on a 37-slide deck. The "38 on 37" is **slide 35** (36th slide) carrying **three** matches:
  - shape 20: `392821 / 390026`  → clobbered to `36 / 37`
  - shape 80: `229722 / 237301`  → clobbered to `36 / 37`
  - shape 93: `34 / 35`          → correctly renumbered to `36 / 37`
- **Repro (verified against the shipped file, not hypothetical):**
  ```
  v20-orig    shape20='392821 / 390026'  shape80='229722 / 237301'  shape93='34 / 35'
  v21-tracked shape20='36 / 37'          shape80='36 / 37'          shape93='36 / 37'
  ```
  The first two are content numbers (parameter-count ratios) that are **gone** from the deck that was shipped to the supervisor.
- **Also clobberable (latent):** any body string that is exactly `N / M` — aspect ratios `16 / 9`, fractions `3 / 4`, `1 / 2`. Predicate matches on full text only, so `"loss 3 / 4 of runs"` is safe but a standalone `16 / 9` box is not.
- **Refuse vs drop:** the fixer must NOT rewrite by shape geometry-agnostic text match. It should restrict to the actual page-number box (e.g. by position band / a marker / the single lowest box on the slide), or at minimum skip boxes whose numbers don't look like `n / total` (denominator == deck length, numerator ≤ total). As written it silently destroys data — should be considered a bug, not a feature, until scoped.

### C2. Duplicate shape-id collision from raw-XML re-insertion (latent corruption; **not** triggered in this deck, but reproduced)
- **Behaviour.** `unsupported` shapes (here: 53, all `LINE (9)` connectors) are re-inserted with `slide.shapes._spTree.append(etree.fromstring(sh["xml"]))` — the original `<p:cNvPr id=...>` is kept verbatim. Native shapes get ids via python-pptx `max(existing)+1`. If a re-inserted shape's original id is **lower** than the count of native shapes preceding it in z-order, you get two shapes with the same `cNvPr/@id` on one slide → PowerPoint "needs to repair", may drop a shape.
- **This deck:** 0 duplicate ids (v20 was generator-built with id == z-order, so ids never clash). **Fragile coincidence, not robustness.**
- **Repro (minimal spec, 2 native textboxes + 1 unsupported connector whose stored xml has `id="2"`, connector last in z-order):**
  ```
  cNvPr ids after build: ['1','2','3','2']  -> duplicate id '2'
  ```
  Hand-edited PowerPoint decks routinely have id ≠ z-order (reordering/duplication reassigns), so this WILL fire on a real hand-edited future deck.
- **Refuse vs drop:** builder should re-id every re-inserted element to a fresh unique id before append (and rewrite internal references), or refuse. Currently silently produces an invalid file.

### C3. Re-inserted raw XML carrying a relationship id → dangling reference (latent corruption; future decks)
- **Behaviour.** `_add` for `unsupported` appends the raw XML but **never copies the slide's relationships** (`grep`: no `rels`/`r:embed` handling anywhere in the builder). Any unsupported shape whose XML contains `r:embed` / `r:id` / `r:link` — a chart (`<p:graphicFrame>`), an embedded video/audio (`MSO_MEDIA`), an OLE object, or a linked picture — will re-insert a `r:id` that does not exist in the rebuilt slide's `.rels` → PowerPoint fails to load that part / repair dialog.
- **This deck:** not triggered — all 53 unsupported shapes are `<a:ln>` connectors with an **explicit `<a:srgbClr val="1F4E79">`** and no relationships, so they survive verbatim with correct geometry (0 geometry diffs measured). Charts/media/OLE: none present (`chart parts: 0, video/audio media: 0`).
- **Refuse vs drop:** builder must copy referenced parts+rels, or refuse when an unsupported shape's XML references any `r:` relationship. Currently would emit a broken file silently.

---

## SEVERITY 2 — SILENT FIDELITY LOSS (opens fine, looks wrong)

### F1. Every non-rectangle autoshape flattened to a plain RECTANGLE (88 shapes) — **worst visual bug**
- **Root cause.** Importer records the *generic* type: `autoshape=str(sh.shape_type)` → `"AUTO_SHAPE (1)"` for every autoshape (it never reads `sh.auto_shape_type`). Builder then does `_enum(MSO_SHAPE, "AUTO_SHAPE (1)")`; the regex `\((\d+)\)` matches `(1)` → `MSO_SHAPE(1)` = `RECTANGLE`. So **all** autoshapes rebuild as rectangles.
- **Measured:**
  ```
  ORIG autoshape types: RECTANGLE 157, ROUNDED_RECTANGLE 44, OVAL 44
  REB  autoshape types: RECTANGLE 245
  ```
  44 rounded-rectangles and **44 ovals** become sharp rectangles. Ovals→rectangles is a blatant silhouette change (diagram nodes / bullets).
- Surfaced indirectly as the `adjustments` diff (44: `[0.16667]`/`[0.08]` → `[]`) because the rebuilt rectangle has no adjustment handle; the generic `shape_type` (== `AUTO_SHAPE`) matches, so the coarse check missed it entirely.
- **Refuse vs drop:** importer should store `auto_shape_type`; this is a plain bug (drops shape identity silently).

### F2. `text_frame.auto_size` forced to SHAPE_TO_FIT_TEXT on 511 text boxes
- **Root cause.** Importer never captures `auto_size`. `slide.shapes.add_textbox(...)` in python-pptx injects `<a:spAutoFit/>` by default. Every rebuilt textbox therefore gets `SHAPE_TO_FIT_TEXT`.
- **Measured:** 511 text frames, `None → SHAPE_TO_FIT_TEXT (1)` (all `textbox`-kind shapes). Original boxes were fixed-size (no autofit); rebuilt boxes will resize to their text in PowerPoint, shifting/clipping layout.
- **Refuse vs drop:** capture + reapply `auto_size` (and its `<a:normAutofit fontScale>` if present). Silent drop.

### F3. Autoshape paragraphs forced to CENTER alignment (245 paragraphs)
- **Root cause.** `add_shape`'s default paragraph carries `algn="ctr"`. When the spec's alignment is `None` (original = inherit/left), `_apply_text` never overrides it, so it stays centered.
- **Measured:** 245 paragraphs, `None → CENTER (2)`. Every autoshape text paragraph whose original alignment was unset is now centered (visible wherever such an autoshape holds left-aligned text).
- **Refuse vs drop:** builder should explicitly set alignment (incl. clearing to left) rather than inherit the add_shape default. Silent.

### F4. Autoshape text vertical-anchor forced to MIDDLE (23 text frames)
- **Root cause.** `add_shape`'s default `bodyPr anchor="ctr"`; importer stored `None`, builder doesn't set it, default MIDDLE remains.
- **Measured:** 23, `None → MIDDLE (3)`.
- **Refuse vs drop:** same as F3 — set explicitly. Silent.

### Preserved correctly (measured 0 diffs — reported so, not padded):
geometry (left/top/width/height/rotation), z-order, generic shape_type, **fill colour**, **line colour/width**, all **run font** (name/size/bold/italic/underline/**colour**), paragraph **level / line_spacing / space_before / space_after**, text-frame **word_wrap** and **margins**, **table** col-widths/row-heights/per-cell fill, **picture crops**, backgrounds, notes, and the 53 **connectors** (verbatim XML, correct geometry, explicit RGB).

---

## SEVERITY 3 — FUTURE-DECK FAILURE MODES (refuse-vs-drop verdicts)

Determined from code; none present in v20 (so no current impact), all latent for a future hand-edited deck.

| Feature | Fate in round-trip | Should REFUSE (raise) vs silently drop? |
|---|---|---|
| **Hyperlinks** (run `hlinkClick`, shape click-action) | **Silently dropped** — importer `_text` never reads `run.hyperlink`; builder never writes it. (0 in this deck.) | Drop is acceptable but should at least warn; a deck built for navigation loses it silently. |
| **Grouped shapes** (`GroupShape`) | **Flattened + likely mis-placed.** Builder `_add` for `kind=="group"` recurses `_add(slide, child)` — children are added straight onto the slide, no group is created. Child `left/top/width/height` are in the group's **child coordinate space** (`chOff/chExt`), so unless the group transform is identity they land at the wrong slide position/scale. (0 groups here.) | Should **refuse** or build a real group — silent flatten corrupts layout. |
| **Charts** (`graphicFrame`) | Falls to `unsupported` (no text frame) → raw XML re-inserted with dangling chart `r:id` (see C3) → **broken part**. | Should **refuse**. |
| **Embedded video/audio** (`MSO_MEDIA`) | `unsupported` raw XML with dangling media `r:id`/`r:link` (C3) → **broken**. Or, if it presents as a picture, only the poster frame survives, media dropped. | Should **refuse**. |
| **Theme-colour fill/font** (`schemeClr`) | **Silently dropped to default.** `_color` records `{"theme":..., "brightness":...}` but `_set_color` only handles `"rgb"` and returns `False`; for a fill, `fill.solid()` is still called → a default/near-black solid; for a font, colour is left to inherit. (v20 has 1100 `schemeClr` refs, but all are in re-inserted connector `<p:style>` blocks overridden by explicit `srgbClr`, or in add_shape default style blocks overridden by explicit spPr — so **no measured colour diff here**; purely future risk.) | Should map theme colours to the master's palette, or **refuse** — dropping to black is a silent miscolour. |
| **Slide layouts / masters / placeholders** | **Discarded** — builder puts every slide on blank `slide_layouts[6]`. Placeholder-inherited fonts/bullets/backgrounds from the master are lost. This deck is all explicit textboxes, so no visible loss. | Drop acceptable for fully-explicit decks; risky for placeholder-driven ones. |
| **Animations / transitions** | **Silently dropped** — importer reads only shapes + notes + background. | Drop acceptable (static export). |

---

## Bottom line
- **Shipped-file data loss:** v21 already lost two real numbers on slide 35 to `--fix-page-numbers` (C1).
- **Round-trip is lossy even on this "matching" deck:** 88 autoshapes changed silhouette (F1), 511 boxes gained autofit (F2), 245 paragraphs re-centered (F3), 23 re-anchored (F4) — none caught by count/text/notes checks.
- **Structurally fragile:** raw-XML re-insertion keeps original ids (duplicate-id corruption on the next hand-edited deck, C2) and copies no relationships (chart/media/OLE decks will emit broken files, C3).
