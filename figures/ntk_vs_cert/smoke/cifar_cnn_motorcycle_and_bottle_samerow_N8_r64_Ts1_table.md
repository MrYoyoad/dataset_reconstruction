RECOVERY (compare only within a chart: each chart defines its own on-chart privates and release)

| chart | scope | k | T | raw proj. err (median) | control NN err (median) | model floor at truth | NTK images (lora/varpro) | NTK residual | verdict | certificate images | certificate landed | top-20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pca | pooled | 16 | 1 | 0.338 | 0.176 | 2.11e-16 | **0/8** | 7.57e-02 | search failure: residual ABOVE the | **2/8** | 2/4 | 2/20 |

RAW-image projection error per chart and k (the true held-out test images BEFORE projection). Privates are on-chart, so each chart
defines its own privates and release: recovery counts above are comparable only WITHIN a chart; THIS table is the only cross-chart comparison.
(ctrl = nearest public image on the same chart, at the same metric.)

| chart | scope | k | median | range | motorcycle[23] | motorcycle[32] | motorcycle[89] | motorcycle[56] | bottle[82] | bottle[74] | bottle[92] | bottle[44] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pca | pooled | 16 | 0.338 | 0.147-0.436 | 0.358 (ctrl 0.170) | 0.295 (ctrl 0.181) | 0.325 (ctrl 0.356) | 0.436 (ctrl 0.414) | 0.147 (ctrl 0.049) | 0.350 (ctrl 0.409) | 0.168 (ctrl 0.058) | 0.370 (ctrl 0.076) |
