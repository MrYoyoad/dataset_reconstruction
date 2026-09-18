RECOVERY (compare only within a chart: each chart defines its own on-chart privates and release)

| chart | scope | k | T | raw proj. err (median) | control NN err (median) | model floor at truth | NTK images (lora/varpro) | NTK residual | verdict | certificate images | certificate landed | top-20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pca | pooled | 32 | 1 | 0.294 | 0.281 | 2.90e-16 | **0/8** | 4.59e-02 | search failure: residual ABOVE the | **4/8** | 7/20 | 7/20 |
| pca | pooled | 32 | 400 | 0.294 | 0.281 | 8.33e-16 | **0/8** | 2.38e-02 | search failure: residual ABOVE the | **5/8** | 8/20 | 8/20 |
| ae | pooled | 32 | 1 | 0.186 | 0.402 | 3.92e-16 | **0/8** | 8.19e-02 | search failure: residual ABOVE the | **0/8** | 0/20 | 0/20 |
| ae | pooled | 32 | 400 | 0.186 | 0.402 | 8.87e-16 | **0/8** | 3.09e-02 | search failure: residual ABOVE the | **0/8** | 0/20 | 0/20 |
| pca_perclass | per_class | 32 | 1 | 0.256 | 0.299 | 2.92e-16 | **0/8** | 3.71e-02 | search failure: residual ABOVE the | **5/8** | 11/20 | 11/20 |
| pca_perclass | per_class | 32 | 400 | 0.256 | 0.299 | 9.11e-16 | **0/8** | 1.63e-02 | search failure: residual ABOVE the | **6/8** | 10/20 | 10/20 |

RAW-image projection error per chart and k (the true held-out test images BEFORE projection). Privates are on-chart, so each chart
defines its own privates and release: recovery counts above are comparable only WITHIN a chart; THIS table is the only cross-chart comparison.
(ctrl = nearest public image on the same chart, at the same metric.)

| chart | scope | k | median | range | letter_a[323] | letter_a[693] | letter_a[173] | letter_a[92] | letter_t[305] | letter_t[484] | letter_t[3] | letter_t[27] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pca | pooled | 32 | 0.329 | 0.260-0.486 | 0.273 (ctrl 0.237) | 0.260 (ctrl 0.356) | 0.435 (ctrl 0.308) | 0.294 (ctrl 0.289) | 0.291 (ctrl 0.218) | 0.486 (ctrl 0.295) | 0.386 (ctrl 0.274) | 0.365 (ctrl 0.219) |
| ae | pooled | 32 | 0.209 | 0.160-0.374 | 0.169 (ctrl 0.292) | 0.233 (ctrl 0.437) | 0.374 (ctrl 0.490) | 0.186 (ctrl 0.367) | 0.160 (ctrl 0.315) | 0.364 (ctrl 0.505) | 0.275 (ctrl 0.421) | 0.178 (ctrl 0.382) |
| pca_perclass | per_class | 32 | 0.261 | 0.225-0.433 | 0.233 (ctrl 0.236) | 0.256 (ctrl 0.357) | 0.407 (ctrl 0.289) | 0.235 (ctrl 0.309) | 0.225 (ctrl 0.247) | 0.433 (ctrl 0.358) | 0.324 (ctrl 0.328) | 0.266 (ctrl 0.284) |
