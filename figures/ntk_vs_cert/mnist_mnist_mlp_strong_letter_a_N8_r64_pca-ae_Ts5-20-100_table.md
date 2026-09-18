RECOVERY (compare only within a chart: each chart defines its own on-chart privates and release)

| chart | scope | k | T | raw proj. err (median) | control NN err (median) | model floor at truth | NTK images (lora/varpro) | NTK residual | verdict | certificate images | certificate landed | top-20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pca | pooled | 16 | 5 | 0.320 | 0.245 | 3.71e-16 | **0/8** | 2.29e-02 | search failure: residual ABOVE the | **8/8** | 170/200 | 20/20 |
| pca | pooled | 16 | 20 | 0.320 | 0.245 | 3.19e-16 | **1/8** | 1.47e-02 | search failure: residual ABOVE the | **8/8** | 175/200 | 20/20 |
| pca | pooled | 16 | 100 | 0.320 | 0.245 | 4.47e-16 | **8/8** | 1.58e-03 | recovered | **8/8** | 174/200 | 20/20 |
| pca | pooled | 32 | 5 | 0.245 | 0.313 | 3.54e-16 | **0/8** | 5.71e-03 | search failure: residual ABOVE the | **7/8** | 82/200 | 20/20 |
| pca | pooled | 32 | 20 | 0.245 | 0.313 | 3.96e-16 | **0/8** | 3.15e-03 | search failure: residual ABOVE the | **8/8** | 69/200 | 20/20 |
| pca | pooled | 32 | 100 | 0.245 | 0.313 | 3.90e-16 | **0/8** | 2.71e-03 | search failure: residual ABOVE the | **8/8** | 73/200 | 20/20 |
| pca | pooled | 48 | 5 | 0.196 | 0.346 | 3.17e-16 | **0/8** | 3.34e-03 | search failure: residual ABOVE the | **1/8** | 1/200 | 1/20 |
| pca | pooled | 48 | 20 | 0.196 | 0.346 | 3.15e-16 | **0/8** | 1.92e-03 | search failure: residual ABOVE the | **2/8** | 4/200 | 4/20 |
| pca | pooled | 48 | 100 | 0.196 | 0.346 | 5.16e-16 | **0/8** | 2.16e-03 | search failure: residual ABOVE the | **2/8** | 4/200 | 4/20 |
| ae | pooled | 16 | 5 | 0.236 | 0.409 | 3.61e-16 | **0/8** | 3.73e-02 | search failure: residual ABOVE the | **2/8** | 5/200 | 5/20 |
| ae | pooled | 16 | 20 | 0.236 | 0.409 | 2.47e-16 | **0/8** | 2.83e-02 | search failure: residual ABOVE the | **2/8** | 9/200 | 9/20 |
| ae | pooled | 16 | 100 | 0.236 | 0.409 | 4.78e-16 | **0/8** | 2.24e-02 | search failure: residual ABOVE the | **3/8** | 9/200 | 9/20 |
| ae | pooled | 32 | 5 | 0.201 | 0.420 | 3.51e-16 | **0/8** | 6.85e-03 | search failure: residual ABOVE the | **1/8** | 3/200 | 3/20 |
| ae | pooled | 32 | 20 | 0.201 | 0.420 | 3.77e-16 | **0/8** | 6.16e-03 | search failure: residual ABOVE the | **1/8** | 3/200 | 3/20 |
| ae | pooled | 32 | 100 | 0.201 | 0.420 | 4.25e-16 | **0/8** | 6.37e-03 | search failure: residual ABOVE the | **2/8** | 3/200 | 3/20 |
| ae | pooled | 48 | 5 | 0.225 | 0.397 | 3.04e-16 | **0/8** | 1.09e-02 | search failure: residual ABOVE the | **1/8** | 1/200 | 1/20 |
| ae | pooled | 48 | 20 | 0.225 | 0.397 | 3.52e-16 | **0/8** | 5.67e-03 | search failure: residual ABOVE the | **0/8** | 0/200 | 0/20 |
| ae | pooled | 48 | 100 | 0.225 | 0.397 | 5.19e-16 | **0/8** | 5.95e-03 | search failure: residual ABOVE the | **1/8** | 1/200 | 1/20 |

RAW-image projection error per chart and k (the true held-out test images BEFORE projection). Privates are on-chart, so each chart
defines its own privates and release: recovery counts above are comparable only WITHIN a chart; THIS table is the only cross-chart comparison.
(ctrl = nearest public image on the same chart, at the same metric.)

| chart | scope | k | median | range | letter_a[323] | letter_a[693] | letter_a[173] | letter_a[92] | letter_a[129] | letter_a[696] | letter_a[298] | letter_a[767] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pca | pooled | 16 | 0.320 | 0.265-0.790 | 0.301 (ctrl 0.200) | 0.328 (ctrl 0.292) | 0.495 (ctrl 0.233) | 0.325 (ctrl 0.243) | 0.280 (ctrl 0.259) | 0.790 (ctrl 0.357) | 0.265 (ctrl 0.248) | 0.316 (ctrl 0.224) |
| pca | pooled | 32 | 0.245 | 0.187-0.691 | 0.233 (ctrl 0.236) | 0.256 (ctrl 0.357) | 0.407 (ctrl 0.289) | 0.235 (ctrl 0.309) | 0.228 (ctrl 0.342) | 0.691 (ctrl 0.515) | 0.187 (ctrl 0.318) | 0.263 (ctrl 0.293) |
| pca | pooled | 48 | 0.196 | 0.150-0.631 | 0.192 (ctrl 0.258) | 0.221 (ctrl 0.378) | 0.328 (ctrl 0.349) | 0.199 (ctrl 0.332) | 0.157 (ctrl 0.366) | 0.631 (ctrl 0.648) | 0.150 (ctrl 0.343) | 0.185 (ctrl 0.332) |
| ae | pooled | 16 | 0.236 | 0.195-0.565 | 0.207 (ctrl 0.331) | 0.255 (ctrl 0.418) | 0.442 (ctrl 0.423) | 0.195 (ctrl 0.444) | 0.237 (ctrl 0.365) | 0.565 (ctrl 0.793) | 0.214 (ctrl 0.400) | 0.235 (ctrl 0.360) |
| ae | pooled | 32 | 0.201 | 0.167-0.623 | 0.174 (ctrl 0.290) | 0.283 (ctrl 0.429) | 0.436 (ctrl 0.462) | 0.199 (ctrl 0.428) | 0.203 (ctrl 0.357) | 0.623 (ctrl 0.825) | 0.192 (ctrl 0.413) | 0.167 (ctrl 0.373) |
| ae | pooled | 48 | 0.225 | 0.178-0.544 | 0.178 (ctrl 0.316) | 0.262 (ctrl 0.414) | 0.337 (ctrl 0.460) | 0.233 (ctrl 0.426) | 0.187 (ctrl 0.372) | 0.544 (ctrl 0.740) | 0.216 (ctrl 0.375) | 0.205 (ctrl 0.379) |
