| cell | landed / starts | images found | residual at truths | best start | median start | top-20 by residual landed | SSIM vs raw (attack) | SSIM vs raw (chart floor) | SSIM vs raw (control) |
|---|---|---|---|---|---|---|---|---|---|
| LoRA on the pixel layer, conv-AE decoder k=32, on-chart privates | 30/400 | 2/8 | 2.9e-14 | 2.8e-14 | 5.5e-03 | 20/20 | 0.38 | 0.59 | 0.28 |
| LoRA on the hidden layer, conv-AE decoder k=32, on-chart privates | 87/400 | 5/8 | 3.0e-14 | 3.0e-14 | 7.2e-03 | 20/20 | 0.47 | 0.59 | 0.25 |
| LoRA on the hidden layer, conv-AE decoder k=32, raw privates | 0/400 | 0/8 | 2.5e-14 | 6.3e-03 | 9.6e-03 | 0/20 | 0.42 | 0.59 | 0.20 |
| LoRA on the hidden layer, public PCA k=32, on-chart privates | 253/400 | 8/8 | 2.9e-14 | 2.7e-14 | 2.8e-14 | 20/20 | 0.58 | 0.58 | 0.21 |
| LoRA on the hidden layer, public PCA k=48, on-chart privates | 84/400 | 7/8 | 2.8e-14 | 2.7e-14 | 4.9e-03 | 20/20 | 0.58 | 0.61 | 0.20 |
| LoRA on the hidden layer, public PCA k=32, raw privates | 0/400 | 0/8 | 2.5e-14 | 7.3e-03 | 1.6e-02 | 0/20 | 0.38 | 0.58 | 0.20 |
| LoRA on the head (features), conv-AE decoder k=32, on-chart privates | 20/400 | 4/8 | 1.4e-14 | 9.1e-15 | 2.3e-02 | 20/20 | 0.48 | 0.59 | 0.45 |
| LoRA on the head (features), oracle chart (eps=0.0) k=32, raw privates | 191/400 | 7/8 | 3.6e-14 | 6.7e-16 | 9.4e-03 | 20/20 | 0.90 | 1.00 | 0.50 |
| LoRA on the head (features), oracle chart (eps=0.02) k=32, raw privates | 169/400 | 4/8 | 3.6e-14 | 7.0e-05 | 2.4e-04 | 20/20 | 0.89 | 0.99 | 0.51 |
| LoRA on the head (features), oracle chart (eps=0.05) k=32, raw privates | 0/400 | 0/8 | 3.6e-14 | 1.8e-04 | 5.9e-04 | 0/20 | 0.85 | 0.94 | 0.48 |
| LoRA on the head (features), oracle chart (eps=0.1) k=32, raw privates | 0/400 | 0/8 | 3.6e-14 | 3.7e-04 | 1.1e-03 | 0/20 | 0.76 | 0.84 | 0.47 |
| LoRA on the head (features), oracle chart (eps=0.2) k=32, raw privates | 0/400 | 0/8 | 3.6e-14 | 7.7e-04 | 2.2e-03 | 0/20 | 0.62 | 0.67 | 0.44 |
| LoRA on the head (features), public PCA k=32, on-chart privates, WRONG-RELEASE CONTROL | 0/400 | 0/8 | 5.1e-01 | 4.7e-15 | 1.8e-02 | 0/20 | 0.33 | 0.58 | 0.44 |
| LoRA on the head (features), public PCA k=32, on-chart privates | 171/400 | 8/8 | 1.3e-14 | 7.1e-15 | 1.3e-02 | 20/20 | 0.58 | 0.58 | 0.60 |
| LoRA on the head (features), public PCA k=48, on-chart privates | 50/400 | 6/8 | 1.8e-14 | 2.1e-15 | 1.1e-02 | 20/20 | 0.55 | 0.61 | 0.45 |
| LoRA on the head (features), public PCA k=32, raw privates | 0/400 | 0/8 | 4.0e-14 | 1.3e-02 | 3.5e-02 | 0/20 | 0.34 | 0.58 | 0.41 |
| LoRA on the head (features), public PCA k=32, raw privates | 0/400 | 0/8 | 3.6e-14 | 1.1e-02 | 3.5e-02 | 0/20 | 0.35 | 0.58 | 0.51 |
