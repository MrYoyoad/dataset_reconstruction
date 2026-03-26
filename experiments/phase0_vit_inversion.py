"""Phase 0: ViT Gradient Inversion Gate Experiment.

Tests whether gradient inversion can reconstruct images from a single
ViT fine-tuning step. This is the gate experiment that determines
feasibility of ALL thesis directions on real (non-MNIST) data.

Setup:
    1. Load pre-trained ViT-B/16 from timm
    2. Add LoRA (rank r) via peft
    3. Fine-tune on 1 image for 1 SGD step (binary classification)
    4. Record exact full-rank gradient during backward pass
    5. Run gradient inversion to reconstruct the image from the gradient
    6. (Phase 0b) Add noise to gradient, measure reconstruction vs noise

Usage:
    python -u -m experiments.phase0_vit_inversion --device cuda
    python -u -m experiments.phase0_vit_inversion --device cuda --noise_sweep
"""

import os
import sys
import argparse
import csv
import time
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime

# Lazy imports for optional deps (timm, peft, torchvision)
def _lazy_imports():
    import timm
    import torchvision
    import torchvision.transforms as T
    from peft import get_peft_model, LoraConfig, TaskType
    return timm, torchvision, T, get_peft_model, LoraConfig, TaskType


def load_vit_with_lora(rank=8, num_classes=2, device='cuda'):
    """Load ViT-B/16 with LoRA adapter."""
    timm, _, _, get_peft_model, LoraConfig, _ = _lazy_imports()

    # Load pre-trained ViT-B/16
    model = timm.create_model('vit_base_patch16_224', pretrained=True,
                               num_classes=num_classes)
    model = model.to(device).to(torch.float32)

    # Apply LoRA to attention layers
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,  # scaling = alpha/r = 1
        target_modules=['qkv'],  # ViT uses fused qkv in timm
        lora_dropout=0.0,
        bias='none',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def get_sample_images(n_images=2, seed=42, device='cuda'):
    """Get sample images for fine-tuning.

    Uses CIFAR-10 as a simple source (224x224 resize).
    Returns (images, labels) where images are [N, 3, 224, 224].
    """
    _, torchvision, T, _, _, _ = _lazy_imports()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # Pick n_images with deterministic seed
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=rng)[:n_images]

    images = []
    labels = []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label % 2)  # binary: even vs odd class

    images = torch.stack(images).to(device)
    labels = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)
    return images, labels


def capture_gradient(model, images, labels):
    """Do one forward+backward pass and capture the full gradient."""
    model.train()
    model.zero_grad()

    logits = model(images)
    # Binary classification with single output
    if logits.shape[1] == 2:
        loss = F.cross_entropy(logits, labels.squeeze(1).long())
    else:
        loss = F.binary_cross_entropy_with_logits(logits, labels)

    loss.backward()

    # Collect gradients from ALL parameters (not just LoRA)
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().clone()

    return gradients, loss.item()


def invert_gradient(model, gradients, labels, image_shape,
                    n_iters=3000, lr=0.1, tv_weight=1e-4,
                    device='cuda', verbose=True):
    """Reconstruct images from gradients via optimization.

    Implements Geiping et al. (2020) style cosine-similarity inversion:
        max_x cos(nabla L(x), nabla_true)

    With Total Variation regularization for image smoothness.
    """
    n_images = labels.shape[0]

    # Initialize random image
    x_recon = torch.randn(n_images, *image_shape[1:],
                           device=device, dtype=torch.float32,
                           requires_grad=True)

    optimizer = torch.optim.Adam([x_recon], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters)

    best_loss = float('inf')
    best_x = x_recon.detach().clone()

    for i in range(n_iters):
        optimizer.zero_grad()
        model.zero_grad()

        logits = model(x_recon)
        if logits.shape[1] == 2:
            loss = F.cross_entropy(logits, labels.squeeze(1).long())
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward(retain_graph=True)

        # Compute cosine similarity between current and target gradients
        cos_sim = 0.0
        n_params = 0
        for name, param in model.named_parameters():
            if name in gradients and param.grad is not None:
                g_true = gradients[name]
                g_pred = param.grad
                # Flatten and compute cosine similarity
                cos = F.cosine_similarity(
                    g_pred.reshape(1, -1), g_true.reshape(1, -1)
                )
                cos_sim += cos
                n_params += 1

        if n_params > 0:
            cos_sim = cos_sim / n_params

        # Inversion loss: minimize negative cosine similarity
        inv_loss = -cos_sim

        # Total Variation regularization
        tv_loss = 0.0
        if tv_weight > 0:
            diff_h = (x_recon[:, :, 1:, :] - x_recon[:, :, :-1, :]).pow(2).sum()
            diff_w = (x_recon[:, :, :, 1:] - x_recon[:, :, :, :-1]).pow(2).sum()
            tv_loss = tv_weight * (diff_h + diff_w)

        total_loss = inv_loss + tv_loss

        # Backward on the inversion loss (not the model loss)
        model.zero_grad()
        x_recon.grad = None
        total_loss.backward()

        optimizer.step()
        scheduler.step()

        # Clamp to valid image range (approximately)
        with torch.no_grad():
            # ImageNet normalization bounds roughly [-2.1, 2.6]
            x_recon.clamp_(-3.0, 3.0)

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_x = x_recon.detach().clone()

        if verbose and (i % 500 == 0 or i == n_iters - 1):
            print(f"  iter {i:5d}: cos_sim={cos_sim.item():.4f}, "
                  f"tv={tv_loss if isinstance(tv_loss, float) else tv_loss.item():.6f}, "
                  f"total={total_loss.item():.4f}")

    return best_x


def add_noise_to_gradient(gradients, target_cosine_sim, seed=42):
    """Add Gaussian noise to gradients to achieve target cosine similarity.

    Returns noisy gradients dict.
    """
    torch.manual_seed(seed)
    noisy_grads = {}

    # Flatten all gradients into one vector for global noise scaling
    flat_true = torch.cat([g.reshape(-1) for g in gradients.values()])
    norm_true = flat_true.norm()

    # cos(g, g+noise) = cos(g, g+noise)
    # For Gaussian noise with std sigma:
    #   E[cos] ≈ 1 / sqrt(1 + (sigma * sqrt(d) / ||g||)^2)
    # Solve for sigma given target cos:
    #   sigma = ||g|| * sqrt(1/cos^2 - 1) / sqrt(d)
    d = flat_true.numel()
    if target_cosine_sim >= 0.999:
        sigma = 0.0
    else:
        sigma = (norm_true * (1.0 / target_cosine_sim**2 - 1.0).sqrt()
                 / (d ** 0.5)).item()

    # Add noise to each gradient tensor
    offset = 0
    for name, g in gradients.items():
        noise = torch.randn_like(g) * sigma
        noisy_grads[name] = g + noise
        offset += g.numel()

    # Verify achieved cosine similarity
    flat_noisy = torch.cat([g.reshape(-1) for g in noisy_grads.values()])
    actual_cos = F.cosine_similarity(
        flat_true.unsqueeze(0), flat_noisy.unsqueeze(0)
    ).item()

    return noisy_grads, actual_cos


def compute_metrics(x_true, x_recon, denorm_mean, denorm_std):
    """Compute reconstruction quality metrics."""
    # Denormalize both
    mean = torch.tensor(denorm_mean, device=x_true.device).reshape(1, 3, 1, 1)
    std = torch.tensor(denorm_std, device=x_true.device).reshape(1, 3, 1, 1)

    x_true_pixel = (x_true * std + mean).clamp(0, 1)
    x_recon_pixel = (x_recon * std + mean).clamp(0, 1)

    # MSE
    mse = F.mse_loss(x_recon_pixel, x_true_pixel).item()

    # PSNR
    psnr = -10.0 * np.log10(mse + 1e-10)

    # Simple SSIM (per-channel mean)
    # Using a basic implementation compatible with any PyTorch version
    ssim = _simple_ssim(x_true_pixel, x_recon_pixel)

    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
    }


def _simple_ssim(img1, img2, window_size=11):
    """Compute SSIM between two image batches. Simple implementation."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Average over spatial dims
    mu1 = img1.mean(dim=(-2, -1))
    mu2 = img2.mean(dim=(-2, -1))

    sigma1_sq = ((img1 - mu1.unsqueeze(-1).unsqueeze(-1)) ** 2).mean(dim=(-2, -1))
    sigma2_sq = ((img2 - mu2.unsqueeze(-1).unsqueeze(-1)) ** 2).mean(dim=(-2, -1))
    sigma12 = ((img1 - mu1.unsqueeze(-1).unsqueeze(-1)) *
               (img2 - mu2.unsqueeze(-1).unsqueeze(-1))).mean(dim=(-2, -1))

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def save_comparison_image(x_true, x_recon, metrics, save_path,
                          denorm_mean, denorm_std, title=None):
    """Save side-by-side comparison of ground truth and reconstruction."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mean = torch.tensor(denorm_mean).reshape(1, 3, 1, 1)
    std = torch.tensor(denorm_std).reshape(1, 3, 1, 1)

    x_true_np = (x_true.cpu() * std + mean).clamp(0, 1).numpy()
    x_recon_np = (x_recon.cpu() * std + mean).clamp(0, 1).numpy()

    n = x_true.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        # Ground truth
        axes[0, i].imshow(x_true_np[i].transpose(1, 2, 0))
        axes[0, i].set_title('Ground Truth')
        axes[0, i].axis('off')

        # Reconstruction
        axes[1, i].imshow(x_recon_np[i].transpose(1, 2, 0))
        ssim = metrics.get('ssim', 0)
        psnr = metrics.get('psnr', 0)
        axes[1, i].set_title(f'Recon (SSIM={ssim:.3f}, PSNR={psnr:.1f})')
        axes[1, i].axis('off')

    fig.suptitle(title or 'Phase 0: ViT Gradient Inversion', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def run_phase0(rank=8, n_images=1, seed=42, n_iters=3000,
               device='cuda', verbose=True):
    """Run Phase 0: exact gradient inversion on ViT."""
    print("=" * 70)
    print(f"PHASE 0: ViT Gradient Inversion (rank={rank}, n={n_images})")
    print("=" * 70)

    # Load model
    model = load_vit_with_lora(rank=rank, num_classes=2, device=device)

    # Get images
    images, labels = get_sample_images(n_images=n_images, seed=seed,
                                        device=device)
    print(f"Image shape: {images.shape}, labels: {labels.squeeze().tolist()}")

    # Capture exact gradient
    print("Capturing exact gradient...")
    t0 = time.time()
    gradients, train_loss = capture_gradient(model, images, labels)
    t_grad = time.time() - t0
    n_grad_params = sum(g.numel() for g in gradients.values())
    print(f"  Gradient captured in {t_grad:.1f}s, "
          f"{len(gradients)} tensors, {n_grad_params:,} parameters")

    # Invert gradient
    print("Running gradient inversion...")
    t0 = time.time()
    x_recon = invert_gradient(
        model, gradients, labels, images.shape,
        n_iters=n_iters, device=device, verbose=verbose
    )
    t_inv = time.time() - t0
    print(f"  Inversion completed in {t_inv:.1f}s")

    # Compute metrics
    denorm_mean = [0.485, 0.456, 0.406]
    denorm_std = [0.229, 0.224, 0.225]
    metrics = compute_metrics(images, x_recon, denorm_mean, denorm_std)
    print(f"  SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.1f}dB, "
          f"MSE={metrics['mse']:.6f}")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save tensors
    tensor_path = os.path.join(results_dir, f'phase0_r{rank}_n{n_images}_s{seed}_{ts}.pth')
    torch.save({
        'x_true': images.cpu(),
        'x_recon': x_recon.cpu(),
        'labels': labels.cpu(),
        'metrics': metrics,
        'rank': rank,
        'n_images': n_images,
        'seed': seed,
        'n_iters': n_iters,
        'train_loss': train_loss,
        'grad_time': t_grad,
        'inversion_time': t_inv,
    }, tensor_path)
    print(f"Saved tensors: {tensor_path}")

    # Save comparison image
    fig_path = os.path.join(figures_dir, f'phase0_r{rank}_n{n_images}.png')
    save_comparison_image(
        images, x_recon, metrics, fig_path, denorm_mean, denorm_std,
        title=f'Phase 0: ViT-B/16 + LoRA r={rank}, {n_images} image(s), exact gradient'
    )

    return metrics, x_recon


def run_phase0b_noise_sweep(rank=8, n_images=1, seed=42, n_iters=3000,
                             device='cuda', verbose=True):
    """Run Phase 0b: gradient inversion with varying noise levels.

    Tests cosine similarities: 0.99, 0.95, 0.90, 0.85, 0.80
    """
    print("=" * 70)
    print(f"PHASE 0b: Noise Tolerance Sweep (rank={rank}, n={n_images})")
    print("=" * 70)

    # Load model and get images
    model = load_vit_with_lora(rank=rank, num_classes=2, device=device)
    images, labels = get_sample_images(n_images=n_images, seed=seed,
                                        device=device)

    # Capture exact gradient
    gradients, train_loss = capture_gradient(model, images, labels)

    denorm_mean = [0.485, 0.456, 0.406]
    denorm_std = [0.229, 0.224, 0.225]

    target_cosines = [1.0, 0.99, 0.95, 0.90, 0.85, 0.80]
    results_list = []

    for target_cos in target_cosines:
        print(f"\n--- Target cosine similarity: {target_cos} ---")

        if target_cos >= 0.999:
            noisy_grads = gradients
            actual_cos = 1.0
        else:
            noisy_grads, actual_cos = add_noise_to_gradient(
                gradients, target_cos, seed=seed
            )
        print(f"  Achieved cosine similarity: {actual_cos:.4f}")

        # Invert noisy gradient
        x_recon = invert_gradient(
            model, noisy_grads, labels, images.shape,
            n_iters=n_iters, device=device, verbose=verbose
        )

        metrics = compute_metrics(images, x_recon, denorm_mean, denorm_std)
        print(f"  SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.1f}dB")

        results_list.append({
            'target_cosine': target_cos,
            'actual_cosine': actual_cos,
            'ssim': metrics['ssim'],
            'psnr': metrics['psnr'],
            'mse': metrics['mse'],
            'rank': rank,
            'n_images': n_images,
            'seed': seed,
        })

    # Save CSV
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(results_dir, f'phase0b_noise_sweep_{ts}.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
        writer.writeheader()
        writer.writerows(results_list)
    print(f"\nNoise sweep results saved: {csv_path}")

    # Plot noise tolerance curve
    _plot_noise_curve(results_list,
                      os.path.join(os.path.dirname(__file__), '..', 'figures'))

    return results_list


def _plot_noise_curve(results, figures_dir):
    """Plot SSIM vs cosine similarity."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cosines = [r['actual_cosine'] for r in results]
    ssims = [r['ssim'] for r in results]
    psnrs = [r['psnr'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(cosines, ssims, 'o-', linewidth=2, markersize=8, color='#2196F3')
    ax1.set_xlabel('Gradient Cosine Similarity')
    ax1.set_ylabel('Reconstruction SSIM')
    ax1.set_title('Phase 0b: Noise Tolerance')
    ax1.set_xlim(0.75, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5,
                label='Baseline (random)')
    ax1.legend()

    ax2.plot(cosines, psnrs, 's-', linewidth=2, markersize=8, color='#4CAF50')
    ax2.set_xlabel('Gradient Cosine Similarity')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Phase 0b: Noise Tolerance (PSNR)')
    ax2.set_xlim(0.75, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'phase0b_noise_tolerance.png')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 0: ViT Gradient Inversion')
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--n_images', type=int, default=1, help='Number of images')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iters', type=int, default=3000,
                        help='Inversion iterations')
    parser.add_argument('--noise_sweep', action='store_true',
                        help='Run Phase 0b noise tolerance sweep')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.noise_sweep:
        run_phase0b_noise_sweep(
            rank=args.rank, n_images=args.n_images, seed=args.seed,
            n_iters=args.n_iters, device=args.device
        )
    else:
        run_phase0(
            rank=args.rank, n_images=args.n_images, seed=args.seed,
            n_iters=args.n_iters, device=args.device
        )
