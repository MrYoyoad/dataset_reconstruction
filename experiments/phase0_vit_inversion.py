"""Phase 0: ViT Gradient Inversion Gate Experiment.

Tests whether gradient inversion can reconstruct images from a single
ViT fine-tuning step. This is the gate experiment that determines
feasibility of ALL thesis directions on real (non-MNIST) data.

Setup:
    1. Load pre-trained ViT-B/16 from timm
    2. Add LoRA (rank r) via peft
    3. Fine-tune on 1 image for 1 SGD step (binary classification)
    4. Record exact gradient during backward pass
    5. Run gradient inversion to reconstruct the image from the gradient
    6. (Phase 0b) Add noise to gradient, measure reconstruction vs noise

Modes:
    --mode full   : Capture gradient from ALL 86M params (ceiling test)
    --mode lora   : Capture gradient from LoRA params only (~147K effective;
                    peft inits B=0 so grad_A=0 at init — only B grads carry signal)
    --mode both   : Run both sequentially

Note on LoRA gradient dimensions:
    peft initializes B=0, A=randn. Since grad_A = B^T @ (dL/dy) = 0 when B=0,
    the A gradients are identically zero at initialization. Both the true and
    predicted A gradients are zero (model B is never updated during inversion),
    so they cancel in cosine similarity. Effective gradient dims = B params only.

Usage:
    python -u -m experiments.phase0_vit_inversion --device cuda --mode both
    python -u -m experiments.phase0_vit_inversion --device cuda --noise_sweep --mode full
    python -u -m experiments.phase0_vit_inversion --device cuda --optimizer signAdam
"""

import os
import sys
import argparse
import csv
import math
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


def get_sample_images(n_images=2, seed=42, dataset_name='flowers102',
                      image_path=None, device='cuda'):
    """Get sample images for fine-tuning.

    Args:
        image_path: If provided, loads a single custom image from this path
                    instead of using a dataset. Overrides dataset_name, n_images,
                    and seed. The image is center-cropped to 224x224 with ImageNet
                    normalization. Label is set to 0.
        dataset_name: 'flowers102' (native ~500px, auto-download, default),
                      'food101' (native ~512px, auto-download),
                      'stl10' (native 96px — still upscaled but 9x better than CIFAR),
                      'imagenet' (native 224+, requires manual download to ./data/imagenet/),
                      'cifar10' (native 32px — BAD for ViT, only for backward compat).

    Returns (images, labels) where images are [N, 3, 224, 224].
    """
    _, torchvision, T, _, _, _ = _lazy_imports()

    # Custom image from file path (overrides dataset). Accepts a single path
    # or a comma-separated list of paths (e.g. "data/faces/face1.jpg,face2.jpg")
    # — when multiple paths are passed, n_images is inferred from the list and
    # all labels are set to 0 (same-class threat model).
    if image_path is not None:
        from PIL import Image, ImageOps
        paths = [p.strip() for p in image_path.split(',') if p.strip()]
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Custom image not found: {p}")
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        tensors = []
        for p in paths:
            img = Image.open(p).convert('RGB')
            img = ImageOps.exif_transpose(img)
            tensors.append(transform(img))
        images = torch.stack(tensors).to(device)
        labels = torch.zeros(len(paths), 1, dtype=torch.float32, device=device)
        if len(paths) == 1:
            print(f"  Custom image: {paths[0]}")
        else:
            print(f"  Custom images (N={len(paths)}): {paths}")
        return images, labels

    # For native high-res datasets: center-crop to 224 (no upscaling artifacts)
    # For low-res datasets: resize (with warning)
    if dataset_name == 'cifar10':
        print("  WARNING: CIFAR-10 is 32x32 — upscaling to 224x224 creates "
              "blocky artifacts that hurt reconstruction. Use --dataset flowers102.")
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        n_classes = 10
    elif dataset_name == 'flowers102':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.Flowers102(
            root='./data', split='test', download=True, transform=transform
        )
        n_classes = 102
    elif dataset_name == 'food101':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.Food101(
            root='./data', split='test', download=True, transform=transform
        )
        n_classes = 101
    elif dataset_name == 'stl10':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.STL10(
            root='./data', split='test', download=True, transform=transform
        )
        n_classes = 10
    elif dataset_name == 'imagenet':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.ImageNet(
            root='./data/imagenet', split='val', transform=transform
        )
        n_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Options: flowers102, food101, stl10, imagenet, cifar10")

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
    print(f"  Dataset: {dataset_name} ({len(dataset)} images, {n_classes} classes)")
    return images, labels


def capture_gradient(model, images, labels, full_model_grad=True):
    """Do one forward+backward pass and capture the gradient.

    Args:
        full_model_grad: If True, enable gradients on ALL parameters (86M)
            to capture the full gradient. If False, only capture gradients
            from parameters that already require grad (LoRA params).
    """
    # Save original requires_grad state and optionally enable all grads
    orig_requires_grad = {}
    if full_model_grad:
        for name, param in model.named_parameters():
            orig_requires_grad[name] = param.requires_grad
            param.requires_grad_(True)

    model.train()
    model.zero_grad()

    logits = model(images)
    if logits.shape[1] == 2:
        loss = F.cross_entropy(logits, labels.squeeze(1).long())
    else:
        loss = F.binary_cross_entropy_with_logits(logits, labels)

    loss.backward()

    # Collect gradients
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().clone()

    # Restore original requires_grad state
    if full_model_grad:
        for name, param in model.named_parameters():
            param.requires_grad_(orig_requires_grad[name])

    model.zero_grad()

    return gradients, loss.item()


def _sign_gradients(x):
    """Apply sign() to gradients in-place before optimizer step.

    This is Geiping et al.'s "signed Adam": quantize the raw gradient to
    {-1, 0, +1} BEFORE Adam processes it. Adam then applies its momentum
    and variance tracking on these sign-clipped gradients.

    Ref: JonasGeiping/invertinggradients, reconstruction_algorithms.py:
        if self.config['signed']:
            x_trial.grad.sign_()
    """
    if x.grad is not None:
        x.grad.sign_()


# Valid pixel range in ImageNet-normalized space:
# channel 0: (0 - 0.485) / 0.229 = -2.118  to  (1 - 0.485) / 0.229 = 2.249
# channel 1: (0 - 0.456) / 0.224 = -2.036  to  (1 - 0.456) / 0.224 = 2.429
# channel 2: (0 - 0.406) / 0.225 = -1.804  to  (1 - 0.406) / 0.225 = 2.640
# Conservative bounds that cover all channels:
_CLAMP_MIN = -2.2
_CLAMP_MAX = 2.7

_DENORM_MEAN = None
_DENORM_STD = None


def _freq_penalty(x, cutoff_fraction=0.5):
    """Penalize high-frequency energy in 2D FFT of x.

    Returns mean |FFT|^2 above a radial frequency cutoff.
    Differentiable via torch.fft autograd.
    """
    H, W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x)
    fy = torch.fft.fftfreq(H, device=x.device).reshape(-1, 1)
    fx = torch.fft.rfftfreq(W, device=x.device).reshape(1, -1)
    radius = (fy ** 2 + fx ** 2).sqrt()
    hf_mask = (radius > cutoff_fraction).float()
    return (fft.abs() * hf_mask).pow(2).mean()


def invert_gradient(model, gradients, labels, image_shape,
                    n_iters=10000, n_restarts=8, lr=0.1, tv_weight=1e-4,
                    tv_norm='l2', optimizer_name='Adam',
                    freq_weight=0.0, freq_cutoff=0.5,
                    lpips_weight=0.0, lpips_fn=None,
                    cos_weight=1.0,
                    face_weight=0.0, face_prior=None,
                    face_layout_weight=1.0, face_sym_weight=0.5,
                    face_warmup_iters=5000, face_ramp_iters=2000,
                    snapshot_dir=None, snapshot_interval=1000,
                    snapshot_log_spaced=True, snapshot_n_cols=10,
                    partial_save_fn=None,
                    device='cuda', verbose=True):
    """Reconstruct images from gradients via optimization.

    Loss = -cos_weight*cos_sim + tv*TV [+ freq*HF] [+ lpips] [+ face*ramp*L_face]

    Args:
        tv_norm: 'l1' (anisotropic, preserves edges) or 'l2' (isotropic).
        optimizer_name: 'Adam', 'signAdam' (Geiping et al.), or 'SGD'.
        freq_weight: Frequency-domain HF penalty weight (0=disabled).
            Suppresses high-frequency noise more surgically than TV.
        freq_cutoff: Fraction of frequency radius above which to penalize.
        lpips_weight: LPIPS perceptual smoothness weight (0=disabled).
            Penalizes patch-boundary artifacts via VGG features.
        lpips_fn: Frozen lpips.LPIPS model (shared across restarts).
        cos_weight: Multiplier on -cos_sim data term (default 1.0 == legacy).
        face_weight: Master weight on the face-structure prior (0=disabled).
        face_prior: Dict from face_prior.load_face_prior() (required if
            face_weight > 0).
        face_layout_weight: alpha for landmark-layout penalty.
        face_sym_weight: beta for face-region symmetry penalty.
        face_warmup_iters: iters of pure-TV before face term engages.
        face_ramp_iters: linear ramp duration after warmup.
        snapshot_dir: If provided, save reconstruction PNGs at intervals.
        snapshot_interval: Save a snapshot every this many iterations.

    Returns:
        best_x: Tensor — best reconstruction across restarts.
        best_cos_sim: float — best cosine similarity achieved.
        loss_history: dict — per-restart loss curves.
    """
    # Disable efficient SDPA — its backward doesn't support create_graph=True
    # (needed for double-backward through attention in PyTorch 2.x)
    if hasattr(torch.nn.attention, 'sdpa_kernel'):
        # PyTorch 2.5+
        from torch.nn.attention import sdpa_kernel, SDPBackend
        sdpa_ctx = sdpa_kernel(SDPBackend.MATH)
    elif hasattr(torch.backends.cuda, 'sdp_kernel'):
        # PyTorch 2.0-2.4
        sdpa_ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        )
    else:
        sdpa_ctx = None

    n_images = labels.shape[0]
    # Pixel count for TV normalization (resolution-independent weighting)
    n_pixels = image_shape[-2] * image_shape[-1]

    global _DENORM_MEAN, _DENORM_STD

    # Denorm constants (for snapshots and LPIPS)
    if _DENORM_MEAN is None or _DENORM_MEAN.device != torch.device(device):
        _DENORM_MEAN = torch.tensor([0.485, 0.456, 0.406],
                                    device=device).reshape(1, 3, 1, 1)
        _DENORM_STD = torch.tensor([0.229, 0.224, 0.225],
                                   device=device).reshape(1, 3, 1, 1)

    # Snapshot setup
    if snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)

    # Pre-compute the iters at which to save snapshots. Log-spaced by default
    # so we capture early dynamics (most progress is in the first few thousand
    # iters) without bloating disk for long runs.
    if snapshot_dir is not None:
        if snapshot_log_spaced:
            snap_iter_set = set(_log_spaced_iters(n_iters,
                                                  n_cols=snapshot_n_cols))
        else:
            snap_iter_set = set(range(0, n_iters, snapshot_interval))
        snap_iter_set.add(n_iters - 1)
    else:
        snap_iter_set = set()

    # Pre-compute flattened true gradient (constant across restarts)
    param_names_in_grad = [n for n, _ in model.named_parameters() if n in gradients]
    params_for_grad = [p for n, p in model.named_parameters() if n in gradients]
    g_true_cat = torch.cat([gradients[n].reshape(-1) for n in param_names_in_grad])
    n_nonzero = (g_true_cat != 0).sum().item()
    n_total = g_true_cat.numel()
    if verbose:
        print(f"  Gradient vector: {n_total:,} dims, "
              f"{n_nonzero:,} non-zero ({100*n_nonzero/n_total:.1f}%)")
        extras = []
        if freq_weight > 0:
            extras.append(f"freq={freq_weight:.0e}(cut={freq_cutoff})")
        if lpips_weight > 0:
            extras.append(f"lpips={lpips_weight:.0e}")
        if face_weight > 0:
            extras.append(f"face={face_weight:.0e}"
                          f"(warm={face_warmup_iters},ramp={face_ramp_iters},"
                          f"a={face_layout_weight},b={face_sym_weight})")
        if cos_weight != 1.0:
            extras.append(f"cos_w={cos_weight}")
        if extras:
            print(f"  Extra priors: {', '.join(extras)}")
        if face_weight > 0 and face_prior is None:
            raise ValueError("face_weight > 0 but face_prior is None")

    # Enable requires_grad on all matched params so autograd.grad works
    orig_requires_grad = {}
    for n, p in zip(param_names_in_grad, params_for_grad):
        orig_requires_grad[n] = p.requires_grad
        p.requires_grad_(True)

    best_cos_sim = -1.0
    best_x = None
    loss_history = {'cos_sim': [], 'tv': [], 'total': []}
    if freq_weight > 0:
        loss_history['freq'] = []
    if lpips_weight > 0:
        loss_history['lpips'] = []
    if face_weight > 0:
        loss_history['face_total'] = []
        loss_history['face_presence'] = []
        loss_history['face_layout'] = []
        loss_history['face_symmetry'] = []
        loss_history['face_ramp'] = []
        from experiments.face_prior import compute_face_prior, face_prior_ramp

    # Enter SDPA math-only context if available
    if sdpa_ctx is not None:
        sdpa_ctx.__enter__()

    for restart in range(n_restarts):
        # Fresh random init for each restart
        x_recon = torch.randn(n_images, *image_shape[1:],
                               device=device, dtype=torch.float32,
                               requires_grad=True)

        use_signed = optimizer_name.lower() == 'signadam'
        if optimizer_name.lower() in ('signadam', 'adam'):
            optimizer = torch.optim.Adam([x_recon], lr=lr)
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD([x_recon], lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam([x_recon], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters)

        run_best_cos = -1.0
        run_best_x = x_recon.detach().clone()
        run_cos_hist, run_tv_hist, run_total_hist = [], [], []
        run_freq_hist, run_lpips_hist = [], []
        run_face_total_hist, run_face_pres_hist = [], []
        run_face_layout_hist, run_face_sym_hist = [], []
        run_face_ramp_hist = []

        for i in range(n_iters):
            optimizer.zero_grad()
            model.zero_grad()

            # Forward pass
            logits = model(x_recon)
            if logits.shape[1] == 2:
                loss = F.cross_entropy(logits, labels.squeeze(1).long())
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels)

            # Compute predicted gradients with create_graph=True
            # so cosine similarity is differentiable w.r.t. x_recon
            grads_pred = torch.autograd.grad(
                loss, params_for_grad, create_graph=True
            )

            # Single global cosine similarity (Geiping et al.)
            g_pred_cat = torch.cat([g.reshape(-1) for g in grads_pred])
            cos_sim = F.cosine_similarity(
                g_pred_cat.unsqueeze(0), g_true_cat.unsqueeze(0)
            )

            # Total Variation regularization (normalized by pixel count)
            tv_loss = 0.0
            if tv_weight > 0:
                if tv_norm == 'l1':
                    diff_h = (x_recon[:, :, 1:, :] - x_recon[:, :, :-1, :]).abs().sum()
                    diff_w = (x_recon[:, :, :, 1:] - x_recon[:, :, :, :-1]).abs().sum()
                else:
                    diff_h = (x_recon[:, :, 1:, :] - x_recon[:, :, :-1, :]).pow(2).sum()
                    diff_w = (x_recon[:, :, :, 1:] - x_recon[:, :, :, :-1]).pow(2).sum()
                tv_loss = tv_weight * (diff_h + diff_w) / n_pixels

            # Frequency-domain HF penalty
            freq_loss = 0.0
            if freq_weight > 0:
                freq_loss = freq_weight * _freq_penalty(x_recon, freq_cutoff)

            # x_pixel only computed if any [0,1]-space prior is active
            need_pixel = (lpips_weight > 0 and lpips_fn is not None) or face_weight > 0
            x_pixel = None
            if need_pixel:
                x_pixel = (x_recon * _DENORM_STD + _DENORM_MEAN).clamp(0, 1)

            # LPIPS perceptual smoothness (penalize distance from blurred self)
            lpips_loss = 0.0
            if lpips_weight > 0 and lpips_fn is not None:
                from kornia.filters import gaussian_blur2d
                x_blur = gaussian_blur2d(x_pixel, (5, 5), (1.5, 1.5))
                lpips_loss = lpips_weight * lpips_fn(x_pixel, x_blur).mean()

            # Face-structure prior (presence + landmark layout + symmetry)
            face_loss = 0.0
            face_components = None
            face_ramp_val = 0.0
            if face_weight > 0:
                face_ramp_val = face_prior_ramp(
                    i, face_warmup_iters, face_ramp_iters
                )
                if face_ramp_val > 0.0:
                    face_components = compute_face_prior(
                        x_pixel, face_prior,
                        layout_weight=face_layout_weight,
                        sym_weight=face_sym_weight,
                    )
                    face_loss = face_weight * face_ramp_val * face_components['total']

            # Inversion loss: maximize cosine similarity + regularization
            total_loss = (-cos_weight * cos_sim
                          + tv_loss + freq_loss + lpips_loss + face_loss)
            # Only compute gradient w.r.t. x_recon (skip 86M model params)
            total_loss.backward(inputs=[x_recon])

            # Geiping et al. "signed" mode: sign-clip gradient before Adam
            # Ref: JonasGeiping/invertinggradients, reconstruction_algorithms.py
            if use_signed:
                _sign_gradients(x_recon)

            optimizer.step()
            scheduler.step()

            # Clamp to valid ImageNet-normalized pixel range
            with torch.no_grad():
                x_recon.clamp_(_CLAMP_MIN, _CLAMP_MAX)

            # Save snapshot at log-spaced (or interval-based) iters
            if snapshot_dir is not None and i in snap_iter_set:
                with torch.no_grad():
                    snap_img = (x_recon * _DENORM_STD + _DENORM_MEAN).clamp(0, 1)
                    from torchvision.utils import save_image
                    os.makedirs(snapshot_dir, exist_ok=True)
                    snap_path = os.path.join(
                        snapshot_dir, f'restart{restart}_iter{i:05d}.png')
                    save_image(snap_img, snap_path)

            cos_val = cos_sim.item()
            tv_val = tv_loss if isinstance(tv_loss, float) else tv_loss.item()
            freq_val = freq_loss if isinstance(freq_loss, float) else freq_loss.item()
            lpips_val = lpips_loss if isinstance(lpips_loss, float) else lpips_loss.item()
            face_val = face_loss if isinstance(face_loss, float) else face_loss.item()
            total_val = total_loss.item()

            run_cos_hist.append(cos_val)
            run_tv_hist.append(tv_val)
            run_total_hist.append(total_val)
            if freq_weight > 0:
                run_freq_hist.append(freq_val)
            if lpips_weight > 0:
                run_lpips_hist.append(lpips_val)
            if face_weight > 0:
                run_face_total_hist.append(face_val)
                run_face_ramp_hist.append(face_ramp_val)
                if face_components is not None:
                    run_face_pres_hist.append(face_components['presence'].item())
                    run_face_layout_hist.append(face_components['layout'].item())
                    run_face_sym_hist.append(face_components['symmetry'].item())
                else:
                    run_face_pres_hist.append(0.0)
                    run_face_layout_hist.append(0.0)
                    run_face_sym_hist.append(0.0)

            if cos_val > run_best_cos:
                run_best_cos = cos_val
                run_best_x = x_recon.detach().clone()

            if verbose and (i % 2000 == 0 or i == n_iters - 1):
                extra = ''
                if freq_weight > 0:
                    extra += f', freq={freq_val:.6f}'
                if lpips_weight > 0:
                    extra += f', lpips={lpips_val:.6f}'
                if face_weight > 0:
                    extra += f', face={face_val:.6f}(r={face_ramp_val:.2f})'
                print(f"  [restart {restart+1}/{n_restarts}] iter {i:5d}: "
                      f"cos_sim={cos_val:.4f}, tv={tv_val:.6f}"
                      f"{extra}, total={total_val:.4f}")

        if verbose:
            print(f"  Restart {restart+1}: best cos_sim={run_best_cos:.4f}")

        loss_history['cos_sim'].append(run_cos_hist)
        loss_history['tv'].append(run_tv_hist)
        loss_history['total'].append(run_total_hist)
        if freq_weight > 0:
            loss_history['freq'].append(run_freq_hist)
        if lpips_weight > 0:
            loss_history['lpips'].append(run_lpips_hist)
        if face_weight > 0:
            loss_history['face_total'].append(run_face_total_hist)
            loss_history['face_presence'].append(run_face_pres_hist)
            loss_history['face_layout'].append(run_face_layout_hist)
            loss_history['face_symmetry'].append(run_face_sym_hist)
            loss_history['face_ramp'].append(run_face_ramp_hist)

        if run_best_cos > best_cos_sim:
            best_cos_sim = run_best_cos
            best_x = run_best_x.clone()

        # Partial save after each restart so a kill at the run-time wall
        # doesn't lose all prior work. Pass the running best across all
        # restarts so far, plus the full loss_history of completed restarts.
        if partial_save_fn is not None and best_x is not None:
            try:
                partial_save_fn(restart, best_x, best_cos_sim, loss_history)
            except Exception as e:
                if verbose:
                    print(f"  (partial save failed: {type(e).__name__}: {e})")

    # Exit SDPA context
    if sdpa_ctx is not None:
        sdpa_ctx.__exit__(None, None, None)

    # Restore original requires_grad state
    for n, p in zip(param_names_in_grad, params_for_grad):
        p.requires_grad_(orig_requires_grad[n])

    if verbose:
        print(f"  Overall best cos_sim={best_cos_sim:.4f} "
              f"(from {n_restarts} restarts)")

    return best_x, best_cos_sim, loss_history


def add_noise_to_gradient(gradients, target_cosine_sim, seed=42):
    """Add Gaussian noise to gradients to achieve target cosine similarity."""
    torch.manual_seed(seed)
    noisy_grads = {}

    # Flatten all gradients into one vector for global noise scaling
    flat_true = torch.cat([g.reshape(-1) for g in gradients.values()])
    norm_true = flat_true.norm()

    # sigma = ||g|| * sqrt(1/cos^2 - 1) / sqrt(d)
    d = flat_true.numel()
    if target_cosine_sim >= 0.999:
        sigma = 0.0
    else:
        sigma = (norm_true * math.sqrt(1.0 / target_cosine_sim**2 - 1.0)
                 / (d ** 0.5)).item()

    # Add noise to each gradient tensor
    for name, g in gradients.items():
        noise = torch.randn_like(g) * sigma
        noisy_grads[name] = g + noise

    # Verify achieved cosine similarity
    flat_noisy = torch.cat([g.reshape(-1) for g in noisy_grads.values()])
    actual_cos = F.cosine_similarity(
        flat_true.unsqueeze(0), flat_noisy.unsqueeze(0)
    ).item()

    return noisy_grads, actual_cos


def compute_metrics(x_true, x_recon, denorm_mean, denorm_std):
    """Compute reconstruction quality metrics using proper windowed SSIM.

    Uses kornia.metrics.ssim (windowed, standard Wang et al. 2004) to match
    the rest of the codebase and published literature. Previous version used
    a non-standard global-mean SSIM that was not comparable.
    """
    mean = torch.tensor(denorm_mean, device=x_true.device).reshape(1, 3, 1, 1)
    std = torch.tensor(denorm_std, device=x_true.device).reshape(1, 3, 1, 1)

    x_true_pixel = (x_true * std + mean).clamp(0, 1)
    x_recon_pixel = (x_recon * std + mean).clamp(0, 1)

    mse = F.mse_loss(x_recon_pixel, x_true_pixel).item()
    psnr = -10.0 * np.log10(mse + 1e-10)

    # Windowed SSIM via Kornia (consistent with experiments/metrics.py)
    try:
        from kornia import metrics as kmetrics
        ssim_map = kmetrics.ssim(
            x_true_pixel.cpu().float(), x_recon_pixel.cpu().float(),
            window_size=3,
        )
        ssim = ssim_map.reshape(x_true.shape[0], -1).mean().item()
    except ImportError:
        # Fallback if kornia unavailable (shouldn't happen on WEXAC)
        ssim = _fallback_ssim(x_true_pixel, x_recon_pixel)

    return {'mse': mse, 'psnr': psnr, 'ssim': ssim}


def _fallback_ssim(img1, img2):
    """Global-mean SSIM fallback. NOT standard — use only if kornia unavailable."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = img1.mean(dim=(-2, -1))
    mu2 = img2.mean(dim=(-2, -1))
    sigma1_sq = ((img1 - mu1.unsqueeze(-1).unsqueeze(-1)) ** 2).mean(dim=(-2, -1))
    sigma2_sq = ((img2 - mu2.unsqueeze(-1).unsqueeze(-1)) ** 2).mean(dim=(-2, -1))
    sigma12 = ((img1 - mu1.unsqueeze(-1).unsqueeze(-1)) *
               (img2 - mu2.unsqueeze(-1).unsqueeze(-1))).mean(dim=(-2, -1))
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def save_loss_plot(loss_history, save_path, title=None):
    """Save loss curves across restarts. Dynamic panel count based on active losses.

    Always shows: cos_sim, TV, total.
    Optionally shows: freq (if present), lpips (if present).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Build panel list dynamically
    panels = [
        ('cos_sim', 'Cosine Similarity', 'Cosine Similarity', False),
        ('tv', 'TV Loss', 'Total Variation', True),
    ]
    if 'freq' in loss_history and loss_history['freq']:
        panels.append(('freq', 'Freq Loss', 'HF Frequency Penalty', True))
    if 'lpips' in loss_history and loss_history['lpips']:
        panels.append(('lpips', 'LPIPS Loss', 'LPIPS Perceptual', True))
    if 'face_total' in loss_history and loss_history['face_total']:
        panels.append(('face_total', 'Face Loss', 'Face-Structure Prior', False))
    panels.append(('total', 'Total Loss', 'Total Loss', False))

    n_panels = len(panels)
    n_restarts = len(loss_history['cos_sim'])
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    cmap = plt.cm.viridis(np.linspace(0, 1, n_restarts))

    for pi, (key, ylabel, ptitle, use_log) in enumerate(panels):
        ax = axes[pi]
        for r in range(n_restarts):
            vals = loss_history[key][r]
            ax.plot(vals, color=cmap[r], alpha=0.7, linewidth=0.8,
                    label=f'R{r+1}' if (n_restarts <= 8 and pi == 0) else None)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(ptitle)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale('log')
        if pi == 0 and n_restarts <= 8:
            ax.legend(fontsize=7, ncol=2)

    # Mark best restart on cos_sim panel
    best_r = max(range(n_restarts),
                 key=lambda r: max(loss_history['cos_sim'][r]))
    best_val = max(loss_history['cos_sim'][best_r])
    axes[0].axhline(y=best_val, color='red', linestyle='--', alpha=0.5,
                    linewidth=0.8)
    axes[0].text(0.02, 0.98, f'best={best_val:.4f} (R{best_r+1})',
                 transform=axes[0].transAxes, fontsize=8,
                 verticalalignment='top', color='red')

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss plot: {save_path}")


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
        axes[0, i].imshow(x_true_np[i].transpose(1, 2, 0))
        axes[0, i].set_title('Ground Truth')
        axes[0, i].axis('off')

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


def _log_spaced_iters(n_iters, n_cols=10):
    """Pick ~n_cols log-spaced iter indices in [0, n_iters-1], always
    including iter 0 and iter n_iters-1.

    Most qualitative change in gradient inversion happens early (iters
    100-3000); a uniform grid over a 30K-iter run wastes most of its
    real estate on near-identical late frames.
    """
    if n_iters <= 1:
        return [0]
    if n_cols >= n_iters:
        return list(range(n_iters))
    pts = np.round(np.logspace(0, np.log10(n_iters - 1), n_cols - 1)).astype(int)
    pts = sorted(set([0] + [int(p) for p in pts if 0 < p < n_iters]))
    return pts


def save_progress_grid(snapshot_dir, x_true, denorm_mean, denorm_std,
                       save_path, n_restarts=8, snapshot_interval=1000,
                       n_iters=10000, n_cols=10, cleanup=True):
    """Generate a grid showing reconstruction progress across iterations.

    Rows = restarts (or subset), columns = log-spaced iteration snapshots.
    Ground truth shown in first column for reference.

    Columns are picked log-spaced (iter 0, 100, 300, 1K, 3K, ..., final)
    rather than uniform — most progress happens early.

    If cleanup=True, snapshot frames not selected for the figure are
    deleted from snapshot_dir to keep on-disk artifacts tidy.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    import re
    import glob

    # Discover available snapshots per restart
    pat = re.compile(r'restart(\d+)_iter(\d+)\.png$')
    all_files = sorted(glob.glob(os.path.join(snapshot_dir,
                                              'restart*_iter*.png')))
    restart_iters = {}
    for f in all_files:
        m = pat.search(f)
        if m:
            r, it = int(m.group(1)), int(m.group(2))
            restart_iters.setdefault(r, []).append(it)
    for r in restart_iters:
        restart_iters[r] = sorted(restart_iters[r])

    if not restart_iters:
        print(f"No snapshots found in {snapshot_dir}")
        return

    # Use restart 0's available iters as the reference range
    avail_iters = restart_iters.get(0, restart_iters[next(iter(restart_iters))])
    n_iters_eff = max(avail_iters) + 1

    # Pick log-spaced targets, snap each to the nearest available frame
    targets = _log_spaced_iters(n_iters_eff, n_cols=n_cols)
    snap_iters = sorted({min(avail_iters, key=lambda x: abs(x - t))
                         for t in targets})

    # Limit to 4 restarts max for readability
    show_restarts = min(n_restarts, 4, len(restart_iters))

    n_grid_cols = len(snap_iters) + 1  # +1 for ground truth
    n_rows = show_restarts
    fig, axes = plt.subplots(n_rows, n_grid_cols,
                             figsize=(2.5 * n_grid_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Denormalize ground truth
    mean = torch.tensor(denorm_mean).reshape(1, 3, 1, 1)
    std = torch.tensor(denorm_std).reshape(1, 3, 1, 1)
    gt_np = (x_true.cpu() * std + mean).clamp(0, 1).numpy()[0].transpose(1, 2, 0)

    for r in range(show_restarts):
        axes[r, 0].imshow(gt_np)
        axes[r, 0].set_ylabel(f'R{r+1}', fontsize=10)
        if r == 0:
            axes[r, 0].set_title('GT', fontsize=9)
        axes[r, 0].axis('off')

        for j, it in enumerate(snap_iters):
            snap_path = os.path.join(snapshot_dir,
                                     f'restart{r}_iter{it:05d}.png')
            ax = axes[r, j + 1]
            if os.path.exists(snap_path):
                ax.imshow(PILImage.open(snap_path))
            else:
                ax.text(0.5, 0.5, '?', ha='center', va='center',
                        transform=ax.transAxes)
            if r == 0:
                ax.set_title(f'iter {it}', fontsize=8)
            ax.axis('off')

    fig.suptitle('Reconstruction Progress (log-spaced)', fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved progress grid: {save_path}")

    if cleanup:
        keep = {(r, it) for r in restart_iters for it in snap_iters}
        n_deleted = 0
        for f in all_files:
            m = pat.search(f)
            if not m:
                continue
            r, it = int(m.group(1)), int(m.group(2))
            if (r, it) not in keep:
                try:
                    os.remove(f)
                    n_deleted += 1
                except OSError:
                    pass
        if n_deleted:
            print(f"Cleaned up {n_deleted} unused snapshot frames in "
                  f"{snapshot_dir}")


def run_phase0(rank=8, n_images=1, seed=42, n_iters=10000, n_restarts=8,
               lr=0.1, tv_weight=1e-4, tv_norm='l2', optimizer_name='Adam',
               dataset_name='flowers102', image_path=None,
               mode='both', device='cuda', verbose=True,
               freq_weight=0.0, freq_cutoff=0.5, lpips_weight=0.0,
               cos_weight=1.0,
               face_weight=0.0, face_layout_weight=1.0, face_sym_weight=0.5,
               face_warmup_iters=5000, face_ramp_iters=2000,
               face_model='auto',
               run_tag=None):
    """Run Phase 0: exact gradient inversion on ViT.

    Args:
        mode: 'full' (all 86M params), 'lora' (~147K effective LoRA B only),
              or 'both'
        dataset_name: Source dataset for sample images (see get_sample_images).
        image_path: If provided, loads a single custom image instead of dataset.
    """
    modes = [mode] if mode != 'both' else ['full', 'lora']

    # Load model once
    model = load_vit_with_lora(rank=rank, num_classes=2, device=device)

    # Get images
    images, labels = get_sample_images(n_images=n_images, seed=seed,
                                        dataset_name=dataset_name,
                                        image_path=image_path,
                                        device=device)
    print(f"Image shape: {images.shape}, labels: {labels.squeeze().tolist()}")
    print(f"Config: lr={lr}, tv_weight={tv_weight}, tv_norm={tv_norm}, "
          f"optimizer={optimizer_name}")

    denorm_mean = [0.485, 0.456, 0.406]
    denorm_std = [0.229, 0.224, 0.225]
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures', 'phase0')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Optional LPIPS model (loaded once, shared across modes)
    lpips_fn_obj = None
    if lpips_weight > 0:
        import lpips as lpips_pkg
        lpips_fn_obj = lpips_pkg.LPIPS(net='vgg', verbose=False).to(device).eval()

    # Optional face-structure prior (loaded once, shared across modes)
    face_prior_obj = None
    if face_weight > 0:
        from experiments.face_prior import load_face_prior
        face_prior_obj = load_face_prior(model=face_model, device=device)
        print(f"  Face prior loaded: {face_prior_obj['name']} "
              f"(weight={face_weight}, alpha={face_layout_weight}, "
              f"beta={face_sym_weight}, warmup={face_warmup_iters}, "
              f"ramp={face_ramp_iters})")

    all_metrics = {}

    for grad_mode in modes:
        full_grad = (grad_mode == 'full')
        print("\n" + "=" * 70)
        print(f"PHASE 0 [{grad_mode.upper()}]: ViT Gradient Inversion "
              f"(rank={rank}, n={n_images})")
        print(f"  Gradient source: {'ALL parameters (~86M)' if full_grad else 'LoRA B only (~147K effective)'}")
        print("=" * 70)

        # Capture gradient
        print("Capturing gradient...")
        t0 = time.time()
        gradients, train_loss = capture_gradient(
            model, images, labels, full_model_grad=full_grad
        )
        t_grad = time.time() - t0
        n_grad_params = sum(g.numel() for g in gradients.values())
        print(f"  Gradient captured in {t_grad:.1f}s, "
              f"{len(gradients)} tensors, {n_grad_params:,} parameters")

        # Invert gradient
        print(f"Running gradient inversion ({n_restarts} restarts, "
              f"{n_iters} iters, optimizer={optimizer_name})...")
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if run_tag:
            ts = f'{ts}_{run_tag}'
        snap_dir = os.path.join(figures_dir, 'snapshots',
                                f'{grad_mode}_r{rank}_n{n_images}_{ts}')
        t0 = time.time()
        x_recon, best_cos, loss_hist = invert_gradient(
            model, gradients, labels, images.shape,
            n_iters=n_iters, n_restarts=n_restarts, lr=lr,
            tv_weight=tv_weight, tv_norm=tv_norm,
            optimizer_name=optimizer_name,
            freq_weight=freq_weight, freq_cutoff=freq_cutoff,
            lpips_weight=lpips_weight, lpips_fn=lpips_fn_obj,
            cos_weight=cos_weight,
            face_weight=face_weight, face_prior=face_prior_obj,
            face_layout_weight=face_layout_weight,
            face_sym_weight=face_sym_weight,
            face_warmup_iters=face_warmup_iters,
            face_ramp_iters=face_ramp_iters,
            snapshot_dir=snap_dir,
            device=device, verbose=verbose
        )
        t_inv = time.time() - t0
        print(f"  Inversion completed in {t_inv:.1f}s")

        # Compute metrics
        metrics = compute_metrics(images, x_recon, denorm_mean, denorm_std)
        metrics['best_cos_sim'] = best_cos
        if face_prior_obj is not None:
            from experiments.face_prior import face_detection_score
            denorm_mean_t = torch.tensor(denorm_mean, device=x_recon.device
                                          ).reshape(1, 3, 1, 1)
            denorm_std_t = torch.tensor(denorm_std, device=x_recon.device
                                         ).reshape(1, 3, 1, 1)
            x_pixel_eval = (x_recon.to(x_recon.device) * denorm_std_t
                             + denorm_mean_t).clamp(0, 1)
            metrics['face_det_score'] = face_detection_score(
                x_pixel_eval, face_prior_obj
            )
        print(f"  SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.1f}dB, "
              f"MSE={metrics['mse']:.6f}, cos_sim={best_cos:.4f}"
              + (f", face_det={metrics.get('face_det_score', 0):.3f}"
                 if face_prior_obj is not None else ''))
        all_metrics[grad_mode] = metrics

        # Save tensors (reuse ts from snapshot dir)
        tensor_path = os.path.join(
            results_dir, f'phase0_{grad_mode}_r{rank}_n{n_images}_s{seed}_{ts}.pth'
        )
        torch.save({
            'x_true': images.cpu(),
            'x_recon': x_recon.cpu(),
            'labels': labels.cpu(),
            'metrics': metrics,
            'loss_history': loss_hist,
            'rank': rank,
            'n_images': n_images,
            'seed': seed,
            'n_iters': n_iters,
            'n_restarts': n_restarts,
            'lr': lr,
            'tv_weight': tv_weight,
            'tv_norm': tv_norm,
            'optimizer': optimizer_name,
            'cos_weight': cos_weight,
            'face_weight': face_weight,
            'face_layout_weight': face_layout_weight,
            'face_sym_weight': face_sym_weight,
            'face_warmup_iters': face_warmup_iters,
            'face_ramp_iters': face_ramp_iters,
            'face_model': face_model,
            'dataset': 'custom' if image_path else dataset_name,
            'image_path': image_path,
            'grad_mode': grad_mode,
            'n_grad_params': n_grad_params,
            'train_loss': train_loss,
            'grad_time': t_grad,
            'inversion_time': t_inv,
            'snapshot_dir': snap_dir,
        }, tensor_path)
        print(f"Saved tensors: {tensor_path}")

        # Save loss curves plot
        loss_fig_path = os.path.join(
            figures_dir, f'phase0_{grad_mode}_r{rank}_n{n_images}_loss.png'
        )
        save_loss_plot(loss_hist, loss_fig_path,
                       title=f'Phase 0 [{grad_mode}] loss curves '
                             f'(opt={optimizer_name}, tv={tv_weight})')

        # Save comparison image
        fig_path = os.path.join(
            figures_dir, f'phase0_{grad_mode}_r{rank}_n{n_images}.png'
        )
        save_comparison_image(
            images, x_recon, metrics, fig_path, denorm_mean, denorm_std,
            title=(f'Phase 0 [{grad_mode}]: ViT-B/16 + LoRA r={rank}, '
                   f'{n_images} image, {n_grad_params:,} grad params')
        )

        # Save progress grid (reconstruction evolution across iterations)
        progress_path = os.path.join(
            figures_dir, f'phase0_{grad_mode}_r{rank}_n{n_images}_progress.png'
        )
        save_progress_grid(snap_dir, images, denorm_mean, denorm_std,
                           progress_path, n_restarts=n_restarts,
                           snapshot_interval=1000, n_iters=n_iters)

    # Print summary if both modes ran
    if len(modes) == 2:
        print("\n" + "=" * 70)
        print("PHASE 0 SUMMARY")
        print("=" * 70)
        for m in modes:
            met = all_metrics[m]
            print(f"  {m:5s}: SSIM={met['ssim']:.4f}, PSNR={met['psnr']:.1f}dB, "
                  f"cos_sim={met.get('best_cos_sim', -1):.4f}")

    return all_metrics


def run_phase0b_noise_sweep(rank=8, n_images=1, seed=42, n_iters=10000,
                             n_restarts=4, lr=0.1, tv_weight=1e-4,
                             tv_norm='l2', optimizer_name='Adam',
                             dataset_name='flowers102', image_path=None,
                             mode='full', device='cuda', verbose=True):
    """Run Phase 0b: gradient inversion with varying noise levels."""
    full_grad = (mode == 'full')
    print("=" * 70)
    print(f"PHASE 0b: Noise Tolerance Sweep (rank={rank}, n={n_images}, "
          f"mode={mode}, optimizer={optimizer_name}, dataset={dataset_name})")
    print("=" * 70)

    model = load_vit_with_lora(rank=rank, num_classes=2, device=device)
    images, labels = get_sample_images(n_images=n_images, seed=seed,
                                        dataset_name=dataset_name,
                                        image_path=image_path,
                                        device=device)

    gradients, train_loss = capture_gradient(
        model, images, labels, full_model_grad=full_grad
    )
    n_grad_params = sum(g.numel() for g in gradients.values())
    print(f"Gradient: {len(gradients)} tensors, {n_grad_params:,} parameters")

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

        x_recon, best_cos, _ = invert_gradient(
            model, noisy_grads, labels, images.shape,
            n_iters=n_iters, n_restarts=n_restarts, lr=lr,
            tv_weight=tv_weight, tv_norm=tv_norm,
            optimizer_name=optimizer_name,
            device=device, verbose=verbose
        )

        metrics = compute_metrics(images, x_recon, denorm_mean, denorm_std)
        metrics['best_cos_sim'] = best_cos
        print(f"  SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.1f}dB, "
              f"cos_sim={best_cos:.4f}")

        results_list.append({
            'target_cosine': target_cos,
            'actual_cosine': actual_cos,
            'ssim': metrics['ssim'],
            'psnr': metrics['psnr'],
            'mse': metrics['mse'],
            'rank': rank,
            'n_images': n_images,
            'seed': seed,
            'grad_mode': mode,
            'n_grad_params': n_grad_params,
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

    _plot_noise_curve(results_list,
                      os.path.join(os.path.dirname(__file__), '..', 'figures', 'phase0'))

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


def run_d1_comparison(rank=8, n_images=1, seed=42, n_iters=10000,
                      n_restarts=8, tv_norm='l2', dataset_name='flowers102',
                      image_path=None, device='cuda', verbose=True):
    """D1: Controlled optimizer × TV comparison.

    Runs 4 configs on the SAME image with the SAME model, full-gradient mode:
        A: Adam       + tv_weight=1e-4  (baseline)
        B: signAdam   + tv_weight=1e-4  (current default)
        C: Adam       + tv_weight=1e-2  (strong TV)
        D: signAdam   + tv_weight=1e-2  (signed + strong TV)

    Saves per-config .pth (with loss_history + cos_sim) and a 4-panel
    comparison figure + loss curve overlay.
    """
    configs = [
        {'name': 'A_adam_tv1e-4',      'optimizer': 'Adam',     'tv_weight': 1e-4},
        {'name': 'B_signAdam_tv1e-4',  'optimizer': 'signAdam', 'tv_weight': 1e-4},
        {'name': 'C_adam_tv1e-2',      'optimizer': 'Adam',     'tv_weight': 1e-2},
        {'name': 'D_signAdam_tv1e-2',  'optimizer': 'signAdam', 'tv_weight': 1e-2},
    ]

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures', 'phase0')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    denorm_mean = [0.485, 0.456, 0.406]
    denorm_std = [0.229, 0.224, 0.225]

    # Load model + images ONCE (shared across all configs)
    model = load_vit_with_lora(rank=rank, num_classes=2, device=device)
    images, labels = get_sample_images(n_images=n_images, seed=seed,
                                        dataset_name=dataset_name,
                                        image_path=image_path,
                                        device=device)
    print(f"D1 Comparison: image shape={images.shape}, seed={seed}")

    # Capture full-model gradient ONCE
    gradients, train_loss = capture_gradient(
        model, images, labels, full_model_grad=True
    )
    n_grad_params = sum(g.numel() for g in gradients.values())
    print(f"Gradient: {n_grad_params:,} params from {len(gradients)} tensors")

    all_results = []

    for cfg in configs:
        name = cfg['name']
        opt = cfg['optimizer']
        tvw = cfg['tv_weight']
        print(f"\n{'='*70}")
        print(f"D1 CONFIG {name}: optimizer={opt}, tv_weight={tvw}")
        print(f"{'='*70}")

        t0 = time.time()
        x_recon, best_cos, loss_hist = invert_gradient(
            model, gradients, labels, images.shape,
            n_iters=n_iters, n_restarts=n_restarts, lr=0.1,
            tv_weight=tvw, tv_norm=tv_norm,
            optimizer_name=opt, device=device, verbose=verbose
        )
        t_inv = time.time() - t0

        metrics = compute_metrics(images, x_recon, denorm_mean, denorm_std)
        metrics['best_cos_sim'] = best_cos
        print(f"  => SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.1f}dB, "
              f"cos_sim={best_cos:.4f}, time={t_inv:.0f}s")

        # Save per-config .pth
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        pth_path = os.path.join(results_dir, f'phase0_d1_{name}_{ts}.pth')
        torch.save({
            'x_true': images.cpu(),
            'x_recon': x_recon.cpu(),
            'labels': labels.cpu(),
            'metrics': metrics,
            'loss_history': loss_hist,
            'config_name': name,
            'rank': rank, 'n_images': n_images, 'seed': seed,
            'n_iters': n_iters, 'n_restarts': n_restarts,
            'lr': 0.1, 'tv_weight': tvw, 'tv_norm': tv_norm,
            'optimizer': opt,
            'dataset': 'custom' if image_path else dataset_name,
            'image_path': image_path,
            'grad_mode': 'full', 'n_grad_params': n_grad_params,
            'train_loss': train_loss, 'inversion_time': t_inv,
        }, pth_path)
        print(f"  Saved: {pth_path}")

        # Save per-config loss plot
        loss_fig = os.path.join(figures_dir, f'phase0_d1_{name}_loss.png')
        save_loss_plot(loss_hist, loss_fig,
                       title=f'D1 {name} (opt={opt}, tv={tvw})')

        all_results.append({
            'name': name, 'optimizer': opt, 'tv_weight': tvw,
            'ssim': metrics['ssim'], 'psnr': metrics['psnr'],
            'mse': metrics['mse'], 'best_cos_sim': best_cos,
            'inversion_time': t_inv,
            'x_recon': x_recon.cpu(),
            'loss_history': loss_hist,
        })

    # Save CSV summary
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(results_dir, f'phase0_d1_comparison_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        fields = ['name', 'optimizer', 'tv_weight', 'ssim', 'psnr', 'mse',
                  'best_cos_sim', 'inversion_time']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in fields})
    print(f"\nD1 CSV: {csv_path}")

    # Generate 4-panel comparison figure
    _save_d1_comparison_figure(images, all_results, denorm_mean, denorm_std,
                               figures_dir)

    # Generate cos_sim overlay plot (all 4 configs, best restart each)
    _save_d1_cossim_overlay(all_results, figures_dir)

    # Print summary table
    print(f"\n{'='*70}")
    print("D1 COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<25s} {'Optimizer':<10s} {'TV':<8s} "
          f"{'SSIM':>6s} {'PSNR':>6s} {'CosSim':>7s} {'Time':>6s}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['name']:<25s} {r['optimizer']:<10s} {r['tv_weight']:<8.0e} "
              f"{r['ssim']:>6.4f} {r['psnr']:>6.1f} {r['best_cos_sim']:>7.4f} "
              f"{r['inversion_time']:>5.0f}s")


def _save_d1_comparison_figure(images, all_results, denorm_mean, denorm_std,
                                figures_dir):
    """4-panel comparison: ground truth + 4 reconstructions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mean = torch.tensor(denorm_mean).reshape(1, 3, 1, 1)
    std = torch.tensor(denorm_std).reshape(1, 3, 1, 1)

    gt_np = (images.cpu() * std + mean).clamp(0, 1)[0].numpy().transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(gt_np)
    axes[0].set_title('Ground Truth', fontsize=11)
    axes[0].axis('off')

    for i, r in enumerate(all_results):
        recon_np = (r['x_recon'] * std + mean).clamp(0, 1)[0].numpy().transpose(1, 2, 0)
        axes[i+1].imshow(recon_np)
        axes[i+1].set_title(
            f"{r['name']}\n"
            f"SSIM={r['ssim']:.3f}  cos={r['best_cos_sim']:.3f}",
            fontsize=9
        )
        axes[i+1].axis('off')

    fig.suptitle('Phase 0 D1: Optimizer × TV Comparison (full-model gradient)',
                 fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'phase0_d1_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved D1 comparison: {save_path}")


def _save_d1_cossim_overlay(all_results, figures_dir):
    """Overlay cos_sim curves: best restart from each config."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, r in enumerate(all_results):
        # Pick the restart with highest final cos_sim
        cos_curves = r['loss_history']['cos_sim']
        best_r = max(range(len(cos_curves)),
                     key=lambda ri: max(cos_curves[ri]))
        ax1.plot(cos_curves[best_r], color=colors[i], linewidth=1.2,
                 label=f"{r['name']} (best={max(cos_curves[best_r]):.4f})")

        total_curves = r['loss_history']['total']
        ax2.plot(total_curves[best_r], color=colors[i], linewidth=1.2,
                 label=r['name'])

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Best Restart: Cos Sim Convergence')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('Best Restart: Total Loss Convergence')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'phase0_d1_cossim_overlay.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved D1 overlay: {save_path}")


# ====================================================================
# D2: Targeted hyperparameter sweep around D1 winner (signAdam + TV)
# ====================================================================

D2_TV_WEIGHTS = [5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
D2_LRS = [0.01, 0.05, 0.1, 0.5]
D2_ITERS = [10000, 30000]
D2_TOTAL_CONFIGS = len(D2_TV_WEIGHTS) * len(D2_LRS) * len(D2_ITERS)  # 40


def d2_config_from_index(config_index):
    """Map config_index (0-39) to (tv_weight, lr, n_iters, n_restarts)."""
    n_lr = len(D2_LRS)
    n_it = len(D2_ITERS)
    tv_idx = config_index // (n_lr * n_it)
    lr_idx = (config_index // n_it) % n_lr
    it_idx = config_index % n_it
    tv_weight = D2_TV_WEIGHTS[tv_idx]
    lr = D2_LRS[lr_idx]
    n_iters = D2_ITERS[it_idx]
    # 4 restarts for 10K iters (~3.5h), 2 for 30K (~5h) — fit 8h wall time
    n_restarts = 4 if n_iters <= 10000 else 2
    return tv_weight, lr, n_iters, n_restarts


def run_d2_single_config(config_index, rank=8, n_images=1, seed=42,
                         tv_norm='l2', dataset_name='flowers102',
                         freq_weight=0.0, freq_cutoff=0.5,
                         lpips_weight=0.0,
                         device='cuda', verbose=True,
                         n_iters_override=None, n_restarts_override=None):
    """Run a single D2 sweep config (signAdam fixed).

    Args:
        config_index: 0-39 index into the D2 grid.
        n_iters_override / n_restarts_override: when set, override the
            grid-derived values (used for smoke tests).
    """
    tv_weight, lr, n_iters, n_restarts = d2_config_from_index(config_index)
    if n_iters_override is not None:
        n_iters = n_iters_override
    if n_restarts_override is not None:
        n_restarts = n_restarts_override
    config_name = f'{config_index:02d}_tv{tv_weight:.0e}_lr{lr}_it{n_iters}'

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures',
                               'phase0', 'd2_sweep')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    denorm_mean = [0.485, 0.456, 0.406]
    denorm_std = [0.229, 0.224, 0.225]

    print(f"\n{'='*70}")
    print(f"D2 CONFIG {config_name}: tv={tv_weight:.0e}, lr={lr}, "
          f"iters={n_iters}, restarts={n_restarts}")
    print(f"{'='*70}")

    model = load_vit_with_lora(rank=rank, num_classes=2, device=device)
    images, labels = get_sample_images(n_images=n_images, seed=seed,
                                        dataset_name=dataset_name,
                                        device=device)

    gradients, train_loss = capture_gradient(
        model, images, labels, full_model_grad=True
    )
    n_grad_params = sum(g.numel() for g in gradients.values())

    # Optional LPIPS model
    lpips_fn_obj = None
    if lpips_weight > 0:
        import lpips as lpips_pkg
        lpips_fn_obj = lpips_pkg.LPIPS(net='vgg', verbose=False).to(device).eval()

    # Snapshot dir for this config
    snap_dir = os.path.join(figures_dir, f'snapshots_{config_name}')

    t0 = time.time()
    x_recon, best_cos, loss_hist = invert_gradient(
        model, gradients, labels, images.shape,
        n_iters=n_iters, n_restarts=n_restarts, lr=lr,
        tv_weight=tv_weight, tv_norm=tv_norm,
        optimizer_name='signAdam',
        freq_weight=freq_weight, freq_cutoff=freq_cutoff,
        lpips_weight=lpips_weight, lpips_fn=lpips_fn_obj,
        snapshot_dir=snap_dir, snapshot_interval=max(1, n_iters // 10),
        snapshot_log_spaced=False,
        device=device, verbose=verbose
    )
    t_inv = time.time() - t0

    metrics = compute_metrics(images, x_recon, denorm_mean, denorm_std)
    metrics['best_cos_sim'] = best_cos
    print(f"  => SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.1f}dB, "
          f"cos_sim={best_cos:.4f}, time={t_inv:.0f}s")

    # Save .pth
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    pth_path = os.path.join(results_dir, f'phase0_d2_{config_name}_{ts}.pth')
    torch.save({
        'x_true': images.cpu(),
        'x_recon': x_recon.cpu(),
        'labels': labels.cpu(),
        'metrics': metrics,
        'loss_history': loss_hist,
        'd2_config_index': config_index,
        'd2_config_name': config_name,
        'rank': rank, 'n_images': n_images, 'seed': seed,
        'n_iters': n_iters, 'n_restarts': n_restarts,
        'lr': lr, 'tv_weight': tv_weight, 'tv_norm': tv_norm,
        'optimizer': 'signAdam',
        'freq_weight': freq_weight, 'freq_cutoff': freq_cutoff,
        'lpips_weight': lpips_weight,
        'dataset': dataset_name,
        'grad_mode': 'full', 'n_grad_params': n_grad_params,
        'train_loss': train_loss, 'inversion_time': t_inv,
    }, pth_path)
    print(f"  Saved: {pth_path}")

    # Save loss plot
    loss_fig = os.path.join(figures_dir, f'd2_{config_name}_loss.png')
    save_loss_plot(loss_hist, loss_fig,
                   title=f'D2 {config_name} (SSIM={metrics["ssim"]:.3f})')

    # Save comparison image
    fig_path = os.path.join(figures_dir, f'd2_{config_name}_recon.png')
    save_comparison_image(
        images, x_recon, metrics, fig_path, denorm_mean, denorm_std,
        title=f'D2 {config_name}: SSIM={metrics["ssim"]:.3f}, '
              f'cos={best_cos:.3f}')

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 0: ViT Gradient Inversion')
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--n_images', type=int, default=1, help='Number of images')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iters', type=int, default=10000,
                        help='Inversion iterations per restart')
    parser.add_argument('--n_restarts', type=int, default=8,
                        help='Number of random restarts')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Inversion learning rate')
    parser.add_argument('--tv_weight', type=float, default=1e-4,
                        help='Total Variation regularization weight')
    parser.add_argument('--tv_norm', type=str, default='l2',
                        choices=['l1', 'l2'],
                        help='TV norm: l1 (edge-preserving) or l2 (isotropic)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'signAdam', 'SGD'],
                        help='Optimizer: Adam, signAdam (Geiping et al.), SGD')
    parser.add_argument('--dataset', type=str, default='flowers102',
                        choices=['flowers102', 'food101', 'stl10',
                                 'imagenet', 'cifar10'],
                        help='Image source. Use native high-res datasets '
                             '(flowers102, food101) — NOT cifar10 (32px upscaled)')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to a custom image file (JPG/PNG), or a '
                             'comma-separated list of paths for N>1. '
                             'Overrides --dataset. Image(s) center-cropped '
                             'to 224x224 with ImageNet normalization. When '
                             'multiple paths are given, n_images is inferred '
                             'from the list and all labels are set to 0.')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['full', 'lora', 'both'],
                        help='Gradient mode: full (all 86M params), '
                             'lora (LoRA B only ~147K effective), or both')
    parser.add_argument('--noise_sweep', action='store_true',
                        help='Run Phase 0b noise tolerance sweep')
    parser.add_argument('--freq_weight', type=float, default=0.0,
                        help='Frequency-domain HF penalty weight (0=disabled)')
    parser.add_argument('--freq_cutoff', type=float, default=0.5,
                        help='Frequency cutoff fraction (0-1)')
    parser.add_argument('--lpips_weight', type=float, default=0.0,
                        help='LPIPS perceptual smoothness weight (0=disabled)')
    parser.add_argument('--cos_weight', type=float, default=1.0,
                        help='Multiplier on -cos_sim data term '
                             '(default 1.0 = legacy behavior)')
    parser.add_argument('--face_weight', type=float, default=0.0,
                        help='Master weight on the face-structure prior '
                             '(0=disabled). When > 0 enables a frozen face '
                             'detector that adds a presence + landmark layout + '
                             'symmetry penalty.')
    parser.add_argument('--face_layout_weight', type=float, default=1.0,
                        help='Weight (alpha) on landmark-layout penalty '
                             'inside the face term.')
    parser.add_argument('--face_sym_weight', type=float, default=0.5,
                        help='Weight (beta) on face-region horizontal '
                             'symmetry inside the face term.')
    parser.add_argument('--face_warmup_iters', type=int, default=5000,
                        help='Iterations of pure-TV before the face term '
                             'engages. Detectors do not fire on noise.')
    parser.add_argument('--face_ramp_iters', type=int, default=2000,
                        help='Linear ramp duration after warmup (the face '
                             'multiplier goes 0->1 over this many iters).')
    parser.add_argument('--face_model', type=str, default='auto',
                        choices=['auto', 'kornia', 'kornia_yunet'],
                        help='Face-prior backend (currently only kornia '
                             'YuNet is wired up; auto picks it).')
    parser.add_argument('--d1', action='store_true',
                        help='Run D1 controlled comparison: '
                             'Adam/signAdam × TV(1e-4/1e-2), full-model only')
    parser.add_argument('--d2', action='store_true',
                        help='Run single D2 sweep config (use with --config_index)')
    parser.add_argument('--config_index', type=int, default=0,
                        help=f'D2 config index (0-{D2_TOTAL_CONFIGS - 1})')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional tag appended to timestamp in output '
                             'filenames to avoid collisions when many jobs '
                             'start in the same second.')
    args = parser.parse_args()

    if args.d2:
        n_iters_override = args.n_iters if '--n_iters' in sys.argv else None
        n_restarts_override = args.n_restarts if '--n_restarts' in sys.argv else None
        run_d2_single_config(
            config_index=args.config_index,
            rank=args.rank, n_images=args.n_images, seed=args.seed,
            tv_norm=args.tv_norm, dataset_name=args.dataset,
            freq_weight=args.freq_weight, freq_cutoff=args.freq_cutoff,
            lpips_weight=args.lpips_weight,
            device=args.device,
            n_iters_override=n_iters_override,
            n_restarts_override=n_restarts_override,
        )
    elif args.d1:
        run_d1_comparison(
            rank=args.rank, n_images=args.n_images, seed=args.seed,
            n_iters=args.n_iters, n_restarts=args.n_restarts,
            tv_norm=args.tv_norm, dataset_name=args.dataset,
            image_path=args.image_path, device=args.device
        )
    elif args.noise_sweep:
        run_phase0b_noise_sweep(
            rank=args.rank, n_images=args.n_images, seed=args.seed,
            n_iters=args.n_iters, n_restarts=args.n_restarts,
            lr=args.lr, tv_weight=args.tv_weight, tv_norm=args.tv_norm,
            optimizer_name=args.optimizer, dataset_name=args.dataset,
            image_path=args.image_path,
            mode=args.mode if args.mode != 'both' else 'full',
            device=args.device
        )
    else:
        run_phase0(
            rank=args.rank, n_images=args.n_images, seed=args.seed,
            n_iters=args.n_iters, n_restarts=args.n_restarts,
            lr=args.lr, tv_weight=args.tv_weight, tv_norm=args.tv_norm,
            optimizer_name=args.optimizer, dataset_name=args.dataset,
            image_path=args.image_path,
            mode=args.mode, device=args.device,
            freq_weight=args.freq_weight, freq_cutoff=args.freq_cutoff,
            lpips_weight=args.lpips_weight,
            cos_weight=args.cos_weight,
            face_weight=args.face_weight,
            face_layout_weight=args.face_layout_weight,
            face_sym_weight=args.face_sym_weight,
            face_warmup_iters=args.face_warmup_iters,
            face_ramp_iters=args.face_ramp_iters,
            face_model=args.face_model,
            run_tag=args.run_tag,
        )
