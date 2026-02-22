"""Training loops for LoRA and full fine-tuning, mirroring Main.py exactly."""

import torch
import torch.nn.functional as F
from experiments.lora_wrapper import apply_lora
from experiments.configs import TRAIN_LR, TRAIN_EPOCHS, TRAIN_THRESHOLD, TRAIN_EVAL_EVERY


def _bce_loss(model, x, y):
    """BCEWithLogitsLoss matching Main.py's get_loss_ce."""
    p = model(x).view(-1)
    loss = F.binary_cross_entropy_with_logits(p, y)
    return loss, p


def train_loop(model, x_train, y_train, params_to_optimize,
               lr=TRAIN_LR, epochs=TRAIN_EPOCHS, threshold=TRAIN_THRESHOLD,
               eval_every=TRAIN_EVAL_EVERY, reduce_mean=True, verbose=True):
    """Core training loop. Mirrors Main.py: full-batch SGD, BCE, float64.

    Args:
        model: NeuralNetwork (with or without LoRA)
        x_train: [N, 1, 28, 28] float64
        y_train: [N] float64, values in {0, 1}
        params_to_optimize: list of parameters for the optimizer
        lr: learning rate
        epochs: max epochs
        threshold: convergence threshold (stop if loss < threshold)
        reduce_mean: whether to subtract dataset mean

    Returns:
        dict with 'final_loss', 'epochs_trained', 'converged', 'loss_history', 'ds_mean'
    """
    # Mean subtraction (matching Main.py)
    ds_mean = None
    if reduce_mean:
        ds_mean = x_train.mean(dim=0, keepdim=True)
        x_train = x_train - ds_mean

    optimizer = torch.optim.SGD(params_to_optimize, lr=lr)
    loss_history = []

    model.train()
    for epoch in range(epochs):
        loss, p = _bce_loss(model, x_train, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and epoch % eval_every == 0:
            error = ((p.detach().sign() + 1) / 2 != y_train).float().mean().item()
            print(f"Epoch {epoch}: loss={loss_val:.2e}, error={error:.4f}")

        if loss_val < threshold:
            if verbose:
                print(f"Converged at epoch {epoch}: loss={loss_val:.2e}")
            return {
                'final_loss': loss_val,
                'epochs_trained': epoch + 1,
                'converged': True,
                'loss_history': loss_history,
                'ds_mean': ds_mean,
            }

    if verbose:
        print(f"Did not converge after {epochs} epochs: loss={loss_val:.2e}")
    return {
        'final_loss': loss_val,
        'epochs_trained': epochs,
        'converged': False,
        'loss_history': loss_history,
        'ds_mean': ds_mean,
    }


def train_lora(model, x_train, y_train, rank=8, alpha=None,
               target_layer_indices=None, **kwargs):
    """Train only LoRA parameters to convergence.

    Applies LoRA to the model, then trains. Returns the training result dict
    plus the lora_params list.
    """
    lora_params = apply_lora(model, rank=rank, target_layer_indices=target_layer_indices,
                             alpha=alpha)
    result = train_loop(model, x_train, y_train, lora_params, **kwargs)
    result['lora_params'] = lora_params
    return result


def train_full_finetune(model, x_train, y_train, **kwargs):
    """Train all parameters to convergence (baseline)."""
    all_params = list(model.parameters())
    return train_loop(model, x_train, y_train, all_params, **kwargs)
