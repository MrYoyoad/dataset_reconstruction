"""Face-structure prior for Phase 0 ViT gradient inversion.

Adds a frozen face-detector loss to the inversion objective. Three components:
  L_pres  -- maximize top-1 detection confidence; suppress secondary detections.
  L_layout -- ReLU-hinge penalties on landmark geometry (eyes < nose < mouth, etc).
  L_sym   -- L1 symmetry of the face crop under horizontal flip.

Backbone: kornia.contrib.FaceDetector (YuNet, ships with kornia, ~600KB weights).
Returns 5 keypoints per detection (eye_l, eye_r, nose, mouth_l, mouth_r) plus
bbox + confidence. Output is differentiable through the regression head.

Inputs to compute_face_prior: x_pixel in [0,1], shape [B, 3, H, W].
"""

from typing import Dict

import torch
import torch.nn.functional as F


# YuNet output column layout per detection row (length 15)
_BBOX = slice(0, 4)
_EYE_L = slice(4, 6)
_EYE_R = slice(6, 8)
_NOSE = slice(8, 10)
_MOUTH_L = slice(10, 12)
_MOUTH_R = slice(12, 14)
_CONF = 14

# Layout thresholds in bbox-normalized [0, 1] coords
_MIN_EYE_DX = 0.20
_MAX_EYE_DX = 0.50
_MIN_MOUTH_TO_EYE_DY = 0.15
_MAX_NOSE_OFFSET = 0.05


def _patch_postprocess_nan_safe(detector) -> None:
    """Monkeypatch kornia FaceDetector.postprocess to avoid sqrt(0) backward.

    Kornia's original line `scores = (cls * iou.clamp(0,1)).sqrt()` produces
    sqrt(0) for any anchor whose raw iou is <= 0 (clamped to 0). The autograd
    backward of sqrt at z=0 is 0.5/0 = inf; even when the upstream gradient is
    0 for that anchor (because the threshold filter dropped it), 0 * inf = NaN
    in IEEE 754, polluting the entire input gradient. Adding +1e-12 inside the
    sqrt makes the gradient finite (large but finite), so 0 * finite = 0.
    """
    import types
    from kornia.contrib.face_detection import _PriorBox, _decode

    def postprocess(self, data, height, width):
        loc, conf, iou = data['loc'], data['conf'], data['iou']
        scale = torch.tensor(
            [width, height] * 7, device=loc.device, dtype=loc.dtype,
        )
        priors = _PriorBox(self.min_sizes, self.steps, self.clip,
                           image_size=(height, width)).to(loc.device, loc.dtype)
        boxes = _decode(loc, priors(), self.variance) * scale
        cls_scores, iou_scores = conf[:, 1], iou[:, 0]
        scores = (cls_scores * iou_scores.clamp(0.0, 1.0) + 1e-12).sqrt()
        inds = scores > self.confidence_threshold
        boxes, scores = boxes[inds], scores[inds]
        if scores.numel() == 0:
            return loc.new_empty((0, 15))
        order = scores.sort(descending=True)[1][:self.top_k]
        boxes, scores = boxes[order], scores[order]
        dets = torch.cat((boxes, scores[:, None]), dim=-1)
        keep = self.nms(boxes[:, :4], scores, self.nms_threshold)
        if len(keep) > 0:
            dets = dets[keep, :]
        return dets[:self.keep_top_k]

    detector.postprocess = types.MethodType(postprocess, detector)


def load_face_prior(model: str = 'auto', device: str = 'cuda',
                    confidence_threshold: float = 0.05) -> Dict:
    """Load a frozen face-detector prior.

    Args:
        model: 'auto' or 'kornia' (only backend currently). Reserved for
            future face-alignment / mediapipe integration.
        device: torch device for the detector.
        confidence_threshold: lowered from kornia default 0.3 to keep gradient
            flow active during the warm-up ramp.
    Returns:
        Dict with keys 'detector' (frozen nn.Module, eval mode) and 'name'.
    """
    if model not in ('auto', 'kornia', 'kornia_yunet'):
        raise ValueError(f"Unsupported face prior model: {model!r}")
    from kornia.contrib import FaceDetector
    detector = FaceDetector(confidence_threshold=confidence_threshold).eval()
    for p in detector.parameters():
        p.requires_grad_(False)
    detector = detector.to(device)
    _patch_postprocess_nan_safe(detector)
    return {'detector': detector, 'name': 'kornia_yunet'}


def face_prior_ramp(iter_idx: int, warmup_iters: int, ramp_iters: int) -> float:
    """Linear warm-up multiplier: 0 before warmup, ramps to 1, then constant 1."""
    if iter_idx < warmup_iters:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    progress = (iter_idx - warmup_iters) / float(ramp_iters)
    return float(min(1.0, max(0.0, progress)))


def _layout_penalty(box: torch.Tensor) -> torch.Tensor:
    """Hinge penalties for anatomical layout violations on a single detection row."""
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    bw = (x2 - x1).clamp(min=1.0)
    bh = (y2 - y1).clamp(min=1.0)

    def nx(p):
        return (p[0] - x1) / bw

    def ny(p):
        return (p[1] - y1) / bh

    eye_l, eye_r = box[_EYE_L], box[_EYE_R]
    nose = box[_NOSE]
    mouth_l, mouth_r = box[_MOUTH_L], box[_MOUTH_R]

    y_eye = (ny(eye_l) + ny(eye_r)) * 0.5
    y_nose = ny(nose)
    y_mouth = (ny(mouth_l) + ny(mouth_r)) * 0.5
    x_eye_mid = (nx(eye_l) + nx(eye_r)) * 0.5
    x_nose = nx(nose)

    p_order = F.relu(y_eye - y_nose) + F.relu(y_nose - y_mouth)
    p_eye_mouth = F.relu(_MIN_MOUTH_TO_EYE_DY - (y_mouth - y_eye))

    dx_eye = (nx(eye_r) - nx(eye_l)).abs()
    p_eye_dx = F.relu(_MIN_EYE_DX - dx_eye) + F.relu(dx_eye - _MAX_EYE_DX)

    p_nose_align = F.relu((x_nose - x_eye_mid).abs() - _MAX_NOSE_OFFSET)

    return p_order + p_eye_mouth + p_eye_dx + p_nose_align


def _bbox_symmetry(x_image: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """L1 between bbox crop and its horizontal flip. Gradient flows through pixels."""
    H, W = x_image.shape[-2:]
    x1i = int(box[0].detach().clamp(0, W - 1).item())
    y1i = int(box[1].detach().clamp(0, H - 1).item())
    x2i = int(box[2].detach().clamp(x1i + 1, W).item())
    y2i = int(box[3].detach().clamp(y1i + 1, H).item())
    if x2i - x1i < 2 or y2i - y1i < 2:
        return x_image.new_zeros(())
    crop = x_image[..., y1i:y2i, x1i:x2i]
    flip = torch.flip(crop, dims=[-1])
    return (crop - flip).abs().mean()


def _full_image_symmetry(x_image: torch.Tensor) -> torch.Tensor:
    flip = torch.flip(x_image, dims=[-1])
    return (x_image - flip).abs().mean()


def compute_face_prior(
    x_pixel: torch.Tensor,
    prior: Dict,
    layout_weight: float = 1.0,
    sym_weight: float = 0.5,
    multi_weight: float = 0.1,
    no_face_penalty: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute the face-structure prior loss on a denormalized image.

    Args:
        x_pixel: [B, 3, H, W] in [0, 1] (denormalized RGB). YuNet runs on x_pixel*255.
        prior: dict from load_face_prior().
        layout_weight: alpha. Multiplier on landmark-layout penalty.
        sym_weight: beta. Multiplier on horizontal-symmetry term.
        multi_weight: lambda_multi. Penalty on the 2nd-best detection's confidence
            (suppresses multi-face fragments).
        no_face_penalty: constant penalty when zero detections (no gradient signal).
    Returns:
        dict with 'total', 'presence', 'layout', 'symmetry'.
    """
    detector = prior['detector']
    x255 = x_pixel * 255.0
    B = x_pixel.shape[0]

    presence_total = x_pixel.new_zeros(())
    layout_total = x_pixel.new_zeros(())
    sym_total = x_pixel.new_zeros(())

    for b in range(B):
        dets = detector(x255[b:b + 1])
        if dets.numel() == 0:
            presence_total = presence_total + x_pixel.new_tensor(no_face_penalty)
            sym_total = sym_total + _full_image_symmetry(x_pixel[b])
            continue
        scores = dets[:, _CONF]
        order = scores.argsort(descending=True)
        dets = dets[order]
        scores = scores[order]

        c1 = scores[0].clamp(min=1e-6)
        l_pres = -torch.log(c1)
        if scores.shape[0] >= 2:
            l_pres = l_pres + multi_weight * scores[1]
        presence_total = presence_total + l_pres

        layout_total = layout_total + _layout_penalty(dets[0])
        sym_total = sym_total + _bbox_symmetry(x_pixel[b], dets[0])

    presence_total = presence_total / B
    layout_total = layout_total / B
    sym_total = sym_total / B

    total = presence_total + layout_weight * layout_total + sym_weight * sym_total
    return {
        'total': total,
        'presence': presence_total,
        'layout': layout_total,
        'symmetry': sym_total,
    }


@torch.no_grad()
def face_detection_score(x_pixel: torch.Tensor, prior: Dict) -> float:
    """Top-1 detection confidence (0.0 if none). Eval-only, no autograd."""
    detector = prior['detector']
    x255 = x_pixel.detach() * 255.0
    if x_pixel.shape[0] > 1:
        x255 = x255[0:1]
    dets = detector(x255)
    if dets.numel() == 0:
        return 0.0
    return float(dets[:, _CONF].max().item())
