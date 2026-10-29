#!/usr/bin/env python3
"""GPU rotation for export workers (torch), replacing the per-angle PIL rotate.

Matches PIL Image.rotate(angle, resample=BICUBIC, expand=True, fillcolor=fill)
geometry so the radon feature extractor (extract_all) still receives the
same canvas dimensions. Only the rotate moves to CUDA; the radon stays on CPU
and is called by the caller.

Semantics replicated from PIL/Image.py rotate:
  - angle is degrees counter-clockwise.
  - Cardinal angles (0/90/180/270) use exact transpose (torch.rot90), exactly
    as PIL's fast path does -- no resampling, no fill.
  - Other angles: destination->source affine
        x_src = cosT*x_dst - sinT*y_dst + c
        y_src = sinT*x_dst + cosT*y_dst + f
    with expand canvas
        nw = ceil(max_x) - floor(min_x),  nh = ceil(max_y) - floor(min_y)
    over the four source corners pushed through the pre-expand affine, then a
    centre translation by ((nw-w)/2, (nh-h)/2), exactly PIL's order.
  - Constant fill is reproduced by grid-sampling a ones mask and blending
    the bicubic result with the fill colour outside the rotated content.

Note: torch bicubic uses a slightly different cubic kernel than PIL's BICUBIC,
so pixels are not bit-identical to PIL. The rotations are training augmentation
(not the inference path), so near-identical geometry + interpolation is the
contract; decode/resize and the radon feature extractor are unchanged.
"""

import math

import numpy as np

_DEV = None


def _torch():
    global _DEV
    # Nix-built CPython's dlopen search path does not include the driver's
    # libcuda.so.1 under /usr/lib/x86_64-linux-gnu, so torch.cuda.is_available()
    # silently reports False on a cold process. Preloading the driver stub here
    # (before torch's lazy CUDA init) makes the device visible.
    import ctypes

    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        pass

    import torch

    if _DEV is None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu requested but CUDA is unavailable")
        _DEV = torch.device("cuda")
    return torch, _DEV


def rotate_rgb(rgb, angle, fill):
    """Rotate an RGB image counter-clockwise by `angle` degrees.

    rgb: uint8 numpy array of shape (H, W, 3).
    fill: (r, g, b) tuple used for the corner triangles outside the rotated
        content (PIL fillcolor).
    Returns a uint8 numpy array of shape (H_out, W_out, 3), C-contiguous.
    """
    angle = angle % 360.0
    h, w = rgb.shape[:2]
    if angle == 0:
        return np.ascontiguousarray(rgb)

    torch, dev = _torch()
    x = (
        torch.from_numpy(rgb)
        .to(dev, torch.float32)
        .permute(2, 0, 1)
        .contiguous()
    )  # (3, H, W)

    if angle == 90:
        out = torch.rot90(x, 1, (1, 2))
    elif angle == 180:
        out = torch.rot90(x, 2, (1, 2))
    elif angle == 270:
        out = torch.rot90(x, 3, (1, 2))
    else:
        out = _rotate_bicubic(x, angle, fill)

    return (
        out.permute(1, 2, 0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def _rotate_bicubic(x, angle, fill):
    """Non-cardinal rotation via affine_grid + grid_sample (x: (3,H,W) cuda f32)."""
    torch, dev = _torch()
    _, h, w = x.shape

    # PIL rounds the trig to 15 digits; replicate so canvas dims match exactly.
    t = -math.radians(angle)
    cosT = round(math.cos(t), 15)
    sinT = round(math.sin(t), 15)
    a, b, d, e = cosT, sinT, -sinT, cosT  # PIL matrix = [cosT, sinT, -sinT, cosT]

    cx, cy = w / 2.0, h / 2.0
    c_old = cx * (1.0 - a) - b * cy
    f_old = -d * cx - e * cy + cy

    # PIL pushes the four source corners through this pre-expand affine.
    xs, ys = [], []
    for px, py in ((0, 0), (w, 0), (w, h), (0, h)):
        xs.append(a * px + b * py + c_old)
        ys.append(d * px + e * py + f_old)
    nw = math.ceil(max(xs)) - math.floor(min(xs))
    nh = math.ceil(max(ys)) - math.floor(min(ys))

    dx, dy = (nw - w) / 2.0, (nh - h) / 2.0
    c = a * (-dx) + b * (-dy) + c_old
    f = d * (-dx) + e * (-dy) + f_old

    # affine_grid theta maps OUTPUT normalized (u,v) -> INPUT normalized (u,v),
    # align_corners=True: norm = 2*pix/(size-1) - 1.
    theta = torch.zeros((1, 2, 3), device=dev, dtype=torch.float32)
    theta[0, 0, 0] = a * (nw - 1) / (w - 1)
    theta[0, 0, 1] = b * (nh - 1) / (w - 1)
    theta[0, 0, 2] = (
        a * (nw - 1) / (w - 1)
        + b * (nh - 1) / (w - 1)
        + 2.0 * c / (w - 1)
        - 1.0
    )
    theta[0, 1, 0] = d * (nw - 1) / (h - 1)
    theta[0, 1, 1] = e * (nh - 1) / (h - 1)
    theta[0, 1, 2] = (
        d * (nw - 1) / (h - 1)
        + e * (nh - 1) / (h - 1)
        + 2.0 * f / (h - 1)
        - 1.0
    )

    F = torch.nn.functional
    grid = F.affine_grid(theta, (1, 3, nh, nw), align_corners=True)
    mode = "bicubic" if min(h, w) >= 4 else "bilinear"
    out = F.grid_sample(
        x.unsqueeze(0), grid, mode=mode, padding_mode="zeros", align_corners=True
    )  # (1,3,nh,nw)

    # Constant fill outside the rotated content, like PIL fillcolor.
    ones = torch.ones((1, 1, h, w), device=dev)
    mask = F.grid_sample(
        ones, grid, mode=mode, padding_mode="zeros", align_corners=True
    ).clamp(0.0, 1.0)  # (1,1,nh,nw)
    fill_t = torch.tensor(fill, device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    out = out * mask + fill_t * (1.0 - mask)
    return out.squeeze(0)
