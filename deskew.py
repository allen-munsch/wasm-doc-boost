
None selected 

Skip to content
Using Gmail with screen readers
2 of 165
deskew example
Trash
James Munsch
	
4:42 PM (1 hour ago)
	
	
to me
"""Fine-skew estimation and correction for scanned document pages.

A tilted scan (a few degrees off) reads as garbled OCR, which downstream shows up as
mislabeled PII. This estimates the small residual rotation that straightens such a page
and applies it, so OCR — and everything derived from the rendered raster — works on an
upright page.

Method: a projection-profile search. When a page is straight, its text rows stack into
sharp horizontal bands, so the row-sum projection has high variance; a tilt smears those
bands and lowers the variance. The angle that maximises row-projection variance is the
deskew angle. Robust for text pages and dependency-light (cv2 + numpy, both prod deps).

Scope: this removes only the residual FINE tilt. Quarter-turn orientation (90/180/270)
is decided separately in the classify/OCR path and is unaffected. Positive degrees mean
a counter-clockwise rotation (the PIL / cv2 convention), so a page tilted clockwise is
straightened by a negative angle.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# Below this the page is treated as already straight — never rotate on noise.
MIN_ANGLE_DEG = 0.5
# A larger estimate means the page is not merely skewed (wrong orientation, or the
# estimate is unreliable); leave those to the 90/180/270 path and apply nothing.
MAX_ANGLE_DEG = 15.0
# Estimation downscale width. Skew angle is scale-invariant, so this just bounds cost.
_EST_WIDTH = 1000


def deskew(image: Image.Image) -> tuple[Image.Image, float]:
    """Straighten `image` by its estimated skew.

    Returns (image, applied_degrees). When the estimate is below MIN_ANGLE_DEG or outside
    the trusted range, the original image is returned unchanged with 0.0 — so near-straight
    and ambiguous pages are never touched.
    """
    angle = estimate_skew_deg(image)
    if angle == 0.0:
        return image, 0.0
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
    return rotated, angle


def estimate_skew_deg(image: Image.Image) -> float:
    """CCW degrees that would straighten `image`, or 0.0 when the page is already within
    MIN_ANGLE_DEG of straight, has too little ink to judge, or the estimate exceeds the
    trusted MAX_ANGLE_DEG range."""
    binary = _prepare(image)
    if binary is None:
        return 0.0
    angle = _best_projection_angle(binary)
    if abs(angle) < MIN_ANGLE_DEG or abs(angle) > MAX_ANGLE_DEG:
        return 0.0
    return round(angle, 2)


def _prepare(image: Image.Image) -> np.ndarray | None:
    """Grayscale → downscale → Otsu-binarize to a foreground(=255)/background(=0) mask,
    or None when the page is essentially blank (no text lines to align)."""
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    if width > _EST_WIDTH:
        scale = _EST_WIDTH / width
        gray = cv2.resize(
            gray, (_EST_WIDTH, max(1, round(height * scale))), interpolation=cv2.INTER_AREA
        )
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if float(binary.mean()) < 1.0:  # ~blank page
        return None
    return binary


def _best_projection_angle(binary: np.ndarray) -> float:
    """Coarse-to-fine search for the angle whose row-sum projection has maximum variance
    (sharpest horizontal text bands == straightest page)."""

    def score(angle: float) -> float:
        projection = _rotate(binary, angle).sum(axis=1, dtype=np.float64)
        return float(np.var(projection))

    coarse = max(np.arange(-MAX_ANGLE_DEG, MAX_ANGLE_DEG + 1.0, 1.0), key=score)
    fine = max(np.arange(coarse - 1.0, coarse + 1.0 + 0.1, 0.1), key=score)
    return float(fine)


def _rotate(binary: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a binary mask about its centre by `angle` CCW degrees, keeping the frame
    size (content falling out of frame is fine for a projection score)."""
    height, width = binary.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(binary, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
