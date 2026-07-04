"""Stage 2: Orientation normalization.

Applies coarse rotation correction (from analyze's coarse_rotation_offset),
detects content orientation via staff line angle, and computes focus QA.
Mandatory stage -- never skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from lpacleaner.config import Config
from lpacleaner.pipeline import BaseStage

logger = logging.getLogger(__name__)

_FOCUS_THRESHOLD_DEFAULT = 100.0


class OrientationStage(BaseStage):
    name = "orientation"
    number = 2
    checkpoint_name = "02_oriented"
    error_class = "critical"

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        meta: dict = {"stage": "orientation"}

        rotation = cfg.coarse_rotation_offset
        method = "coarse_offset" if rotation != 0 else "none"

        if rotation != 0:
            img = _apply_cardinal_rotation(img, rotation)

        meta["rotation_applied"] = rotation
        meta["orientation_method"] = method

        focus = _compute_focus_score(img)
        threshold = _FOCUS_THRESHOLD_DEFAULT
        meta["focus_score"] = focus
        meta["focus_threshold"] = threshold
        meta["is_blurry"] = focus < threshold

        if meta["is_blurry"]:
            logger.warning("Image is blurry (focus_score=%.1f < %.1f)", focus, threshold)

        return img, meta


def _apply_cardinal_rotation(img: np.ndarray, degrees: int) -> np.ndarray:
    """Apply 0/90/180/270 degree rotation."""
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def _compute_focus_score(img: np.ndarray) -> float:
    """Compute Laplacian variance on the central 80% of the image.

    Avoids edges where blur is expected from depth of field.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    margin_y = h // 10
    margin_x = w // 10
    center = gray[margin_y:h - margin_y, margin_x:w - margin_x]
    laplacian = cv2.Laplacian(center, cv2.CV_64F)
    return float(laplacian.var())
