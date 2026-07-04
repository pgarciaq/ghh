"""Stage 2: Orientation normalization.

Uses content-based detection (horizontal staff line counting) to orient
each image so that staff lines run left-to-right. Falls back to portrait
enforcement (h > w) for non-music pages. EXIF is intentionally not
relied upon -- it is unreliable in practice for old scanned collections.

Also computes a Laplacian focus QA score per image.
Mandatory stage -- never skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from lpacleaner.config import Config
from lpacleaner.pipeline import BaseStage
from lpacleaner.utils.line_detect import count_horizontal_lines

logger = logging.getLogger(__name__)

_FOCUS_THRESHOLD_DEFAULT = 100.0
_HORIZONTAL_LINE_MIN_COUNT = 5


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

        # Content-based orientation: try the current image and a 90° CCW
        # rotation, pick the one with more horizontal line segments
        # (staff lines should be horizontal in the correct orientation).
        img, rotation, method = _orient_by_content(img)

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


def _orient_by_content(img: np.ndarray) -> tuple[np.ndarray, int, str]:
    """Orient image so that staff lines are horizontal.

    Tries the image as-is and rotated 90° CCW. Picks whichever has more
    horizontal line segments. If neither exceeds the minimum threshold,
    falls back to portrait enforcement (taller than wider).

    Returns (oriented_image, degrees_applied, method_name).
    """
    # Downscale for speed -- line detection doesn't need full resolution
    scale = 1.0
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1200:
        scale = 1200.0 / max_dim
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img

    h_lines_0 = count_horizontal_lines(small)

    rotated_small = cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h_lines_90 = count_horizontal_lines(rotated_small)

    logger.debug(
        "Orientation check: h_lines(0°)=%d, h_lines(90°)=%d",
        h_lines_0, h_lines_90,
    )

    # If one orientation has clearly more horizontal lines, use it.
    # Require a 2:1 ratio to be confident -- a weak ratio means the
    # background (desk, table) is contributing lines and the signal is
    # ambiguous.
    max_lines = max(h_lines_0, h_lines_90)
    min_lines = max(min(h_lines_0, h_lines_90), 1)
    ratio = max_lines / min_lines

    if max_lines >= _HORIZONTAL_LINE_MIN_COUNT and ratio >= 2.0:
        if h_lines_90 > h_lines_0:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 90, "staff_lines"
        else:
            return img, 0, "staff_lines"

    # Fallback: neither orientation has enough staff lines (cover, text
    # page, blank). Enforce portrait (taller than wider).
    if w > h:
        logger.info("No staff lines detected; falling back to portrait enforcement")
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 90, "portrait_fallback"

    return img, 0, "portrait_fallback"


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
