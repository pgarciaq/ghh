"""Shared fixtures: synthetic test images, temporary directories, Config factory."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------

def make_music_page(
    width: int = 800,
    height: int = 600,
    staff_color: tuple[int, int, int] = (0, 0, 200),
    num_staves: int = 4,
    skew_deg: float = 0.0,
    curve_amount: float = 0.0,
    noise_level: int = 0,
    bg_color: tuple[int, int, int] = (230, 220, 200),
) -> np.ndarray:
    """Generate a synthetic music page with staff lines, text placeholders, and border.

    Staff lines are drawn as horizontal lines (5 per stave, grouped into
    ``num_staves`` stave systems). Notes are drawn as small filled circles on
    the lines. A thin border rectangle is drawn around the page.

    Returns a BGR uint8 image.
    """
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    margin_x = int(width * 0.08)
    margin_y = int(height * 0.08)
    usable_h = height - 2 * margin_y
    usable_w = width - 2 * margin_x

    # Border rectangle (same color as staff ink)
    cv2.rectangle(img, (margin_x, margin_y),
                  (width - margin_x, height - margin_y), staff_color, 2)

    # Staff lines: 5 lines per stave, evenly spaced staves
    line_spacing = max(3, usable_h // (num_staves * 8))
    stave_height = line_spacing * 4
    stave_gap = (usable_h - num_staves * stave_height) // max(num_staves + 1, 1)

    for s in range(num_staves):
        stave_top = margin_y + stave_gap * (s + 1) + stave_height * s
        for line_idx in range(5):
            y = stave_top + line_idx * line_spacing
            if curve_amount > 0:
                pts = []
                for x in range(margin_x, width - margin_x, 4):
                    dy = int(curve_amount * np.sin(np.pi * (x - margin_x) / usable_w) * line_spacing)
                    pts.append([x, y + dy])
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(img, [pts], isClosed=False, color=staff_color, thickness=2)
            else:
                cv2.line(img, (margin_x, y), (width - margin_x, y), staff_color, 2)

        # Notes: small filled circles at roughly regular intervals
        note_y_base = stave_top + 2 * line_spacing
        for nx in range(margin_x + 30, width - margin_x - 30, usable_w // 10):
            note_y = note_y_base + np.random.randint(-line_spacing, line_spacing + 1)
            cv2.circle(img, (nx, note_y), line_spacing // 2, (20, 20, 20), -1)

    # Optional skew
    if abs(skew_deg) > 0.01:
        center = (width / 2, height / 2)
        mat = cv2.getRotationMatrix2D(center, skew_deg, 1.0)
        img = cv2.warpAffine(img, mat, (width, height),
                             borderValue=bg_color, flags=cv2.INTER_LINEAR)

    # Optional noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def make_text_page(
    width: int = 800,
    height: int = 600,
    bg_color: tuple[int, int, int] = (230, 220, 200),
    text_color: tuple[int, int, int] = (30, 30, 30),
    num_lines: int = 20,
) -> np.ndarray:
    """Generate a synthetic text-only page with horizontal dark lines (simulating text).

    Returns a BGR uint8 image.
    """
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    margin_x = int(width * 0.1)
    margin_y = int(height * 0.08)
    usable_h = height - 2 * margin_y
    line_gap = usable_h // (num_lines + 1)

    for i in range(1, num_lines + 1):
        y = margin_y + i * line_gap
        line_len = width - 2 * margin_x - np.random.randint(0, int(width * 0.15))
        cv2.line(img, (margin_x, y), (margin_x + line_len, y), text_color, 2)

    return img


def make_page_on_background(
    page: np.ndarray,
    angle: float = 0.0,
    perspective_skew: float = 0.0,
    bg: tuple[int, int, int] = (40, 30, 25),
    border: int = 80,
) -> np.ndarray:
    """Place a page image on a dark background with optional rotation and perspective.

    Simulates a photographed page on a table.
    Returns a BGR uint8 image larger than the input page.
    """
    ph, pw = page.shape[:2]
    out_h = ph + 2 * border
    out_w = pw + 2 * border
    canvas = np.full((out_h, out_w, 3), bg, dtype=np.uint8)

    canvas[border:border + ph, border:border + pw] = page

    if abs(angle) > 0.01 or abs(perspective_skew) > 0.01:
        center = (out_w / 2, out_h / 2)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        canvas = cv2.warpAffine(canvas, mat, (out_w, out_h),
                                borderValue=bg, flags=cv2.INTER_LINEAR)

    if abs(perspective_skew) > 0.01:
        s = perspective_skew
        src = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
        dst = np.float32([
            [s * out_w, s * out_h],
            [out_w - s * out_w, 0],
            [out_w, out_h],
            [0, out_h - s * out_h],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        canvas = cv2.warpPerspective(canvas, M, (out_w, out_h),
                                     borderValue=bg, flags=cv2.INTER_LINEAR)

    return canvas


def add_finger(
    img: np.ndarray,
    position: str = "top-right",
    size_frac: float = 0.05,
) -> np.ndarray:
    """Draw a skin-colored ellipse on the image border simulating a finger."""
    out = img.copy()
    h, w = out.shape[:2]
    size = int(max(h, w) * size_frac)

    positions = {
        "top-right": (w - size // 2, size // 2),
        "top-left": (size // 2, size // 2),
        "bottom-right": (w - size // 2, h - size // 2),
        "bottom-left": (size // 2, h - size // 2),
    }
    center = positions.get(position, positions["top-right"])
    skin_color = (130, 160, 200)  # BGR skin-ish tone
    cv2.ellipse(out, center, (size, int(size * 1.5)), 0, 0, 360, skin_color, -1)
    return out


def add_barrel_distortion(img: np.ndarray, k1: float = 0.3) -> np.ndarray:
    """Apply synthetic barrel distortion with known k1 coefficient."""
    h, w = img.shape[:2]
    fx = fy = w
    cx, cy = w / 2, h / 2
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None,
                                              camera_matrix, (w, h), cv2.CV_32FC1)
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)


def add_hotspot(img: np.ndarray, center: tuple[int, int] | None = None,
                radius: int = 40) -> np.ndarray:
    """Add a bright white circular hotspot (simulating flash reflection)."""
    out = img.copy()
    h, w = out.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    cv2.circle(out, center, radius, (255, 255, 255), -1)
    cv2.GaussianBlur(out, (0, 0), radius / 3, dst=out)
    # re-apply hard white center
    cv2.circle(out, center, radius // 2, (255, 255, 255), -1)
    return out


def blur_image(img: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Apply Gaussian blur simulating an out-of-focus photo."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(img, (k, k), 0)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def music_page():
    """800x600 synthetic music page with red staff lines."""
    return make_music_page()


@pytest.fixture
def text_page():
    """800x600 synthetic text-only page."""
    return make_text_page()


@pytest.fixture
def music_page_on_bg(music_page):
    """Music page placed on a dark background (simulating a photo)."""
    return make_page_on_background(music_page)


@pytest.fixture
def sample_jpeg(tmp_path, music_page) -> Path:
    """Save a music page as JPEG with EXIF orientation=1 (normal)."""
    path = tmp_path / "test_image.jpg"
    pil_img = Image.fromarray(cv2.cvtColor(music_page, cv2.COLOR_BGR2RGB))
    exif = pil_img.getexif()
    exif[0x0112] = 1  # Orientation tag = normal
    pil_img.save(path, "JPEG", quality=95, exif=exif.tobytes())
    return path


@pytest.fixture
def sample_jpeg_rotated_90cw(tmp_path, music_page) -> Path:
    """Save a music page as JPEG with EXIF orientation=6 (90 CW).

    The pixel data is stored rotated 90 CCW, and EXIF says to rotate 90 CW
    to view correctly — this matches how cameras save portrait photos.
    """
    rotated_pixels = cv2.rotate(music_page, cv2.ROTATE_90_COUNTERCLOCKWISE)
    path = tmp_path / "rotated_90cw.jpg"
    pil_img = Image.fromarray(cv2.cvtColor(rotated_pixels, cv2.COLOR_BGR2RGB))
    exif = pil_img.getexif()
    exif[0x0112] = 6  # Orientation = 90 CW
    pil_img.save(path, "JPEG", quality=95, exif=exif.tobytes())
    return path


@pytest.fixture
def sample_png(tmp_path, music_page) -> Path:
    """Save a music page as PNG (no EXIF)."""
    path = tmp_path / "test_image.png"
    cv2.imwrite(str(path), music_page)
    return path
