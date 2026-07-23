"""Tests for Stage 5 (GentleCropStage): gentle bounding-box crop.

Covers the BaseStage contract, bounding-box crop with margin,
quad_corners propagation, passthrough on missing/degenerate quads,
metadata output, and integration via run().
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ghh.config import Config
from ghh.pipeline import BaseStage, PipelineState, StageResult
from tests.conftest import make_music_page, make_page_on_background

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path, **overrides) -> Config:
    defaults = {"input_dir": tmp_path}
    defaults.update(overrides)
    return Config(**defaults)


def _setup_stage4_output(
    tmp_path: Path,
    images: dict[str, tuple[np.ndarray, dict]],
) -> Path:
    input_dir = tmp_path / "04_page_detected"
    input_dir.mkdir()
    for name, (img, meta) in images.items():
        cv2.imwrite(str(input_dir / name), img)
        stem = Path(name).stem
        (input_dir / f"{stem}.json").write_text(json.dumps(meta))
    return input_dir


# ---------------------------------------------------------------------------
# TestGentleCropStageContract
# ---------------------------------------------------------------------------

class TestGentleCropStageContract:

    def test_has_correct_name(self):
        from ghh.stages.gentle_crop import GentleCropStage
        assert GentleCropStage().name == "gentle_crop"

    def test_has_correct_number(self):
        from ghh.stages.gentle_crop import GentleCropStage
        assert GentleCropStage().number == 5

    def test_has_correct_checkpoint_name(self):
        from ghh.stages.gentle_crop import GentleCropStage
        assert GentleCropStage().checkpoint_name == "05_gentle_crop"

    def test_is_base_stage_subclass(self):
        from ghh.stages.gentle_crop import GentleCropStage
        assert issubclass(GentleCropStage, BaseStage)

    def test_error_class_is_skippable(self):
        from ghh.stages.gentle_crop import GentleCropStage
        assert GentleCropStage().error_class == "skippable"

    def test_is_not_skippable_by_default(self):
        from ghh.stages.gentle_crop import GentleCropStage
        cfg = Config(input_dir=Path("/tmp"))
        assert GentleCropStage().should_skip(cfg) is False

    def test_registered_in_stage_registry(self):
        from ghh.stages import STAGE_BY_NUMBER
        assert 5 in STAGE_BY_NUMBER
        assert STAGE_BY_NUMBER[5].name == "gentle_crop"


# ---------------------------------------------------------------------------
# TestGentleCropPassthrough
# ---------------------------------------------------------------------------

class TestGentleCropPassthrough:

    def test_passthrough_when_no_quad(self):
        from ghh.stages.gentle_crop import GentleCropStage

        stage = GentleCropStage()
        img = make_music_page(width=600, height=400)
        cfg = Config(input_dir=Path("/tmp"))

        result, meta = stage.process_image(img, {}, cfg)
        np.testing.assert_array_equal(result, img)
        assert meta["method"] == "passthrough"

    def test_passthrough_when_quad_wrong_shape(self):
        from ghh.stages.gentle_crop import GentleCropStage

        stage = GentleCropStage()
        img = make_music_page(width=600, height=400)
        cfg = Config(input_dir=Path("/tmp"))

        metadata = {"quad_corners": [[0, 0], [100, 0], [100, 100]]}
        result, meta = stage.process_image(img, metadata, cfg)
        np.testing.assert_array_equal(result, img)
        assert meta["method"] == "passthrough"

    def test_forwards_page_type_on_passthrough(self):
        from ghh.stages.gentle_crop import GentleCropStage

        stage = GentleCropStage()
        img = make_music_page(width=600, height=400)
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {"page_type": "music"}, cfg)
        assert meta["page_type"] == "music"


# ---------------------------------------------------------------------------
# TestGentleCropBBox
# ---------------------------------------------------------------------------

class TestGentleCropBBox:

    def test_crops_to_quad_bbox_with_margin(self):
        from ghh.stages.gentle_crop import GentleCropStage

        stage = GentleCropStage()
        img = np.full((1000, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"), gentle_crop_margin_frac=0.0)

        quad = [[100, 200], [700, 200], [700, 800], [100, 800]]
        result, meta = stage.process_image(img, {"quad_corners": quad}, cfg)

        assert meta["method"] == "bbox_crop"
        assert result.shape[0] == 600  # y: 200..800
        assert result.shape[1] == 600  # x: 100..700

    def test_margin_expands_crop(self):
        from ghh.stages.gentle_crop import GentleCropStage

        stage = GentleCropStage()
        img = np.full((1000, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"), gentle_crop_margin_frac=0.05)

        quad = [[100, 200], [700, 200], [700, 800], [100, 800]]
        result, meta = stage.process_image(img, {"quad_corners": quad}, cfg)

        assert meta["method"] == "bbox_crop"
        assert result.shape[1] > 600
        assert result.shape[0] > 600

    def test_clamps_to_image_bounds(self):
        from ghh.stages.gentle_crop import GentleCropStage

        stage = GentleCropStage()
        img = np.full((500, 400, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"), gentle_crop_margin_frac=0.10)

        quad = [[10, 10], [390, 10], [390, 490], [10, 490]]
        result, meta = stage.process_image(img, {"quad_corners": quad}, cfg)

        assert meta["method"] == "bbox_crop"
        assert result.shape[0] <= 500
        assert result.shape[1] <= 400

    def test_full_image_quad_returns_full_image(self):
        from ghh.stages.gentle_crop import GentleCropStage

        stage = GentleCropStage()
        img = np.full((600, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"), gentle_crop_margin_frac=0.0)

        quad = [[0, 0], [800, 0], [800, 600], [0, 600]]
        result, meta = stage.process_image(img, {"quad_corners": quad}, cfg)

        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# TestQuadPropagation
# ---------------------------------------------------------------------------

class TestQuadPropagation:

    def test_quad_corners_shifted_by_crop_offset(self):
        from ghh.stages.gentle_crop import GentleCropStage

        stage = GentleCropStage()
        img = np.full((1000, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"), gentle_crop_margin_frac=0.0)

        quad = [[100, 200], [700, 200], [700, 800], [100, 800]]
        _, meta = stage.process_image(img, {"quad_corners": quad}, cfg)

        new_quad = np.array(meta["quad_corners"])
        assert new_quad.shape == (4, 2)
        np.testing.assert_allclose(new_quad[0], [0, 0], atol=1.0)
        np.testing.assert_allclose(new_quad[1], [600, 0], atol=1.0)
        np.testing.assert_allclose(new_quad[2], [600, 600], atol=1.0)
        np.testing.assert_allclose(new_quad[3], [0, 600], atol=1.0)

    def test_quad_corners_with_margin_still_valid(self):
        from ghh.stages.gentle_crop import GentleCropStage

        stage = GentleCropStage()
        img = np.full((1000, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"), gentle_crop_margin_frac=0.05)

        quad = [[100, 200], [700, 200], [700, 800], [100, 800]]
        result, meta = stage.process_image(img, {"quad_corners": quad}, cfg)

        new_quad = np.array(meta["quad_corners"])
        rh, rw = result.shape[:2]
        assert np.all(new_quad[:, 0] >= 0)
        assert np.all(new_quad[:, 1] >= 0)
        assert np.all(new_quad[:, 0] <= rw)
        assert np.all(new_quad[:, 1] <= rh)


# ---------------------------------------------------------------------------
# TestGentleCropMetadata
# ---------------------------------------------------------------------------

class TestGentleCropMetadata:

    def test_metadata_has_required_fields(self):
        from ghh.stages.gentle_crop import GentleCropStage

        stage = GentleCropStage()
        img = np.full((600, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        quad = [[20, 20], [780, 20], [780, 580], [20, 580]]
        _, meta = stage.process_image(img, {"quad_corners": quad}, cfg)

        assert meta["stage"] == "gentle_crop"
        assert meta["method"] == "bbox_crop"
        assert "crop_box" in meta
        assert "margin_frac" in meta
        assert "quad_corners" in meta
        assert len(meta["crop_box"]) == 4

    def test_forwards_page_type(self):
        from ghh.stages.gentle_crop import GentleCropStage

        stage = GentleCropStage()
        img = np.full((600, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        metadata = {
            "quad_corners": [[20, 20], [780, 20], [780, 580], [20, 580]],
            "page_type": "music",
        }
        _, meta = stage.process_image(img, metadata, cfg)
        assert meta["page_type"] == "music"


# ---------------------------------------------------------------------------
# TestGentleCropStageRun (integration)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGentleCropStageRun:

    def test_produces_checkpoint_directory(self, tmp_path):
        from ghh.stages.gentle_crop import GentleCropStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=30)
        quad = [[30, 30], [630, 30], [630, 430], [30, 430]]
        s4_meta = {"stage": "page_detect", "quad_corners": quad}

        input_dir = _setup_stage4_output(
            tmp_path, {"IMG_0001.png": (photo, s4_meta)},
        )
        cfg = _cfg(tmp_path, input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = GentleCropStage()

        stage.run(input_dir, tmp_path, cfg, state)
        assert (tmp_path / "05_gentle_crop").exists()

    def test_processes_multiple_images(self, tmp_path):
        from ghh.stages.gentle_crop import GentleCropStage

        quad = [[30, 30], [630, 30], [630, 430], [30, 430]]
        s4_meta = {"stage": "page_detect", "quad_corners": quad}

        images = {}
        for i in range(3):
            page = make_music_page(width=600, height=400)
            photo = make_page_on_background(page, border=30)
            images[f"IMG_{i:04d}.png"] = (photo, s4_meta)

        input_dir = _setup_stage4_output(tmp_path, images)
        cfg = _cfg(tmp_path, input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = GentleCropStage()

        result = stage.run(input_dir, tmp_path, cfg, state)
        assert result.processed == 3
        for i in range(3):
            assert (tmp_path / "05_gentle_crop" / f"IMG_{i:04d}.png").exists()

    def test_writes_metadata_sidecar(self, tmp_path):
        from ghh.stages.gentle_crop import GentleCropStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=30)
        quad = [[30, 30], [630, 30], [630, 430], [30, 430]]
        s4_meta = {"stage": "page_detect", "quad_corners": quad}

        input_dir = _setup_stage4_output(
            tmp_path, {"IMG_0001.png": (photo, s4_meta)},
        )
        cfg = _cfg(tmp_path, input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = GentleCropStage()

        stage.run(input_dir, tmp_path, cfg, state)

        sidecar = tmp_path / "05_gentle_crop" / "IMG_0001.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["stage"] == "gentle_crop"
        assert "quad_corners" in meta

    def test_sidecar_quad_is_downstream_consumable(self, tmp_path):
        """quad_corners in output sidecar should be valid for downstream stages."""
        from ghh.stages.gentle_crop import GentleCropStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=30)
        quad = [[30, 30], [630, 30], [630, 430], [30, 430]]
        s4_meta = {"stage": "page_detect", "quad_corners": quad}

        input_dir = _setup_stage4_output(
            tmp_path, {"IMG_0001.png": (photo, s4_meta)},
        )
        cfg = _cfg(
            tmp_path, input_dir=input_dir, output_dir=tmp_path,
            gentle_crop_margin_frac=0.0,
        )
        state = PipelineState(tmp_path)
        stage = GentleCropStage()
        stage.run(input_dir, tmp_path, cfg, state)

        sidecar = tmp_path / "05_gentle_crop" / "IMG_0001.json"
        meta = json.loads(sidecar.read_text())
        new_quad = np.array(meta["quad_corners"])
        assert new_quad.shape == (4, 2)
        assert np.all(new_quad >= 0)

    def test_returns_stage_result(self, tmp_path):
        from ghh.stages.gentle_crop import GentleCropStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=30)
        quad = [[30, 30], [630, 30], [630, 430], [30, 430]]
        s4_meta = {"stage": "page_detect", "quad_corners": quad}

        input_dir = _setup_stage4_output(
            tmp_path, {"IMG_0001.png": (photo, s4_meta)},
        )
        cfg = _cfg(tmp_path, input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = GentleCropStage()

        result = stage.run(input_dir, tmp_path, cfg, state)
        assert isinstance(result, StageResult)
        assert result.stage_name == "gentle_crop"
        assert result.processed == 1
        assert result.failed == 0

    def test_resume_skips_completed(self, tmp_path):
        from ghh.stages.gentle_crop import GentleCropStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=30)
        quad = [[30, 30], [630, 30], [630, 430], [30, 430]]
        s4_meta = {"stage": "page_detect", "quad_corners": quad}

        input_dir = _setup_stage4_output(
            tmp_path, {"IMG_0001.png": (photo, s4_meta)},
        )
        cfg = _cfg(tmp_path, input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = GentleCropStage()

        r1 = stage.run(input_dir, tmp_path, cfg, state)
        assert r1.processed == 1

        r2 = stage.run(input_dir, tmp_path, cfg, state)
        assert r2.processed == 0
        assert r2.skipped == 1
