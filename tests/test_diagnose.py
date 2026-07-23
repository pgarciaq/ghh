"""Tests for ghh diagnose — sidecar metadata aggregation and smart warnings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ghh.cli import main
from ghh.diagnose import (
    FieldStats,
    aggregate_checkpoint,
    diagnose,
    format_summary,
    format_summary_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_sidecars(stage_dir: Path, records: list[dict]) -> None:
    """Write sidecar JSON files named IMG_0001.json, IMG_0002.json, …"""
    stage_dir.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(records, start=1):
        path = stage_dir / f"IMG_{i:04d}.json"
        path.write_text(json.dumps(rec))


# ---------------------------------------------------------------------------
# FieldStats
# ---------------------------------------------------------------------------

class TestFieldStats:
    def test_string_values(self):
        fs = FieldStats(name="method")
        fs.add("hough_border")
        fs.add("ink_density")
        fs.add("ink_density")
        assert fs.dominant_type == "str"
        assert fs.str_counts["ink_density"] == 2
        assert fs.str_counts["hough_border"] == 1

    def test_bool_values(self):
        fs = FieldStats(name="is_blurry")
        fs.add(True)
        fs.add(False)
        fs.add(False)
        assert fs.dominant_type == "bool"
        assert fs.bool_counts[True] == 1
        assert fs.bool_counts[False] == 2

    def test_numeric_values(self):
        fs = FieldStats(name="skew_angle")
        for v in [0.5, -1.2, 0.8, 2.0]:
            fs.add(v)
        assert fs.dominant_type == "number"
        assert fs.num_min == pytest.approx(-1.2)
        assert fs.num_max == pytest.approx(2.0)
        assert fs.num_mean == pytest.approx(0.525)

    def test_list_values_skipped(self):
        fs = FieldStats(name="quad_corners")
        fs.add([[0, 0], [100, 0], [100, 200], [0, 200]])
        assert fs.dominant_type == "list"
        assert len(fs.num_values) == 0

    def test_mixed_types(self):
        fs = FieldStats(name="mixed")
        fs.add("hello")
        fs.add(42)
        assert fs.count == 2
        assert fs.type_counts["str"] == 1
        assert fs.type_counts["number"] == 1

    def test_empty(self):
        fs = FieldStats(name="empty")
        assert fs.dominant_type == "unknown"
        assert fs.num_min is None
        assert fs.num_max is None
        assert fs.num_mean is None


# ---------------------------------------------------------------------------
# aggregate_checkpoint
# ---------------------------------------------------------------------------

class TestAggregateCheckpoint:
    def test_basic(self, tmp_path):
        stage_dir = tmp_path / "06_content"
        _write_sidecars(stage_dir, [
            {"stage": "content_area", "method": "ink_density", "margin_px": 1},
            {"stage": "content_area", "method": "hough_border", "margin_px": 5},
            {"stage": "content_area", "method": "ink_density", "margin_px": 2},
        ])
        fields = aggregate_checkpoint(stage_dir)

        assert "stage" in fields
        assert fields["stage"].str_counts["content_area"] == 3
        assert fields["method"].str_counts["ink_density"] == 2
        assert fields["method"].str_counts["hough_border"] == 1
        assert fields["margin_px"].num_min == 1
        assert fields["margin_px"].num_max == 5

    def test_empty_dir(self, tmp_path):
        stage_dir = tmp_path / "06_content"
        stage_dir.mkdir()
        fields = aggregate_checkpoint(stage_dir)
        assert fields == {}

    def test_malformed_json_skipped(self, tmp_path):
        stage_dir = tmp_path / "06_content"
        stage_dir.mkdir()
        (stage_dir / "IMG_0001.json").write_text("{not valid json")
        (stage_dir / "IMG_0002.json").write_text('{"stage": "ok"}')
        fields = aggregate_checkpoint(stage_dir)
        assert fields["stage"].count == 1

    def test_heterogeneous_keys(self, tmp_path):
        stage_dir = tmp_path / "07_staff_extract"
        _write_sidecars(stage_dir, [
            {"stage": "s", "staff_extract_action": "cropped", "staff_extract_coverage": 0.8},
            {"stage": "s", "staff_extract_action": "passthrough",
             "staff_extract_reason": "too_small"},
        ])
        fields = aggregate_checkpoint(stage_dir)
        assert fields["staff_extract_action"].str_counts["cropped"] == 1
        assert fields["staff_extract_action"].str_counts["passthrough"] == 1
        assert "staff_extract_reason" in fields
        assert fields["staff_extract_reason"].count == 1
        assert fields["staff_extract_coverage"].count == 1


# ---------------------------------------------------------------------------
# diagnose (integration)
# ---------------------------------------------------------------------------

class TestDiagnose:
    def test_root_checkpoints(self, tmp_path):
        _write_sidecars(tmp_path / "00_preprocessed", [
            {"stage": "preprocess", "hotspot_detected": False},
            {"stage": "preprocess", "hotspot_detected": True},
        ])
        _write_sidecars(tmp_path / "06_content", [
            {"stage": "content_area", "method": "ink_density", "margin_px": 1},
        ])
        summaries = diagnose(tmp_path)
        assert len(summaries) == 2
        assert summaries[0].checkpoint_name == "00_preprocessed"
        assert summaries[0].branch is None
        assert summaries[1].checkpoint_name == "06_content"

    def test_branch_checkpoints(self, tmp_path):
        _write_sidecars(tmp_path / "05_gentle_crop", [
            {"stage": "gentle_crop", "method": "bbox_crop"},
        ])
        _write_sidecars(tmp_path / "book" / "08_deskewed", [
            {"stage": "deskew", "skew_angle": 0.5},
        ])
        _write_sidecars(tmp_path / "score" / "06_content", [
            {"stage": "content_area", "method": "ink_density", "margin_px": 1},
        ])
        summaries = diagnose(tmp_path)
        assert len(summaries) == 3
        names = [(s.checkpoint_name, s.branch) for s in summaries]
        assert ("05_gentle_crop", None) in names
        assert ("08_deskewed", "book") in names
        assert ("06_content", "score") in names

    def test_stage_filter(self, tmp_path):
        _write_sidecars(tmp_path / "00_preprocessed", [{"stage": "preprocess"}])
        _write_sidecars(tmp_path / "06_content", [{"stage": "content_area"}])
        summaries = diagnose(tmp_path, stage_filter=6)
        assert len(summaries) == 1
        assert summaries[0].checkpoint_name == "06_content"

    def test_empty_output(self, tmp_path):
        summaries = diagnose(tmp_path)
        assert summaries == []

    def test_non_checkpoint_dirs_ignored(self, tmp_path):
        (tmp_path / "some_random_dir").mkdir()
        (tmp_path / "book.toml").write_text("")
        summaries = diagnose(tmp_path)
        assert summaries == []


# ---------------------------------------------------------------------------
# Smart warnings
# ---------------------------------------------------------------------------

class TestWarnings:
    def test_preprocess_hotspot(self, tmp_path):
        _write_sidecars(tmp_path / "00_preprocessed", [
            {"stage": "preprocess", "hotspot_detected": True},
            {"stage": "preprocess", "hotspot_detected": True},
            {"stage": "preprocess", "hotspot_detected": False},
        ])
        summaries = diagnose(tmp_path)
        assert any("hotspot" in w.lower() for w in summaries[0].warnings)

    def test_orientation_blurry(self, tmp_path):
        records = [{"stage": "orientation", "is_blurry": True, "rotation_applied": 0}] * 5 + \
                  [{"stage": "orientation", "is_blurry": False, "rotation_applied": 0}] * 5
        _write_sidecars(tmp_path / "02_oriented", records)
        summaries = diagnose(tmp_path)
        assert any("blurry" in w.lower() for w in summaries[0].warnings)

    def test_content_area_noop(self, tmp_path):
        _write_sidecars(tmp_path / "06_content", [
            {"stage": "content_area", "method": "ink_density", "margin_px": 1},
            {"stage": "content_area", "method": "ink_density", "margin_px": 1},
        ])
        summaries = diagnose(tmp_path)
        warnings_text = " ".join(summaries[0].warnings)
        assert "ink_density" in warnings_text.lower() or "no-op" in warnings_text.lower()

    def test_content_area_warnings_not_on_staff_extract(self, tmp_path):
        """Inherited stage field should NOT trigger content area warnings."""
        _write_sidecars(tmp_path / "07_staff_extract", [
            {"stage": "content_area", "method": "ink_density", "margin_px": 1,
             "staff_extract_action": "cropped", "staff_extract_coverage": 0.8},
        ])
        summaries = diagnose(tmp_path)
        assert not any("ink_density fallback" in w for w in summaries[0].warnings)

    def test_staff_extract_high_passthrough(self, tmp_path):
        records = [
            {"stage": "s", "staff_extract_action": "passthrough",
             "staff_extract_reason": "staff_region_too_small"},
        ] * 4 + [
            {"stage": "s", "staff_extract_action": "cropped",
             "staff_extract_coverage": 0.6},
        ] * 6
        _write_sidecars(tmp_path / "07_staff_extract", records)
        summaries = diagnose(tmp_path)
        assert any("passthrough" in w.lower() or "passed through" in w.lower()
                    for w in summaries[0].warnings)

    def test_deskew_extreme_angles(self, tmp_path):
        records = [
            {"stage": "deskew", "skew_angle": 4.5},
            {"stage": "deskew", "skew_angle": 0.2},
            {"stage": "deskew", "skew_angle": -0.3},
        ]
        _write_sidecars(tmp_path / "08_deskewed", records)
        summaries = diagnose(tmp_path)
        assert any("extreme" in w.lower() or ">3°" in w for w in summaries[0].warnings)

    def test_deskew_systematic_tilt(self, tmp_path):
        records = [{"stage": "deskew", "skew_angle": 1.5}] * 10
        _write_sidecars(tmp_path / "08_deskewed", records)
        summaries = diagnose(tmp_path)
        assert any("systematic" in w.lower() or "camera" in w.lower()
                    for w in summaries[0].warnings)

    def test_stitch_failures(self, tmp_path):
        _write_sidecars(tmp_path / "01_stitched", [
            {"stage": "stitch", "stitch_success": False},
            {"stage": "stitch", "stitch_success": True},
        ])
        summaries = diagnose(tmp_path)
        assert any("failed" in w.lower() for w in summaries[0].warnings)

    def test_no_warnings_when_clean(self, tmp_path):
        _write_sidecars(tmp_path / "08_deskewed", [
            {"stage": "deskew", "skew_angle": 0.3},
            {"stage": "deskew", "skew_angle": -0.2},
            {"stage": "deskew", "skew_angle": 0.1},
        ])
        summaries = diagnose(tmp_path)
        assert summaries[0].warnings == []


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_format_summary_text(self, tmp_path):
        _write_sidecars(tmp_path / "08_deskewed", [
            {"stage": "deskew", "method": "staff_lines", "skew_angle": 0.5},
            {"stage": "deskew", "method": "staff_lines", "skew_angle": -0.3},
        ])
        summaries = diagnose(tmp_path)
        text = format_summary(summaries[0])
        assert "08_deskewed" in text
        assert "2 images" in text
        assert "staff_lines" in text
        assert "skew_angle" in text

    def test_format_summary_json(self, tmp_path):
        _write_sidecars(tmp_path / "06_content", [
            {"stage": "content_area", "method": "ink_density", "margin_px": 1},
        ])
        summaries = diagnose(tmp_path)
        data = format_summary_json(summaries[0])
        assert data["checkpoint"] == "06_content"
        assert data["n_files"] == 1
        assert "method" in data["fields"]
        assert data["fields"]["method"]["distribution"]["ink_density"] == 1

    def test_format_branch_label(self, tmp_path):
        _write_sidecars(tmp_path / "score" / "06_content", [
            {"stage": "content_area", "method": "ink_density"},
        ])
        summaries = diagnose(tmp_path)
        text = format_summary(summaries[0])
        assert "[score branch]" in text

    def test_constant_numeric_shows_all_same(self, tmp_path):
        _write_sidecars(tmp_path / "08_deskewed", [
            {"stage": "deskew", "skew_angle": 0.0},
            {"stage": "deskew", "skew_angle": 0.0},
        ])
        summaries = diagnose(tmp_path)
        text = format_summary(summaries[0])
        assert "all same" in text


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestCLI:
    def test_diagnose_text(self, tmp_path):
        _write_sidecars(tmp_path / "06_content", [
            {"stage": "content_area", "method": "ink_density", "margin_px": 1},
        ])
        runner = CliRunner()
        result = runner.invoke(main, ["diagnose", str(tmp_path)])
        assert result.exit_code == 0
        assert "06_content" in result.output
        assert "ink_density" in result.output

    def test_diagnose_json(self, tmp_path):
        _write_sidecars(tmp_path / "06_content", [
            {"stage": "content_area", "method": "ink_density"},
        ])
        runner = CliRunner()
        result = runner.invoke(main, ["diagnose", str(tmp_path), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data[0]["checkpoint"] == "06_content"

    def test_diagnose_stage_filter(self, tmp_path):
        _write_sidecars(tmp_path / "00_preprocessed", [{"stage": "preprocess"}])
        _write_sidecars(tmp_path / "06_content", [{"stage": "content_area"}])
        runner = CliRunner()
        result = runner.invoke(main, ["diagnose", str(tmp_path), "--stage", "6"])
        assert result.exit_code == 0
        assert "06_content" in result.output
        assert "00_preprocessed" not in result.output

    def test_diagnose_empty(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["diagnose", str(tmp_path)])
        assert result.exit_code == 0
        assert "No checkpoint" in result.output
