"""Aggregate sidecar metadata from checkpoint directories into per-stage summaries.

The ``ghh diagnose`` command reads all ``.json`` sidecar files produced by
pipeline stages and prints a statistical summary for each checkpoint.  This
is purely a read-only post-hoc analysis tool—no images are opened or
modified.

The aggregation is **fully generic**: for every JSON key found across the
sidecars, string/bool values are counted, numeric values get min/max/mean
statistics, and list/dict values are silently skipped.

On top of the generic stats, a set of **smart warnings** flag common
problems (e.g., a content-area stage that crops nothing, a staff-extract
stage with a high passthrough rate, many blurry images, etc.).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FieldStats:
    """Aggregated statistics for a single JSON key across all sidecars."""

    name: str
    count: int = 0
    str_counts: Counter = field(default_factory=Counter)
    bool_counts: Counter = field(default_factory=Counter)
    num_values: list[float] = field(default_factory=list)
    type_counts: Counter = field(default_factory=Counter)

    def add(self, value) -> None:
        self.count += 1
        if isinstance(value, bool):
            self.type_counts["bool"] += 1
            self.bool_counts[value] += 1
        elif isinstance(value, str):
            self.type_counts["str"] += 1
            self.str_counts[value] += 1
        elif isinstance(value, (int, float)):
            self.type_counts["number"] += 1
            self.num_values.append(float(value))
        elif isinstance(value, list):
            self.type_counts["list"] += 1
        elif isinstance(value, dict):
            self.type_counts["dict"] += 1
        else:
            self.type_counts["other"] += 1

    @property
    def dominant_type(self) -> str:
        if not self.type_counts:
            return "unknown"
        return self.type_counts.most_common(1)[0][0]

    @property
    def num_min(self) -> float | None:
        return min(self.num_values) if self.num_values else None

    @property
    def num_max(self) -> float | None:
        return max(self.num_values) if self.num_values else None

    @property
    def num_mean(self) -> float | None:
        return sum(self.num_values) / len(self.num_values) if self.num_values else None


@dataclass
class CheckpointSummary:
    """Summary for one checkpoint directory."""

    checkpoint_name: str
    branch: str | None
    n_files: int
    fields: dict[str, FieldStats]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate_checkpoint(ckpt_dir: Path) -> dict[str, FieldStats]:
    """Read all .json sidecars in *ckpt_dir* and aggregate field statistics."""
    fields: dict[str, FieldStats] = {}

    for json_path in sorted(ckpt_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", json_path, exc)
            continue

        if not isinstance(data, dict):
            continue

        for key, value in data.items():
            if key not in fields:
                fields[key] = FieldStats(name=key)
            fields[key].add(value)

    return fields


def _find_checkpoints(
    output_dir: Path,
    stage_filter: int | None = None,
) -> list[tuple[Path, str | None]]:
    """Discover checkpoint directories, including branch subdirs.

    Returns a list of ``(directory, branch_label)`` tuples.
    ``branch_label`` is ``None`` for root-level checkpoints, or
    ``"book"`` / ``"score"`` for branch checkpoints.
    """
    results: list[tuple[Path, str | None]] = []

    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name in ("book", "score"):
            for sub in sorted(d.iterdir()):
                if sub.is_dir() and _is_checkpoint(sub, stage_filter):
                    results.append((sub, name))
        elif _is_checkpoint(d, stage_filter):
            results.append((d, None))

    return results


def _is_checkpoint(d: Path, stage_filter: int | None) -> bool:
    """Return True if *d* looks like a checkpoint directory."""
    name = d.name
    if not any(d.glob("*.json")):
        return False
    if stage_filter is not None:
        prefix = f"{stage_filter:02d}_"
        return name.startswith(prefix)
    return name[:2].isdigit() and name[2] == "_"


def diagnose(
    output_dir: Path,
    stage_filter: int | None = None,
) -> list[CheckpointSummary]:
    """Run full diagnosis on an output directory.

    Returns a list of :class:`CheckpointSummary` objects.
    """
    checkpoints = _find_checkpoints(output_dir, stage_filter)
    summaries: list[CheckpointSummary] = []

    for ckpt_path, branch in checkpoints:
        n_files = len(list(ckpt_path.glob("*.json")))
        fields = aggregate_checkpoint(ckpt_path)
        warnings = _generate_warnings(ckpt_path.name, fields, n_files)

        summaries.append(CheckpointSummary(
            checkpoint_name=ckpt_path.name,
            branch=branch,
            n_files=n_files,
            fields=fields,
            warnings=warnings,
        ))

    return summaries


# ---------------------------------------------------------------------------
# Smart warnings
# ---------------------------------------------------------------------------

def _generate_warnings(
    checkpoint_name: str,
    fields: dict[str, FieldStats],
    n_files: int,
) -> list[str]:
    """Generate context-aware warnings based on field statistics.

    Uses the checkpoint directory name (e.g. ``"06_content"``) rather than
    the ``stage`` JSON field to decide which warnings to fire, because some
    stages inherit metadata from their predecessor and the ``stage`` key may
    not match the actual checkpoint.
    """
    warnings: list[str] = []
    if n_files == 0:
        warnings.append("No sidecar files found")
        return warnings

    # Derive the stage identity from the checkpoint dir name, not the
    # potentially-inherited "stage" JSON field.
    ckpt_suffix = checkpoint_name.lstrip("0123456789").lstrip("_")

    _warn_preprocess(warnings, fields, n_files, ckpt_suffix)
    _warn_orientation(warnings, fields, n_files, ckpt_suffix)
    _warn_page_detect(warnings, fields, n_files, ckpt_suffix)
    _warn_content_area(warnings, fields, n_files, ckpt_suffix)
    _warn_staff_extract(warnings, fields, n_files, ckpt_suffix)
    _warn_deskew(warnings, fields, n_files, ckpt_suffix)
    _warn_stitch(warnings, fields, n_files, ckpt_suffix)

    return warnings


def _warn_preprocess(w: list[str], fields: dict[str, FieldStats], n: int, ckpt: str) -> None:
    if ckpt != "preprocessed":
        return
    hotspot = fields.get("hotspot_detected")
    if hotspot and hotspot.bool_counts.get(True, 0) > 0:
        pct = 100.0 * hotspot.bool_counts[True] / n
        w.append(f"Flash hotspots detected on {hotspot.bool_counts[True]}/{n} images ({pct:.0f}%)")


def _warn_orientation(w: list[str], fields: dict[str, FieldStats], n: int, ckpt: str) -> None:
    if ckpt != "oriented":
        return
    blur = fields.get("is_blurry")
    if blur and blur.bool_counts.get(True, 0) > n * 0.1:
        pct = 100.0 * blur.bool_counts[True] / n
        w.append(
            f"{blur.bool_counts[True]}/{n} images flagged as blurry ({pct:.0f}%) "
            f"— consider checking focus_threshold calibration"
        )

    rot = fields.get("rotation_applied")
    if rot and len(rot.str_counts) + len(set(rot.num_values)) > 2:
        vals = rot.str_counts if rot.str_counts else Counter(int(v) for v in rot.num_values)
        dist = ", ".join(f"{k}°={c}" for k, c in vals.most_common())
        w.append(f"Mixed rotation values ({dist}) — some images may have inconsistent orientation")

    focus = fields.get("focus_score")
    if focus and focus.num_values:
        low = sum(1 for v in focus.num_values if v < 50.0)
        if low > n * 0.2:
            w.append(f"{low}/{n} images have very low focus scores (<50)")


def _warn_page_detect(w: list[str], fields: dict[str, FieldStats], n: int, ckpt: str) -> None:
    if ckpt != "page_detected":
        return
    pt = fields.get("page_type")
    if pt and pt.str_counts:
        dist = ", ".join(f"{k}={c}" for k, c in pt.str_counts.most_common())
        if len(pt.str_counts) > 1:
            w.append(f"Mixed page types detected ({dist})")

    method = fields.get("method")
    if method and method.str_counts.get("fallback", 0) > n * 0.2:
        w.append(
            f"{method.str_counts['fallback']}/{n} images used fallback detection "
            f"— page boundaries may be unreliable"
        )


def _warn_content_area(w: list[str], fields: dict[str, FieldStats], n: int, ckpt: str) -> None:
    if ckpt != "content":
        return
    rect = fields.get("content_rect")
    if rect and rect.type_counts.get("list", 0) > 0:
        pass

    method = fields.get("method")
    if method and method.str_counts:
        if method.str_counts.get("inset_fallback", 0) == n:
            w.append("All images used inset_fallback — no border frames or ink regions detected")
        if method.str_counts.get("ink_density", 0) == n:
            w.append(
                "All images used ink_density fallback — no border frame lines detected. "
                "If has_border_frame=false in book.toml, this is expected"
            )

    margin = fields.get("margin_px")
    if margin and margin.num_values:
        if margin.num_max is not None and margin.num_max <= 2:
            w.append(
                "Margin is \u22642px on all images \u2014 content rect likely "
                "covers the full page (stage may be a no-op)"
            )


def _warn_staff_extract(w: list[str], fields: dict[str, FieldStats], n: int, ckpt: str) -> None:
    if ckpt != "staff_extract":
        return
    action = fields.get("staff_extract_action")
    if not action or not action.str_counts:
        return
    passthrough = action.str_counts.get("passthrough", 0)
    if passthrough > n * 0.15:
        pct = 100.0 * passthrough / n
        w.append(
            f"{passthrough}/{n} images passed through ({pct:.0f}%) "
            "\u2014 staff detection may need tuning"
        )

        reason = fields.get("staff_extract_reason")
        if reason and reason.str_counts:
            dist = ", ".join(f"{k}={c}" for k, c in reason.str_counts.most_common())
            w.append(f"  Passthrough reasons: {dist}")

    coverage = fields.get("staff_extract_coverage")
    if coverage and coverage.num_values:
        full_page = sum(1 for v in coverage.num_values if v >= 0.95)
        if full_page > n * 0.2:
            w.append(
                f"{full_page}/{n} images have ≥95% coverage — staff extract "
                f"is keeping the full page on many images"
            )


def _warn_deskew(w: list[str], fields: dict[str, FieldStats], n: int, ckpt: str) -> None:
    if ckpt != "deskewed":
        return
    angle = fields.get("skew_angle")
    if angle and angle.num_values:
        extreme = sum(1 for v in angle.num_values if abs(v) > 3.0)
        if extreme > 0:
            w.append(f"{extreme}/{n} images have extreme skew (>3°)")
        if angle.num_mean is not None and abs(angle.num_mean) > 1.0:
            w.append(
                f"Mean skew angle is {angle.num_mean:.2f}° — "
                f"systematic tilt suggests a camera alignment issue"
            )


def _warn_stitch(w: list[str], fields: dict[str, FieldStats], n: int, ckpt: str) -> None:
    if ckpt != "stitched":
        return
    success = fields.get("stitch_success")
    if success and success.bool_counts.get(False, 0) > 0:
        w.append(
            f"{success.bool_counts[False]}/{n} stitch operations failed"
        )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_summary(summary: CheckpointSummary) -> str:
    """Format a single checkpoint summary as human-readable text."""
    lines: list[str] = []

    label = summary.checkpoint_name
    if summary.branch:
        label += f"  [{summary.branch} branch]"
    lines.append(f"=== {label} ({summary.n_files} images) ===")

    for key in sorted(summary.fields, key=_field_sort_key):
        fs = summary.fields[key]
        if key == "stage":
            continue

        if fs.dominant_type == "str":
            dist = ", ".join(f"{v}={c}" for v, c in fs.str_counts.most_common())
            lines.append(f"  {key}: {dist}")

        elif fs.dominant_type == "bool":
            true_c = fs.bool_counts.get(True, 0)
            false_c = fs.bool_counts.get(False, 0)
            lines.append(f"  {key}: True={true_c}  False={false_c}")

        elif fs.dominant_type == "number":
            if fs.num_min == fs.num_max:
                lines.append(f"  {key}: {_fmt_num(fs.num_min)} (all same)")
            else:
                lines.append(
                    f"  {key}: min={_fmt_num(fs.num_min)}  "
                    f"max={_fmt_num(fs.num_max)}  "
                    f"mean={_fmt_num(fs.num_mean)}"
                )

        # Skip list/dict/other types — not useful to aggregate

    if summary.warnings:
        lines.append("")
        for w in summary.warnings:
            lines.append(f"  ⚠ {w}")

    return "\n".join(lines)


def format_summary_json(summary: CheckpointSummary) -> dict:
    """Format a single checkpoint summary as a JSON-serializable dict."""
    result: dict = {
        "checkpoint": summary.checkpoint_name,
        "branch": summary.branch,
        "n_files": summary.n_files,
        "fields": {},
        "warnings": summary.warnings,
    }
    for key, fs in summary.fields.items():
        entry: dict = {"count": fs.count, "type": fs.dominant_type}
        if fs.dominant_type == "str":
            entry["distribution"] = dict(fs.str_counts.most_common())
        elif fs.dominant_type == "bool":
            entry["true"] = fs.bool_counts.get(True, 0)
            entry["false"] = fs.bool_counts.get(False, 0)
        elif fs.dominant_type == "number":
            entry["min"] = fs.num_min
            entry["max"] = fs.num_max
            entry["mean"] = fs.num_mean
        result["fields"][key] = entry
    return result


def _fmt_num(v: float | None) -> str:
    if v is None:
        return "?"
    if v == int(v) and abs(v) < 1e9:
        return str(int(v))
    return f"{v:.4f}"


def _field_sort_key(key: str) -> tuple[int, str]:
    """Sort fields: stage first, method/action early, then alphabetical."""
    priority = {
        "stage": 0, "method": 1, "stitch_method": 1,
        "staff_extract_action": 1, "page_type": 2,
    }
    return (priority.get(key, 10), key)
