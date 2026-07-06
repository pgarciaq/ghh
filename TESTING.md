# Testing Strategy (TDD)

Development follows strict red-green-refactor TDD: write a failing test
first, implement just enough code to make it pass, then refactor.

Extracted from [PLAN.md](PLAN.md) to keep the technical spec focused on
stage specifications and pipeline architecture.

---

### Project Structure

```
tests/
  conftest.py               # Shared fixtures: test images, temp dirs, Config factory
  fixtures/                  # Synthetic test images (generated, not real photos)
    music_page_4x3.png       # Synthetic music page with red staff lines
    text_page_4x3.png        # Synthetic text-only page
    rotated_90cw.jpg         # Same page with EXIF orientation=6
    partial_top.png           # Top half of a page (for stitch testing)
    partial_bottom.png        # Bottom half with overlap
    blurry_page.png           # Synthetically blurred page
    finger_on_edge.png        # Page with skin-colored region at border
    barrel_distorted.png      # Synthetically distorted with known k1
    book_cover.png            # Dark uniform image (non-content)
  test_image_io.py
  test_line_detect.py
  test_geometry.py
  test_preprocess.py
  test_accel.py
  test_config.py
  test_stage_stitch.py
  test_stage_orientation.py
  test_stage_lens_correct.py
  test_stage_page_detect.py
  test_stage_perspective.py
  test_stage_content_area.py
  test_stage_dewarp.py
  test_stage_deskew.py
  test_stage_enhance.py
  test_stage_normalize.py
  test_stage_ocr.py
  test_stage_pdf_assembly.py
  test_pipeline.py           # Integration tests
  test_cli.py                # CLI invocation tests
```

### Test Fixtures: Synthetic Images

Test images are **generated programmatically**, not extracted from real
books (which are large, copyrighted, and non-reproducible). Each fixture
is a function that creates a controlled test image:

```python
def make_music_page(width=800, height=600, staff_color=(0, 0, 200),
                    num_staves=4, skew_deg=0, curve_amount=0,
                    noise_level=0, bg_color=(230, 220, 200)):
    """Generate a synthetic music page with staff lines, text, and border."""

def make_text_page(width=800, height=600, ...): ...
def make_page_on_background(page, angle=0, perspective_skew=0, bg=(40,30,25)): ...
def add_finger(img, position="top-right", size_frac=0.05): ...
def add_barrel_distortion(img, k1=0.3): ...
def add_hotspot(img, center, radius): ...
def blur_image(img, kernel_size=15): ...
```

Small images (800x600) for fast tests. A few 4000x3000 fixtures
(marked `@pytest.mark.slow`) for realistic integration tests.

### Test Tiers

Two tiers for developer workflow:

| Tier | Command | When to run | Target time |
|------|---------|-------------|-------------|
| **Fast** | `pytest -m "not slow"` | Every edit, during TDD | <15 seconds |
| **Full** | `pytest` | Before commit, in CI | <5 minutes |

Tests marked `@pytest.mark.slow` are those that:
- Process full-resolution (4000x3000) synthetic images
- Run multi-image stage integration tests (e.g., StitchStage with grouping)
- Perform batch operations (e.g., analyze over multiple samples)
- Individually take >2 seconds

Fast tests use small (800x600) images and test single-function behavior.
The `slow` marker is registered in `pyproject.toml`.

### Test Levels

#### 1. Unit Tests (per function, fast, <1s each)

Every public function in `utils/` gets tests before implementation:

```python
# test_image_io.py
class TestLoadImage:
    def test_loads_jpeg(self, tmp_path): ...
    def test_loads_png(self, tmp_path): ...
    def test_applies_exif_rotation(self): ...
    def test_extracts_exif_metadata(self): ...
    def test_returns_empty_exif_for_png(self): ...

class TestSaveCheckpoint:
    def test_saves_as_png(self, tmp_path): ...
    def test_atomic_write_no_partial_on_interrupt(self, tmp_path): ...
    def test_writes_metadata_sidecar(self, tmp_path): ...
    def test_cleans_up_tmp_files(self, tmp_path): ...

# test_line_detect.py
class TestDetectInkMask:
    def test_detects_red_staff_lines(self): ...
    def test_detects_brown_staff_lines(self): ...
    def test_ignores_background(self): ...
    def test_fallback_to_channel_difference(self): ...

class TestDetectInkMaskGeometric:
    def test_filters_round_foxing_spots(self): ...
    def test_preserves_horizontal_lines(self): ...

class TestDetectStaffLines:
    def test_finds_expected_number_of_lines(self): ...
    def test_returns_polynomial_coefficients(self): ...
    def test_returns_empty_for_text_page(self): ...

# test_geometry.py
class TestOrderCorners:
    def test_already_ordered(self): ...
    def test_shuffled_corners(self): ...
    def test_near_rectangular(self): ...
```

#### 2. Stage Tests (per stage, medium, <5s each)

Each stage is tested end-to-end with synthetic inputs:

```python
# test_stage_orientation.py
class TestOrientationStage:
    def test_corrects_90cw_rotation(self): ...
    def test_corrects_90ccw_rotation(self): ...
    def test_detects_upside_down_via_text_direction(self): ...
    def test_passthrough_when_already_correct(self): ...
    def test_computes_focus_score(self): ...
    def test_flags_blurry_image(self): ...

# test_stage_page_detect.py
class TestPageDetectStage:
    def test_finds_page_on_dark_background(self): ...
    def test_fallback_to_inverted_otsu(self): ...
    def test_fallback_to_canny(self): ...
    def test_detects_spread(self): ...
    def test_classifies_music_page(self): ...
    def test_classifies_text_page(self): ...
    def test_classifies_blank_page(self): ...

# test_stage_deskew.py
class TestDeskewStage:
    def test_corrects_3_degree_skew(self): ...
    def test_skips_when_angle_below_threshold(self): ...
    def test_uses_projection_profile_for_text_page(self): ...
    def test_fills_border_with_background_color(self): ...
    def test_post_geometry_trim(self): ...

# test_stage_enhance.py
class TestEnhanceStage:
    def test_sub_steps_run_in_correct_order(self): ...
    def test_skips_disabled_sub_steps(self): ...
    def test_color_cast_correction(self): ...
    def test_shadow_removal(self): ...
    def test_denoise_reduces_noise(self): ...
    def test_sharpen_increases_laplacian_variance(self): ...
```

#### 3. Integration Tests (multi-stage, slow, <30s each)

Test the pipeline end-to-end on synthetic data:

```python
# test_pipeline.py
class TestPipeline:
    def test_full_pipeline_produces_pdf(self, tmp_path): ...
    def test_geometry_profile_skips_enhance(self, tmp_path): ...
    def test_resume_after_interrupt(self, tmp_path): ...
    def test_auto_analyze_when_no_book_toml(self, tmp_path): ...
    def test_skip_ocr_when_tesseract_missing(self, tmp_path): ...
    def test_parallel_processing(self, tmp_path): ...

class TestPipelineOutputValidation:
    def test_pdf_has_correct_page_count(self, tmp_path): ...
    def test_pdf_pages_have_correct_dpi(self, tmp_path): ...
    def test_pdf_has_text_layer_when_ocr_enabled(self, tmp_path): ...
    def test_all_checkpoint_dirs_exist(self, tmp_path): ...
    def test_pipeline_json_has_all_stages(self, tmp_path): ...
    def test_flagged_pages_reported(self, tmp_path): ...

# test_cli.py
class TestCLI:
    def test_run_with_only_input_dir(self, tmp_path): ...
    def test_run_with_profile(self, tmp_path): ...
    def test_run_with_skip_flags(self, tmp_path): ...
    def test_analyze_generates_book_toml(self, tmp_path): ...
    def test_review_generates_contact_sheet(self, tmp_path): ...
```

#### 4. Real Image Smoke Test (after Stage 2)

Planned after Stage 2 (orientation) is complete. Runs Stages 0-1-2 on
a small set (5-10) of real LPA-1 images to validate that:

1. Stage 0: hotspot/finger detection doesn't produce artifacts on real photos
2. Stage 1: grouping correctly identifies the known partial photo set
   (IMG_0232-0234) and excludes the book cover (IMG_0231)
3. Stage 2: orientation produces correctly rotated pages

The smoke test is **not automated** -- it's a manual visual inspection of
checkpoint outputs. Results inform whether synthetic test parameters need
adjustment and which integration test paths to add (GitHub #3).

Select images that cover common scenarios:
- A normal standalone page
- The book cover (IMG_0231)
- The 3-image partial set (IMG_0232-0234)
- A page with visible finger at the border
- A page with uneven lighting

#### 5. Regression Tests (golden reference, run on real images)

Not part of the standard test suite (requires actual book photos),
but available via `pytest -m regression`:

```python
@pytest.mark.regression
class TestRealBookRegression:
    """Run on a small set of real images with known-good outputs.
    
    Golden references are stored in tests/golden/ as PNG files.
    Tests compare stage outputs against golden references using
    structural similarity (SSIM > 0.95) rather than pixel-exact
    comparison, allowing for minor algorithmic improvements.
    """
    def test_lpa1_orientation(self): ...
    def test_lpa1_page_detect(self): ...
    def test_lpa1_dewarp(self): ...
```

### TDD Workflow per Implementation Step

```
For each item in the Implementation Order:
1. RED:   Write test(s) for the function/stage (they fail)
2. GREEN: Implement just enough code to pass the tests
3. REFACTOR: Clean up, extract helpers, improve naming
4. VERIFY: Run full test suite (pytest), check no regressions
5. COMMIT: One commit per red-green-refactor cycle
```

### Running Tests

```bash
pytest -m "not slow"             # fast tests only (~15s) -- use during TDD
pytest                           # full suite (all tiers)
pytest -x                        # stop at first failure
pytest --cov=ghh          # with coverage report
pytest -m slow                   # only slow tests (4000x3000 images)
pytest -m regression             # only regression tests (real images)
pytest tests/test_stage_deskew.py  # single test file
```
