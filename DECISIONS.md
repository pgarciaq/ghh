# Known Risks, Mitigations, and Design Decisions

This document records architectural decisions, identified risks, and their
mitigations for the Guido's Helping Hand pipeline. Each entry follows an
ADR-like format: risk statement, resolution or mitigation strategy, and
rationale for the chosen approach.

Extracted from [PLAN.md](PLAN.md) to keep the technical spec focused on
stage specifications and pipeline architecture.

---

### K1. 180-Degree Disambiguation — Resolved

**Risk**: The orientation stage detects 90-degree rotation reliably (staff
lines are either horizontal or vertical), but distinguishing right-side-up
from upside-down is much harder. Page numbers are small, faded, or absent.
If this fails, the page is upside down and every downstream stage fails
silently.

**Resolution** (implemented):

The 180° ambiguity is resolved by a cascading polarity detector that
achieved 224/224 (100%) accuracy on the LPA-1 test set:

1. **Tesseract OSD (letter shapes)**: Primary detector.  Analyses the
   shapes of individual characters (ascenders, descenders, letter
   geometry) to determine if text is at 0° or 180°.  Two passes:
   standard (1200 px, colour) and adaptive (2000 px, binarized,
   `--dpi 300`).  Handles ~98% of pages including aged manuscripts
   with stains and decorative elements.
2. **Red title edge comparison**: Chant-book-specific fallback.
   Compares proximity-weighted title-eligible red ink in the top 10%
   vs bottom 10% of the image.  Title-eligible = red rows with no
   dark text in the centre (distinguishes titles from body rubrics).
3. **Spine S/V detection**: Last-resort fallback for covers and blank
   pages.  Compares saturation-to-brightness ratio of left/right
   edges; the more-worn edge is placed on the left (Western
   convention).
4. **Manual override**: The `[orientation_overrides]` table in
   `book.toml` can force a specific rotation for individual images:
   ```toml
   [orientation_overrides]
   "IMG_0080.JPG" = 180  # force 180-degree rotation
   ```

**Design decisions**:
- EXIF is not used (unreliable across devices and transfer tools).
- OSD is preferred over page-number detection (page numbers are too
  small and faded in this corpus).
- Sequential consistency pass was not needed (per-page detection is
  accurate enough).

### K2. Stage 9 Enhancement Chain Ordering

**Risk**: The 12-step enhancement chain has order-dependent interactions.
Wrong ordering could cause one step to amplify artifacts from another.

**Mitigations**:

1. **Principled ordering** (coarse-to-fine, low-frequency-to-high-frequency):
   - First: color cast correction (global color shift)
   - Second: illumination normalization (low-frequency spatial)
   - Third: shadow removal (medium-frequency spatial)
   - Fourth: stain correction (medium-frequency spatial)
   - Fifth: halo reduction (localized contrast)
   - Sixth: show-through removal (pixel-level classification)
   - Seventh: white balance (global color adjustment on clean signal)
   - Eighth: CLAHE (adaptive contrast on clean signal)
   - Ninth: salt correction (localized CLAHE)
   - Tenth: denoise (high-frequency noise removal)
   - Eleventh: sharpen (high-frequency detail restoration)
   - Last: binarize (if requested)
2. **Checkpoint each sub-step during development**: Temporarily save
   intermediate results after each sub-step to visually verify the chain.
   Remove intermediate checkpoints once ordering is validated.
3. **A/B comparison**: The `inspect` command should show before/after for
   each sub-step to identify problematic interactions.

### K3. No QA/Validation Mechanism

**Risk**: With 225+ pages per book, failures on individual pages (wrong
orientation, bad crop, distorted dewarp) can go unnoticed until the final
PDF is reviewed.

**Mitigations**:

1. **Per-page confidence scores**: Each stage computes a confidence metric
   and writes it to `pipeline.json`:
   - Orientation: number of agreeing signals (0-3), focus score (Laplacian variance)
   - Page detection: contour area as fraction of image
   - Dewarping: number of staff lines found, polynomial fit R-squared
   - Deskewing: skew angle magnitude
   - Enhancement: background uniformity score
2. **Flagged pages**: Pages with low confidence in any stage are flagged.
   The CLI reports them at the end: `"3 pages flagged for review: IMG_0060,
   IMG_0080, IMG_0230"`.
3. **Contact sheet generation**: New `ghh review OUTPUT_DIR` command
   that generates a single image showing thumbnails of all pages from a
   given stage (e.g., `--stage 09_enhanced`), so the user can visually
   scan all 225 pages at once and spot problems.
4. **Stage-level summary stats**: After each stage completes, log summary
   statistics:    "Stage 8: 210/225 pages dewarped (93%), 12 passed through
   (no lines), 3 used AI fallback."

### K4. Text-Only and Special Pages

**Risk**: Title pages, text-only pages, blank pages, index pages, and
heavily damaged pages have zero staff lines. The plan's fallback paths
exist but are secondary. In practice 10-20% of pages per book may be
non-standard.

**Mitigations**:

1. **Page type classification** (in Stage 4 or as part of analyze):
   - "music": has staff lines (detected via ink mask)
   - "text": has text but no staff lines
   - "decorative": title page, elaborate borders
   - "blank": stddev < threshold
   - "damaged": very low contrast or mostly stained
   Stored in metadata, used to select appropriate algorithms per page.
2. **First-class fallback paths**: The projection profile deskew and
   pass-through dewarp paths are not "fallbacks" -- they're the correct
   algorithms for text pages. Code and test them to the same standard as
   the staff-line paths.
3. **Analyze detects page type distribution**: `book.toml` records how
   many pages of each type were found in samples, so the user knows what
   to expect.

### K5. OCR Produces Garbage on Music Notation

**Risk**: Tesseract running on pages that are 70% musical notation produces
nonsense for the notation areas, polluting the searchable text layer.

**Mitigations**:

1. **Notation masking before OCR**: Detect staff line regions (using the
   same line_detect.py output from Stage 8) and mask them with white
   before feeding to Tesseract. This leaves only text lines visible to OCR.
2. **Text region extraction**: Use horizontal projection profiles to find
   text lines between staves. These inter-staff text bands are the only
   regions that should be OCR'd.
3. **PSM selection**: Use `--psm 6` (uniform block of text) on extracted
   text regions rather than full-page OCR.
4. **Confidence filtering**: Tesseract outputs per-word confidence. Discard
   words below a threshold (e.g., 40%) to avoid nonsense in the text layer.

### K6. Page Ordering May Not Match Filename Order

**Risk**: Filenames (IMG_0001 through IMG_0225) might include cover shots,
spine photos, retakes, or be out of order. Filename order != book page order.

**Mitigations**:

1. **Page number extraction**: After orientation is fixed, run a targeted
   OCR on the page number region (configured position, small crop) to
   extract the printed page number. Use this for ordering.
2. **Duplicate/retake detection**: If two images have the same extracted
   page number, flag as possible retake. Let the user choose which to keep
   via `book.toml`:
   ```toml
   [page_overrides]
   exclude = ["IMG_0045.JPG", "IMG_0046.JPG"]  # retakes, use IMG_0047
   ```
3. **Fallback to filename order**: If page number extraction fails (no
   number detected), fall back to filename order. Flag these pages.
4. **Manual ordering**: Support an optional `page_order` list in `book.toml`
   for fully manual control:
   ```toml
   page_order = ["IMG_0011.JPG", "IMG_0012.JPG", ...]
   ```

### K7. Possible Two-Page Spreads

**Risk**: Some photos might capture a full two-page spread (both pages
fully visible). The largest-contour approach grabs both as one page.

**Mitigations**:

1. **Spread detection** (in Stage 4): After finding the largest contour,
   check its aspect ratio. If width/height > 1.5 (landscape-oriented and
   much wider than a single page), suspect a spread.
2. **Vertical split detection**: Look for a vertical line/gap near the
   center of the detected quad (the book spine). Use edge detection or
   luminance valley to find the split point.
3. **Split into two pages**: If spread detected, split into left and right
   halves, process each as a separate page.
4. **The `analyze` command detects spreads** in sample images and records
   `has_spreads = true` in `book.toml`. If detected, the split logic
   activates automatically.

### K8. Performance at Scale

**Risk**: 225 images x 10+ stages = 2000+ operations per book. Some
operations are slow (denoise ~500ms, inpainting ~200ms, AI inference).
Total per-book: 30-60 minutes. With 15+ books: 8-15 hours.

**Mitigations**:

1. **Image-level parallelism**: Process multiple images simultaneously
   using `concurrent.futures.ProcessPoolExecutor`. Most stages are
   per-image with no inter-image dependencies.
   - Default: `--workers N` where N = CPU count / 2 (leave headroom for GPU)
   - Stage 10 (normalize) is the exception: it needs all images first
2. **Skip unchanged**: If a checkpoint exists, the source image mtime
   hasn't changed, AND the stage's config hash matches the previous run,
   skip that image. Config changes automatically invalidate affected
   stages and all downstream stages (see Resumability section).
3. **GPU batching**: For operations using UMat (CLAHE, remap), batch the
   GPU transfers to amortize overhead.
4. **Progress reporting**: `tqdm` progress bar per stage showing
   images/second throughput and ETA.
5. **Profile first**: Before optimizing, profile a 10-image run to find
   the actual bottleneck. It might not be where we expect.

### K9. Partial Photo Stitching Failures

**Risk**: Partial photographs of the same page may have significantly
different perspective, lighting, or white balance between shots. The
photographer may have moved the book between shots. Stitching may produce
visible seams, ghosting, or misalignment.

**Mitigations**:

1. **Pre-stitching homogenization**: Before stitching, normalize white
   balance and exposure across images in the same group using the
   overlapping region as reference. This reduces visible seams.
2. **Multi-band blending**: `cv2.Stitcher` uses multi-band blending by
   default, which handles gradual lighting differences well.
3. **Fallback to best single image**: If `cv2.Stitcher` fails (returns
   `ERR_NEED_MORE_IMGS` or `ERR_HOMOGRAPHY_EST_FAIL`), fall back to the
   single image in the group that captures the largest page area. Flag
   the page for manual review.
4. **Manual group override**: Allow explicit grouping in `book.toml`:
   ```toml
   [stitch_groups]
   "INDEX_1" = ["IMG_0232.JPG", "IMG_0233.JPG", "IMG_0234.JPG"]
   ```
   Also allow forcing images as standalone:
   ```toml
   [stitch_overrides]
   standalone = ["IMG_0233.JPG"]  # this is a retake, not a partial
   ```
5. **Visual QA**: The `review` command includes stitched results with
   overlay lines showing the seam positions, so the user can spot
   misalignment.

### K10. Incomplete Page Coverage (Cut Corners)

**Risk**: Many photographs do not capture the full page -- one or two
corners are outside the camera frame. This is common when photographing
thick bound books where the photographer cannot frame the page perfectly.
The missing corners affect multiple stages differently.

**Impact per stage**:

| Stage | Impact | Severity |
|-------|--------|----------|
| 4 (page detect) | Quad can't find real corners, falls back to full-image quad | Low |
| 5 (perspective) | Full-image fallback = near-identity warp, so minimal distortion | Low |
| 6 (content area) | Border detection may fail on cut side; ink-density fallback works; feathering hides the cut edge | Low |
| 7 (deskew) | Rotation shifts the cut corner, background fill makes it visible; `trim_to_content()` cleans up | Medium |
| 8 (dewarp) | Staff lines near the cut corner are shorter/missing, weakening polynomial fit on that side | Medium |

**Mitigations** (most already in place):

1. **Full-image quad fallback** (Stage 4): When the quad detection fails to
   find 4 real corners, it returns the full image bounds. This prevents
   incorrect perspective correction in Stage 5.
2. **`page_detect_expand_frac`** (Stage 4): Expands the detected quad
   outward by a fraction, reducing the chance of cropping at the edge.
3. **Feathered edge masking** (Stage 6): Gaussian feathering at image
   edges fades out the cut-corner region instead of creating a hard edge.
4. **Background fill** (Stages 5, 7, 8): All geometric transforms use
   `borderMode=BORDER_CONSTANT` with the estimated background color.
   Cut corners are filled with a plausible color rather than black.
5. **`trim_to_content()`** (Stages 7, 8): Post-geometry trim crops the
   background-filled artifacts from cut corners after rotation/dewarp.
6. **This is fundamentally a photography limitation**: The pipeline
   cannot recover content that was never captured. The mitigations make
   the result look clean, but the missing corner content is gone.
