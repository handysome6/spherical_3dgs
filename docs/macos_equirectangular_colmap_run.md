# macOS Equirectangular to COLMAP Registration Run

Date: 2026-07-02

This note documents the first practical path we implemented and tested on macOS:

```text
360 equirectangular panoramas -> virtual perspective views -> COLMAP/pycolmap registration
```

No 3DGS training was run in this experiment. The goal was to validate all preprocessing and SfM registration artifacts before moving to a CUDA-supported platform.

## Environment

- Project: `/Users/andyliu/workspace/ensightfful/spherical_3dgs`
- Input data: `/Users/andyliu/Downloads/sphere_images`
- Input count: 304 JPEG panoramas
- Input resolution: 5760x2880, 2:1 equirectangular
- Python environment: `uv`
- Main dependency: `pycolmap==4.1.0`
- Platform used for this run: macOS
- Current pycolmap CUDA status on this machine: `pycolmap.has_cuda == False`

The macOS run should be treated as a correctness baseline, not a performance baseline.

## Pipeline Implemented

The CLI entrypoint is:

```bash
uv run spherical-3dgs-prep register \
  --input /Users/andyliu/Downloads/sphere_images \
  --output runs/sphere_images_full \
  --stride 1
```

The implemented steps are:

1. Validate that selected panoramas are 2:1 equirectangular images.
2. Build a coarse static self-mask for visible operator/body/arms.
3. Render virtual perspective images from each equirectangular panorama.
4. Project the self-mask into every virtual perspective view.
5. Extract COLMAP/SIFT features with pycolmap.
6. Apply COLMAP rig configuration so virtual cameras from one panorama are treated as one rig frame.
7. Run sequential matching with loop detection.
8. Run incremental mapping.
9. Select the largest/best reconstruction.
10. Write the virtual-view sparse model.
11. Convert poses back to one equirectangular image per panorama for inspection/debugging.
12. Write `registration_summary.json`.

## Virtual Camera Layout

The full run used `perspective_overlapping`:

- 4 yaw steps
- 3 pitch rows: `-35`, `0`, `35` degrees
- 90 degree horizontal FOV
- 90 degree vertical FOV
- 12 virtual perspective images per panorama

For 304 panoramas, this produced:

- Expected rendered views: 3648
- Rendered views: 3648
- Masks: 3648

## Masking

We used a static coarse self-mask by default because the sample sequence has visible operator/body/arms near the seam and lower band.

Default normalized mask rectangles:

```text
x=[0.00, 0.18], y=[0.34, 1.00]
x=[0.88, 1.00], y=[0.34, 1.00]
x=[0.00, 1.00], y=[0.82, 1.00]
```

This mask is projected into every virtual view and combined with COLMAP's virtual-camera ownership mask before feature extraction.

## Runs Completed

### Tiny Debug Run

Command:

```bash
uv run spherical-3dgs-prep register \
  --input /Users/andyliu/Downloads/sphere_images \
  --output runs/debug_tiny \
  --start 1 \
  --end 4 \
  --stride 1 \
  --min-registered-pano-ratio 0
```

Result:

- Selected panoramas: 4
- Virtual views: 48
- Registered panoramas: 4/4
- Registered virtual images: 48/48
- Points3D: 2712
- Mean reprojection error: about 0.9990
- Output size: about 70 MB

### Smoke Run

Command:

```bash
uv run spherical-3dgs-prep register \
  --input /Users/andyliu/Downloads/sphere_images \
  --output runs/sphere_images_smoke \
  --start 1 \
  --end 80 \
  --stride 2
```

Result:

- Selected panoramas: 40
- Virtual views: 480
- Registered panoramas: 40/40
- Registered virtual images: 480/480
- Points3D: 23,879
- Mean reprojection error: about 0.8918
- Output size: about 661 MB

### Full Run

Command:

```bash
uv run spherical-3dgs-prep register \
  --input /Users/andyliu/Downloads/sphere_images \
  --output runs/sphere_images_full \
  --stride 1
```

Result:

- Selected panoramas: 304
- Virtual views: 3648
- Registered panoramas: 304/304
- Registered virtual images: 3648/3648
- Model count: 1
- Points3D: 183,516
- Observations: 1,429,606
- Mean track length: 7.79009
- Mean observations per image: 391.888
- Mean reprojection error: 0.992294
- Passed full-run threshold: yes, threshold was 75%
- Output size: about 4.6 GB

Important artifacts:

```text
runs/sphere_images_full/images/
runs/sphere_images_full/masks/
runs/sphere_images_full/database.db
runs/sphere_images_full/sparse/0/
runs/sphere_images_full/sparse_equirectangular/0/
runs/sphere_images_full/reports/registration_summary.json
```

Sparse model files were written in both `sparse/0` and `sparse_equirectangular/0`:

```text
rigs.bin
cameras.bin
images.bin
frames.bin
points3D.bin
```

## Time Analysis

The full run was successful but slow on macOS.

Measured or inferred timing:

| Stage | Approximate time | Source |
| --- | ---: | --- |
| Render virtual views and masks | about 3 minutes | image/mask file mtimes, `10:26:27` to `10:29:23` |
| Feature extraction | about 25 minutes | inferred from render end and matching start |
| Sequential matching with loop detection | 31.571 minutes | COLMAP log |
| Incremental mapping | 168.123 minutes | COLMAP log |
| Write sparse and equirectangular models | less than 1 minute | sparse/report mtimes |

The longest stage was incremental mapping, especially repeated retriangulation and global bundle adjustment as the model grew.

## Why It Took So Long

The run produced a large SfM problem:

- 304 panoramas became 3648 virtual images.
- COLMAP extracted about 10,138,032 keypoints.
- Average keypoints per virtual image: about 2779.
- Maximum keypoints in a virtual image: 9052.
- Database had 440,521 match/two-view-geometry rows.
- `database.db` alone was about 1.6 GB.
- The final model had 183,516 3D points and 1,429,606 observations.

Incremental SfM is expensive here because it repeatedly:

- registers more frames,
- triangulates new points,
- filters tracks,
- performs local bundle adjustment,
- periodically performs global bundle adjustment.

As the model grows, global BA becomes progressively more expensive. During the full run, the terminal often appeared quiet for long periods because COLMAP was inside those optimization passes.

## macOS and GPU Findings

The current macOS pycolmap environment is CPU-only:

```text
pycolmap.__version__ = 4.1.0
pycolmap.has_cuda = False
FeatureExtractionOptions.use_gpu = False
FeatureMatchingOptions.use_gpu = False
IncrementalPipelineOptions.ba_use_gpu = False
```

This means enabling GPU options in this environment would not give CUDA acceleration.

For the CUDA rerun, use a Linux machine with NVIDIA GPU support and a COLMAP/pycolmap build that reports CUDA support. Then explicitly test GPU options for:

- SIFT extraction
- SIFT matching
- Bundle adjustment, if supported by the chosen COLMAP/pycolmap build

## Recommended Improvements Before CUDA Rerun

### Add Better Timing

The CLI should record per-stage timings into `registration_summary.json`:

- validation
- self-mask generation
- rendering
- feature extraction
- rig config application
- matching
- mapping
- sparse write
- equirectangular conversion

This will make future CPU/CUDA comparisons much cleaner.

### Add Performance Presets

Add `--preset fast|balanced|quality`.

Suggested first-pass presets:

| Preset | Virtual views per pano | Max image size | Max SIFT features | Intended use |
| --- | ---: | ---: | ---: | --- |
| `fast` | 4 or 6 | 1024 | 4096 | iteration/debug |
| `balanced` | 12 | 1200 or 1440 | 4096-8192 | likely default |
| `quality` | 12 | full rendered size | 8192+ | final registration |

The current full run is closer to `quality`.

### Reduce Virtual View Count

The current 12-view overlapping layout is robust but expensive. Try:

1. 4-view non-overlapping layout using only pitch `0`.
2. 6-view cubemap-like layout.
3. 8-view layout if 4 or 6 loses too much registration quality.
4. 12-view layout only when registration quality needs it.

Registration success should be compared against runtime and 3DGS downstream quality.

### Limit SIFT Work

Expose options for:

- `max_image_size`
- `max_num_features`
- `peak_threshold`
- `first_octave`
- `num_octaves`

The first CUDA rerun should compare:

1. Current default SIFT settings.
2. `max_image_size=1200`, `max_num_features=4096`.
3. `max_image_size=1024`, `max_num_features=4096`.

### Tune Matching

Current matching uses sequential matching with loop detection. It worked well, but the database became large.

Try:

- reducing sequential overlap,
- reducing loop detection frequency,
- reducing loop detection candidate count,
- comparing vocab-tree matching on CUDA platform,
- using known video order to limit pairs more aggressively.

### Tune Mapper/BA

The largest gain is likely from mapper tuning.

Try:

- global mapper as a benchmark,
- lower global BA frequency,
- lower global BA max iterations,
- lower local BA max iterations,
- GPU BA if available and stable,
- fixed intrinsics/extrinsics as we already do for the rig,
- stricter feature/match filtering if mapper becomes too large.

The current configuration is robust but not tuned for speed.

## CUDA Platform Rerun Checklist

1. Verify CUDA:

```bash
nvidia-smi
```

2. Verify COLMAP or pycolmap CUDA support:

```bash
uv run python - <<'PY'
import pycolmap
print(pycolmap.__version__)
print(pycolmap.has_cuda)
PY
```

3. Run the same smoke command first.
4. Compare registration ratio, points3D, reprojection error, and per-stage time.
5. Run the full sequence only after the smoke run passes.
6. Save the exact `registration_summary.json` and machine/GPU details with the run.

## Open Questions

- Is 12 virtual views per panorama necessary for downstream 3DGS quality?
- Can we get near-identical registration with 4 to 6 virtual views?
- Does CUDA acceleration reduce only SIFT/matching time, or also BA time in the chosen build?
- Does global mapping produce a comparable model faster than incremental mapping on this dataset?
- Are the static self-mask rectangles sufficient for all frames, or should we add learned/person segmentation later?

