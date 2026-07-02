# Raw Fisheye SfM and Training Plan

This document outlines the second practical path:

```text
raw fisheye frames -> fisheye-aware SfM -> fisheye-aware 3DGS training/rendering
```

This path should be cleaner than equirectangular-to-perspective rendering if the tooling supports the camera model end to end. It avoids resampling panoramas into many virtual pinhole views, but it requires reliable calibration and downstream support for fisheye or nonlinear camera projection.

## Motivation

The equirectangular path worked, but it expanded 304 panoramas into 3648 virtual perspective images. That made feature extraction, matching, and bundle adjustment expensive.

The raw fisheye path may reduce:

- resampling artifacts,
- cubemap/perspective seam issues,
- number of derived images,
- redundant keypoints,
- matching database size,
- mapper bundle-adjustment cost.

The tradeoff is that raw fisheye introduces stricter requirements for camera calibration, camera model selection, and 3DGS training support.

## COLMAP Camera Model Basis

COLMAP documents multiple camera models, including fisheye models:

- Official docs: [COLMAP camera models](https://colmap.github.io/cameras.html)
- Relevant fisheye-capable models listed by COLMAP include:
  - `SIMPLE_RADIAL_FISHEYE`
  - `RADIAL_FISHEYE`
  - `OPENCV_FISHEYE`
  - `FOV`
  - `THIN_PRISM_FISHEYE`
  - `RAD_TAN_THIN_PRISM_FISHEYE`
  - `SIMPLE_FISHEYE`
  - `FISHEYE`

COLMAP's guidance is to use the simplest camera model that is complex enough for the lens distortion. If a model is too simple, COLMAP may need many local/global bundle adjustments or fail to model the lens well. If a model is too complex and intrinsics are poorly constrained, calibration can become unstable.

## Key Decision: What Are the Raw Frames?

Before implementation, identify the exact raw format from the 360 camera:

1. Single full-frame fisheye image.
2. Dual circular fisheye image in one frame.
3. Two separate fisheye streams/images.
4. Already stitched equirectangular panorama.
5. Vendor-specific encoded raw format.

This matters because dual-fisheye input is naturally a rig with two fisheye sensors, while a single fisheye stream is one camera.

## Proposed Experiment Stages

### Stage 0: Data and Metadata Audit

Collect:

- raw fisheye frames,
- original video metadata,
- camera model,
- lens field of view,
- resolution,
- frame rate,
- whether stabilization is enabled,
- whether horizon leveling is enabled,
- whether internal stitching is already applied,
- any vendor calibration metadata.

Avoid mixing frames from different stabilization modes or different resolutions in one reconstruction.

### Stage 1: Calibration

Goal: obtain stable intrinsics before running full SfM.

Possible approaches:

1. Use vendor calibration if available.
2. Calibrate with checkerboard/Charuco using OpenCV fisheye calibration.
3. Let COLMAP estimate intrinsics on a small subset, then inspect whether the values are plausible.
4. Share intrinsics across all frames from the same physical sensor.
5. Fix intrinsics during the full mapping run once a stable calibration is found.

For dual-fisheye cameras, also estimate or recover the fixed rig transform between the two lenses.

### Stage 2: COLMAP SfM on Raw Fisheye Frames

For single fisheye:

```text
raw_fisheye_images/
  frame_000001.jpg
  frame_000002.jpg
  ...
```

For dual fisheye, prefer two sensor folders:

```text
raw_fisheye_images/
  fisheye_left/
    frame_000001.jpg
    frame_000002.jpg
  fisheye_right/
    frame_000001.jpg
    frame_000002.jpg
```

Then configure COLMAP with:

- shared intrinsics per physical sensor,
- a fisheye camera model,
- fixed or estimated rig extrinsics for dual fisheye,
- sequential matching for video order,
- loop detection if the capture path revisits areas,
- masks for operator/body/vehicle regions.

Candidate first camera model:

- `OPENCV_FISHEYE` if we have calibration or enough shared-intrinsic constraints.
- `SIMPLE_FISHEYE` or `FISHEYE` if the lens is close to equidistant and distortion is already corrected.
- `RADIAL_FISHEYE` as a simpler alternative if `OPENCV_FISHEYE` is unstable.

### Stage 3: Masking for Raw Fisheye

The operator/arms problem still exists. The mask should be defined in raw fisheye image coordinates rather than equirectangular coordinates.

Possible masking approaches:

1. Static mask per fisheye sensor if the hand/body occupies a stable region.
2. Per-frame segmentation mask for moving hands/body.
3. Hybrid static mask plus optional segmentation.
4. Lens-circle mask for circular fisheye frames, so black borders do not produce features.

The first test should use simple static masks, then inspect feature distribution.

### Stage 4: Export and Inspect

Expected SfM output:

```text
runs/<name>/database.db
runs/<name>/sparse/0/
runs/<name>/reports/registration_summary.json
```

Inspect:

- registered frame ratio,
- number of connected components/models,
- points3D,
- mean reprojection error,
- estimated intrinsics,
- camera/rig trajectory,
- whether points cluster on operator/body artifacts,
- whether geometry bends because of poor fisheye calibration.

### Stage 5: Fisheye-Aware 3DGS Training

The downstream training stack must support the same camera projection used by SfM.

Options:

1. Use a 3DGS implementation with fisheye/nonlinear camera support.
2. Convert raw fisheye frames into perspective samples only after obtaining fisheye SfM poses.
3. Undistort fisheye frames into pinhole views, accepting some resampling.
4. Extend the training dataloader/renderer to consume COLMAP fisheye camera models directly.

The cleanest path is direct fisheye-aware training, but it depends on renderer support and calibration correctness.

## First Test Matrix

Run small experiments before full scale:

| Experiment | Input | Camera model | Intrinsics | Mapper | Goal |
| --- | --- | --- | --- | --- | --- |
| A | 50 raw fisheye frames | `OPENCV_FISHEYE` | shared, estimated | incremental | see if COLMAP registers |
| B | 50 raw fisheye frames | `RADIAL_FISHEYE` | shared, estimated | incremental | simpler model stability |
| C | 50 raw fisheye frames | `FISHEYE` | shared, fixed/estimated | incremental | equidistant assumption check |
| D | 50 dual-fisheye frames | `OPENCV_FISHEYE` | shared per sensor | rig incremental | test physical rig |
| E | same as best small run | best model | fixed intrinsics | global/incremental | speed and stability comparison |

Only run the full sequence after a small run has stable intrinsics and a coherent trajectory.

## CUDA Platform Notes

The CUDA platform should be used for:

- SIFT extraction,
- SIFT matching,
- possibly bundle adjustment if the COLMAP/pycolmap build supports it.

But fisheye correctness is mostly a modeling/calibration issue, not only a speed issue. A fast CUDA run with the wrong camera model can still produce bad geometry.

## Success Criteria

A raw fisheye run is promising if:

- at least 75% of frames register on the first full run,
- the reconstruction is one dominant connected component,
- estimated intrinsics are plausible and stable,
- reprojection error is reasonable for fisheye imagery,
- trajectory is smooth,
- operator/body artifacts are not driving major points,
- downstream 3DGS can consume the camera model or a controlled conversion.

## Risks and Blockers

- Raw frames may not be available, only stitched equirectangular images.
- Dual-fisheye frames may require vendor calibration to recover lens alignment.
- Incorrect fisheye model can cause warped geometry.
- COLMAP may estimate unstable intrinsics if every image has independent calibration.
- Some 3DGS implementations assume pinhole COLMAP cameras only.
- Direct fisheye rendering/training support may require custom code.
- Operator/hand masks need to be redone in raw fisheye coordinates.

## Recommended Next Steps

1. Collect a small raw fisheye sample set from the camera.
2. Identify whether the data is single fisheye, dual fisheye, or vendor-specific.
3. Extract 50 to 100 frames with stable motion and overlap.
4. Build static lens/body masks in raw fisheye coordinates.
5. Run COLMAP with shared intrinsics and 2 to 3 candidate fisheye camera models.
6. Inspect intrinsics, trajectory, sparse points, and registered ratio.
7. Pick the best model and rerun on a larger subset.
8. Decide whether downstream training will be direct fisheye-aware or converted to pinhole/perspective views.

