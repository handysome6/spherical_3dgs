# Jetson COLMAP CUDA Comparison

Date: 2026-07-03

This note compares two Jetson SfM backends for the spherical image dataset:

```text
/home/jetson/sphere_images
```

Both runs use the same 304 equirectangular panoramas and the same virtual camera
layout from `docs/macos_equirectangular_colmap_run.md`:

- 12 perspective views per panorama
- 3648 virtual images total
- 1440x1440 virtual image resolution
- mask projection enabled
- COLMAP rig frames used to group the 12 virtual cameras for each panorama

The comparison is focused on the COLMAP/SfM stages. The self-built COLMAP 4.1
run reused the already rendered images and masks from the Pixi/PyCOLMAP run, so
rendering time is excluded from the direct COLMAP stage comparison.

## Backends Compared

### Conda-Forge / Pixi Backend

Environment from `pixi.toml`:

```toml
platforms = [{ platform = "linux-aarch64", cuda = "12.6" }]
cuda-version = "12.6.*"
pycolmap = { version = "==3.12.1", build = "cuda126py312h44145ad_0" }
```

Resolved COLMAP stack:

```text
pycolmap 3.12.1 cuda126py312h44145ad_0
colmap   3.12.5 cuda_126h106f28d_0
```

Important behavior:

- `pycolmap.has_cuda == True`
- SIFT feature extraction used CUDA.
- SIFT matching used CUDA.
- Bundle adjustment was kept on CPU because the first BA-GPU attempt hit the
  conda-forge Ceres limitation:

```text
Can't use DENSE_SCHUR with dense_linear_algebra_library_type = CUDA because
support not enabled when Ceres was built.
```

So the conda-forge/Pixi run should not be interpreted as a CPU-only COLMAP
baseline. It was GPU accelerated for feature calculation and matching. Its
major limitation was mapping/BA, where it had to fall back to CPU.

Other limitations:

- PyCOLMAP 3.12.1 does not expose `CameraModelId.EQUIRECTANGULAR`, so the
  equirectangular sparse export is skipped.
- This package is still useful as a GUI viewer. Its `colmap gui` command works
  and can read the self-built COLMAP 4.1 sparse model.

### Self-Built COLMAP 4.1 CUDA Backend

Installed package:

```text
dist/colmap-cuda-sm87_4.1.0+cuda12.6.sm87-1_arm64.deb
```

Package checksum:

```text
SHA256 59b5aa46fdd5b258b4b8a79e5962adcdf07db65e141366d737b978364bbe8f4d
```

Build summary:

- COLMAP 4.1.0, commit `fa8e3b3f`
- CUDA 12.6
- Ceres 2.3.0 built with CUDA and cuDSS
- cuDSS 0.8.0
- CUDA architecture restricted to `sm_87`
- GUI/OpenGL/ONNX/tests disabled
- Installed under `/opt/colmap-cuda-4.1.0`
- Wrapper command: `colmap-cuda`

Verification:

```bash
colmap-cuda help
```

reported:

```text
COLMAP 4.1.0 (Commit fa8e3b3f on 2026-06-26 with CUDA)
```

Important behavior:

- SIFT feature extraction uses the COLMAP CUDA SIFT extractor.
- SIFT matching uses CUDA and showed sustained GR3D activity during matching.
- Mapping was run with `--Mapper.ba_use_gpu 1 --Mapper.ba_gpu_index 0`.
- The mapper still appeared mostly CPU-bound in `tegrastats`, with only brief
  GPU bursts observed during BA. The log did contain `cuSolverDN` warnings, so
  the CUDA-backed solver path was exercised at least in some BA steps.
- The binary was intentionally built without GUI support. Use the Pixi
  conda-forge COLMAP binary to view the model.

## Input And Outputs

Input dataset:

```text
/home/jetson/sphere_images
```

Conda/PyCOLMAP output:

```text
runs/jetson_sphere_images_full
runs/jetson_sphere_images_full/sparse/0
runs/logs/jetson_sphere_images_full.log
```

Self-built COLMAP 4.1 output:

```text
runs/jetson_sphere_images_colmap_cuda41_full
runs/jetson_sphere_images_colmap_cuda41_full/sparse/0
runs/logs/jetson_sphere_images_colmap_cuda41_full.log
```

The COLMAP 4.1 run symlinked/reused:

```text
runs/jetson_sphere_images_full/images
runs/jetson_sphere_images_full/masks
```

This avoided retiming the panorama-to-perspective rendering stage and isolated
the COLMAP stage comparison.

## Conda-Forge / Pixi Run

Command:

```bash
pixi run prep register \
  --input /home/jetson/sphere_images \
  --output runs/jetson_sphere_images_full \
  --stride 1
```

Pipeline:

1. Select 304 equirectangular panoramas.
2. Render 3648 virtual perspective images.
3. Render/project masks.
4. Extract SIFT features with PyCOLMAP using CUDA.
5. Apply rig configuration.
6. Run sequential matching with loop detection using CUDA SIFT matching.
7. Run incremental mapping with CPU bundle adjustment.
8. Write `sparse/0`.
9. Skip `sparse_equirectangular/0` because PyCOLMAP 3.12 lacks
   `CameraModelId.EQUIRECTANGULAR`.

Recorded timing from `runs/logs/jetson_sphere_images_full.log`:

| Stage | Acceleration | Time |
| --- | --- | ---: |
| Rendering virtual views | CPU/OpenCV | about `22.8 min` |
| Feature extraction | CUDA SIFT | `7.269 min` |
| Sequential matching | CUDA SIFT matching | `45.825 min` |
| Rig/database overhead | CPU | `0.049 min` |
| Incremental mapping / BA | CPU BA | `228.450 min` |
| Full pipeline wall time | mixed | `18278 s` / `304.6 min` |

Model analysis:

```bash
colmap-cuda model_analyzer \
  --path runs/jetson_sphere_images_full/sparse/0
```

Result:

| Metric | Value |
| --- | ---: |
| Rigs | 1 |
| Cameras | 12 |
| Frames | 304 |
| Registered frames | 304 |
| Images | 3648 |
| Registered images | 3648 |
| Points | 187586 |
| Observations | 1343190 |
| Mean track length | 7.160396 |
| Mean observations per image | 368.199013 |
| Mean reprojection error | `1.042097 px` |

## Self-Built COLMAP 4.1 Run

The 4.1 benchmark reused the rendered images and masks from
`runs/jetson_sphere_images_full`.

Main stages:

1. Verify the installed `colmap-cuda` binary.
2. Create a fresh database under
   `runs/jetson_sphere_images_colmap_cuda41_full/database.db`.
3. Run `feature_extractor` against the reused images/masks.
4. Run `rig_configurator` with the generated 12-camera rig config.
5. Run `sequential_matcher` with GPU SIFT matching and loop detection.
6. Run `mapper` with GPU BA enabled and camera/rig intrinsics fixed.
7. Validate `sparse/0` with `model_analyzer`.

Mapper command:

```bash
/opt/colmap-cuda-4.1.0/bin/colmap mapper \
  --database_path runs/jetson_sphere_images_colmap_cuda41_full/database.db \
  --image_path runs/jetson_sphere_images_colmap_cuda41_full/images \
  --output_path runs/jetson_sphere_images_colmap_cuda41_full/sparse \
  --Mapper.ba_use_gpu 1 \
  --Mapper.ba_gpu_index 0 \
  --Mapper.ba_refine_sensor_from_rig 0 \
  --Mapper.ba_refine_focal_length 0 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 0
```

Recorded timing from
`runs/logs/jetson_sphere_images_colmap_cuda41_full.log`:

| Stage | Internal Time | Wrapper Time |
| --- | ---: | ---: |
| Version check | - | `1 s` |
| Feature extraction | `7.484 min` | `449 s` |
| Rig configuration | - | `10 s` |
| Sequential matching core | `36.673 min` | - |
| Rig verification after matching | `0.622 min` | - |
| Sequential matcher wrapper total | - | `2250 s` / `37.5 min` |
| Mapper | `59.158 min` | `3552 s` / `59.2 min` |

COLMAP-only wrapper total:

```text
449 + 10 + 2250 + 3552 = 6261 s = 104.4 min
```

Model analysis:

```bash
colmap-cuda model_analyzer \
  --path runs/jetson_sphere_images_colmap_cuda41_full/sparse/0
```

Result:

| Metric | Value |
| --- | ---: |
| Rigs | 1 |
| Cameras | 12 |
| Frames | 304 |
| Registered frames | 304 |
| Images | 3648 |
| Registered images | 3648 |
| Points | 168177 |
| Observations | 1252594 |
| Mean track length | 7.448070 |
| Mean observations per image | 343.364583 |
| Mean reprojection error | `1.017088 px` |

Warnings:

The mapper completed successfully but emitted 18 Ceres linear-solver warnings:

- 10 warnings of the form:

```text
Linear solver failure. Failed to compute a step: Success.
```

- 8 warnings from the CUDA solver path:

```text
cuSolverDN::cusolverDnDpotrf numerical failure. The leading minor ... is not positive definite.
```

These warnings did not abort mapping. The final model registered all frames and
all virtual images.

## Timing Comparison

Direct COLMAP-stage comparison:

| Stage | Conda/PyCOLMAP 3.12.1 | Self-Built COLMAP 4.1 | Speedup |
| --- | ---: | ---: | ---: |
| Feature extraction | `7.269 min` GPU | `7.484 min` GPU | `0.97x` |
| Matching | `45.825 min` GPU | `37.5 min` GPU | `1.22x` |
| Mapping / BA | `228.450 min` CPU BA | `59.158 min` GPU-enabled BA | `3.86x` |
| COLMAP stages total | `281.6 min` | `104.4 min` | `2.70x` |

The first two rows are not a GPU-versus-CPU comparison. Both backends used GPU
acceleration for SIFT extraction and matching. The decisive difference is the
mapping/BA row: conda-forge PyCOLMAP 3.12 had to run BA on CPU, while the
self-built COLMAP 4.1 package could use the CUDA/Ceres/cuDSS build.

If the reused rendering time from the first run is added back to the self-built
COLMAP path, the estimated end-to-end time is:

```text
22.8 min rendering + 104.4 min COLMAP = 127.2 min
```

Compared with the conda/PyCOLMAP full pipeline:

```text
304.6 min / 127.2 min = 2.39x estimated end-to-end speedup
```

## Reconstruction Comparison

| Metric | Conda/PyCOLMAP 3.12.1 | Self-Built COLMAP 4.1 |
| --- | ---: | ---: |
| Registered frames | `304/304` | `304/304` |
| Registered images | `3648/3648` | `3648/3648` |
| Points | `187586` | `168177` |
| Observations | `1343190` | `1252594` |
| Mean track length | `7.160396` | `7.448070` |
| Mean observations per image | `368.199013` | `343.364583` |
| Mean reprojection error | `1.042097 px` | `1.017088 px` |

Both reconstructions are complete for this dataset. The self-built 4.1 model
contains fewer points and observations, but has slightly lower mean reprojection
error. This is a version/backend comparison, not a strict quality regression:
different COLMAP versions, solver paths, and mapper internals can produce
different triangulation and filtering decisions while registering the same
frames.

## GPU Observations

Conda/PyCOLMAP 3.12.1:

- Good CUDA usage for SIFT extraction and matching.
- BA GPU was not usable in this environment because Ceres was not built with the
  required CUDA dense solver support.
- Mapping therefore became the main bottleneck.

Self-built COLMAP 4.1:

- Feature extraction created a `SIFT GPU feature extractor`.
- Sequential matching showed clear GR3D utilization and improved from
  `45.825 min` to `37.5 min`.
- Mapping improved dramatically from `228.450 min` to `59.158 min`.
- During BA, `tegrastats` was usually CPU-heavy with GR3D near `0%`, but
  occasional GPU bursts and `cuSolverDN` warnings indicate that parts of the
  CUDA solver path were used.
- The practical result is still much faster mapping, even if BA GPU utilization
  is not sustained in every `tegrastats` sample.

## GUI Viewing

The self-built COLMAP 4.1 binary cannot open the GUI:

```text
Cannot start graphical user interface. COLMAP was built without GUI support or Qt dependency was not found.
```

Use the Pixi conda-forge COLMAP binary as a viewer:

```bash
cd /home/jetson/workspace/spherical_3dgs

CONDA_OVERRIDE_CUDA=12.6 pixi run colmap gui \
  --database_path /home/jetson/workspace/spherical_3dgs/runs/jetson_sphere_images_colmap_cuda41_full/database.db \
  --image_path /home/jetson/workspace/spherical_3dgs/runs/jetson_sphere_images_colmap_cuda41_full/images \
  --import_path /home/jetson/workspace/spherical_3dgs/runs/jetson_sphere_images_colmap_cuda41_full/sparse/0
```

`CONDA_OVERRIDE_CUDA=12.6` is needed in this shell because Pixi did not detect
the CUDA virtual package automatically.

## Recommendation

Use the self-built COLMAP 4.1 CUDA package for production Jetson SfM runs:

- It preserves full registration on this dataset.
- It is about `2.70x` faster for COLMAP stages.
- It is about `3.86x` faster for mapping/BA, the dominant bottleneck.
- It avoids the conda-forge Ceres BA GPU limitation.
- It is compiled specifically for Jetson `sm_87` and CUDA 12.6.

Keep the conda-forge Pixi environment for:

- fallback PyCOLMAP compatibility tests,
- GUI viewing,
- quick checks with `model_analyzer`,
- reproducing the original conda/PyCOLMAP baseline.

## Open Follow-Ups

- Add a checked-in script for the self-built COLMAP replay so the command
  sequence is not reconstructed manually.
- Add stage timing output to `registration_summary.json`.
- Run one more full self-built COLMAP 4.1 pass without reusing rendered images
  to capture a true end-to-end wall time in one log.
- If GUI is needed from the same binary, build a separate COLMAP 4.1 viewer
  package with Qt/OpenGL enabled and CUDA/mapper features disabled.
