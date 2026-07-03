# Jetson Equirectangular COLMAP Run Plan

Date: 2026-07-02

This note records the Jetson audit for rerunning the macOS baseline from
`docs/macos_equirectangular_colmap_run.md`:

```text
360 equirectangular panoramas -> virtual perspective views -> COLMAP/pycolmap registration
```

The same dataset is available locally at:

```bash
/home/jetson/sphere_images
```

## Dataset Check

Local dataset findings:

- Image count: 304 JPEG files
- Resolution spot check: `5760x2880`
- Aspect ratio: 2:1 equirectangular
- Dataset size: about 1015 MB

This matches the macOS full-run input described in
`docs/macos_equirectangular_colmap_run.md`.

## Jetson Environment Check

Observed local environment:

- Platform: Linux aarch64, Jetson kernel `5.15.148-tegra`
- CPU: 8 Cortex-A78AE cores
- RAM: 15 GiB total, about 13 GiB available during the audit
- Disk: about 29 GB available on `/`
- CUDA: 12.6 via `/usr/local/cuda-12.6`
- GPU: `nvidia-smi` reports `Orin (nvgpu)`
- Docker: installed, user is in the `docker` group, default runtime is `nvidia`
- Host `sudo -n`: unavailable because a password is required

The disk budget is enough for one smoke/full output, but it is tight for large
Docker image pulls, source builds, and repeated 4 GB to 5 GB run directories.

## Python Backend Status

The original `uv` environment still cannot install unchanged on this Jetson:

```bash
uv run python --version
```

fails during dependency resolution because the pinned package
`pycolmap==4.1.0` has no compatible Linux aarch64 wheel. The lockfile only
contains wheels for:

- macOS arm64
- Linux x86_64
- Windows amd64

System Python also has no `pycolmap` module installed.

`pycolmap-cuda12` was also not resolvable for this aarch64 platform in the local
pip check.

The chosen Jetson backend is now Pixi with conda-forge CUDA 12.6 packages:

```text
pycolmap 3.12.1 cuda126py312h44145ad_0
colmap 3.12.5 cuda_126h106f28d_0
cuda-version 12.6
```

This backend reports:

```text
pycolmap.has_cuda == True
```

It accelerates SIFT extraction and SIFT matching on GPU. It does not support
the newer PyCOLMAP 4.x API exactly, so the CLI now includes compatibility paths
for PyCOLMAP 3.12 matching, camera construction, and database opening.

Important limitation: PyCOLMAP 3.12.1 does not expose
`CameraModelId.EQUIRECTANGULAR`, so `sparse_equirectangular/0` is skipped on
this backend. The normal virtual-camera COLMAP model is still written to
`sparse/0`.

## COLMAP Version Context

COLMAP's install docs say that default distro packages generally do not include
CUDA support and CUDA support requires a source build. The PyCOLMAP docs say
that source PyCOLMAP builds require COLMAP to be installed from source first.

Relevant references:

- https://colmap.github.io/install.html
- https://colmap.github.io/pycolmap/index.html

The official `colmap/colmap:latest` Docker image manifest inspected locally is
`amd64`, so it is not a direct Jetson solution. NVIDIA's
`nvcr.io/nvidia/l4t-jetpack:r36.4.0` image is `arm64` and is the more plausible
container base if we later build inside Docker.

The conda-forge package
`colmap-4.1.0-cuda_129h32c2424_0` exists for `linux-aarch64`, but it is not
usable on this Jetson because it requires:

```text
__cuda >=12.9
cuda-version >=12.9,<13
arm-variant * sbsa
```

This Jetson reports `__cuda=12.6`, so Pixi correctly refuses that package.
`linux-aarch64` only means Linux on ARM64; it does not mean every ARM64 CUDA
machine is compatible with every ARM64 CUDA build.

## Recommended Run Path

Use Pixi on the host with the checked-in `pixi.toml`:

```bash
pixi run prep-tiny
pixi run prep-smoke
```

The manifest pins:

```toml
platforms = [{ platform = "linux-aarch64", cuda = "12.6" }]
cuda-version = "12.6.*"
pycolmap = { version = "==3.12.1", build = "cuda126py312h44145ad_0" }
```

This prevents Pixi from drifting to CUDA 12.9 packages that do not match the
Jetson CUDA runtime.

## Commands

Tiny debug run:

```bash
pixi run prep-tiny
```

Smoke run:

```bash
pixi run prep-smoke
```

Full run:

```bash
pixi run prep register \
  --input /home/jetson/sphere_images \
  --output runs/jetson_sphere_images_full \
  --stride 1
```

## Runs Completed

### Tiny Debug Run

Command:

```bash
pixi run prep-tiny
```

Result:

- Selected panoramas: 4
- Virtual views: 48
- Registered panoramas: 4/4
- Registered virtual images: 48/48
- Points3D: 2938
- Mean reprojection error: about 1.064
- Output size: about 63 MB
- `sparse/0` written
- `sparse_equirectangular/0` skipped because PyCOLMAP 3.12 lacks
  `CameraModelId.EQUIRECTANGULAR`

### Smoke Run

Command:

```bash
pixi run prep-smoke
```

The first smoke attempt completed rendering, GPU SIFT extraction, and GPU SIFT
matching, then failed during mapping because `ba_use_gpu=True` hit a conda Ceres
limitation:

```text
Can't use DENSE_SCHUR with dense_linear_algebra_library_type = CUDA because
support not enabled when Ceres was built.
```

The CLI now leaves bundle adjustment on CPU for this backend while preserving
GPU SIFT extraction and matching. Mapping was resumed from the existing
database with `ba_use_gpu=False`.

Result:

- Selected panoramas: 40
- Virtual views: 480
- Registered panoramas: 40/40
- Registered virtual images: 480/480
- Model count: 1
- Points3D: 22,810
- Observations: 140,727
- Mean track length: 6.16953
- Mean observations per image: 293.181
- Mean reprojection error: about 0.94121
- Output size: about 580 MB
- `sparse/0` written
- `sparse_equirectangular/0` skipped because PyCOLMAP 3.12 lacks
  `CameraModelId.EQUIRECTANGULAR`

Important artifacts:

```text
runs/jetson_debug_tiny/reports/registration_summary.json
runs/jetson_sphere_images_smoke/reports/registration_summary.json
runs/jetson_sphere_images_smoke/sparse/0/
runs/logs/jetson_debug_tiny_final.log
runs/logs/jetson_sphere_images_smoke.log
```

## First Jetson Comparisons

Compare against the macOS doc:

- registered panoramas
- registered virtual images
- model count
- points3D
- mean reprojection error
- `database.db` size
- total output size
- wall time for feature extraction, matching, and mapping

The current CLI does not record per-stage timings, so use shell timing around
the command until timing instrumentation is added.

```bash
time pixi run prep-smoke
```

For Jetson resource monitoring:

```bash
tegrastats --interval 1000 --logfile runs/jetson_sphere_images_smoke/tegrastats.log
```

Stop it after the run:

```bash
pkill tegrastats
```

## Fallback Paths

If the PyCOLMAP 3.12 compatibility path is not enough:

1. Build COLMAP and PyCOLMAP 4.x from source against Jetson CUDA 12.6, ideally
   inside an L4T container.
2. Upgrade the Jetson CUDA/L4T stack only if NVIDIA provides a supported path to
   CUDA 12.9 or newer, then retest conda-forge COLMAP 4.x CUDA packages.
3. Build and run on an x86_64 Linux CUDA workstation using the existing
   `pycolmap==4.1.0` or matching CUDA wheels.
4. Add an alternate repo backend that renders images locally and shells out to a
   `colmap` executable, then converts the sparse model back to equirectangular
   poses. This is more engineering work because the current implementation uses
   PyCOLMAP objects throughout rendering, rig setup, mapping, and model export.
5. Try the Ubuntu `colmap` 3.7 package only as a CPU-only registration
   experiment. It is not equivalent to the current PyCOLMAP 4.x rig workflow.

## Code Improvements Before Long Runs

These would make the Jetson rerun easier to interpret:

- Add per-stage timings to `registration_summary.json`.
- Add explicit SIFT/matching GPU flags to the CLI.
- Keep BA GPU disabled for conda-forge PyCOLMAP 3.12 unless a Ceres/cuDSS-capable
  build is available.
- Add SIFT limits such as max image size and max feature count.
- Add a fast preset using the existing `perspective_non_overlapping` render type
  for quick Jetson iteration.
