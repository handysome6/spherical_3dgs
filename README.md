# spherical-3dgs-prep

Utilities for registering 360 equirectangular image sequences before Gaussian Splatting training.

This project currently implements the conservative first path:

1. Render each panorama into a fixed rig of virtual perspective cameras.
2. Project a coarse self-mask into those views.
3. Run pycolmap feature extraction, matching, and mapping.
4. Export both the virtual-camera sparse model and an equirectangular pose model.

No 3DGS training is run by this tool.

## Raw Fisheye SfM

The experimental raw-fisheye path runs COLMAP directly on fisheye frames:

Status: implemented but untested on real raw fisheye data. So far it has only
been validated with command/help checks and dry-run plumbing tests.

```bash
pixi run prep register-fisheye \
  --input /path/to/raw_fisheye_images \
  --output runs/raw_fisheye_A \
  --camera-model OPENCV_FISHEYE \
  --lens-circle 0.5,0.5,0.48 \
  --mask-rect 0.0,0.85,1.0,1.0
```

For calibrated intrinsics, pass COLMAP camera parameters and fix them during BA:

```bash
pixi run prep register-fisheye \
  --input /path/to/raw_fisheye_images \
  --output runs/raw_fisheye_fixed \
  --camera-model OPENCV_FISHEYE \
  --camera-params fx,fy,cx,cy,k1,k2,k3,k4 \
  --fix-intrinsics
```

For dual-fisheye folders:

```bash
pixi run prep register-fisheye \
  --input /path/to/raw_dual_fisheye \
  --output runs/raw_dual_fisheye_A \
  --layout dual \
  --sensor-folder fisheye_left \
  --sensor-folder fisheye_right
```

The command writes staged images, optional raw-coordinate masks, COLMAP logs,
`colmap_commands.json`, `database.db`, `sparse/`, and
`reports/registration_summary.json`. Use `--dry-run` to prepare and inspect the
run directory without launching COLMAP.

## Smoke Run

```bash
uv run spherical-3dgs-prep register \
  --input /Users/andyliu/Downloads/sphere_images \
  --output runs/sphere_images_smoke \
  --start 1 \
  --end 80 \
  --stride 2
```

## Full Run

```bash
uv run spherical-3dgs-prep register \
  --input /Users/andyliu/Downloads/sphere_images \
  --output runs/sphere_images_full \
  --stride 1
```

Important outputs:

- `images/pano_camera*/`: rendered perspective views
- `masks/pano_camera*/`: combined COLMAP feature masks
- `database.db`: COLMAP feature/match database
- `sparse/0/`: best virtual-camera sparse reconstruction
- `sparse_equirectangular/0/`: best reconstruction mapped back to original panoramas
- `reports/registration_summary.json`: registration metrics
- `previews/`: mask overlays and rendered view previews
