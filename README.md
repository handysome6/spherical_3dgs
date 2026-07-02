# spherical-3dgs-prep

Utilities for registering 360 equirectangular image sequences before Gaussian Splatting training.

This project currently implements the conservative first path:

1. Render each panorama into a fixed rig of virtual perspective cameras.
2. Project a coarse self-mask into those views.
3. Run pycolmap feature extraction, matching, and mapping.
4. Export both the virtual-camera sparse model and an equirectangular pose model.

No 3DGS training is run by this tool.

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
