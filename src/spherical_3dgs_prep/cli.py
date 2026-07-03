from __future__ import annotations

import collections
import enum
import json
import os
import shutil
from collections.abc import Sequence
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, TypeVar, cast

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image
import pycolmap
import typer
from pycolmap import logging
from rich.console import Console
from scipy.spatial.transform import Rotation
from tqdm import tqdm

app = typer.Typer(
    add_completion=False,
    help="Preprocess 360 equirectangular panoramas and run COLMAP registration.",
)
console = Console()

N = TypeVar("N", bound=int)
NDArrayNx2 = np.ndarray[tuple[N, Literal[2]], np.dtype[np.float64]]
NDArray3x1 = np.ndarray[tuple[Literal[3], Literal[1]], np.dtype[np.float64]]
NDArray3x3 = np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]]


class Matcher(enum.StrEnum):
    SEQUENTIAL = enum.auto()
    EXHAUSTIVE = enum.auto()
    VOCABTREE = enum.auto()
    SPATIAL = enum.auto()


class Mapper(enum.StrEnum):
    INCREMENTAL = enum.auto()
    GLOBAL = enum.auto()


class PanoRenderType(enum.StrEnum):
    PERSPECTIVE_OVERLAPPING = enum.auto()
    PERSPECTIVE_NON_OVERLAPPING = enum.auto()


@dataclass(frozen=True)
class PanoRenderOptions:
    num_steps_yaw: int
    pitches_deg: Sequence[float]
    hfov_deg: float
    vfov_deg: float


@dataclass(frozen=True)
class ModelWriteResult:
    equirectangular_exported: bool
    equirectangular_export_skip_reason: str | None = None


PANO_RENDER_OPTIONS: dict[PanoRenderType, PanoRenderOptions] = {
    PanoRenderType.PERSPECTIVE_OVERLAPPING: PanoRenderOptions(
        num_steps_yaw=4,
        pitches_deg=(-35.0, 0.0, 35.0),
        hfov_deg=90.0,
        vfov_deg=90.0,
    ),
    PanoRenderType.PERSPECTIVE_NON_OVERLAPPING: PanoRenderOptions(
        num_steps_yaw=4,
        pitches_deg=(0.0,),
        hfov_deg=90.0,
        vfov_deg=90.0,
    ),
}


@app.callback()
def root() -> None:
    """Preprocess 360 equirectangular panoramas for registration."""

DEFAULT_SELF_MASK_RECTS: tuple[tuple[float, float, float, float], ...] = (
    (0.00, 0.34, 0.18, 1.00),
    (0.88, 0.34, 1.00, 1.00),
    (0.00, 0.82, 1.00, 1.00),
)

MANAGED_OUTPUT_DIRS = (
    "images",
    "masks",
    "previews",
    "reports",
    "sparse",
    "sparse_equirectangular",
)
MANAGED_OUTPUT_FILES = (
    "database.db",
    "database.db-shm",
    "database.db-wal",
    "self_mask.png",
)


def create_virtual_camera(
    *,
    pano_width: int,
    pano_height: int,
    hfov_deg: float,
    vfov_deg: float,
) -> pycolmap.Camera:
    image_width = int(pano_width * hfov_deg / 360)
    image_height = int(pano_height * vfov_deg / 180)
    focal = image_width / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
    camera = create_colmap_camera(
        camera_id=0,
        model=pycolmap.CameraModelId.SIMPLE_PINHOLE,
        focal_length=focal,
        width=image_width,
        height=image_height,
    )
    camera.has_prior_focal_length = True
    return camera


def camera_model_id_name(model: object) -> str:
    if isinstance(model, str):
        return model
    name = getattr(model, "name", None)
    if isinstance(name, str):
        return name
    return str(model).rsplit(".", 1)[-1]


def camera_model_name(camera: pycolmap.Camera) -> str:
    model_name = getattr(camera, "model_name", None)
    if isinstance(model_name, str):
        return model_name
    return camera_model_id_name(camera.model)


def camera_params_string(camera: pycolmap.Camera) -> str:
    params = np.asarray(camera.params, dtype=np.float64)
    return ",".join(f"{value:.17g}" for value in params)


def create_colmap_camera(
    *,
    camera_id: int,
    model: object,
    focal_length: float,
    width: int,
    height: int,
) -> pycolmap.Camera:
    create_from_model_id = getattr(pycolmap.Camera, "create_from_model_id", None)
    if callable(create_from_model_id):
        return create_from_model_id(
            camera_id=camera_id,
            model=model,
            focal_length=focal_length,
            width=width,
            height=height,
        )

    create = getattr(pycolmap.Camera, "create", None)
    if callable(create):
        return create(
            camera_id=camera_id,
            model=camera_model_id_name(model),
            focal_length=focal_length,
            width=width,
            height=height,
        )

    return pycolmap.Camera(
        camera_id=camera_id,
        model=model,
        width=width,
        height=height,
        params=[focal_length, width / 2, height / 2],
    )


def camera_rays_from_img(
    camera: pycolmap.Camera,
    image_points: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    cam_ray_from_img = getattr(camera, "cam_ray_from_img", None)
    if callable(cam_ray_from_img):
        return np.asarray(cam_ray_from_img(image_points=image_points))

    xy_norm: npt.NDArray[np.floating] = np.asarray(
        camera.cam_from_img(image_points=image_points)
    )
    return np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)


def get_virtual_camera_rays(camera: pycolmap.Camera) -> npt.NDArray[np.floating]:
    size = (camera.width, camera.height)
    x, y = np.indices(size).astype(np.float32)
    xy: NDArrayNx2 = np.column_stack([x.ravel(), y.ravel()])
    xy += 0.5
    xy_norm: NDArrayNx2 = camera.cam_from_img(image_points=xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(
    image_size: tuple[int, int], rays_in_cam: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 2:1 equirectangular panoramas are supported.")
    if rays_in_cam.ndim != 2 or rays_in_cam.shape[1] != 3:
        raise ValueError(f"{rays_in_cam.shape=} but expected (N, 3).")
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size


def get_virtual_rotations(
    num_steps_yaw: int, pitches_deg: Sequence[float]
) -> Sequence[npt.NDArray[np.floating]]:
    cams_from_pano_r = []
    yaws = np.linspace(0, 360, num_steps_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        yaw_offset = (360 / num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            cam_from_pano_r = Rotation.from_euler(
                "XY", [-pitch_deg, -yaw_deg], degrees=True
            ).as_matrix()
            cams_from_pano_r.append(cam_from_pano_r)
    return cams_from_pano_r


def create_pano_rig_config(
    cams_from_pano_rotation: Sequence[npt.NDArray[np.floating]],
    ref_idx: int = 0,
) -> pycolmap.RigConfig:
    rig_cameras = []
    zero_translation = cast(NDArray3x1, np.zeros((3, 1), dtype=np.float64))
    for idx, cam_from_pano_rotation in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = (
                cam_from_pano_rotation @ cams_from_pano_rotation[ref_idx].T
            )
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation),
                zero_translation,
            )
        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=idx == ref_idx,
                image_prefix=f"pano_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)


class PanoProcessor:
    def __init__(
        self,
        pano_image_dir: Path,
        output_image_dir: Path,
        mask_dir: Path,
        render_options: PanoRenderOptions,
        self_mask: npt.NDArray[np.uint8] | None,
        preview_dir: Path,
        preview_count: int,
    ) -> None:
        self.render_options = render_options
        self.pano_image_dir = pano_image_dir
        self.output_image_dir = output_image_dir
        self.mask_dir = mask_dir
        self.self_mask = self_mask
        self.preview_dir = preview_dir
        self.preview_count = preview_count
        self.cams_from_pano_rotation = get_virtual_rotations(
            num_steps_yaw=render_options.num_steps_yaw,
            pitches_deg=render_options.pitches_deg,
        )
        self.rig_config = create_pano_rig_config(self.cams_from_pano_rotation)
        self.cam_centers_in_pano = np.einsum(
            "nij,i->nj", self.cams_from_pano_rotation, [0, 0, 1]
        )
        self._lock = Lock()
        self._camera: pycolmap.Camera | None = None
        self._pano_size: tuple[int, int] | None = None
        self._rays_in_cam: npt.NDArray[np.floating] | None = None
        self._preview_names: set[str] = set()

    @property
    def virtual_camera_count(self) -> int:
        return len(self.cams_from_pano_rotation)

    def process(self, pano_name: str) -> None:
        pano_path = self.pano_image_dir / pano_name
        try:
            pano_pil_image = PIL.Image.open(pano_path).convert("RGB")
        except PIL.Image.UnidentifiedImageError:
            logging.info(f"Skipping unreadable image {pano_path}")
            return

        pano_image = np.asarray(pano_pil_image)
        pano_height, pano_width, *_ = pano_image.shape
        if pano_width != pano_height * 2:
            raise ValueError(
                f"{pano_path} is {pano_width}x{pano_height}, expected 2:1."
            )

        with self._lock:
            if self._camera is None:
                self._camera = create_virtual_camera(
                    pano_width=pano_width,
                    pano_height=pano_height,
                    hfov_deg=self.render_options.hfov_deg,
                    vfov_deg=self.render_options.vfov_deg,
                )
                for rig_camera in self.rig_config.cameras:
                    rig_camera.camera = self._camera
                self._pano_size = (pano_width, pano_height)
                self._rays_in_cam = get_virtual_camera_rays(self._camera)
            elif (pano_width, pano_height) != self._pano_size:
                raise ValueError(
                    "Panoramas of different sizes are not supported: "
                    f"{pano_path} is {pano_width}x{pano_height}, "
                    f"expected {self._pano_size}."
                )

            camera = self._camera
            pano_size = self._pano_size
            rays_in_cam = self._rays_in_cam

        assert camera is not None
        assert pano_size is not None
        assert rays_in_cam is not None

        for cam_idx, cam_from_pano_r in enumerate(self.cams_from_pano_rotation):
            rays_in_pano = rays_in_cam @ cam_from_pano_r
            xy_in_pano = spherical_img_from_cam(pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(
                camera.width, camera.height, 2
            ).astype(np.float32)
            xy_in_pano -= 0.5

            x_coords, y_coords = np.moveaxis(xy_in_pano, [0, 1, 2], [2, 1, 0])
            image = cv2.remap(
                pano_image,
                x_coords,
                y_coords,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )

            closest_camera = np.argmax(rays_in_pano @ self.cam_centers_in_pano.T, -1)
            ownership_mask = (
                ((closest_camera == cam_idx) * 255)
                .astype(np.uint8)
                .reshape(camera.width, camera.height)
                .transpose()
            )
            mask = ownership_mask
            if self.self_mask is not None:
                projected_self_mask = cv2.remap(
                    self.self_mask,
                    x_coords,
                    y_coords,
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_WRAP,
                )
                mask = np.minimum(mask, projected_self_mask)

            image_name = self.rig_config.cameras[cam_idx].image_prefix + pano_name
            mask_name = f"{image_name}.png"
            image_path = self.output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            PIL.Image.fromarray(image).save(image_path, quality=95)

            mask_path = self.mask_dir / mask_name
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            if not pycolmap.Bitmap.from_array(mask).write(str(mask_path)):
                raise RuntimeError(f"Cannot write {mask_path}")

            self._maybe_write_preview(image_name, image, mask)

    def _maybe_write_preview(
        self, image_name: str, image: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8]
    ) -> None:
        if self.preview_count <= 0:
            return
        with self._lock:
            if len(self._preview_names) >= self.preview_count:
                return
            if image_name in self._preview_names:
                return
            self._preview_names.add(image_name)

        preview_name = image_name.replace("/", "_")
        PIL.Image.fromarray(image).save(self.preview_dir / f"{preview_name}")
        overlay = overlay_mask(image, mask)
        PIL.Image.fromarray(overlay).save(self.preview_dir / f"{preview_name}.mask.jpg")

    def split_image_name(self, image_name: str) -> tuple[int, str]:
        for cam_idx, rig_camera in enumerate(self.rig_config.cameras):
            prefix = rig_camera.image_prefix
            if image_name.startswith(prefix):
                return cam_idx, image_name[len(prefix) :]
        raise ValueError(f"Unknown virtual camera for image {image_name!r}.")

    def convert_to_equirectangular(
        self, reconstruction: pycolmap.Reconstruction
    ) -> pycolmap.Reconstruction:
        if self._camera is None or self._pano_size is None:
            raise RuntimeError("No panorama was rendered yet.")

        pano_width, pano_height = self._pano_size
        equirect = pycolmap.Reconstruction()
        equirect_camera = create_colmap_camera(
            camera_id=1,
            model=pycolmap.CameraModelId.EQUIRECTANGULAR,
            focal_length=0.0,
            width=pano_width,
            height=pano_height,
        )
        equirect.add_camera_with_trivial_rig(equirect_camera)

        pano_from_ref = pycolmap.Rigid3d(
            pycolmap.Rotation3d(
                cast(NDArray3x3, self.cams_from_pano_rotation[0])
            ),
            cast(NDArray3x1, np.zeros((3, 1), dtype=np.float64)),
        ).inverse()

        images_by_frame: dict[int, list[pycolmap.Image]] = collections.defaultdict(list)
        for image in reconstruction.images.values():
            if image.has_pose:
                images_by_frame[image.frame_id].append(image)

        old_to_new_point2D: dict[int, dict[int, tuple[int, str]]] = {}
        pano_to_image_id: dict[str, int] = {}
        frame_images = sorted(
            images_by_frame.values(),
            key=lambda images: self.split_image_name(images[0].name)[1],
        )

        for image_id, images in enumerate(frame_images, start=1):
            pano_name = self.split_image_name(images[0].name)[1]
            pano_to_image_id[pano_name] = image_id
            frame = images[0].frame
            assert frame is not None
            rig_from_world = frame.rig_from_world
            assert rig_from_world is not None
            pano_from_world = pano_from_ref * rig_from_world

            keypoints: list[npt.NDArray[np.floating]] = []
            for image in images:
                cam_idx = self.split_image_name(image.name)[0]
                num_points2D = len(image.points2D)
                if num_points2D == 0:
                    old_to_new_point2D[image.image_id] = {}
                    continue

                xy = np.array([point2D.xy for point2D in image.points2D])
                rays_in_cam = camera_rays_from_img(self._camera, xy)
                rays_in_cam /= np.linalg.norm(rays_in_cam, axis=-1, keepdims=True)
                rays_in_pano = rays_in_cam @ self.cams_from_pano_rotation[cam_idx]
                xy_in_pano = spherical_img_from_cam(self._pano_size, rays_in_pano)

                base_idx = len(keypoints)
                keypoints.extend(xy_in_pano)
                old_to_new_point2D[image.image_id] = {
                    point2D_idx: (base_idx + point2D_idx, pano_name)
                    for point2D_idx in range(num_points2D)
                }

            equirect.add_image_with_trivial_frame(
                pycolmap.Image(
                    name=pano_name,
                    keypoints=np.asarray(keypoints, dtype=np.float64),
                    camera_id=equirect_camera.camera_id,
                    image_id=image_id,
                ),
                pano_from_world,
            )

        for point3D_id, point3D in reconstruction.points3D.items():
            track = pycolmap.Track()
            for element in point3D.track.elements:
                new_point2D_idx, pano_name = old_to_new_point2D[element.image_id][
                    element.point2D_idx
                ]
                track.add_element(pano_to_image_id[pano_name], new_point2D_idx)
            equirect.add_point3D_with_id(
                point3D_id,
                pycolmap.Point3D(
                    xyz=point3D.xyz,
                    color=point3D.color,
                    track=track,
                ),
            )
        return equirect


def natural_image_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.as_posix().lower())


def numeric_stem(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def collect_pano_names(
    input_path: Path,
    start: int | None,
    end: int | None,
    stride: int,
) -> list[str]:
    if stride < 1:
        raise typer.BadParameter("--stride must be >= 1.")

    suffixes = {".jpg", ".jpeg", ".png"}
    paths = sorted(
        [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes],
        key=natural_image_key,
    )
    selected: list[Path] = []
    for path in paths:
        idx = numeric_stem(path)
        if start is not None or end is not None:
            if idx is None:
                continue
            if start is not None and idx < start:
                continue
            if end is not None and idx > end:
                continue
        selected.append(path)

    selected = selected[::stride]
    if not selected:
        raise typer.BadParameter("No input panoramas matched the requested range.")

    return [p.relative_to(input_path).as_posix() for p in selected]


def validate_panoramas(input_path: Path, pano_names: Sequence[str]) -> tuple[int, int]:
    expected_size: tuple[int, int] | None = None
    for pano_name in pano_names:
        with PIL.Image.open(input_path / pano_name) as image:
            width, height = image.size
        if width != height * 2:
            raise ValueError(
                f"{input_path / pano_name} is {width}x{height}, expected 2:1."
            )
        if expected_size is None:
            expected_size = (width, height)
        elif (width, height) != expected_size:
            raise ValueError(
                f"{input_path / pano_name} is {width}x{height}, "
                f"expected {expected_size[0]}x{expected_size[1]}."
            )
    assert expected_size is not None
    return expected_size


def parse_mask_rect(rect: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in rect.split(",")]
    if len(parts) != 4:
        raise typer.BadParameter(
            f"Mask rect {rect!r} must use x0,y0,x1,y1 normalized coordinates."
        )
    try:
        x0, y0, x1, y1 = [float(part) for part in parts]
    except ValueError as exc:
        raise typer.BadParameter(f"Mask rect {rect!r} contains a non-number.") from exc
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise typer.BadParameter(
            f"Mask rect {rect!r} must satisfy 0 <= x0 < x1 <= 1 "
            "and 0 <= y0 < y1 <= 1."
        )
    return (x0, y0, x1, y1)


def build_self_mask(
    pano_size: tuple[int, int],
    use_default_mask: bool,
    custom_rects: Sequence[tuple[float, float, float, float]],
) -> npt.NDArray[np.uint8] | None:
    rects: list[tuple[float, float, float, float]] = []
    if use_default_mask:
        rects.extend(DEFAULT_SELF_MASK_RECTS)
    rects.extend(custom_rects)
    if not rects:
        return None

    width, height = pano_size
    mask = np.full((height, width), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in rects:
        left = int(round(x0 * width))
        top = int(round(y0 * height))
        right = int(round(x1 * width))
        bottom = int(round(y1 * height))
        mask[top:bottom, left:right] = 0
    return mask


def overlay_mask(
    image: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    overlay = image.copy()
    masked = mask == 0
    if masked.any():
        red = np.array([255, 0, 0], dtype=np.float32)
        overlay_float = overlay.astype(np.float32)
        overlay_float[masked] = overlay_float[masked] * 0.45 + red * 0.55
        overlay = np.clip(overlay_float, 0, 255).astype(np.uint8)
    return overlay


def write_equirect_mask_preview(
    input_path: Path,
    pano_name: str,
    self_mask: npt.NDArray[np.uint8] | None,
    preview_dir: Path,
) -> None:
    if self_mask is None:
        return
    with PIL.Image.open(input_path / pano_name) as image:
        pano = np.asarray(image.convert("RGB"))
    PIL.Image.fromarray(self_mask).save(preview_dir / "equirect_self_mask.png")
    PIL.Image.fromarray(overlay_mask(pano, self_mask)).save(
        preview_dir / "equirect_self_mask_overlay.jpg", quality=95
    )


def prepare_output_path(output_path: Path, overwrite: bool) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for dirname in MANAGED_OUTPUT_DIRS:
        shutil.rmtree(output_path / dirname, ignore_errors=True)
    for filename in MANAGED_OUTPUT_FILES:
        (output_path / filename).unlink(missing_ok=True)


def render_perspective_images(
    pano_image_names: Sequence[str],
    pano_image_dir: Path,
    output_image_dir: Path,
    mask_dir: Path,
    render_options: PanoRenderOptions,
    self_mask: npt.NDArray[np.uint8] | None,
    preview_dir: Path,
    preview_count: int,
) -> PanoProcessor:
    processor = PanoProcessor(
        pano_image_dir,
        output_image_dir,
        mask_dir,
        render_options,
        self_mask,
        preview_dir,
        preview_count,
    )
    max_workers = min(32, max(1, (os.cpu_count() or 2) - 1))
    with tqdm(total=len(pano_image_names), desc="render") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as thread_pool:
            futures = [
                thread_pool.submit(processor.process, pano_name)
                for pano_name in pano_image_names
            ]
            for future in as_completed(futures):
                future.result()
                pbar.update(1)
    return processor


def create_sift_extraction_options() -> object | None:
    options_cls = getattr(pycolmap, "SiftExtractionOptions", None)
    if options_cls is None:
        options_cls = getattr(pycolmap, "FeatureExtractionOptions", None)
    if options_cls is None:
        return None

    options = options_cls()
    if hasattr(options, "use_gpu"):
        options.use_gpu = bool(getattr(pycolmap, "has_cuda", False))
    return options


def create_sift_matching_options() -> object:
    options_cls = getattr(pycolmap, "SiftMatchingOptions", None)
    if options_cls is None:
        options_cls = getattr(pycolmap, "FeatureMatchingOptions")

    options = options_cls()
    if hasattr(options, "use_gpu"):
        options.use_gpu = bool(getattr(pycolmap, "has_cuda", False))
    if hasattr(options, "rig_verification"):
        options.rig_verification = True
    if hasattr(options, "skip_image_pairs_in_same_frame"):
        options.skip_image_pairs_in_same_frame = True
    return options


@contextmanager
def open_colmap_database(path: Path):
    try:
        database_context = pycolmap.Database.open(str(path))
    except TypeError:
        database = pycolmap.Database()
        database.open(str(path))
        try:
            yield database
        finally:
            database.close()
    else:
        with database_context as database:
            yield database


def run_matcher(
    matcher: Matcher,
    database_path: Path,
) -> None:
    if hasattr(pycolmap, "FeatureMatchingOptions"):
        matching_options = create_sift_matching_options()
        if matcher == Matcher.SEQUENTIAL:
            pycolmap.match_sequential(
                str(database_path),
                pairing_options=pycolmap.SequentialPairingOptions(
                    loop_detection=True
                ),
                matching_options=matching_options,
            )
        elif matcher == Matcher.EXHAUSTIVE:
            pycolmap.match_exhaustive(
                str(database_path), matching_options=matching_options
            )
        elif matcher == Matcher.VOCABTREE:
            pycolmap.match_vocabtree(
                str(database_path), matching_options=matching_options
            )
        elif matcher == Matcher.SPATIAL:
            pycolmap.match_spatial(str(database_path), matching_options=matching_options)
        else:
            raise ValueError(f"Unknown matcher: {matcher}")
        return

    sift_options = create_sift_matching_options()
    verification_options = pycolmap.TwoViewGeometryOptions()
    if matcher == Matcher.SEQUENTIAL:
        sequential_options = pycolmap.SequentialMatchingOptions()
        sequential_options.loop_detection = True
        pycolmap.match_sequential(
            str(database_path),
            sift_options=sift_options,
            matching_options=sequential_options,
            verification_options=verification_options,
        )
    elif matcher == Matcher.EXHAUSTIVE:
        pycolmap.match_exhaustive(
            str(database_path),
            sift_options=sift_options,
            matching_options=pycolmap.ExhaustiveMatchingOptions(),
            verification_options=verification_options,
        )
    elif matcher == Matcher.VOCABTREE:
        pycolmap.match_vocabtree(
            str(database_path),
            sift_options=sift_options,
            matching_options=pycolmap.VocabTreeMatchingOptions(),
            verification_options=verification_options,
        )
    elif matcher == Matcher.SPATIAL:
        pycolmap.match_spatial(
            str(database_path),
            sift_options=sift_options,
            matching_options=pycolmap.SpatialMatchingOptions(),
            verification_options=verification_options,
        )
    else:
        raise ValueError(f"Unknown matcher: {matcher}")


def choose_best_reconstruction(
    recs: dict[int, pycolmap.Reconstruction],
) -> tuple[int | None, pycolmap.Reconstruction | None]:
    if not recs:
        return None, None

    def score(item: tuple[int, pycolmap.Reconstruction]) -> tuple[int, int]:
        _, rec = item
        registered = sum(1 for image in rec.images.values() if image.has_pose)
        return (registered, len(rec.points3D))

    return max(recs.items(), key=score)


def safe_float_method(obj: object, method_names: Sequence[str]) -> float | None:
    for method_name in method_names:
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                value = method()
            except TypeError:
                continue
            if value is not None:
                return float(value)
    return None


def write_best_models(
    best_rec: pycolmap.Reconstruction,
    processor: PanoProcessor,
    sparse_path: Path,
    sparse_equirect_path: Path,
) -> ModelWriteResult:
    best_sparse_path = sparse_path / "0"
    shutil.rmtree(best_sparse_path, ignore_errors=True)
    best_sparse_path.mkdir(parents=True, exist_ok=True)
    best_rec.write(str(best_sparse_path))

    if not hasattr(pycolmap.CameraModelId, "EQUIRECTANGULAR"):
        return ModelWriteResult(
            equirectangular_exported=False,
            equirectangular_export_skip_reason=(
                "PyCOLMAP does not expose CameraModelId.EQUIRECTANGULAR."
            ),
        )

    equirect_rec = processor.convert_to_equirectangular(best_rec)
    best_equirect_path = sparse_equirect_path / "0"
    shutil.rmtree(best_equirect_path, ignore_errors=True)
    best_equirect_path.mkdir(parents=True, exist_ok=True)
    equirect_rec.write(str(best_equirect_path))
    return ModelWriteResult(equirectangular_exported=True)


def summarize_registration(
    *,
    output_path: Path,
    input_path: Path,
    pano_names: Sequence[str],
    processor: PanoProcessor,
    recs: dict[int, pycolmap.Reconstruction],
    best_model_id: int | None,
    best_rec: pycolmap.Reconstruction | None,
    expected_rendered_images: int,
    min_registered_pano_ratio: float,
    model_write_result: ModelWriteResult,
) -> dict[str, object]:
    rendered_image_count = sum(1 for _ in (output_path / "images").rglob("*.jpg"))
    mask_count = sum(1 for _ in (output_path / "masks").rglob("*.png"))
    model_summaries = {
        str(idx): {
            "summary": rec.summary(),
            "registered_virtual_images": sum(
                1 for image in rec.images.values() if image.has_pose
            ),
            "points3D": len(rec.points3D),
        }
        for idx, rec in recs.items()
    }

    registered_virtual_images = 0
    registered_panos: set[str] = set()
    point_count = 0
    mean_reprojection_error = None
    if best_rec is not None:
        registered_virtual_images = sum(
            1 for image in best_rec.images.values() if image.has_pose
        )
        for image in best_rec.images.values():
            if image.has_pose:
                _, pano_name = processor.split_image_name(image.name)
                registered_panos.add(pano_name)
        point_count = len(best_rec.points3D)
        mean_reprojection_error = safe_float_method(
            best_rec,
            (
                "compute_mean_reprojection_error",
                "mean_reprojection_error",
            ),
        )

    selected_pano_count = len(pano_names)
    registered_pano_ratio = (
        len(registered_panos) / selected_pano_count if selected_pano_count else 0.0
    )
    registered_virtual_image_ratio = (
        registered_virtual_images / expected_rendered_images
        if expected_rendered_images
        else 0.0
    )
    passed = registered_pano_ratio >= min_registered_pano_ratio

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "selected_pano_count": selected_pano_count,
        "selected_panos_first": list(pano_names[:5]),
        "selected_panos_last": list(pano_names[-5:]),
        "virtual_camera_count": processor.virtual_camera_count,
        "expected_rendered_images": expected_rendered_images,
        "rendered_image_count": rendered_image_count,
        "mask_count": mask_count,
        "best_model_id": best_model_id,
        "model_count": len(recs),
        "model_summaries": model_summaries,
        "registered_virtual_images": registered_virtual_images,
        "registered_virtual_image_ratio": registered_virtual_image_ratio,
        "registered_panos": len(registered_panos),
        "registered_pano_ratio": registered_pano_ratio,
        "points3D": point_count,
        "mean_reprojection_error": mean_reprojection_error,
        "min_registered_pano_ratio": min_registered_pano_ratio,
        "passed_registration_threshold": passed,
        "equirectangular_exported": model_write_result.equirectangular_exported,
        "equirectangular_export_skip_reason": (
            model_write_result.equirectangular_export_skip_reason
        ),
        "output_size_bytes": directory_size(output_path),
    }


def directory_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            try:
                total += (Path(root) / filename).stat().st_size
            except OSError:
                pass
    return total


def choose_default_threshold(selected_count: int, stride: int) -> float:
    if selected_count >= 250 and stride == 1:
        return 0.75
    return 0.70


@app.command()
def register(
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing 2:1 equirectangular panorama images.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        file_okay=False,
        dir_okay=True,
        help="Run output directory.",
    ),
    start: int | None = typer.Option(None, help="Minimum numeric image stem to use."),
    end: int | None = typer.Option(None, help="Maximum numeric image stem to use."),
    stride: int = typer.Option(1, help="Use every Nth selected panorama."),
    matcher: Matcher = typer.Option(Matcher.SEQUENTIAL, help="COLMAP matcher."),
    mapper: Mapper = typer.Option(Mapper.INCREMENTAL, help="COLMAP mapper."),
    pano_render_type: PanoRenderType = typer.Option(
        PanoRenderType.PERSPECTIVE_OVERLAPPING,
        help="Virtual perspective rendering layout.",
    ),
    no_self_mask: bool = typer.Option(
        False,
        "--no-self-mask",
        help="Disable the default coarse operator/arms mask.",
    ),
    mask_rect: list[str] | None = typer.Option(
        None,
        "--mask-rect",
        help="Extra normalized self-mask rectangle as x0,y0,x1,y1. Repeatable.",
    ),
    preview_count: int = typer.Option(6, help="Number of rendered view previews."),
    min_registered_pano_ratio: float | None = typer.Option(
        None,
        help="Registration pass threshold. Defaults to 0.70 for smoke, 0.75 for full.",
    ),
    overwrite: bool = typer.Option(
        True,
        "--overwrite/--no-overwrite",
        help="Clean this tool's managed outputs inside the run directory first.",
    ),
) -> None:
    """Render virtual perspective views, apply masks, and run pycolmap mapping."""
    pycolmap.set_random_seed(0)
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    custom_rects = [parse_mask_rect(rect) for rect in (mask_rect or [])]

    prepare_output_path(output_path, overwrite=overwrite)
    image_dir = output_path / "images"
    masks_dir = output_path / "masks"
    previews_dir = output_path / "previews"
    reports_dir = output_path / "reports"
    sparse_path = output_path / "sparse"
    sparse_equirect_path = output_path / "sparse_equirectangular"
    for path in (image_dir, masks_dir, previews_dir, reports_dir, sparse_path):
        path.mkdir(parents=True, exist_ok=True)

    pano_names = collect_pano_names(input_path, start, end, stride)
    pano_size = validate_panoramas(input_path, pano_names)
    self_mask = build_self_mask(
        pano_size,
        use_default_mask=not no_self_mask,
        custom_rects=custom_rects,
    )
    if self_mask is not None:
        PIL.Image.fromarray(self_mask).save(output_path / "self_mask.png")
    write_equirect_mask_preview(input_path, pano_names[0], self_mask, previews_dir)

    threshold = (
        min_registered_pano_ratio
        if min_registered_pano_ratio is not None
        else choose_default_threshold(len(pano_names), stride)
    )
    render_options = PANO_RENDER_OPTIONS[pano_render_type]
    expected_rendered_images = (
        len(pano_names) * len(get_virtual_rotations(
            render_options.num_steps_yaw,
            render_options.pitches_deg,
        ))
    )

    console.print(
        f"Selected {len(pano_names)} panoramas from {input_path} "
        f"({pano_size[0]}x{pano_size[1]})."
    )
    console.print(f"Rendering {expected_rendered_images} virtual perspective views.")
    processor = render_perspective_images(
        pano_names,
        input_path,
        image_dir,
        masks_dir,
        render_options,
        self_mask,
        previews_dir,
        preview_count,
    )

    database_path = output_path / "database.db"
    rendered_camera = processor.rig_config.cameras[0].camera
    assert rendered_camera is not None

    console.print("Extracting features with pycolmap.")
    extraction_kwargs: dict[str, object] = {}
    sift_extraction_options = create_sift_extraction_options()
    if sift_extraction_options is not None:
        if hasattr(pycolmap, "SiftExtractionOptions"):
            extraction_kwargs["sift_options"] = sift_extraction_options
            extraction_kwargs["camera_model"] = camera_model_name(rendered_camera)
        else:
            extraction_kwargs["extraction_options"] = sift_extraction_options

    pycolmap.extract_features(
        str(database_path),
        str(image_dir),
        reader_options=pycolmap.ImageReaderOptions(
            mask_path=str(masks_dir),
            camera_model=camera_model_name(rendered_camera),
            camera_params=camera_params_string(rendered_camera),
        ),
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
        **extraction_kwargs,
    )

    with open_colmap_database(database_path) as db:
        pycolmap.apply_rig_config([processor.rig_config], db)

    console.print(f"Matching features with {matcher.value}.")
    run_matcher(matcher, database_path)

    console.print(f"Running {mapper.value} mapping.")
    if mapper == Mapper.INCREMENTAL:
        opts = pycolmap.IncrementalPipelineOptions()
        opts.ba_refine_sensor_from_rig = False
        opts.ba_refine_focal_length = False
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params = False
        if hasattr(opts, "ba_use_gpu"):
            # The conda-forge Jetson CUDA build accelerates SIFT, but its Ceres
            # build does not currently support CUDA bundle adjustment reliably.
            opts.ba_use_gpu = False
        recs = pycolmap.incremental_mapping(
            str(database_path), str(image_dir), str(sparse_path), opts
        )
    elif mapper == Mapper.GLOBAL:
        if not hasattr(pycolmap, "global_mapping"):
            raise typer.BadParameter(
                "The selected PyCOLMAP build does not expose global_mapping. "
                "Use --mapper incremental."
            )
        global_opts = pycolmap.GlobalPipelineOptions(
            mapper=pycolmap.GlobalMapperOptions(refine_sensor_from_rig=False)
        )
        global_opts.mapper.bundle_adjustment.refine_focal_length = False
        global_opts.mapper.bundle_adjustment.refine_principal_point = False
        global_opts.mapper.bundle_adjustment.refine_extra_params = False
        recs = pycolmap.global_mapping(
            str(database_path), str(image_dir), str(sparse_path), global_opts
        )
    else:
        raise ValueError(f"Unknown mapper: {mapper}")

    best_model_id, best_rec = choose_best_reconstruction(recs)
    model_write_result = ModelWriteResult(
        equirectangular_exported=False,
        equirectangular_export_skip_reason="No reconstruction was produced.",
    )
    if best_rec is not None:
        console.print(f"Writing best model {best_model_id} to sparse/0.")
        model_write_result = write_best_models(
            best_rec, processor, sparse_path, sparse_equirect_path
        )
        if model_write_result.equirectangular_export_skip_reason:
            console.print(
                "Skipped sparse_equirectangular export: "
                f"{model_write_result.equirectangular_export_skip_reason}"
            )

    summary = summarize_registration(
        output_path=output_path,
        input_path=input_path,
        pano_names=pano_names,
        processor=processor,
        recs=recs,
        best_model_id=best_model_id,
        best_rec=best_rec,
        expected_rendered_images=expected_rendered_images,
        min_registered_pano_ratio=threshold,
        model_write_result=model_write_result,
    )
    summary_path = reports_dir / "registration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    console.print(f"Wrote registration report: {summary_path}")
    console.print(
        "Registered "
        f"{summary['registered_panos']}/{summary['selected_pano_count']} panoramas "
        f"({float(summary['registered_pano_ratio']):.1%})."
    )
    if not summary["passed_registration_threshold"]:
        raise typer.Exit(code=2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
