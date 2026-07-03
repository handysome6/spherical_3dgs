from __future__ import annotations

import collections
import enum
import json
import os
import shlex
import shutil
import subprocess
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


class FisheyeLayout(enum.StrEnum):
    SINGLE = enum.auto()
    DUAL = enum.auto()


class FisheyeCameraModel(enum.StrEnum):
    OPENCV_FISHEYE = "OPENCV_FISHEYE"
    RADIAL_FISHEYE = "RADIAL_FISHEYE"
    SIMPLE_RADIAL_FISHEYE = "SIMPLE_RADIAL_FISHEYE"
    FOV = "FOV"
    THIN_PRISM_FISHEYE = "THIN_PRISM_FISHEYE"
    RAD_TAN_THIN_PRISM_FISHEYE = "RAD_TAN_THIN_PRISM_FISHEYE"


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


@dataclass(frozen=True)
class FisheyeImageSelection:
    image_names: list[str]
    image_counts_by_sensor: dict[str, int]
    image_sizes_by_sensor: dict[str, tuple[int, int]]


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
    "logs",
    "masks",
    "previews",
    "reports",
    "sparse",
    "sparse_equirectangular",
)
MANAGED_OUTPUT_FILES = (
    "colmap_commands.json",
    "database.db",
    "database.db-shm",
    "database.db-wal",
    "fisheye_config.json",
    "image_list.txt",
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


def collect_fisheye_images(
    input_path: Path,
    layout: FisheyeLayout,
    sensor_folders: Sequence[str],
    start: int | None,
    end: int | None,
    stride: int,
) -> FisheyeImageSelection:
    if stride < 1:
        raise typer.BadParameter("--stride must be >= 1.")

    suffixes = {".jpg", ".jpeg", ".png"}
    if layout == FisheyeLayout.SINGLE:
        sensors = [(".", input_path)]
    elif layout == FisheyeLayout.DUAL:
        if not sensor_folders:
            sensor_folders = ("fisheye_left", "fisheye_right")
        if len(sensor_folders) < 2:
            raise typer.BadParameter(
                "Dual fisheye layout requires at least two --sensor-folder values."
            )
        sensors = []
        for folder in sensor_folders:
            folder_path = Path(folder)
            if folder_path.is_absolute() or ".." in folder_path.parts:
                raise typer.BadParameter(
                    f"Sensor folder {folder!r} must be a relative path."
                )
            sensors.append((folder_path.as_posix(), input_path / folder_path))
    else:
        raise ValueError(f"Unknown fisheye layout: {layout}")

    image_names_by_sensor: dict[str, list[str]] = {}
    image_counts_by_sensor: dict[str, int] = {}
    image_sizes_by_sensor: dict[str, tuple[int, int]] = {}

    for sensor_name, sensor_root in sensors:
        if not sensor_root.is_dir():
            raise typer.BadParameter(f"Sensor image directory does not exist: {sensor_root}")

        paths = sorted(
            [
                p
                for p in sensor_root.rglob("*")
                if p.is_file() and p.suffix.lower() in suffixes
            ],
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
            raise typer.BadParameter(
                f"No images matched the requested range for sensor {sensor_name}."
            )

        expected_size: tuple[int, int] | None = None
        image_names: list[str] = []
        for path in selected:
            with PIL.Image.open(path) as image:
                size = image.size
            if expected_size is None:
                expected_size = size
            elif size != expected_size:
                raise ValueError(
                    f"{path} is {size[0]}x{size[1]}, "
                    f"expected {expected_size[0]}x{expected_size[1]} for "
                    f"sensor {sensor_name}."
                )
            image_names.append(path.relative_to(input_path).as_posix())

        assert expected_size is not None
        image_names_by_sensor[sensor_name] = image_names
        image_counts_by_sensor[sensor_name] = len(image_names)
        image_sizes_by_sensor[sensor_name] = expected_size

    if layout == FisheyeLayout.SINGLE:
        ordered_image_names = next(iter(image_names_by_sensor.values()))
    else:
        ordered_image_names = []
        max_sensor_count = max(len(names) for names in image_names_by_sensor.values())
        for idx in range(max_sensor_count):
            for sensor_name, _ in sensors:
                sensor_names = image_names_by_sensor[sensor_name]
                if idx < len(sensor_names):
                    ordered_image_names.append(sensor_names[idx])

    return FisheyeImageSelection(
        image_names=ordered_image_names,
        image_counts_by_sensor=image_counts_by_sensor,
        image_sizes_by_sensor=image_sizes_by_sensor,
    )


def stage_fisheye_images(
    input_path: Path,
    output_image_dir: Path,
    image_names: Sequence[str],
    copy_images: bool,
) -> None:
    for image_name in image_names:
        src = input_path / image_name
        dst = output_image_dir / image_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy_images:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)


def write_image_list(image_list_path: Path, image_names: Sequence[str]) -> None:
    image_list_path.write_text(
        "".join(f"{image_name}\n" for image_name in image_names),
        encoding="utf-8",
    )


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


def parse_lens_circle(circle: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in circle.split(",")]
    if len(parts) != 3:
        raise typer.BadParameter(
            f"Lens circle {circle!r} must use cx,cy,r normalized coordinates."
        )
    try:
        cx, cy, radius = [float(part) for part in parts]
    except ValueError as exc:
        raise typer.BadParameter(f"Lens circle {circle!r} contains a non-number.") from exc
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < radius <= 1.0):
        raise typer.BadParameter(
            f"Lens circle {circle!r} must satisfy 0 <= cx,cy <= 1 "
            "and 0 < r <= 1."
        )
    return (cx, cy, radius)


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


def build_fisheye_mask(
    image_size: tuple[int, int],
    mask_rects: Sequence[tuple[float, float, float, float]],
    lens_circle: tuple[float, float, float] | None,
) -> npt.NDArray[np.uint8] | None:
    if not mask_rects and lens_circle is None:
        return None

    width, height = image_size
    mask = np.full((height, width), 255, dtype=np.uint8)
    if lens_circle is not None:
        cx, cy, radius = lens_circle
        center_x = cx * width
        center_y = cy * height
        pixel_radius = radius * min(width, height)
        yy, xx = np.ogrid[:height, :width]
        outside_lens = (xx - center_x) ** 2 + (yy - center_y) ** 2 > pixel_radius**2
        mask[outside_lens] = 0

    for x0, y0, x1, y1 in mask_rects:
        left = int(round(x0 * width))
        top = int(round(y0 * height))
        right = int(round(x1 * width))
        bottom = int(round(y1 * height))
        mask[top:bottom, left:right] = 0
    return mask


def write_fisheye_masks(
    image_dir: Path,
    mask_dir: Path,
    preview_dir: Path,
    image_names: Sequence[str],
    mask_rects: Sequence[tuple[float, float, float, float]],
    lens_circle: tuple[float, float, float] | None,
    preview_count: int,
) -> int:
    if not mask_rects and lens_circle is None:
        return 0

    mask_count = 0
    for image_name in image_names:
        image_path = image_dir / image_name
        with PIL.Image.open(image_path) as pil_image:
            image = np.asarray(pil_image.convert("RGB"))
            image_size = pil_image.size

        mask = build_fisheye_mask(image_size, mask_rects, lens_circle)
        assert mask is not None
        mask_path = mask_dir / f"{image_name}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not pycolmap.Bitmap.from_array(mask).write(str(mask_path)):
            raise RuntimeError(f"Cannot write {mask_path}")

        if mask_count < preview_count:
            preview_name = image_name.replace("/", "_")
            PIL.Image.fromarray(image).save(preview_dir / preview_name)
            PIL.Image.fromarray(mask).save(preview_dir / f"{preview_name}.mask.png")
            PIL.Image.fromarray(overlay_mask(image, mask)).save(
                preview_dir / f"{preview_name}.mask.jpg",
                quality=95,
            )
        mask_count += 1
    return mask_count


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


def bool_arg(value: bool) -> str:
    return "1" if value else "0"


def command_string(command: Sequence[str]) -> str:
    return shlex.join(command)


def run_logged_command(label: str, command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"$ {command_string(command)}\n\n", encoding="utf-8")
    console.print(f"Running {label}. Log: {log_path}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"COLMAP binary {command[0]!r} was not found."
        ) from exc

    with log_path.open("a", encoding="utf-8") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"{label} failed with exit code {return_code}. See {log_path}."
        )


def build_fisheye_colmap_commands(
    *,
    colmap_binary: str,
    database_path: Path,
    image_dir: Path,
    masks_dir: Path,
    sparse_path: Path,
    image_list_path: Path,
    layout: FisheyeLayout,
    camera_model: FisheyeCameraModel,
    camera_params: str | None,
    mask_count: int,
    matcher: Matcher,
    loop_detection: bool,
    vocab_tree_path: Path | None,
    use_gpu: bool,
    gpu_index: str,
    fix_intrinsics: bool,
    refine_principal_point: bool,
) -> dict[str, list[str]]:
    feature_command = [
        colmap_binary,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--image_list_path",
        str(image_list_path),
        "--ImageReader.camera_model",
        camera_model.value,
        "--FeatureExtraction.use_gpu",
        bool_arg(use_gpu),
        "--FeatureExtraction.gpu_index",
        gpu_index,
    ]
    if camera_params:
        feature_command.extend(["--ImageReader.camera_params", camera_params])
    if mask_count:
        feature_command.extend(["--ImageReader.mask_path", str(masks_dir)])
    if layout == FisheyeLayout.SINGLE:
        feature_command.extend(["--ImageReader.single_camera", "1"])
    else:
        feature_command.extend(["--ImageReader.single_camera_per_folder", "1"])

    matcher_command_name = {
        Matcher.SEQUENTIAL: "sequential_matcher",
        Matcher.EXHAUSTIVE: "exhaustive_matcher",
        Matcher.VOCABTREE: "vocab_tree_matcher",
        Matcher.SPATIAL: "spatial_matcher",
    }[matcher]
    matcher_command = [
        colmap_binary,
        matcher_command_name,
        "--database_path",
        str(database_path),
        "--FeatureMatching.use_gpu",
        bool_arg(use_gpu),
        "--FeatureMatching.gpu_index",
        gpu_index,
    ]
    if matcher == Matcher.SEQUENTIAL:
        matcher_command.extend(
            ["--SequentialMatching.loop_detection", bool_arg(loop_detection)]
        )
    elif matcher == Matcher.VOCABTREE:
        if vocab_tree_path is None:
            raise typer.BadParameter(
                "--vocab-tree-path is required when --matcher vocabtree is used."
            )
        matcher_command.extend(
            ["--VocabTreeMatching.vocab_tree_path", str(vocab_tree_path)]
        )

    refine_intrinsics = not fix_intrinsics
    mapper_command = [
        colmap_binary,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse_path),
        "--Mapper.ba_use_gpu",
        bool_arg(use_gpu),
        "--Mapper.ba_gpu_index",
        gpu_index,
        "--Mapper.ba_refine_focal_length",
        bool_arg(refine_intrinsics),
        "--Mapper.ba_refine_principal_point",
        bool_arg(refine_intrinsics and refine_principal_point),
        "--Mapper.ba_refine_extra_params",
        bool_arg(refine_intrinsics),
    ]

    return {
        "feature_extractor": feature_command,
        matcher.value: matcher_command,
        "mapper": mapper_command,
    }


def sparse_model_paths(sparse_path: Path) -> list[Path]:
    if not sparse_path.exists():
        return []

    def sort_key(path: Path) -> tuple[int, int | str]:
        try:
            return (0, int(path.name))
        except ValueError:
            return (1, path.name)

    paths = []
    for path in sorted(sparse_path.iterdir(), key=sort_key):
        if not path.is_dir():
            continue
        if (path / "cameras.bin").exists() or (path / "cameras.txt").exists():
            paths.append(path)
    return paths


def summarize_sparse_model(model_path: Path) -> dict[str, object]:
    try:
        reconstruction = pycolmap.Reconstruction(str(model_path))
    except Exception as exc:
        return {
            "path": str(model_path),
            "read_error": f"{type(exc).__name__}: {exc}",
        }

    registered_images = sum(
        1 for image in reconstruction.images.values() if image.has_pose
    )
    mean_reprojection_error = safe_float_method(
        reconstruction,
        (
            "compute_mean_reprojection_error",
            "mean_reprojection_error",
        ),
    )
    cameras = {
        str(camera_id): {
            "model": camera_model_name(camera),
            "width": camera.width,
            "height": camera.height,
            "params": [float(value) for value in np.asarray(camera.params)],
            "params_info": getattr(camera, "params_info", None),
        }
        for camera_id, camera in reconstruction.cameras.items()
    }
    return {
        "path": str(model_path),
        "summary": reconstruction.summary(),
        "registered_images": registered_images,
        "images": len(reconstruction.images),
        "points3D": len(reconstruction.points3D),
        "mean_reprojection_error": mean_reprojection_error,
        "cameras": cameras,
    }


def choose_best_sparse_model(
    model_summaries: dict[str, dict[str, object]],
) -> tuple[str | None, dict[str, object] | None]:
    if not model_summaries:
        return None, None

    def score(item: tuple[str, dict[str, object]]) -> tuple[int, int]:
        _, summary = item
        return (
            int(summary.get("registered_images") or 0),
            int(summary.get("points3D") or 0),
        )

    return max(model_summaries.items(), key=score)


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


@app.command("register-fisheye")
def register_fisheye(
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing raw fisheye image frames.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        file_okay=False,
        dir_okay=True,
        help="Run output directory.",
    ),
    layout: FisheyeLayout = typer.Option(
        FisheyeLayout.SINGLE,
        help="Raw fisheye layout: single stream or dual sensor folders.",
    ),
    sensor_folder: list[str] | None = typer.Option(
        None,
        "--sensor-folder",
        help=(
            "Dual-fisheye sensor folder relative to --input. Repeatable. "
            "Defaults to fisheye_left and fisheye_right."
        ),
    ),
    camera_model: FisheyeCameraModel = typer.Option(
        FisheyeCameraModel.OPENCV_FISHEYE,
        help="COLMAP fisheye camera model.",
    ),
    camera_params: str | None = typer.Option(
        None,
        help="Optional COLMAP camera params, for example fx,fy,cx,cy,k1,k2,k3,k4.",
    ),
    start: int | None = typer.Option(None, help="Minimum numeric image stem to use."),
    end: int | None = typer.Option(None, help="Maximum numeric image stem to use."),
    stride: int = typer.Option(1, help="Use every Nth selected raw frame."),
    matcher: Matcher = typer.Option(Matcher.SEQUENTIAL, help="COLMAP matcher."),
    loop_detection: bool = typer.Option(
        True,
        "--loop-detection/--no-loop-detection",
        help="Enable loop detection for sequential matching.",
    ),
    vocab_tree_path: Path | None = typer.Option(
        None,
        help="Vocabulary tree path required by --matcher vocabtree.",
    ),
    mask_rect: list[str] | None = typer.Option(
        None,
        "--mask-rect",
        help="Raw-image normalized mask rectangle as x0,y0,x1,y1. Repeatable.",
    ),
    lens_circle: str | None = typer.Option(
        None,
        help="Keep only a normalized circular lens region as cx,cy,r.",
    ),
    preview_count: int = typer.Option(6, help="Number of raw mask previews."),
    fix_intrinsics: bool = typer.Option(
        False,
        "--fix-intrinsics",
        help="Disable focal length, principal point, and distortion refinement.",
    ),
    refine_principal_point: bool = typer.Option(
        False,
        "--refine-principal-point",
        help="Allow mapper bundle adjustment to refine principal point.",
    ),
    use_gpu: bool = typer.Option(
        True,
        "--use-gpu/--no-use-gpu",
        help="Use CUDA for SIFT extraction, matching, and mapper BA.",
    ),
    gpu_index: str = typer.Option("0", help="COLMAP GPU index."),
    colmap_binary: str = typer.Option("colmap-cuda", help="COLMAP executable."),
    copy_images: bool = typer.Option(
        False,
        "--copy-images/--symlink-images",
        help="Copy selected images into the run directory instead of symlinking.",
    ),
    min_registered_image_ratio: float = typer.Option(
        0.75,
        help="Registration pass threshold for selected raw images.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Prepare files and command logs without running COLMAP.",
    ),
    overwrite: bool = typer.Option(
        True,
        "--overwrite/--no-overwrite",
        help="Clean this tool's managed outputs inside the run directory first.",
    ),
) -> None:
    """Run fisheye-aware COLMAP SfM on raw fisheye image frames."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if layout == FisheyeLayout.DUAL:
        sensor_folders = tuple(sensor_folder or ("fisheye_left", "fisheye_right"))
    else:
        sensor_folders = ()
    if fix_intrinsics and not camera_params:
        raise typer.BadParameter("--fix-intrinsics requires --camera-params.")
    custom_rects = [parse_mask_rect(rect) for rect in (mask_rect or [])]
    parsed_lens_circle = parse_lens_circle(lens_circle) if lens_circle else None

    selection = collect_fisheye_images(
        input_path=input_path,
        layout=layout,
        sensor_folders=sensor_folders,
        start=start,
        end=end,
        stride=stride,
    )

    prepare_output_path(output_path, overwrite=overwrite)
    image_dir = output_path / "images"
    masks_dir = output_path / "masks"
    logs_dir = output_path / "logs"
    previews_dir = output_path / "previews"
    reports_dir = output_path / "reports"
    sparse_path = output_path / "sparse"
    for path in (image_dir, masks_dir, logs_dir, previews_dir, reports_dir, sparse_path):
        path.mkdir(parents=True, exist_ok=True)

    console.print(
        f"Selected {len(selection.image_names)} raw fisheye images from {input_path}."
    )
    stage_fisheye_images(
        input_path=input_path,
        output_image_dir=image_dir,
        image_names=selection.image_names,
        copy_images=copy_images,
    )
    image_list_path = output_path / "image_list.txt"
    write_image_list(image_list_path, selection.image_names)
    mask_count = write_fisheye_masks(
        image_dir=image_dir,
        mask_dir=masks_dir,
        preview_dir=previews_dir,
        image_names=selection.image_names,
        mask_rects=custom_rects,
        lens_circle=parsed_lens_circle,
        preview_count=preview_count,
    )

    database_path = output_path / "database.db"
    commands = build_fisheye_colmap_commands(
        colmap_binary=colmap_binary,
        database_path=database_path,
        image_dir=image_dir,
        masks_dir=masks_dir,
        sparse_path=sparse_path,
        image_list_path=image_list_path,
        layout=layout,
        camera_model=camera_model,
        camera_params=camera_params,
        mask_count=mask_count,
        matcher=matcher,
        loop_detection=loop_detection,
        vocab_tree_path=vocab_tree_path,
        use_gpu=use_gpu,
        gpu_index=gpu_index,
        fix_intrinsics=fix_intrinsics,
        refine_principal_point=refine_principal_point,
    )

    config = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "layout": layout.value,
        "sensor_folders": list(sensor_folders),
        "camera_model": camera_model.value,
        "camera_params": camera_params,
        "start": start,
        "end": end,
        "stride": stride,
        "matcher": matcher.value,
        "loop_detection": loop_detection,
        "mask_rects": custom_rects,
        "lens_circle": parsed_lens_circle,
        "fix_intrinsics": fix_intrinsics,
        "refine_principal_point": refine_principal_point,
        "use_gpu": use_gpu,
        "gpu_index": gpu_index,
        "colmap_binary": colmap_binary,
        "copy_images": copy_images,
    }
    (output_path / "fisheye_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    (output_path / "colmap_commands.json").write_text(
        json.dumps(
            {
                label: {
                    "argv": command,
                    "command": command_string(command),
                }
                for label, command in commands.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if dry_run:
        summary = {
            **config,
            "dry_run": True,
            "selected_image_count": len(selection.image_names),
            "image_counts_by_sensor": selection.image_counts_by_sensor,
            "image_sizes_by_sensor": selection.image_sizes_by_sensor,
            "mask_count": mask_count,
            "commands_path": str(output_path / "colmap_commands.json"),
            "output_size_bytes": directory_size(output_path),
        }
        summary_path = reports_dir / "registration_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        console.print(f"Dry run complete. Wrote report: {summary_path}")
        return

    for label, command in commands.items():
        run_logged_command(label, command, logs_dir / f"{label}.log")

    model_summaries = {
        model_path.name: summarize_sparse_model(model_path)
        for model_path in sparse_model_paths(sparse_path)
    }
    best_model_id, best_model = choose_best_sparse_model(model_summaries)
    registered_images = int(best_model.get("registered_images") or 0) if best_model else 0
    selected_image_count = len(selection.image_names)
    registered_image_ratio = (
        registered_images / selected_image_count if selected_image_count else 0.0
    )
    passed = registered_image_ratio >= min_registered_image_ratio

    summary = {
        **config,
        "dry_run": False,
        "selected_image_count": selected_image_count,
        "image_counts_by_sensor": selection.image_counts_by_sensor,
        "image_sizes_by_sensor": selection.image_sizes_by_sensor,
        "mask_count": mask_count,
        "best_model_id": best_model_id,
        "model_count": len(model_summaries),
        "model_summaries": model_summaries,
        "registered_images": registered_images,
        "registered_image_ratio": registered_image_ratio,
        "points3D": int(best_model.get("points3D") or 0) if best_model else 0,
        "mean_reprojection_error": (
            best_model.get("mean_reprojection_error") if best_model else None
        ),
        "min_registered_image_ratio": min_registered_image_ratio,
        "passed_registration_threshold": passed,
        "output_size_bytes": directory_size(output_path),
    }
    summary_path = reports_dir / "registration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    console.print(f"Wrote registration report: {summary_path}")
    console.print(
        "Registered "
        f"{registered_images}/{selected_image_count} raw images "
        f"({registered_image_ratio:.1%})."
    )
    if not passed:
        raise typer.Exit(code=2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
