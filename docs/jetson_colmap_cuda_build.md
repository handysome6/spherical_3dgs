# Jetson COLMAP CUDA Build Process

Date: 2026-07-03

This note records the validated source build process for COLMAP 4.1.0 on the
Jetson platform, using the Jetson CUDA 12.6 stack and a CUDA-enabled Ceres build
with cuDSS support.

The target machine for this build was:

- Jetson / L4T R36.4.x, aarch64
- CUDA 12.6
- Orin GPU, compute capability `sm_87`
- Ubuntu 22.04 userland in the NVIDIA L4T JetPack container

The final packages are stored in:

```text
dist/colmap-cuda-sm87_4.1.0+cuda12.6.sm87-1_arm64.deb
dist/colmap-cuda-sm87_4.1.0+cuda12.6.sm87.gui-1_arm64.deb
```

## Build Outputs

### CLI Package

```text
dist/colmap-cuda-sm87_4.1.0+cuda12.6.sm87-1_arm64.deb
SHA256 59b5aa46fdd5b258b4b8a79e5962adcdf07db65e141366d737b978364bbe8f4d
```

Build characteristics:

- COLMAP 4.1.0, commit `fa8e3b3ff591552855f8ad2806723c80f963f69c`
- Ceres 2.3.0, commit `bac1127f9ef672405bd0d2d9c84e809ae89bd239`
- CUDA 12.6
- cuDSS 0.8.0
- CUDA code generated only for `sm_87`
- GUI/OpenGL disabled
- ONNX/tests/benchmarks disabled
- Installed under `/opt/colmap-cuda-4.1.0`
- Wrapper command: `colmap-cuda`

### GUI Replacement Package

```text
dist/colmap-cuda-sm87_4.1.0+cuda12.6.sm87.gui-1_arm64.deb
SHA256 394fd6bb2124ed34415585b71cb908c1fd3768088e53ecdbe78c887acaf4055c
```

Build characteristics:

- Same COLMAP, Ceres, CUDA, cuDSS, and `sm_87` target as the CLI package
- GUI enabled
- OpenGL enabled
- ONNX/tests/benchmarks disabled
- Same package name with a higher version, so `dpkg -i` upgrades the CLI-only
  package in place
- Same wrapper command: `colmap-cuda`

The GUI build can replace the CLI package. COLMAP's GUI-enabled binary still
contains the same command-line commands such as `feature_extractor`, `mapper`,
`bundle_adjuster`, `patch_match_stereo`, and `model_analyzer`.

## Container

Use the NVIDIA JetPack image instead of generic Ubuntu. The validated container
was persistent and was not created with `--rm`.

```bash
docker pull nvcr.io/nvidia/l4t-jetpack:r36.4.0

docker run -dit \
  --name colmap_cuda_build_r36_4 \
  --runtime nvidia \
  --network host \
  -v /home/jetson/workspace/spherical_3dgs:/workspace/spherical_3dgs \
  nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
  bash
```

Enter it with:

```bash
docker exec -it colmap_cuda_build_r36_4 bash
```

The first build was run conservatively while another COLMAP job was active:

```bash
docker update --cpus 2 --memory 6g --memory-swap 8g colmap_cuda_build_r36_4
```

After the other job finished, the GUI rebuild used the machine more fully:

```bash
docker update --cpus 8 --memory 14g --memory-swap 22g colmap_cuda_build_r36_4
```

The final inspected container settings were:

```text
Image=nvcr.io/nvidia/l4t-jetpack:r36.4.0
Runtime=nvidia
NetworkMode=host
Memory=15032385536
MemorySwap=23622320128
NanoCpus=8000000000
Mount=/home/jetson/workspace/spherical_3dgs -> /workspace/spherical_3dgs
```

## Base Dependencies

Inside the container, add NVIDIA's CUDA apt repo for Ubuntu 22.04 arm64 if the
image does not already expose the packages:

```bash
apt-get update
apt-get install -y ca-certificates gnupg wget

wget -qO /tmp/cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
```

Install build dependencies:

```bash
export DEBIAN_FRONTEND=noninteractive

apt-get install -y \
  build-essential git ninja-build ccache pkg-config \
  gcc-10 g++-10 gcc-11 g++-11 \
  cuda-toolkit-12-6 cudss-cuda-12 \
  libboost-program-options-dev libboost-filesystem-dev \
  libboost-graph-dev libboost-system-dev libboost-test-dev \
  libboost-thread-dev libeigen3-dev \
  libmetis-dev libgoogle-glog-dev libgflags-dev \
  libsqlite3-dev libglew-dev libgl1-mesa-dev libglu1-mesa-dev \
  libopengl-dev libegl-dev \
  qt6-base-dev qt6-base-dev-tools libqt6opengl6-dev libqt6svg6-dev \
  libcgal-dev libsuitesparse-dev libatlas-base-dev \
  libopenimageio-dev libopenexr-dev \
  libcurl4-openssl-dev libssl-dev \
  dpkg-dev fakeroot ripgrep
```

The validated important package versions were:

```text
cmake                       4.3.4-0kitware3ubuntu22.04.1
cuda-cudart-12-6            12.6.68-1
cudss-cuda-12               0.8.0.10-1
libcublas-12-6              12.6.1.4-1
libcusolver-12-6            11.6.4.69-1
libcusparse-12-6            12.5.3.3-1
libcudss0-cuda-12           0.8.0.10-1
qt6-base-dev                6.2.4+dfsg-2ubuntu1.1
libqt6opengl6-dev           6.2.4+dfsg-2ubuntu1.1
libqt6svg6-dev              6.2.4-1ubuntu1
```

COLMAP 4.1.0 fetches FAISS during configure, and the FAISS CMake project needs
CMake 3.24 or newer. Ubuntu 22.04's stock CMake 3.22 is too old, so add the
Kitware Jammy repo:

```bash
apt-get install -y ca-certificates gpg wget

wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
  | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg

printf '%s\n' \
  'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' \
  > /etc/apt/sources.list.d/kitware.list

apt-get update
apt-get install -y cmake
```

Check the toolchain:

```bash
cmake --version
nvcc --version
```

The validated build used:

```text
cmake version 4.3.4
Cuda compilation tools, release 12.6, V12.6.68
```

## Build Prefixes

Use a separate source/build area and a single install prefix:

```bash
mkdir -p /opt/colmap-build/src
mkdir -p /opt/colmap-build/build
mkdir -p /opt/colmap-build/logs

export PREFIX=/opt/colmap-cuda-4.1.0
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH="${PREFIX}/thirdparty:${PREFIX}/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/libcudss/12:${LD_LIBRARY_PATH:-}"
```

`gcc-11` was used because it is supported by CUDA 12.6 and matches the
container's Ubuntu 22.04 toolchain well.

## Build Ceres With CUDA And cuDSS

Clone Ceres:

```bash
cd /opt/colmap-build/src
git clone https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
git checkout bac1127f9ef672405bd0d2d9c84e809ae89bd239
git submodule update --init --recursive
```

At this commit, Ceres appended its default CUDA architectures even when
`CMAKE_CUDA_ARCHITECTURES=87` was provided. Patch `CMakeLists.txt` so an
explicit architecture list is respected. The effective logic should be:

```cmake
if (DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  message("-- Setting CUDA Architecture to ${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(CMAKE_CUDA_ARCHITECTURES "")
  # Append Ceres defaults only when the caller did not set an arch list.
  ...
  message("-- Setting CUDA Architecture to ${CMAKE_CUDA_ARCHITECTURES}")
endif()
```

After patching, configure Ceres:

```bash
cmake -S /opt/colmap-build/src/ceres-solver \
  -B /opt/colmap-build/build/ceres \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DUSE_CUDA=ON \
  -DSUITESPARSE=ON \
  -DEIGENSPARSE=ON \
  -DLAPACK=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_DOCUMENTATION=OFF \
  -DEXPORT_BUILD_DIR=OFF \
  2>&1 | tee /opt/colmap-build/logs/03_ceres_configure_sm87.log
```

The important configure markers are:

```text
-- Detected Ceres version: 2.3.0
-- Setting CUDA Architecture to 87
-- Found SuiteSparse 5.10.1, building with SuiteSparse.
-- Found cudss ...
```

Build and install:

```bash
{
  cmake --build /opt/colmap-build/build/ceres --parallel "$(nproc)"
  cmake --install /opt/colmap-build/build/ceres
} 2>&1 | tee /opt/colmap-build/logs/04_05_ceres_sm87_build_install.log
```

The validated low-resource Ceres build/install time was:

```text
CERES_SM87_BUILD_START 2026-07-02T10:36:24+00:00
CERES_SM87_BUILD_DONE  2026-07-02T11:16:24+00:00
```

That is exactly 40 minutes under the conservative 2 CPU / 6 GiB container cap.

## Build COLMAP CLI

Clone COLMAP:

```bash
cd /opt/colmap-build/src
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout fa8e3b3ff591552855f8ad2806723c80f963f69c
```

Configure the CLI-only build:

```bash
cmake -S /opt/colmap-build/src/colmap \
  -B /opt/colmap-build/build/colmap \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_PREFIX_PATH="${PREFIX}" \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCUDA_ENABLED=ON \
  -DGUI_ENABLED=OFF \
  -DOPENGL_ENABLED=OFF \
  -DONNX_ENABLED=OFF \
  -DTESTS_ENABLED=OFF \
  -DBENCHMARK_ENABLED=OFF \
  -DCCACHE_ENABLED=ON \
  -DBUILD_SHARED_LIBS=ON \
  2>&1 | tee /opt/colmap-build/logs/03_colmap_configure.log
```

Important configure markers:

```text
-- Found Ceres version: 2.3.0 installed in: /opt/colmap-cuda-4.1.0 with components: [EigenSparse, SparseLinearAlgebraLibrary, LAPACK, SuiteSparse, cuDSS, SchurSpecializations]
-- Enabling CUDA support (version: 12.6.68, archs: 87)
-- Disabling GUI support
-- Disabling OpenGL support
-- Enabling GPU support (OpenGL: OFF, CUDA: ON)
```

Build and install:

```bash
{
  cmake --build /opt/colmap-build/build/colmap --parallel "$(nproc)"
  cmake --install /opt/colmap-build/build/colmap
} 2>&1 | tee /opt/colmap-build/logs/04_colmap_build_install.log
```

The validated low-resource COLMAP CLI build/install time was:

```text
2026-07-02T11:19:47+00:00
2026-07-02T13:07:07+00:00
```

That is 1 hour, 47 minutes, 20 seconds under the conservative 2 CPU / 6 GiB
container cap.

## Build COLMAP GUI Replacement

The GUI build reuses the same Ceres install and COLMAP source tree. Configure it
in a separate build directory:

```bash
cmake -S /opt/colmap-build/src/colmap \
  -B /opt/colmap-build/build/colmap-gui \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_PREFIX_PATH="${PREFIX}" \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCUDA_ENABLED=ON \
  -DGUI_ENABLED=ON \
  -DOPENGL_ENABLED=ON \
  -DONNX_ENABLED=OFF \
  -DTESTS_ENABLED=OFF \
  -DBENCHMARK_ENABLED=OFF \
  -DCCACHE_ENABLED=ON \
  -DBUILD_SHARED_LIBS=ON \
  2>&1 | tee /opt/colmap-build/logs/08_colmap_gui_configure.log
```

Important configure markers:

```text
-- Found Ceres version: 2.3.0 installed in: /opt/colmap-cuda-4.1.0 with components: [EigenSparse, SparseLinearAlgebraLibrary, LAPACK, SuiteSparse, cuDSS, SchurSpecializations]
-- Enabling CUDA support (version: 12.6.68, archs: 87)
-- Found Qt
-- Enabling GUI support
-- Enabling OpenGL support
-- Enabling GPU support (OpenGL: ON, CUDA: ON)
```

Build it:

```bash
cmake --build /opt/colmap-build/build/colmap-gui --parallel "$(nproc)" \
  2>&1 | tee /opt/colmap-build/logs/09_colmap_gui_build.log
```

The validated GUI build time with the expanded 8 CPU / 14 GiB container limit
was:

```text
COLMAP_GUI_SM87_BUILD_START 2026-07-03T06:17:21+00:00
COLMAP_GUI_SM87_BUILD_DONE  2026-07-03T06:24:58+00:00
```

That is 7 minutes, 37 seconds. It was much faster than the first CLI build
because Ceres was already installed and the container could use the full CPU
budget.

## Runtime Wrapper

COLMAP installs its own shared libraries under:

```text
/opt/colmap-cuda-4.1.0/thirdparty
```

Ceres and Abseil install under:

```text
/opt/colmap-cuda-4.1.0/lib
```

The package therefore installs a wrapper at `/usr/bin/colmap-cuda`:

```sh
#!/bin/sh
prefix=/opt/colmap-cuda-4.1.0
export LD_LIBRARY_PATH="${prefix}/thirdparty:${prefix}/lib:/usr/local/cuda/lib64:/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/libcudss/12${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${prefix}/bin/colmap" "$@"
```

The package also installs:

```text
/etc/ld.so.conf.d/colmap-cuda-4.1.0.conf
```

with:

```text
/opt/colmap-cuda-4.1.0/lib
/opt/colmap-cuda-4.1.0/thirdparty
```

and runs `ldconfig` from `postinst` and `postrm`.

## Package Build

The package root mirrors the install prefix and adds the wrapper:

```bash
pkg=colmap-cuda-sm87
ver=4.1.0+cuda12.6.sm87.gui-1
root=/opt/colmap-build/pkgroot/${pkg}_${ver}_arm64
out=/opt/colmap-build/packages

rm -rf "${root}"
mkdir -p "${root}/DEBIAN" "${root}/opt" "${root}/usr/bin" "${root}/etc/ld.so.conf.d" "${out}"

cp -a "${PREFIX}" "${root}/opt/"
DESTDIR="${root}" cmake --install /opt/colmap-build/build/colmap-gui
```

Create `DEBIAN/control` with runtime dependencies. The GUI package used:

```text
Package: colmap-cuda-sm87
Version: 4.1.0+cuda12.6.sm87.gui-1
Architecture: arm64
Depends: libc6 (>= 2.34), libstdc++6, libgcc-s1, libgomp1,
 libboost-program-options1.74.0, libboost-filesystem1.74.0,
 libboost-thread1.74.0, libboost-iostreams1.74.0, libgoogle-glog0v5,
 libgflags2.2, libsqlite3-0, libglew2.2, libgl1, libopengl0,
 libglu1-mesa, libmetis5, libcholmod3, libcxsparse3, libspqr2,
 libblas3, liblapack3, libatlas3-base, libopenimageio2.2,
 libopenexr25, libcurl4, zlib1g, libqt6core6, libqt6gui6,
 libqt6widgets6, libqt6opengl6, libqt6openglwidgets6, libqt6svg6,
 qt6-qpa-plugins, cuda-cudart-12-6, libcublas-12-6, libcusparse-12-6,
 libcusolver-12-6, libcudss0-cuda-12
```

Build the package:

```bash
find "${root}" -type d -exec chmod 0755 {} +
dpkg-deb --build --root-owner-group "${root}" "${out}/${pkg}_${ver}_arm64.deb"
```

Copy it to the host workspace:

```bash
docker cp \
  colmap_cuda_build_r36_4:/opt/colmap-build/packages/colmap-cuda-sm87_4.1.0+cuda12.6.sm87.gui-1_arm64.deb \
  /home/jetson/workspace/spherical_3dgs/dist/
```

## Install And Verify

Install on the host or in the container:

```bash
sudo dpkg -i dist/colmap-cuda-sm87_4.1.0+cuda12.6.sm87.gui-1_arm64.deb
sudo apt-get -f install
```

The target system must have the same NVIDIA CUDA/cuDSS apt repositories
available if `apt-get -f install` needs to resolve CUDA runtime dependencies.

Basic CLI checks:

```bash
colmap-cuda version
colmap-cuda help | grep -E '^  gui$'
colmap-cuda gui -h
ldd /opt/colmap-cuda-4.1.0/bin/colmap | grep 'not found' || true
```

Validated output:

```text
COLMAP 4.1.0 (Commit fa8e3b3f on 2026-06-26 with CUDA)
  gui
```

`colmap-cuda gui -h` was used as the container-safe GUI command validation.
Opening the actual GUI window requires a display server to be available to the
process.

## Verify SM87-Only CUDA Artifacts

Use `cuobjdump` to confirm that the CUDA objects contain only `sm_87` cubins and
PTX:

```bash
for so in \
  /opt/colmap-cuda-4.1.0/thirdparty/libcolmap_mvs_cuda.so \
  /opt/colmap-cuda-4.1.0/thirdparty/libcolmap_sift_gpu.so \
  /opt/colmap-cuda-4.1.0/lib/libceres.so.4
do
  echo "${so}"
  cuobjdump -lelf "${so}"
  cuobjdump -lptx "${so}"
done
```

Validated output:

```text
/opt/colmap-cuda-4.1.0/thirdparty/libcolmap_mvs_cuda.so
ELF file    1: libcolmap_mvs_cuda.1.sm_87.cubin
ELF file    2: libcolmap_mvs_cuda.2.sm_87.cubin
ELF file    3: libcolmap_mvs_cuda.3.sm_87.cubin
PTX file    1: libcolmap_mvs_cuda.1.sm_87.ptx
PTX file    2: libcolmap_mvs_cuda.2.sm_87.ptx
PTX file    3: libcolmap_mvs_cuda.3.sm_87.ptx

/opt/colmap-cuda-4.1.0/thirdparty/libcolmap_sift_gpu.so
ELF file    1: libcolmap_sift_gpu.1.sm_87.cubin
PTX file    1: libcolmap_sift_gpu.1.sm_87.ptx

/opt/colmap-cuda-4.1.0/lib/libceres.so.4
ELF file    1: libceres.so.1.sm_87.cubin
ELF file    2: libceres.so.2.sm_87.cubin
PTX file    1: libceres.so.1.sm_87.ptx
PTX file    2: libceres.so.2.sm_87.ptx
```

## Notes

- Building only `sm_87` matters on Jetson Orin. The first unpatched Ceres
  configure attempted `87;50;60;70;75;80;90`, which wastes compile time and
  package size for this target.
- The GUI package is the preferred replacement package because it keeps all CLI
  functionality and adds `colmap-cuda gui`.
- The CLI-only package is still useful as a smaller rollback artifact.
- The build container is intentionally persistent. It keeps source trees, build
  trees, logs, and package roots under `/opt/colmap-build`.
- Logs from the validated build are under `/opt/colmap-build/logs` inside the
  container.
