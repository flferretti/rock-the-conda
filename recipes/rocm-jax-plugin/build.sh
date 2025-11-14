#!/bin/bash
# Build script for jax-rocm7-plugin, modeled after conda-forge/jaxlib-feedstock
set -euxo pipefail

cd jax_rocm_plugin

# ---------------------------------------------------------------------------
# Python toolchain: use conda's Python instead of Bazel's hermetic download
# ---------------------------------------------------------------------------
$RECIPE_DIR/add_py_toolchain.sh

# ---------------------------------------------------------------------------
# Compiler / linker flags (matching jaxlib-feedstock conventions)
# ---------------------------------------------------------------------------
export LDFLAGS="${LDFLAGS} -lrt"
# See https://github.com/llvm/llvm-project/issues/85656
export CXXFLAGS="${CXXFLAGS} -fclang-abi-compat=17"
# https://github.com/conda-forge/jaxlib-feedstock/issues/310
LDFLAGS+=" -Wl,-z,noexecstack"
export CFLAGS="${CFLAGS} -DNDEBUG -Dabsl_nullable= -Dabsl_nonnull="
export CXXFLAGS="${CXXFLAGS} -DNDEBUG -Dabsl_nullable= -Dabsl_nonnull="

# ---------------------------------------------------------------------------
# ROCm environment
# ---------------------------------------------------------------------------
export ROCM_PATH="${PREFIX}"
export HIP_PATH="${PREFIX}"
export HIP_PLATFORM="amd"
export PATH="${BUILD_PREFIX}/bin:${PATH}"

# ROCm's rocm_configure.bzl expects clang at ${ROCM_PATH}/llvm/bin/clang
# for discovering builtin include directories. Create a symlink so it finds
# conda's clang there.
mkdir -p "${PREFIX}/llvm/bin"
ln -sf "${BUILD_PREFIX}/bin/clang" "${PREFIX}/llvm/bin/clang"

# HIP headers (hip_version.h etc.) are provided by compiler('hip') in
# BUILD_PREFIX. Symlink them into PREFIX so find_rocm_config.py discovers
# them under ROCM_PATH.
if [[ ! -d "${PREFIX}/include/hip" && -d "${BUILD_PREFIX}/include/hip" ]]; then
    ln -sf "${BUILD_PREFIX}/include/hip" "${PREFIX}/include/hip"
fi

# RCCL paths
export RCCL_ROOT="${PREFIX}"
export RCCL_INCLUDE_DIR="${PREFIX}/include"
export RCCL_LIB_DIR="${PREFIX}/lib"

# Extra library paths for Bazel actions
EXTRA_LD="${PREFIX}/lib:${BUILD_PREFIX}/lib"

# ---------------------------------------------------------------------------
# Bazel configuration overrides
# ---------------------------------------------------------------------------
cat >> .bazelrc <<EOF
build --verbose_failures
build --local_resources=cpu=${CPU_COUNT}
EOF

# Remove incompatible clang flag from upstream .bazelrc
sed -i '/Qunused-arguments/d' .bazelrc

# Detect host Python version from the actual binary ($PYTHON is set by
# rattler-build to the host prefix interpreter).
HOST_PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# ---------------------------------------------------------------------------
# Build wheels (plugin + PJRT runtime)
# ---------------------------------------------------------------------------
python build/build.py build \
    --wheels=jax-rocm-plugin,jax-rocm-pjrt \
    --python_version="${HOST_PY_VER}" \
    --bazel_path="${BUILD_PREFIX}/bin/bazel" \
    --use_clang=true \
    --clang_path="${BUILD_PREFIX}/bin/clang" \
    --rocm_path="${PREFIX}" \
    --rocm_amdgpu_targets="gfx906,gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100,gfx1101,gfx1200,gfx1201" \
    --output_path=dist \
    --bazel_options="--action_env=LD_LIBRARY_PATH=${EXTRA_LD}" \
    --bazel_options="--action_env=RCCL_ROOT=${PREFIX}" \
    --bazel_options="--action_env=ROCM_PATH=${PREFIX}" \
    --bazel_options="--action_env=HIP_PATH=${PREFIX}"

# Clean up Bazel cache to speed up post-processing
pushd build
bazel clean --expunge || true
popd

# ---------------------------------------------------------------------------
# Install wheels into the conda prefix
# ---------------------------------------------------------------------------
pip install --no-deps --prefix="${PREFIX}" \
    dist/jax_rocm7_plugin-*.whl \
    dist/jax_rocm7_pjrt-*.whl

# Add INSTALLER file and remove RECORD (conda compatibility workaround,
# see https://github.com/conda-forge/jaxlib-feedstock/issues/293)
pushd $SP_DIR
for DIST_INFO in jax_rocm7_plugin-*.dist-info jax_rocm7_pjrt-*.dist-info; do
    if [[ -d "${DIST_INFO}" ]]; then
        echo "conda" > "${DIST_INFO}/INSTALLER"
        rm -f "${DIST_INFO}/RECORD"
    fi
done
popd

# ---------------------------------------------------------------------------
# Clean up build-only artifacts from PREFIX
# ---------------------------------------------------------------------------
rm -rf "${PREFIX}/llvm"
rm -f "${PREFIX}/include/hip"
