#!/usr/bin/env bash
# Run after cloning the pinned sources and applying the documented patches.
set -euo pipefail
SS_PREFIX="${SS_PREFIX:-$HOME/miniconda3/envs/ss}"
THIRD_PARTY="${THIRD_PARTY:-$HOME/third_party}"
# WSL inherits Windows PATH entries; these can contaminate CMake header discovery.
export PATH="$SS_PREFIX/bin:/usr/local/bin:/usr/bin:/bin"
cd "$THIRD_PARTY/habitat-sim"
"$SS_PREFIX/bin/python" setup.py build_ext --parallel 2 install \
  --headless --audio --no-update-submodules --force-cmake \
  --cmake-args='-DCMAKE_CXX_FLAGS="-include cstdint" -U*ZLIB* -U*OPENGL* -U*X11*'
