#!/usr/bin/env bash
set -e

# Build mlipcpp for WebAssembly using Emscripten
# Requires: Emscripten SDK installed and activated (source emsdk_env.sh)

BUILD_DIR="wasm"
BUILD_TYPE="${BUILD_TYPE:-Release}"

# Check if emcmake is available
if ! command -v emcmake &> /dev/null; then
    echo "Error: emcmake not found. Please install and activate Emscripten SDK:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk && ./emsdk install latest && ./emsdk activate latest"
    echo "  source emsdk_env.sh"
    exit 1
fi

echo "=== Building mlipcpp for WebAssembly ==="
echo "Build directory: ${BUILD_DIR}"
echo "Build type: ${BUILD_TYPE}"

# Configure with CMake via emcmake
emcmake cmake . -B"${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -GNinja

# Build the WASM target
cmake --build "${BUILD_DIR}" --target mlipcpp_wasm

echo ""
echo "=== Build complete ==="
echo "Output files:"
echo "  ${BUILD_DIR}/bin/mlipcpp_wasm.js"
echo ""
echo "To use in Node.js:"
echo "  const createMlipcpp = require('./${BUILD_DIR}/bin/mlipcpp_wasm.js');"
echo "  createMlipcpp().then(Module => { ... });"
