#!/usr/bin/env bash
# ==============================================================================
# Environment Setup Script for D2CO-M (Powered by uv & pip)
# ==============================================================================
# Creates virtual environment using uv, syncs dependencies from pyproject.toml,
# activates the environment, and installs pyakmaxsat using pip.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYAKMAXSAT_DIR="$(dirname "${SCRIPT_DIR}")/pyakmaxsat"

echo "=== 1. Creating Virtual Environment with uv ==="
uv venv "${VENV_DIR}"

echo "=== 2. Syncing dependencies from pyproject.toml ==="
uv sync

echo "=== 3. Activating Virtual Environment ==="
source "${VENV_DIR}/bin/activate"

echo "=== 4. Installing pyakmaxsat with pip ==="
export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
export CXXFLAGS="-include cstdint"

if [ -d "${PYAKMAXSAT_DIR}" ]; then
    echo "Installing pyakmaxsat from local directory: ${PYAKMAXSAT_DIR}..."
    pip install --no-build-isolation "${PYAKMAXSAT_DIR}"
else
    echo "Installing pyakmaxsat from GitHub repository..."
    pip install --no-build-isolation git+https://github.com/mullzhang/pyakmaxsat.git
fi

echo "=== Environment Setup Complete! ==="
echo "To activate environment, run: source .venv/bin/activate"
