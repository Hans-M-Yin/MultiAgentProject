#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="pantheonrl39"
PYTHON_VERSION="3.9"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating conda environment: ${ENV_NAME} (python=${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"

echo "Activating conda environment: ${ENV_NAME}"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "Installing legacy packaging tools for gym==0.21 compatibility"
pip install --upgrade pip
pip install "setuptools==65.5.0" "wheel<0.40.0"

echo "Installing PantheonRL core dependencies"
pip install \
  "gym==0.21.0" \
  "stable-baselines3==1.7.0" \
  "torch==1.13.1" \
  "scipy==1.7.3" \
  "tqdm==4.64.1" \
  "tensorboard"

echo "Initializing git submodules"
git -C "${ROOT_DIR}" submodule update --init --recursive

echo "Installing editable packages"
pip install -e "${ROOT_DIR}"
pip install -e "${ROOT_DIR}/overcookedgym/human_aware_rl/overcooked_ai"

echo
echo "Setup complete."
echo "To use the environment later, run:"
echo "  conda activate ${ENV_NAME}"
