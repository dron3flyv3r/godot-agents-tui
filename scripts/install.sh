#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_PYTHON=""
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
fi

BUILD_ENV=("CONTROLLER_SCRIPTS_ROOT=${REPO_ROOT}")
if [[ -n "${VENV_PYTHON}" ]]; then
    echo "Embedding Python interpreter: ${VENV_PYTHON}"
    BUILD_ENV+=("CONTROLLER_PYTHON_BIN=${VENV_PYTHON}")
fi

INSTALL_DIR="${INSTALL_DIR:-${HOME}/.local/bin}"

echo "Building controller-mk2 in release mode..."
if [[ ${#BUILD_ENV[@]} -gt 0 ]]; then
    env "${BUILD_ENV[@]}" cargo build --release --locked --bin controller-mk2
else
    cargo build --release --locked --bin controller-mk2
fi
mkdir -p "${INSTALL_DIR}"

BIN_SOURCE="${REPO_ROOT}/target/release/controller-mk2"
BIN_TARGET="${INSTALL_DIR}/controller-mk2"

if [[ ! -f "${BIN_SOURCE}" ]]; then
    echo "Error: expected binary at ${BIN_SOURCE} but it was not found." >&2
    exit 1
fi

echo "Installing binary to ${BIN_TARGET}"
install -m 755 "${BIN_SOURCE}" "${BIN_TARGET}"

if [[ ":${PATH}:" != *":${INSTALL_DIR}:"* ]]; then
    echo
    echo "Note: ${INSTALL_DIR} is not on your PATH."
    echo "Add the following line to your shell rc file (e.g., ~/.bashrc or ~/.zshrc):"
    echo "    export PATH=\"${INSTALL_DIR}:\$PATH\""
fi

echo
echo "Done. You can now run 'controller-mk2' from anywhere."
