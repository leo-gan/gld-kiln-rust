#!/usr/bin/env bash
# Install stable Rust on Ubuntu when cargo is not already on PATH.
set -euo pipefail

add_path() {
  local dir="$1"
  if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "${dir}" >> "${GITHUB_PATH}"
  fi
  export PATH="${dir}:${PATH}"
}

if [[ -f "${HOME}/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
fi

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
fi

add_path "${HOME}/.cargo/bin"
rustup component add clippy
rustc --version
cargo --version
