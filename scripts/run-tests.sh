#!/usr/bin/env bash
# Unit tests, then clippy with warnings denied.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"
if [[ -f "${HOME}/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
fi
cargo test
cargo clippy --all-targets -- -D warnings
