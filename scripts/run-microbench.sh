#!/usr/bin/env bash
# Encode and decode a small struct batch and one 1024-element f32 tensor.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"
if [[ -f "${HOME}/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
fi
cargo build --release --bin kiln-bench
bench="${root}/target/release/kiln-bench"
if command -v taskset >/dev/null 2>&1; then
  bench=(taskset -c 1 "${bench}")
else
  bench=("${bench}")
fi
for _ in 1 2 3; do
  "${bench[@]}"
done
