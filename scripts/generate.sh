#!/usr/bin/env bash
# Generate testdata/schema/*.json into tests/generated/.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"
if [[ -f "${HOME}/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
fi
mkdir -p tests/generated
cargo build --bin gld-kilngen-rust
bin="${root}/target/debug/gld-kilngen-rust"
shopt -s nullglob
for schema in testdata/schema/*.json; do
  "${bin}" --schema "${schema}" --out tests/generated
done
