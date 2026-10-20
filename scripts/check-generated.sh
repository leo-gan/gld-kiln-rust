#!/usr/bin/env bash
# Regenerate into a temp directory and diff against tests/generated.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"
if [[ -f "${HOME}/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
fi
cargo build --bin gld-kilngen-rust
bin="${root}/target/debug/gld-kilngen-rust"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
shopt -s nullglob
for schema in testdata/schema/*.json; do
  "${bin}" --schema "${schema}" --out "${tmp}"
done
diff -ru tests/generated "${tmp}"
