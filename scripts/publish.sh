#!/usr/bin/env bash
# Package the crate. Publish to crates.io only when CARGO_REGISTRY_TOKEN is set.
# If that secret is missing, say so and exit 0. Do not publish to prefix.dev.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"
if [[ -f "${HOME}/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
fi

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
elif [[ -n "${1:-}" ]]; then
  echo "usage: scripts/publish.sh [--dry-run]" >&2
  exit 2
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is not on PATH" >&2
  exit 1
fi

version=""
if [[ -f VERSION ]]; then
  version="$(tr -d '[:space:]' < VERSION)"
fi
if [[ -z "${version}" ]]; then
  echo "could not read VERSION" >&2
  exit 1
fi

origin="$(git remote get-url origin 2>/dev/null || true)"
origin="${origin%.git}"
origin="${origin#git@github.com:}"
origin="${origin#https://github.com/}"
if [[ -z "${origin}" ]]; then
  origin="leo-gan/gld-kiln-rust"
fi
repo_url="https://github.com/${origin}"

if [[ "${GITHUB_REF_TYPE:-}" == "tag" ]]; then
  tag_name="${GITHUB_REF_NAME}"
  tag_version="${tag_name#v}"
  if [[ "${tag_version}" != "${version}" ]]; then
    echo "git tag ${tag_name} does not match package version ${version}" >&2
    exit 1
  fi
else
  tag_name="${version}"
fi

archive="${TMPDIR:-/tmp}/gld-kiln-${version}.crate"
rm -f "${archive}"
cargo package --allow-dirty --list >/dev/null
crate="$(find target/package -maxdepth 1 -name "gld-kiln-${version}.crate" -print -quit)"
if [[ -z "${crate}" ]]; then
  echo "cargo package did not write gld-kiln-${version}.crate" >&2
  exit 1
fi
cp "${crate}" "${archive}"

block="$(cat <<EOF
Rust crate \`gld-kiln\` ${version}.

\`\`\`toml
gld-kiln = "${version}"
\`\`\`

Import \`kiln\`. The generator binary is \`gld-kilngen-rust\`.
Repository: ${repo_url}
This script does not publish to prefix.dev.
EOF
)"
printf '%s\n' "${block}"
echo "crate archive: ${archive}"

if [[ "${dry_run}" -eq 1 ]]; then
  echo "dry-run: crates.io was not called and the release was not changed"
  exit 0
fi

if [[ "${GITHUB_ACTIONS:-}" == "true" && "${GITHUB_REF_TYPE:-}" == "tag" ]]; then
  printf '%s\n' "${block}" | python3 scripts/github_release.py notes --tag "${tag_name}"
  python3 scripts/github_release.py upload --tag "${tag_name}" --file "${archive}"
fi

if [[ -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
  echo "CARGO_REGISTRY_TOKEN is unset. Not publishing to crates.io."
  exit 0
fi

cargo publish --token "${CARGO_REGISTRY_TOKEN}"
