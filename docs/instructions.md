# Instructions

These steps assume a Linux machine and Rust 1.70 or newer. The Cargo package
is `gld-kiln`. The import name is `kiln`. The generator binary is
`gld-kilngen-rust`.

## Install

```bash
cargo add gld-kiln
```

```toml
[dependencies]
gld-kiln = "0.2.0"
```

`use kiln::{decode, encode, Schema, Value}` then resolves the dynamic codec.

## Build this repository

```bash
git clone https://github.com/leo-gan/gld-kiln-rust.git
cd gld-kiln-rust
bash scripts/run-tests.sh
```

| Script | What it runs |
| --- | --- |
| `bash scripts/run-tests.sh` | `cargo test` and `cargo clippy --all-targets -- -D warnings`. |
| `bash scripts/generate.sh` | Writes `tests/generated` from `testdata/schema`. |
| `bash scripts/check-generated.sh` | Regenerates into a temp directory and diffs. |
| `bash scripts/run-microbench.sh` | Encodes and decodes a 1024-struct batch and one 1024-element f32 tensor. |

## Decode and encode

```rust
use kiln::{decode, encode, Schema, Value};

let schema = Schema::struct_fields([("x", Schema::I32), ("y", Schema::I32)]);
let value = Value::Struct(vec![Value::I32(3), Value::I32(-4)]);
let raw = encode(&schema, &value)?;
let back = decode(&schema, &raw)?;
```

`DecodeOpts` raises the cap or selects a trusted reader. `encode_framed` writes
a `KLN0` frame whose hash is `type_hash`.

## Generate Rust

```bash
cargo run --bin gld-kilngen-rust -- --schema testdata/schema/point.json --out tests/generated
```

The generator writes one `.rs` file per schema title. Compile that file in a
crate that depends on `gld-kiln`. The generated module imports `kiln`.

## Document the package

```bash
pip install -r requirements-docs.txt
mkdocs build --strict
```

The site is Material for MkDocs with an indigo primary color and a teal accent.
