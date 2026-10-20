# gld-kiln-rust

A from-scratch Kiln implementation for Rust.
The runtime and the code generator are Rust. They do not wrap, link, or vendor
a Kiln library written in another language.

Python `testdata/reference/kiln_ref.py` is a **test oracle** only. It is not
required to encode or decode at runtime.

This repository is a standalone library. It is not the Kiln specification, and
it is not part of any other project.

Documentation: [Why Kiln](https://leo-gan.github.io/gld-kiln-rust/why-kiln/),
[Instructions](https://leo-gan.github.io/gld-kiln-rust/instructions/),
[Examples](https://leo-gan.github.io/gld-kiln-rust/examples/),
[Techniques](https://leo-gan.github.io/gld-kiln-rust/techniques/),
[Test data](https://leo-gan.github.io/gld-kiln-rust/test-data/).

## Install

Rust 1.70 or newer. The Cargo package is `gld-kiln`. The import name is `kiln`.
The generator binary is `gld-kilngen-rust`.

```bash
cargo add gld-kiln
```

```toml
[dependencies]
gld-kiln = "0.2.0"
```

From a git checkout:

```bash
git clone https://github.com/leo-gan/gld-kiln-rust.git
cd gld-kiln-rust
bash scripts/run-tests.sh
```

## What it reads and writes

| Surface | Behavior |
| --- | --- |
| Frozen values | Positional little-endian scalars, strings, lists, options, dicts, and structs. |
| Tensor directory | Version-2 block. Numeric payloads are borrowed slices of the input buffer. |
| Frame | `KLN0`, four zero bytes, and an FNV-1a 64 schema hash. A mismatch is `SCHEMA`. |
| Schema | A JSON Schema subset. `gld-kilngen-rust` emits structs with `encode`, `decode`, and `encoded_size`. |

`list<u8>` and `list<i8>` are length-prefixed raw bytes. Other lists are a count
plus concatenated elements. Dict keys are ordered by raw UTF-8 bytes. A struct
is its fields in declaration order, with no names and no padding on the wire.
A hostile list count fails before the decoder allocates element storage.

## Layout

`src/` is the `kiln` library: Frozen values, the tensor directory, schema
hashes, and the JSON Schema emitter. `src/main.rs` is the `gld-kilngen-rust`
executable.

Requires **Rust 1.70**.

## License

MIT. Copyright (c) 2026 Leonid Ganeline.
