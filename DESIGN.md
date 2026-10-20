# gld-kiln-rust design

gld-kiln-rust is a from-scratch Kiln library for Rust 1.70.
The runtime never calls a Kiln library written in another language. The copied
Python reference under `testdata/reference/` may be used as a test oracle. It
is not a build or run dependency.

This crate is not the Kiln specification.

## Goals

The library encodes and decodes Frozen positional values and a version-2 tensor
directory. Scalars are little-endian. A struct is the schema: fields are
written in declaration order, without names, tags, or padding. The tensor
directory keeps numeric bytes as a slice of the input array. A hostile count
is rejected before the decoder allocates element storage.

A second API reads a JSON Schema subset and emits Rust structs. Generated
structs expose `encode`, `decode`, and `encoded_size`. A `tensors` field is not
produced from that subset. The generator command is `gld-kilngen-rust`.

## Values

`Schema` describes a value. `Value` is the dynamic tree. Strings and tensor
payloads borrow the input buffer. `F32` and `F64` store IEEE bits so negative
zero and a quiet NaN payload round-trip. Dict keys are sorted by raw UTF-8
bytes on encode. Decode rejects keys that are not strictly increasing.

`opt` is a `0` or `1` tag. `list<u8>` and `list<i8>` are a `u32` count plus raw
bytes. Other lists are a count plus concatenated elements. Before a list or
dict allocates, the decoder requires `count * min_size(element)` to fit in the
unread bytes. The default cap is 16,777,216 counts and 16 MiB for one string
or byte run. A caller can raise the cap.

## Tensor directory

Encode places payloads on 64-byte boundaries and writes the canonical name
heap: names in tensor order, then layout names. Decode follows the reference
check order and returns `FRAME`, `LIMIT`, `EOF`, `TYPE`, `UTF8`, `TRAILING`,
or `SCHEMA`. A payload view is a subslice of the input. A trusted reader may
skip UTF-8 checks and quantization-shape checks. It still rejects a length
past the buffer.

A `tensors` field inside a struct is one directory block, then the next field.
The block length comes from the header, so the following field starts at the
end of the directory.

## Frame hash

A frame is `KLN0`, four zero bytes, the FNV-1a 64 `type_hash` of the schema,
and one Frozen value. Each token is mixed, then a trailing `0x00` multiply.
The token for a tensor field is the ASCII bytes `tensors`. A hash mismatch is
`SCHEMA`.

## Code generation

`gld-kilngen-rust --schema FILE --out DIR` reads one JSON Schema document.
Objects with `properties` become structs. A property not listed in `required`
is `Option<T>`. `$ref` must be `#/$defs/Name`. `additionalProperties` with a
schema and no fixed properties is `BTreeMap<String, V>`. Integer `format`
selects the Frozen width. A JSON number is `f64` unless the format asks for
`f32`. String `format: binary` is `list<u8>`.

## Testing

`testdata/kiln-v2-cases.json` holds 46 accept and reject cases. Accept cases
decode, match the recorded value, and re-encode to the same bytes. Reject
cases return the named error. `tests/oracle.rs` asks `kiln_ref.py` for Point,
Message, lists, options, dicts, and framed bytes, including negative zero and
a quiet NaN. `tests/hostile_alloc.rs` checks that a list count of 16,777,216
structs returns `EOF` without allocating element storage.

## Packaging

The Cargo package is `gld-kiln`. The Rust import name is `kiln`. The generator
binary is `gld-kilngen-rust`. The version in `VERSION`, `Cargo.toml`, and
`kiln::VERSION` starts at `0.1.0`. Publishing to crates.io runs only when
`CARGO_REGISTRY_TOKEN` is set. This repository does not publish to prefix.dev.
