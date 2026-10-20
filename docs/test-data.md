# Test data

The tests read files under `testdata/`. That directory is the only local input
the suite needs.

## Wire cases

`testdata/kiln-v2-cases.json` is the version-2 case file. It has 46 cases.
`tests/kiln_v2_cases.rs` loads that file.

| `expect` | What a passing run means |
| --- | --- |
| `accept` | Decode the hex, match `decoded` when the case has one, and re-encode to the same bytes. |
| `reject` | The error name equals `error`. |

Seventeen cases accept and twenty-nine reject. The schemas in the file are
`tensor-v2`, a framed struct, a struct with a `tensors` field, and lists of
`f32`, `str`, or `struct{a:i64}`. A list count of 16777216 structs is `EOF`
because the elements are not in the buffer. The count is inside the default
cap, so it is not `LIMIT`. `tests/hostile_alloc.rs` checks that this path
allocates no element storage.

Tensor payloads of accept cases must be slices of the input buffer.

## Schemas

| File | What it generates |
| --- | --- |
| `testdata/schema/point.json` | `Point`, with `x`, `y`, and an optional string `label`. |
| `testdata/schema/message.json` | `Message` and nested `Meta`. |

`bash scripts/check-generated.sh` writes those files again and diffs them
against `tests/generated`. A drift fails the check.

## Python oracle

`testdata/reference/kiln_ref.py` is a copy of the Frozen reference encoder.
`tests/oracle.rs` runs it with `python3` and compares bytes. The oracle covers
Point, Message, lists, options, dicts, and framed values, including negative
zero and a quiet NaN payload. The library does not import that module at
runtime.

The reference file is test-only. Encoding and decoding in production use the
Rust crate.
