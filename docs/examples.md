# Examples

The dynamic API and the generated API write the same Frozen bytes. Use the
dynamic API when the shape is chosen at runtime. Use a generated struct when a
JSON Schema already names the fields.

## Dynamic struct

`examples/point_roundtrip.rs` encodes a Point and decodes it. Run it with
`cargo run --example point_roundtrip`.

```rust
use kiln::{decode, encode, Schema, Value};

let schema = Schema::struct_fields([("x", Schema::I32), ("y", Schema::I32)]);
let value = Value::Struct(vec![Value::I32(3), Value::I32(-4)]);
let raw = encode(&schema, &value).expect("encode");
let back = decode(&schema, &raw).expect("decode");
assert_eq!(back, value);
```

`Value::F32` and `Value::F64` store IEEE bits. Negative zero is
`Value::from_f32(-0.0)`, not a canonicalized `0.0`.

## One tensor

```rust
use kiln::{decode_tensors, encode_tensors, DecodeOpts, Tensor, DTYPE_F32, DEFAULT_CAP};

let payload = [0x00u8, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0];
let tensor = Tensor::new(DTYPE_F32, &payload).with_shape(&[2]);
let raw = encode_tensors(&[tensor], DEFAULT_CAP).expect("encode");
let dir = decode_tensors(&raw, DecodeOpts::default()).expect("decode");
assert_eq!(dir.tensors[0].data, &payload);
```

`dir.tensors[0].data` is a slice of `raw`. Drop `dir` before mutating `raw`.

## Generated struct

`testdata/schema/point.json` describes `x`, `y`, and an optional `label`.
After `bash scripts/generate.sh`, `tests/generated/Point.rs` has those fields.
`label` is omitted from `required`, so the wire form is an option tag.

```rust
let point = Point { x: 3, y: -4, label: None };
let raw = point.encode().expect("encode");
assert_eq!(point.encoded_size(), raw.len());
let back = Point::decode(&raw).expect("decode");
assert_eq!(back.x, 3);
```

`testdata/schema/message.json` nests `Meta` through `$ref`. Both structs are
emitted into `Message.rs`. `encoded_size` is the exact Frozen length, including
the length prefixes of strings.
