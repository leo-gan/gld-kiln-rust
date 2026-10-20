//! Round trip bytes produced by `testdata/reference/kiln_ref.py`.
//! The Python module is a test oracle only. The library does not call it.

use std::process::Command;

use kiln::{decode, decode_framed, encode, encode_framed, Schema, Value};

fn oracle_hex() -> serde_json::Value {
    let root = env!("CARGO_MANIFEST_DIR");
    let script = r#"
import json, struct, sys
sys.path.insert(0, sys.argv[1])
import kiln_ref

def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]

def f64(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]

out = []
def add(name, raw):
    out.append({"name": name, "hex": raw.hex()})

add("point", kiln_ref.encode(kiln_ref.POINT, {"x": 3, "y": -4}))
add("point_framed", kiln_ref.encode_framed(kiln_ref.POINT, {"x": 3, "y": -4}))
msg = {
    "f_bool": True,
    "f_int32": -3,
    "f_int64": 99,
    "f_float64": f64(0x8000000000000000),
    "f_string": "hi",
    "f_bool_2": False,
    "f_int32_2": 0,
    "f_string_2": "",
}
add("message", kiln_ref.encode(kiln_ref.MESSAGE, msg))
add("message_framed", kiln_ref.encode_framed(kiln_ref.MESSAGE, msg))
add("f32_neg0", kiln_ref.encode(("f32",), f32(0x80000000)))
add("f32_qnan", kiln_ref.encode(("f32",), f32(0x7F800001)))
add("f64_neg0", kiln_ref.encode(("f64",), f64(0x8000000000000000)))
add("f64_qnan", kiln_ref.encode(("f64",), f64(0x7FF8000000000001)))
add("list_i16", kiln_ref.encode(("list", ("i16",)), [1, -2, 3]))
add("list_u8", kiln_ref.encode(("list", ("u8",)), b"\x00\xff"))
add("list_i8", kiln_ref.encode(("list", ("i8",)), [1, -1, 127]))
add("opt_some", kiln_ref.encode(("opt", ("str",)), "a"))
add("opt_none", kiln_ref.encode(("opt", ("str",)), None))
add("dict", kiln_ref.encode(("dict", ("u8",)), {"b": 2, "a": 1}))
add("bytes", kiln_ref.encode(("bytes",), b"\x00\xff"))
print(json.dumps(out))
"#;
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(format!("{root}/testdata/reference"))
        .output()
        .expect("python3");
    assert!(
        output.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("oracle json")
}

fn hex_decode(text: &str) -> Vec<u8> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let val = |c: u8| match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        _ => panic!("hex"),
    };
    for i in (0..bytes.len()).step_by(2) {
        out.push((val(bytes[i]) << 4) | val(bytes[i + 1]));
    }
    out
}

fn point() -> Schema {
    Schema::struct_fields([("x", Schema::I32), ("y", Schema::I32)])
}

fn message() -> Schema {
    Schema::struct_fields([
        ("f_bool", Schema::Bool),
        ("f_int32", Schema::I32),
        ("f_int64", Schema::I64),
        ("f_float64", Schema::F64),
        ("f_string", Schema::Str),
        ("f_bool_2", Schema::Bool),
        ("f_int32_2", Schema::I32),
        ("f_string_2", Schema::Str),
    ])
}

fn schema_for(name: &str) -> (Schema, bool) {
    match name {
        "point" => (point(), false),
        "point_framed" => (point(), true),
        "message" => (message(), false),
        "message_framed" => (message(), true),
        "f32_neg0" | "f32_qnan" => (Schema::F32, false),
        "f64_neg0" | "f64_qnan" => (Schema::F64, false),
        "list_i16" => (Schema::list(Schema::I16), false),
        "list_u8" => (Schema::list(Schema::U8), false),
        "list_i8" => (Schema::list(Schema::I8), false),
        "opt_some" | "opt_none" => (Schema::opt(Schema::Str), false),
        "dict" => (Schema::dict(Schema::U8), false),
        "bytes" => (Schema::Bytes, false),
        other => panic!("unknown oracle case {other}"),
    }
}

#[test]
fn kiln_ref_round_trip() {
    let cases = oracle_hex();
    let cases = cases.as_array().expect("array");
    assert!(cases.len() >= 14);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let raw = hex_decode(case["hex"].as_str().unwrap());
        let (schema, framed) = schema_for(name);
        let value = if framed {
            decode_framed(&schema, &raw).unwrap_or_else(|err| panic!("{name} {}", err.name()))
        } else {
            decode(&schema, &raw).unwrap_or_else(|err| panic!("{name} {}", err.name()))
        };
        match name {
            "f32_neg0" => assert_eq!(value, Value::F32(0x8000_0000)),
            "f32_qnan" => assert_eq!(value, Value::F32(0x7f80_0001)),
            "f64_neg0" => assert_eq!(value, Value::F64(0x8000_0000_0000_0000)),
            "f64_qnan" => assert_eq!(value, Value::F64(0x7ff8_0000_0000_0001)),
            "point" | "point_framed" => {
                assert_eq!(
                    value,
                    Value::Struct(vec![Value::I32(3), Value::I32(-4)])
                );
            }
            "message" | "message_framed" => {
                let Value::Struct(fields) = &value else {
                    panic!("message");
                };
                assert_eq!(fields[3], Value::F64(0x8000_0000_0000_0000));
            }
            "dict" => {
                let Value::Dict(pairs) = &value else { panic!("dict") };
                assert_eq!(pairs[0].0.as_ref(), "a");
                assert_eq!(pairs[1].0.as_ref(), "b");
            }
            _ => {}
        }
        let encoded = if framed {
            encode_framed(&schema, &value).unwrap()
        } else {
            encode(&schema, &value).unwrap()
        };
        assert_eq!(encoded, raw, "{name} round trip");
    }
}
