//! Every record in `testdata/kiln-v2-cases.json`.
//!
//! Accept cases decode, match `decoded` when present, and re-encode to the same bytes.
//! Reject cases return the named error.

use kiln::{decode, decode_framed, dtype_name, encode, encode_framed, DecodeOpts, Schema, Value};

fn cases_doc() -> serde_json::Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/kiln-v2-cases.json");
    let raw = std::fs::read_to_string(path).expect("read testdata/kiln-v2-cases.json");
    serde_json::from_str(&raw).expect("parse kiln-v2-cases.json")
}

fn decode_hex(text: &str) -> Vec<u8> {
    let bytes = text.as_bytes();
    assert!(bytes.len() % 2 == 0, "odd hex");
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let val = |c: u8| match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => panic!("bad hex"),
    };
    for i in (0..bytes.len()).step_by(2) {
        out.push((val(bytes[i]) << 4) | val(bytes[i + 1]));
    }
    out
}

fn encode_hex(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(data.len() * 2);
    for byte in data {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

fn parse_ty(text: &str) -> Schema {
    if let Some(rest) = text.strip_prefix("struct{") {
        let rest = rest.strip_suffix('}').expect("struct brace");
        let mut fields = Vec::new();
        for part in rest.split(',') {
            if part.is_empty() {
                continue;
            }
            let (name, ty) = part.split_once(':').expect("field");
            fields.push((name.to_string(), parse_ty(ty)));
        }
        return Schema::Struct(fields);
    }
    match text {
        "bool" => Schema::Bool,
        "i8" => Schema::I8,
        "i16" => Schema::I16,
        "i32" => Schema::I32,
        "i64" => Schema::I64,
        "int" => Schema::Int,
        "u8" => Schema::U8,
        "u16" => Schema::U16,
        "u32" => Schema::U32,
        "u64" => Schema::U64,
        "uint" => Schema::Uint,
        "f32" => Schema::F32,
        "f64" => Schema::F64,
        "str" => Schema::Str,
        "tensors" => Schema::Tensors,
        other => panic!("unknown type {other}"),
    }
}

fn case_schema(value: &serde_json::Value) -> (Schema, bool) {
    if let Some(name) = value.as_str() {
        return match name {
            "tensor-v2" => (Schema::Tensors, false),
            "frame" => (
                Schema::struct_fields([("id", Schema::I64), ("batch", Schema::Tensors)]),
                true,
            ),
            other => panic!("unknown schema {other}"),
        };
    }
    let obj = value.as_object().expect("schema object");
    match obj["type"].as_str().expect("schema type") {
        "list" => (
            Schema::list(parse_ty(obj["element"].as_str().unwrap())),
            false,
        ),
        "struct" => {
            let mut fields = Vec::new();
            for item in obj["fields"].as_array().expect("fields") {
                let text = item.as_str().expect("field");
                let (name, ty) = text.split_once(':').expect("field");
                fields.push((name.to_string(), parse_ty(ty)));
            }
            (Schema::Struct(fields), false)
        }
        other => panic!("schema type {other}"),
    }
}

fn project(schema: &Schema, value: &Value<'_>) -> serde_json::Value {
    match schema {
        Schema::Bool => match value {
            Value::Bool(v) => serde_json::Value::Bool(*v),
            _ => panic!("bool"),
        },
        Schema::I8 => num_i64(match value {
            Value::I8(v) => i64::from(*v),
            _ => panic!("i8"),
        }),
        Schema::I16 => num_i64(match value {
            Value::I16(v) => i64::from(*v),
            _ => panic!("i16"),
        }),
        Schema::I32 => num_i64(match value {
            Value::I32(v) => i64::from(*v),
            _ => panic!("i32"),
        }),
        Schema::I64 | Schema::Int => num_i64(match value {
            Value::I64(v) => *v,
            _ => panic!("i64"),
        }),
        Schema::U8 => num_u64(match value {
            Value::U8(v) => u64::from(*v),
            _ => panic!("u8"),
        }),
        Schema::U16 => num_u64(match value {
            Value::U16(v) => u64::from(*v),
            _ => panic!("u16"),
        }),
        Schema::U32 => num_u64(match value {
            Value::U32(v) => u64::from(*v),
            _ => panic!("u32"),
        }),
        Schema::U64 | Schema::Uint => num_u64(match value {
            Value::U64(v) => *v,
            _ => panic!("u64"),
        }),
        Schema::F32 => match value {
            Value::F32(bits) => num_f64(f32::from_bits(*bits) as f64),
            _ => panic!("f32"),
        },
        Schema::F64 => match value {
            Value::F64(bits) => num_f64(f64::from_bits(*bits)),
            _ => panic!("f64"),
        },
        Schema::Str => match value {
            Value::Str(text) => serde_json::Value::String(text.to_string()),
            _ => panic!("str"),
        },
        Schema::Bytes => panic!("bytes have no golden projection"),
        Schema::List(elem) => match value {
            Value::List(items) => {
                serde_json::Value::Array(items.iter().map(|item| project(elem, item)).collect())
            }
            Value::F32s(data) => project_f32s(data),
            Value::F64s(data) => project_f64s(data),
            Value::I64s(data) => project_i64s(data),
            _ => panic!("list"),
        },
        Schema::Opt(inner) => match value {
            Value::Opt(None) => serde_json::Value::Null,
            Value::Opt(Some(inner_value)) => project(inner, inner_value),
            _ => panic!("opt"),
        },
        Schema::Dict(_) => panic!("dict projection is not used by the golden file"),
        Schema::Struct(fields) => {
            let Value::Struct(values) = value else {
                panic!("struct");
            };
            let mut obj = serde_json::Map::new();
            for ((name, field), item) in fields.iter().zip(values.iter()) {
                obj.insert(name.clone(), project(field, item));
            }
            serde_json::Value::Object(obj)
        }
        Schema::Tensors => match value {
            Value::Tensors(dir) => project_dir(dir),
            _ => panic!("tensors"),
        },
    }
}

fn project_dir(dir: &kiln::TensorDir<'_>) -> serde_json::Value {
    let mut tensors = Vec::new();
    for tensor in &dir.tensors {
        let dtype = match dtype_name(tensor.dtype) {
            Some(name) => serde_json::Value::String(name.to_string()),
            None => num_u64(u64::from(tensor.dtype)),
        };
        let shape: Vec<serde_json::Value> = tensor.shape[..tensor.rank as usize]
            .iter()
            .map(|dim| num_u64(*dim))
            .collect();
        let name = match tensor.name {
            Some(text) => serde_json::Value::String(text.to_string()),
            None => serde_json::Value::Null,
        };
        tensors.push(serde_json::json!({
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "quant_scheme": tensor.quant_scheme,
            "data_hex": encode_hex(tensor.data),
        }));
    }
    let flags = if dir.names {
        serde_json::json!(["names", "le"])
    } else {
        serde_json::json!(["le"])
    };
    serde_json::json!({
        "version": 2,
        "flags": flags,
        "tensors": tensors,
        "block_length": dir.block_len,
    })
}

fn num_i64(v: i64) -> serde_json::Value {
    serde_json::Value::Number(serde_json::Number::from(v))
}

fn num_u64(v: u64) -> serde_json::Value {
    serde_json::Value::Number(serde_json::Number::from(v))
}

fn num_f64(v: f64) -> serde_json::Value {
    serde_json::Value::Number(serde_json::Number::from_f64(v).expect("finite"))
}

fn json_eq(left: &serde_json::Value, right: &serde_json::Value) -> bool {
    match (left, right) {
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => num_eq(a, b),
        (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| json_eq(x, y))
        }
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, value)| b.get(key).is_some_and(|other| json_eq(value, other)))
        }
        _ => left == right,
    }
}

fn num_eq(a: &serde_json::Number, b: &serde_json::Number) -> bool {
    if let (Some(x), Some(y)) = (a.as_u64(), b.as_u64()) {
        return x == y;
    }
    if let (Some(x), Some(y)) = (a.as_i64(), b.as_i64()) {
        return x == y;
    }
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

fn first_diff(left: &[u8], right: &[u8]) -> String {
    let n = left.len().min(right.len());
    for i in 0..n {
        if left[i] != right[i] {
            return format!(
                "at {i}: got {:02x} exp {:02x} (lens {} vs {})",
                left[i],
                right[i],
                left.len(),
                right.len()
            );
        }
    }
    format!("length got {} exp {}", left.len(), right.len())
}

fn project_f32s(data: &[u8]) -> serde_json::Value {
    serde_json::Value::Array(
        data.chunks_exact(4)
            .map(|chunk| {
                let bits = u32::from_le_bytes(chunk.try_into().unwrap());
                num_f64(f32::from_bits(bits) as f64)
            })
            .collect(),
    )
}

fn project_f64s(data: &[u8]) -> serde_json::Value {
    serde_json::Value::Array(
        data.chunks_exact(8)
            .map(|chunk| {
                let bits = u64::from_le_bytes(chunk.try_into().unwrap());
                num_f64(f64::from_bits(bits))
            })
            .collect(),
    )
}

fn project_i64s(data: &[u8]) -> serde_json::Value {
    serde_json::Value::Array(
        data.chunks_exact(8)
            .map(|chunk| num_i64(i64::from_le_bytes(chunk.try_into().unwrap())))
            .collect(),
    )
}

fn payload_in_input(buf: &[u8], value: &Value<'_>) -> bool {
    match value {
        Value::Tensors(dir) => dir.tensors.iter().all(|tensor| slice_in(buf, tensor.data)),
        Value::Struct(items) | Value::List(items) => {
            items.iter().all(|item| payload_in_input(buf, item))
        }
        Value::Opt(Some(inner)) => payload_in_input(buf, inner),
        Value::Bytes(data)
        | Value::I8s(data)
        | Value::F32s(data)
        | Value::F64s(data)
        | Value::I64s(data) => slice_in(buf, data),
        _ => true,
    }
}

fn slice_in(buf: &[u8], data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }
    let start = buf.as_ptr() as usize;
    let end = start + buf.len();
    let data_start = data.as_ptr() as usize;
    let data_end = data_start + data.len();
    data_start >= start && data_end <= end
}

#[test]
fn loads_kiln_v2_cases() {
    let doc = cases_doc();
    let cases = doc["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 46, "the Kiln v2 file has 46 cases");
    let mut accept = 0;
    let mut reject = 0;
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let expect = case["expect"].as_str().unwrap();
        let input = decode_hex(case["input"].as_str().unwrap());
        let (schema, framed) = case_schema(&case["schema"]);
        let decoded = if framed {
            decode_framed(&schema, &input)
        } else {
            decode(&schema, &input)
        };
        match expect {
            "accept" => {
                accept += 1;
                let value = decoded.unwrap_or_else(|err| panic!("{id}: decode {}", err.name()));
                assert!(
                    payload_in_input(&input, &value),
                    "{id}: payload is not a view"
                );
                if let Some(recorded) = case.get("decoded") {
                    let mut got = project(&schema, &value);
                    if framed {
                        let hash = kiln::type_hash(&schema);
                        let serde_json::Value::Object(map) = &mut got else {
                            panic!("{id}: frame projection");
                        };
                        map.insert("hash".into(), num_u64(hash));
                    }
                    assert!(
                        json_eq(&got, recorded),
                        "{id}: decoded mismatch\n got {got}\n exp {recorded}"
                    );
                }
                let encoded = if framed {
                    encode_framed(&schema, &value)
                        .unwrap_or_else(|err| panic!("{id}: {}", err.name()))
                } else {
                    encode(&schema, &value).unwrap_or_else(|err| panic!("{id}: {}", err.name()))
                };
                assert_eq!(
                    encoded,
                    input,
                    "{id}: re-encode {}",
                    first_diff(&encoded, &input)
                );
            }
            "reject" => {
                reject += 1;
                let err = decoded.expect_err(id);
                let want = case["error"].as_str().unwrap();
                assert_eq!(err.name(), want, "{id}: error");
            }
            other => panic!("{id}: expect {other}"),
        }
    }
    assert_eq!(accept, 17);
    assert_eq!(reject, 29);
    let _ = DecodeOpts::default();
}

#[test]
fn version_file_matches() {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/VERSION")).unwrap();
    assert_eq!(raw.trim(), kiln::VERSION);
}
