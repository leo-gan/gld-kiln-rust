//! Behaviors covered by the Python wire tests beyond the 46 golden cases.

use std::borrow::Cow;

use kiln::{
    decode, decode_framed, decode_tensors, decode_with, encode, encode_framed, encode_tensors,
    encode_with, DecodeOpts, Error, Reader, Schema, Tensor, Value, DEFAULT_CAP, DTYPE_F8E4M3,
    DTYPE_F8E5M2,
};

fn list_of(depth: usize, inner: Value<'static>) -> Value<'static> {
    let mut value = inner;
    for _ in 0..depth {
        value = Value::List(vec![value]);
    }
    value
}

fn nested_list(depth: usize) -> Schema {
    let mut schema = Schema::I32;
    for _ in 0..depth {
        schema = Schema::list(schema);
    }
    schema
}

#[test]
fn negative_zero_and_nan_payload_round_trip() {
    let schema = Schema::list(Schema::F64);
    let mut raw = vec![2, 0, 0, 0];
    raw.extend_from_slice(&(-0.0f64).to_bits().to_le_bytes());
    raw.extend_from_slice(&0x7ff8_0000_0000_0001u64.to_le_bytes());
    let value = decode(&schema, &raw).unwrap();
    let Value::List(items) = &value else {
        panic!("list");
    };
    assert_eq!(items[0], Value::F64((-0.0f64).to_bits()));
    assert_eq!(items[1], Value::F64(0x7ff8_0000_0000_0001));
    assert_eq!(encode(&schema, &value).unwrap(), raw);
}

#[test]
fn dict_keys_sort_and_opt_round_trips() {
    let schema = Schema::dict(Schema::I32);
    let value = Value::Dict(vec![
        (Cow::Borrowed("b"), Value::I32(2)),
        (Cow::Borrowed("a"), Value::I32(1)),
    ]);
    let raw = encode(&schema, &value).unwrap();
    let back = decode(&schema, &raw).unwrap();
    let Value::Dict(pairs) = &back else {
        panic!("dict");
    };
    assert_eq!(pairs[0].0.as_ref(), "a");
    assert_eq!(pairs[0].1, Value::I32(1));
    assert_eq!(pairs[1].0.as_ref(), "b");
    assert_eq!(pairs[1].1, Value::I32(2));

    let opt = Schema::opt(Schema::Str);
    assert_eq!(
        decode(&opt, &encode(&opt, &Value::Opt(None)).unwrap()).unwrap(),
        Value::Opt(None)
    );
    let some = Value::Opt(Some(Box::new(Value::str("z"))));
    assert_eq!(decode(&opt, &encode(&opt, &some).unwrap()).unwrap(), some);
}

#[test]
fn frame_hash_mismatch_is_schema() {
    let schema = Schema::struct_fields([("id", Schema::I64)]);
    let raw = encode_framed(&schema, &Value::Struct(vec![Value::I64(1)])).unwrap();
    let other = Schema::struct_fields([("id", Schema::I32)]);
    let err = decode_framed(&other, &raw).unwrap_err();
    assert_eq!(err, Error::Schema);
}

#[test]
fn nonzero_alignment_gap_is_type() {
    let tensors = [
        Tensor::new(DTYPE_F8E4M3, &[0x38]).with_shape(&[1]),
        Tensor::new(DTYPE_F8E5M2, &[0x3c]).with_shape(&[1]),
    ];
    let mut raw = encode_tensors(&tensors, DEFAULT_CAP).unwrap();
    assert!(raw.len() > 320 + 63, "directory layout changed");
    raw[320 + 63] = 1;
    let err = decode_tensors(&raw, DecodeOpts::default()).unwrap_err();
    assert_eq!(err, Error::Type);
}

#[test]
fn reader_str_fast_skips_the_cap_but_not_the_buffer() {
    let raw = encode_with(&Schema::Str, &Value::str("z".repeat(20)), 100).unwrap();
    let mut fast = Reader::with_opts(
        &raw,
        DecodeOpts {
            cap: 10,
            ..DecodeOpts::default()
        },
    );
    assert_eq!(fast.str_fast().unwrap(), "z".repeat(20));
    let err = decode_with(
        &Schema::Str,
        &raw,
        DecodeOpts {
            cap: 10,
            ..DecodeOpts::default()
        },
    )
    .unwrap_err();
    assert_eq!(err, Error::Limit);

    let mut short = Reader::new(&[0x01, 0x00, 0x00, 0x00]);
    assert_eq!(short.str_fast().unwrap_err(), Error::Eof);
}

#[test]
fn depth_limit_is_sixty_four_containers() {
    let ok_schema = nested_list(64);
    let ok_value = list_of(64, Value::I32(1));
    let raw = encode(&ok_schema, &ok_value).unwrap();
    assert_eq!(decode(&ok_schema, &raw).unwrap(), ok_value);

    let bad_schema = nested_list(65);
    let bad_value = list_of(65, Value::I32(1));
    assert_eq!(encode(&bad_schema, &bad_value).unwrap_err(), Error::Limit);

    let mut hostile = vec![0u8; 65 * 4];
    for chunk in hostile.chunks_exact_mut(4) {
        chunk.copy_from_slice(&1u32.to_le_bytes());
    }
    hostile.extend_from_slice(&1i32.to_le_bytes());
    assert_eq!(decode(&bad_schema, &hostile).unwrap_err(), Error::Limit);
}

#[test]
fn signed_byte_list_round_trips() {
    let schema = Schema::list(Schema::I8);
    let value = Value::List(vec![
        Value::I8(0),
        Value::I8(-1),
        Value::I8(127),
        Value::I8(-128),
    ]);
    let raw = encode(&schema, &value).unwrap();
    let back = decode(&schema, &raw).unwrap();
    let Value::I8s(bytes) = &back else {
        panic!("list<i8> should borrow the payload");
    };
    assert_eq!(&bytes[..], &[0, 0xff, 127, 128]);
    assert_eq!(encode(&schema, &back).unwrap(), raw);
}

#[test]
fn struct_list_pool_reuse_keeps_values() {
    let schema = Schema::list(Schema::struct_fields([
        ("x", Schema::I32),
        ("y", Schema::I32),
        ("label", Schema::Str),
    ]));
    let point = |x, y, label: &'static str| {
        Value::Struct(vec![Value::I32(x), Value::I32(y), Value::str(label)])
    };
    let first = Value::List(vec![
        point(1, 2, "pt"),
        point(3, 4, "ab"),
        point(5, 6, "cd"),
    ]);
    let second = Value::List(vec![point(9, -1, "z")]);
    let raw_first = encode(&schema, &first).unwrap();
    let raw_second = encode(&schema, &second).unwrap();
    assert_eq!(decode(&schema, &raw_first).unwrap(), first);
    assert_eq!(decode(&schema, &raw_second).unwrap(), second);
    assert_eq!(decode(&schema, &raw_first).unwrap(), first);
}
