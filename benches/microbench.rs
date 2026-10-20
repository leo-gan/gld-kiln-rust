//! Encode and decode a small struct batch and one 1024-element f32 tensor.

use std::hint::black_box;
use std::time::Instant;

use kiln::{
    decode, encode_into, encode_tensors_into, write_i32, write_str, DecodeOpts, Reader, Schema,
    Tensor, Value, DEFAULT_CAP, DTYPE_F32,
};

const BATCH: usize = 1024;
const TENSOR_LEN: usize = 1024;

fn point_schema() -> Schema {
    Schema::struct_fields([
        ("x", Schema::I32),
        ("y", Schema::I32),
        ("label", Schema::Str),
    ])
}

fn point_value() -> Value<'static> {
    Value::Struct(vec![Value::I32(7), Value::I32(-2), Value::str("pt")])
}

fn batch_value(schema_item: &Value<'static>) -> Value<'static> {
    Value::List(vec![schema_item.clone(); BATCH])
}

fn time_ns(mut body: impl FnMut()) -> u128 {
    // Warm the instruction cache and the allocator, then take the best of five.
    for _ in 0..3 {
        body();
    }
    let mut best = u128::MAX;
    for _ in 0..5 {
        let start = Instant::now();
        body();
        let elapsed = start.elapsed().as_nanos();
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}

fn bench_struct_batch() {
    let schema = Schema::list(point_schema());
    let value = batch_value(&point_value());
    let mut buf = Vec::new();
    encode_into(&schema, &value, &mut buf, DEFAULT_CAP).unwrap();
    let encoded = buf.clone();
    let encode_ns = time_ns(|| {
        buf.clear();
        encode_into(&schema, &value, &mut buf, DEFAULT_CAP).unwrap();
        black_box(buf.len());
    });
    let decode_ns = time_ns(|| {
        let back = decode(&schema, &encoded).unwrap();
        black_box(back);
    });
    println!(
        "struct_batch count={BATCH} encode_ns={encode_ns} decode_ns={decode_ns} bytes={}",
        encoded.len()
    );

    let direct_encode_ns = time_ns(|| {
        buf.clear();
        for i in 0..BATCH {
            write_i32(&mut buf, i as i32);
            write_i32(&mut buf, -(i as i32));
            write_str(&mut buf, "pt", DEFAULT_CAP).unwrap();
        }
        black_box(buf.len());
    });
    let direct = buf.clone();
    let direct_decode_ns = time_ns(|| {
        let mut reader = Reader::new(&direct);
        let mut sum = 0i64;
        for _ in 0..BATCH {
            sum += i64::from(reader.i32().unwrap());
            sum += i64::from(reader.i32().unwrap());
            black_box(reader.string().unwrap());
        }
        reader.finish().unwrap();
        black_box(sum);
    });
    println!(
        "struct_direct count={BATCH} encode_ns={direct_encode_ns} decode_ns={direct_decode_ns} bytes={}",
        direct.len()
    );
}

fn bench_tensor() {
    let payload = vec![0x3fu8; TENSOR_LEN * 4];
    let tensor = Tensor::new(DTYPE_F32, &payload).with_shape(&[TENSOR_LEN as u64]);
    let mut buf = Vec::new();
    encode_tensors_into(std::slice::from_ref(&tensor), &mut buf, DEFAULT_CAP).unwrap();
    let encoded = buf.clone();
    let encode_ns = time_ns(|| {
        buf.clear();
        encode_tensors_into(std::slice::from_ref(&tensor), &mut buf, DEFAULT_CAP).unwrap();
        black_box(buf.len());
    });
    let opts = DecodeOpts::default();
    let decode_ns = time_ns(|| {
        let dir = kiln::decode_tensors(&encoded, opts).unwrap();
        black_box(dir.tensors[0].data.len());
    });
    println!(
        "tensor_f32 count={TENSOR_LEN} encode_ns={encode_ns} decode_ns={decode_ns} bytes={}",
        encoded.len()
    );
}

fn bench_f32_list() {
    const N: usize = 1024 * 1024;
    let mut body = vec![0u8; N * 4];
    body[0] = 0x00;
    body[1] = 0x00;
    body[2] = 0x00;
    body[3] = 0x80; // negative zero
    let schema = Schema::list(Schema::F32);
    let value = Value::F32s(std::borrow::Cow::Borrowed(&body));
    let mut buf = Vec::new();
    encode_into(&schema, &value, &mut buf, DEFAULT_CAP).unwrap();
    let encoded = buf.clone();
    let encode_ns = time_ns(|| {
        buf.clear();
        encode_into(&schema, &value, &mut buf, DEFAULT_CAP).unwrap();
        black_box(buf.len());
    });
    let decode_ns = time_ns(|| {
        let back = decode(&schema, &encoded).unwrap();
        let Value::F32s(data) = &back else {
            panic!("f32 list");
        };
        black_box(data.len());
    });
    println!(
        "f32_list count={N} encode_ns={encode_ns} decode_ns={decode_ns} bytes={}",
        encoded.len()
    );
}

fn main() {
    bench_struct_batch();
    bench_tensor();
    bench_f32_list();
}
