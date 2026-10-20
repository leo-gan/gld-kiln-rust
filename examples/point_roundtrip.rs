use kiln::{decode, encode, Schema, Value};

fn main() {
    let schema = Schema::struct_fields([("x", Schema::I32), ("y", Schema::I32)]);
    let value = Value::Struct(vec![Value::I32(3), Value::I32(-4)]);
    let bytes = encode(&schema, &value).expect("encode");
    let back = decode(&schema, &bytes).expect("decode");
    assert_eq!(back, value);
    println!("point {} bytes", bytes.len());
}
