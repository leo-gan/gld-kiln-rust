//! Positional schemas and the FNV-1a 64 frame hash.

use crate::error::Error;

const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// A Frozen schema. Field order is the wire order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Schema {
    Bool,
    I8,
    I16,
    I32,
    I64,
    Int,
    U8,
    U16,
    U32,
    U64,
    Uint,
    F32,
    F64,
    Str,
    Bytes,
    List(Box<Schema>),
    Opt(Box<Schema>),
    Dict(Box<Schema>),
    Struct(Vec<(String, Schema)>),
    Tensors,
}

impl Schema {
    pub fn list(inner: Schema) -> Self {
        Schema::List(Box::new(inner))
    }

    pub fn opt(inner: Schema) -> Self {
        Schema::Opt(Box::new(inner))
    }

    pub fn dict(inner: Schema) -> Self {
        Schema::Dict(Box::new(inner))
    }

    pub fn struct_fields<N, I>(fields: I) -> Self
    where
        N: Into<String>,
        I: IntoIterator<Item = (N, Schema)>,
    {
        Schema::Struct(
            fields
                .into_iter()
                .map(|(name, schema)| (name.into(), schema))
                .collect(),
        )
    }
}

/// Minimum byte length of one value, used before a list or dict allocates.
pub fn min_size(schema: &Schema) -> u64 {
    match schema {
        Schema::Bool | Schema::I8 | Schema::U8 => 1,
        Schema::I16 | Schema::U16 => 2,
        Schema::I32 | Schema::U32 | Schema::F32 => 4,
        Schema::I64 | Schema::Int | Schema::U64 | Schema::Uint | Schema::F64 => 8,
        Schema::Str | Schema::Bytes | Schema::List(_) | Schema::Dict(_) => 4,
        Schema::Opt(_) => 1,
        Schema::Tensors => 64,
        Schema::Struct(fields) => fields.iter().map(|(_, field)| min_size(field)).sum(),
    }
}

fn mix(mut hash: u64, data: &[u8]) -> u64 {
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    // Trailing 0x00 is a multiply: xor with zero would change nothing.
    hash.wrapping_mul(FNV_PRIME)
}

fn hash_schema(hash: u64, schema: &Schema) -> u64 {
    match schema {
        Schema::Bool => mix(hash, b"bool"),
        Schema::I8 => mix(hash, b"i8"),
        Schema::I16 => mix(hash, b"i16"),
        Schema::I32 => mix(hash, b"i32"),
        Schema::I64 => mix(hash, b"i64"),
        Schema::Int => mix(hash, b"int"),
        Schema::U8 => mix(hash, b"u8"),
        Schema::U16 => mix(hash, b"u16"),
        Schema::U32 => mix(hash, b"u32"),
        Schema::U64 => mix(hash, b"u64"),
        Schema::Uint => mix(hash, b"uint"),
        Schema::F32 => mix(hash, b"f32"),
        Schema::F64 => mix(hash, b"f64"),
        Schema::Str => mix(hash, b"str"),
        Schema::Bytes => mix(hash, b"bytes"),
        Schema::Tensors => mix(hash, b"tensors"),
        Schema::List(inner) => {
            let hash = mix(hash, b"list<");
            let hash = hash_schema(hash, inner);
            mix(hash, b">")
        }
        Schema::Opt(inner) => {
            let hash = mix(hash, b"opt<");
            let hash = hash_schema(hash, inner);
            mix(hash, b">")
        }
        Schema::Dict(inner) => {
            let hash = mix(hash, b"dict<str,");
            let hash = hash_schema(hash, inner);
            mix(hash, b">")
        }
        Schema::Struct(fields) => {
            let mut hash = mix(hash, b"struct{");
            for (name, field) in fields {
                hash = mix(hash, name.as_bytes());
                hash = mix(hash, b":");
                hash = hash_schema(hash, field);
                hash = mix(hash, b",");
            }
            mix(hash, b"}")
        }
    }
}

/// FNV-1a 64 schema hash written in a `KLN0` frame.
pub fn type_hash(schema: &Schema) -> u64 {
    hash_schema(FNV_OFFSET, schema)
}

pub(crate) fn add_u64(a: u128, b: u128) -> Result<u64, Error> {
    let total = a + b;
    if total > u64::MAX as u128 {
        return Err(Error::Limit);
    }
    Ok(total as u64)
}

pub(crate) fn mul_u64(a: u128, b: u128) -> Result<u64, Error> {
    if a != 0 && b > (u64::MAX as u128) / a {
        return Err(Error::Limit);
    }
    Ok((a * b) as u64)
}

/// Round `n` up to a multiple of 64. The result may exceed `u64`.
pub(crate) fn align64(n: u128) -> u128 {
    (n + 63) & !63
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_hash_includes_tensors_token() {
        let schema = Schema::struct_fields([("id", Schema::I64), ("batch", Schema::Tensors)]);
        assert_eq!(type_hash(&schema), 16_540_223_653_110_812_034);
    }

    #[test]
    fn struct_min_is_sum() {
        let schema = Schema::struct_fields([("a", Schema::I64)]);
        assert_eq!(min_size(&schema), 8);
        assert_eq!(min_size(&Schema::list(Schema::Str)), 4);
        assert_eq!(min_size(&Schema::Tensors), 64);
    }
}
