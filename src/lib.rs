//! From-scratch Kiln codec. Frozen values are positional and little-endian.
//! A tensor directory keeps numeric payloads as slices of the input buffer.
//!
//! This crate is not the Kiln specification. The normative wire is
//! <https://leo-gan.github.io/gld-kiln/spec/>.

#![allow(clippy::result_large_err)]

pub mod emit;
pub mod error;
pub mod frozen;
pub mod schema;
pub mod tensor;
pub mod value;

pub use error::{DecodeOpts, Error, DEFAULT_CAP};
pub use frozen::{
    decode, decode_framed, decode_framed_with, decode_with, encode, encode_framed,
    encode_framed_with, encode_into, encode_with, write_bool, write_bytes, write_count, write_f32,
    write_f64, write_i16, write_i32, write_i64, write_i8, write_str, write_u16, write_u32,
    write_u64, write_u8, Reader,
};
pub use schema::{min_size, type_hash, Schema};
pub use tensor::{
    decode_tensors, dtype_name, encode_tensors, encode_tensors_into, DTYPE_BF16, DTYPE_BOOL,
    DTYPE_E2M1, DTYPE_E8M0, DTYPE_F16, DTYPE_F32, DTYPE_F64, DTYPE_F8E4M3, DTYPE_F8E5M2, DTYPE_I16,
    DTYPE_I32, DTYPE_I4, DTYPE_I64, DTYPE_I8, DTYPE_U16, DTYPE_U32, DTYPE_U4, DTYPE_U64, DTYPE_U8,
};
pub use value::{Tensor, TensorDir, Value};

/// Crate version. Kept in step with `VERSION` and `Cargo.toml`.
pub const VERSION: &str = "0.1.0";
