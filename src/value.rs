//! Borrowed dynamic values. Strings and tensor payloads alias the input.

use std::borrow::Cow;

/// One tensor inside a version-2 directory.
///
/// `data` is a slice of the input buffer when the tensor was decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor<'a> {
    pub name: Option<&'a str>,
    pub dtype: u16,
    pub rank: u8,
    pub shape: [u64; 8],
    pub quant_scheme: u16,
    pub quant_axis: u16,
    pub group_size: u32,
    pub scale_index: u32,
    pub outer_index: u32,
    pub block_size: u32,
    pub layout: Option<&'a str>,
    pub data: &'a [u8],
}

impl<'a> Tensor<'a> {
    /// Scalar-free descriptor defaults: no quant, no name, empty shape.
    pub fn new(dtype: u16, data: &'a [u8]) -> Self {
        Self {
            name: None,
            dtype,
            rank: 0,
            shape: [0; 8],
            quant_scheme: 0,
            quant_axis: 0xffff,
            group_size: 0,
            scale_index: 0xffff_ffff,
            outer_index: 0xffff_ffff,
            block_size: 0,
            layout: None,
            data,
        }
    }

    pub fn with_shape(mut self, shape: &[u64]) -> Self {
        self.rank = shape.len() as u8;
        for (i, dim) in shape.iter().enumerate().take(8) {
            self.shape[i] = *dim;
        }
        self
    }
}

/// A decoded version-2 directory. Payload bytes borrow `data`'s buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDir<'a> {
    pub tensors: Vec<Tensor<'a>>,
    pub block_len: usize,
    pub names: bool,
}

/// A schema-described value. Integers are the raw two's-complement bits.
/// `F32` and `F64` store IEEE bits so negative zero and NaN payloads survive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value<'a> {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(u32),
    F64(u64),
    Str(Cow<'a, str>),
    Bytes(Cow<'a, [u8]>),
    I8s(Cow<'a, [u8]>),
    List(Vec<Value<'a>>),
    Opt(Option<Box<Value<'a>>>),
    Dict(Vec<(Cow<'a, str>, Value<'a>)>),
    Struct(Vec<Value<'a>>),
    Tensors(TensorDir<'a>),
}

impl<'a> Value<'a> {
    pub fn from_f32(v: f32) -> Self {
        Value::F32(v.to_bits())
    }

    pub fn from_f64(v: f64) -> Self {
        Value::F64(v.to_bits())
    }

    pub fn str(s: impl Into<Cow<'a, str>>) -> Self {
        Value::Str(s.into())
    }
}
