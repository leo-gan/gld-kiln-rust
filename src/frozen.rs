//! Frozen positional values. Scalars are little-endian.

use std::borrow::Cow;

use crate::error::{DecodeOpts, Error, DEFAULT_CAP};
use crate::schema::{min_size, type_hash, Schema};
use crate::tensor::{decode_tensors, encode_tensors_into, tensor_block_len};
use crate::value::Value;

/// Cursor over a Frozen buffer.
pub struct Reader<'a> {
    buf: &'a [u8],
    i: usize,
    cap: u64,
    trusted: bool,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self::with_opts(buf, DecodeOpts::default())
    }

    pub fn with_opts(buf: &'a [u8], opts: DecodeOpts) -> Self {
        Self {
            buf,
            i: 0,
            cap: opts.cap,
            trusted: opts.trusted,
        }
    }

    pub fn position(&self) -> usize {
        self.i
    }

    pub fn finish(&self) -> Result<(), Error> {
        if self.i != self.buf.len() {
            Err(Error::Trailing)
        } else {
            Ok(())
        }
    }

    #[inline]
    fn take<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        if self.buf.len().saturating_sub(self.i) < N {
            return Err(Error::Eof);
        }
        let mut raw = [0u8; N];
        raw.copy_from_slice(&self.buf[self.i..self.i + N]);
        self.i += N;
        Ok(raw)
    }

    #[inline]
    pub fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take::<1>()?[0])
    }

    pub fn i8(&mut self) -> Result<i8, Error> {
        Ok(i8::from_le_bytes(self.take()?))
    }

    pub fn u16(&mut self) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(self.take()?))
    }

    pub fn i16(&mut self) -> Result<i16, Error> {
        Ok(i16::from_le_bytes(self.take()?))
    }

    #[inline]
    pub fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    #[inline]
    pub fn i32(&mut self) -> Result<i32, Error> {
        Ok(i32::from_le_bytes(self.take()?))
    }

    pub fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    pub fn i64(&mut self) -> Result<i64, Error> {
        Ok(i64::from_le_bytes(self.take()?))
    }

    pub fn f32(&mut self) -> Result<f32, Error> {
        Ok(f32::from_bits(self.u32()?))
    }

    pub fn f64(&mut self) -> Result<f64, Error> {
        Ok(f64::from_bits(self.u64()?))
    }

    #[inline]
    pub fn bool(&mut self) -> Result<bool, Error> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error::Type),
        }
    }

    /// `true` when the optional value is present.
    pub fn opt_tag(&mut self) -> Result<bool, Error> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error::Type),
        }
    }

    /// Count of a list or a byte run. Does not allocate element storage.
    #[inline]
    pub fn list_count(&mut self, elem_min: u64) -> Result<u32, Error> {
        if self.buf.len().saturating_sub(self.i) < 4 {
            return Err(Error::Eof);
        }
        let count = u32::from_le_bytes([
            self.buf[self.i],
            self.buf[self.i + 1],
            self.buf[self.i + 2],
            self.buf[self.i + 3],
        ]);
        if u64::from(count) > self.cap {
            return Err(Error::Limit);
        }
        if elem_min != 0 && u64::from(count) > u64::MAX / elem_min {
            return Err(Error::Limit);
        }
        let need = u64::from(count) * elem_min;
        let rest = (self.buf.len() - self.i - 4) as u64;
        if need > rest {
            return Err(Error::Eof);
        }
        self.i += 4;
        Ok(count)
    }

    /// Count of a dict. `value_min` is the minimum size of one value.
    pub fn dict_count(&mut self, value_min: u64) -> Result<u32, Error> {
        let elem = 4u64.checked_add(value_min).ok_or(Error::Limit)?;
        self.list_count(elem)
    }

    pub fn raw(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.i.checked_add(n).ok_or(Error::Limit)?;
        if end > self.buf.len() {
            return Err(Error::Eof);
        }
        let slice = &self.buf[self.i..end];
        self.i = end;
        Ok(slice)
    }

    pub fn bytes(&mut self) -> Result<&'a [u8], Error> {
        let n = self.list_count(1)?;
        self.raw(n as usize)
    }

    #[inline]
    pub fn string(&mut self) -> Result<&'a str, Error> {
        if self.buf.len().saturating_sub(self.i) < 4 {
            return Err(Error::Eof);
        }
        let len = u32::from_le_bytes([
            self.buf[self.i],
            self.buf[self.i + 1],
            self.buf[self.i + 2],
            self.buf[self.i + 3],
        ]);
        if u64::from(len) > self.cap {
            return Err(Error::Limit);
        }
        let rest = self.buf.len() - self.i - 4;
        if (len as usize) > rest {
            return Err(Error::Eof);
        }
        self.i += 4;
        let bytes = self.raw(len as usize)?;
        if self.trusted {
            // SAFETY: trusted mode promises the payload is UTF-8.
            Ok(unsafe { std::str::from_utf8_unchecked(bytes) })
        } else {
            std::str::from_utf8(bytes).map_err(|_| Error::Utf8)
        }
    }

    pub fn string_owned(&mut self) -> Result<String, Error> {
        self.string().map(str::to_owned)
    }

    pub(crate) fn read_value(&mut self, schema: &Schema) -> Result<Value<'a>, Error> {
        let buf = self.buf;
        let cap = self.cap;
        let trusted = self.trusted;
        read_value_at(buf, &mut self.i, schema, cap, trusted)
    }
}

fn read_value_at<'a>(
    buf: &'a [u8],
    i: &mut usize,
    schema: &Schema,
    cap: u64,
    trusted: bool,
) -> Result<Value<'a>, Error> {
    match schema {
        Schema::Bool => Ok(Value::Bool(read_bool(buf, i)?)),
        Schema::I8 => Ok(Value::I8(read_i8(buf, i)?)),
        Schema::I16 => Ok(Value::I16(read_i16(buf, i)?)),
        Schema::I32 => Ok(Value::I32(read_i32(buf, i)?)),
        Schema::I64 | Schema::Int => Ok(Value::I64(read_i64(buf, i)?)),
        Schema::U8 => Ok(Value::U8(read_u8(buf, i)?)),
        Schema::U16 => Ok(Value::U16(read_u16(buf, i)?)),
        Schema::U32 => Ok(Value::U32(read_u32(buf, i)?)),
        Schema::U64 | Schema::Uint => Ok(Value::U64(read_u64(buf, i)?)),
        Schema::F32 => Ok(Value::F32(read_u32(buf, i)?)),
        Schema::F64 => Ok(Value::F64(read_u64(buf, i)?)),
        Schema::Str => Ok(Value::Str(Cow::Borrowed(read_string(buf, i, cap, trusted)?))),
        Schema::Bytes => Ok(Value::Bytes(Cow::Borrowed(read_bytes(buf, i, cap)?))),
        Schema::List(elem) => read_list(buf, i, elem, cap, trusted),
        Schema::Opt(inner) => {
            if read_tag(buf, i)? {
                Ok(Value::Opt(Some(Box::new(read_value_at(buf, i, inner, cap, trusted)?))))
            } else {
                Ok(Value::Opt(None))
            }
        }
        Schema::Dict(inner) => read_dict(buf, i, inner, cap, trusted),
        Schema::Struct(fields) => {
            let mut values = Vec::with_capacity(fields.len());
            for (_, field) in fields {
                values.push(read_value_at(buf, i, field, cap, trusted)?);
            }
            Ok(Value::Struct(values))
        }
        Schema::Tensors => {
            let nested = DecodeOpts {
                cap,
                trusted,
                standalone: false,
            };
            let dir = decode_tensors(&buf[*i..], nested)?;
            *i += dir.block_len;
            Ok(Value::Tensors(dir))
        }
    }
}

fn read_list<'a>(
    buf: &'a [u8],
    i: &mut usize,
    elem: &Schema,
    cap: u64,
    trusted: bool,
) -> Result<Value<'a>, Error> {
    match elem {
        Schema::U8 => {
            let n = list_count_at(buf, i, 1, cap)?;
            Ok(Value::Bytes(Cow::Borrowed(read_raw(buf, i, n as usize)?)))
        }
        Schema::I8 => {
            let n = list_count_at(buf, i, 1, cap)?;
            Ok(Value::I8s(Cow::Borrowed(read_raw(buf, i, n as usize)?)))
        }
        _ => {
            let n = list_count_at(buf, i, min_size(elem), cap)?;
            let mut items = Vec::with_capacity(n as usize);
            for _ in 0..n {
                items.push(read_value_at(buf, i, elem, cap, trusted)?);
            }
            Ok(Value::List(items))
        }
    }
}

fn read_dict<'a>(
    buf: &'a [u8],
    i: &mut usize,
    elem: &Schema,
    cap: u64,
    trusted: bool,
) -> Result<Value<'a>, Error> {
    let n = list_count_at(buf, i, 4u64.checked_add(min_size(elem)).ok_or(Error::Limit)?, cap)?;
    let mut pairs = Vec::with_capacity(n as usize);
    let mut prev: Option<(usize, usize)> = None;
    for _ in 0..n {
        let key = read_string(buf, i, cap, trusted)?;
        let at = key.as_ptr() as usize - buf.as_ptr() as usize;
        let len = key.len();
        if let Some((pat, plen)) = prev {
            if key.as_bytes() <= &buf[pat..pat + plen] {
                return Err(Error::Type);
            }
        }
        let value = read_value_at(buf, i, elem, cap, trusted)?;
        let key = &buf[at..at + len];
        let key = if trusted {
            // SAFETY: the same bytes already passed the trusted UTF-8 promise.
            unsafe { std::str::from_utf8_unchecked(key) }
        } else {
            std::str::from_utf8(key).map_err(|_| Error::Utf8)?
        };
        prev = Some((at, len));
        pairs.push((Cow::Borrowed(key), value));
    }
    Ok(Value::Dict(pairs))
}

fn list_count_at(buf: &[u8], i: &mut usize, elem_min: u64, cap: u64) -> Result<u32, Error> {
    if buf.len().saturating_sub(*i) < 4 {
        return Err(Error::Eof);
    }
    let count = u32::from_le_bytes([buf[*i], buf[*i + 1], buf[*i + 2], buf[*i + 3]]);
    if u64::from(count) > cap {
        return Err(Error::Limit);
    }
    if elem_min != 0 && u64::from(count) > u64::MAX / elem_min {
        return Err(Error::Limit);
    }
    let need = u64::from(count) * elem_min;
    let rest = (buf.len() - *i - 4) as u64;
    if need > rest {
        return Err(Error::Eof);
    }
    *i += 4;
    Ok(count)
}

fn read_raw<'a>(buf: &'a [u8], i: &mut usize, n: usize) -> Result<&'a [u8], Error> {
    let end = i.checked_add(n).ok_or(Error::Limit)?;
    if end > buf.len() {
        return Err(Error::Eof);
    }
    let slice = &buf[*i..end];
    *i = end;
    Ok(slice)
}

fn read_bytes<'a>(buf: &'a [u8], i: &mut usize, cap: u64) -> Result<&'a [u8], Error> {
    let n = list_count_at(buf, i, 1, cap)?;
    read_raw(buf, i, n as usize)
}

fn read_string<'a>(buf: &'a [u8], i: &mut usize, cap: u64, trusted: bool) -> Result<&'a str, Error> {
    if buf.len().saturating_sub(*i) < 4 {
        return Err(Error::Eof);
    }
    let len = u32::from_le_bytes([buf[*i], buf[*i + 1], buf[*i + 2], buf[*i + 3]]);
    if u64::from(len) > cap {
        return Err(Error::Limit);
    }
    let rest = buf.len() - *i - 4;
    if len as usize > rest {
        return Err(Error::Eof);
    }
    *i += 4;
    let bytes = read_raw(buf, i, len as usize)?;
    if trusted {
        // SAFETY: trusted mode promises the payload is UTF-8.
        Ok(unsafe { std::str::from_utf8_unchecked(bytes) })
    } else {
        std::str::from_utf8(bytes).map_err(|_| Error::Utf8)
    }
}

fn read_bool(buf: &[u8], i: &mut usize) -> Result<bool, Error> {
    match read_u8(buf, i)? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Error::Type),
    }
}

fn read_tag(buf: &[u8], i: &mut usize) -> Result<bool, Error> {
    read_bool(buf, i)
}

fn read_u8(buf: &[u8], i: &mut usize) -> Result<u8, Error> {
    if *i >= buf.len() {
        return Err(Error::Eof);
    }
    let v = buf[*i];
    *i += 1;
    Ok(v)
}

fn read_i8(buf: &[u8], i: &mut usize) -> Result<i8, Error> {
    Ok(read_u8(buf, i)? as i8)
}

fn read_u16(buf: &[u8], i: &mut usize) -> Result<u16, Error> {
    let raw = read_raw(buf, i, 2)?;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

fn read_i16(buf: &[u8], i: &mut usize) -> Result<i16, Error> {
    Ok(read_u16(buf, i)? as i16)
}

fn read_u32(buf: &[u8], i: &mut usize) -> Result<u32, Error> {
    let raw = read_raw(buf, i, 4)?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

fn read_i32(buf: &[u8], i: &mut usize) -> Result<i32, Error> {
    Ok(read_u32(buf, i)? as i32)
}

fn read_u64(buf: &[u8], i: &mut usize) -> Result<u64, Error> {
    let raw = read_raw(buf, i, 8)?;
    Ok(u64::from_le_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]))
}

fn read_i64(buf: &[u8], i: &mut usize) -> Result<i64, Error> {
    Ok(read_u64(buf, i)? as i64)
}

#[inline]
pub fn write_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

#[inline]
pub fn write_i8(out: &mut Vec<u8>, v: i8) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn write_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn write_i16(out: &mut Vec<u8>, v: i16) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn write_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn write_i64(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn write_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_bits().to_le_bytes());
}

#[inline]
pub fn write_f64(out: &mut Vec<u8>, v: f64) {
    out.extend_from_slice(&v.to_bits().to_le_bytes());
}

#[inline]
pub fn write_bool(out: &mut Vec<u8>, v: bool) {
    out.push(u8::from(v));
}

#[inline]
pub fn write_count(out: &mut Vec<u8>, n: usize, cap: u64) -> Result<(), Error> {
    if n > u32::MAX as usize || n as u64 > cap {
        return Err(Error::Limit);
    }
    write_u32(out, n as u32);
    Ok(())
}

#[inline]
pub fn write_bytes(out: &mut Vec<u8>, data: &[u8], cap: u64) -> Result<(), Error> {
    write_count(out, data.len(), cap)?;
    out.extend_from_slice(data);
    Ok(())
}

#[inline]
pub fn write_str(out: &mut Vec<u8>, text: &str, cap: u64) -> Result<(), Error> {
    write_bytes(out, text.as_bytes(), cap)
}

fn write_value(out: &mut Vec<u8>, schema: &Schema, value: &Value<'_>, cap: u64) -> Result<(), Error> {
    match schema {
        Schema::Bool => match value {
            Value::Bool(v) => write_bool(out, *v),
            _ => return Err(Error::Type),
        },
        Schema::I8 => match value {
            Value::I8(v) => write_i8(out, *v),
            _ => return Err(Error::Type),
        },
        Schema::I16 => match value {
            Value::I16(v) => write_i16(out, *v),
            _ => return Err(Error::Type),
        },
        Schema::I32 => match value {
            Value::I32(v) => write_i32(out, *v),
            _ => return Err(Error::Type),
        },
        Schema::I64 | Schema::Int => match value {
            Value::I64(v) => write_i64(out, *v),
            _ => return Err(Error::Type),
        },
        Schema::U8 => match value {
            Value::U8(v) => write_u8(out, *v),
            _ => return Err(Error::Type),
        },
        Schema::U16 => match value {
            Value::U16(v) => write_u16(out, *v),
            _ => return Err(Error::Type),
        },
        Schema::U32 => match value {
            Value::U32(v) => write_u32(out, *v),
            _ => return Err(Error::Type),
        },
        Schema::U64 | Schema::Uint => match value {
            Value::U64(v) => write_u64(out, *v),
            _ => return Err(Error::Type),
        },
        Schema::F32 => match value {
            Value::F32(bits) => out.extend_from_slice(&bits.to_le_bytes()),
            _ => return Err(Error::Type),
        },
        Schema::F64 => match value {
            Value::F64(bits) => out.extend_from_slice(&bits.to_le_bytes()),
            _ => return Err(Error::Type),
        },
        Schema::Str => match value {
            Value::Str(text) => write_str(out, text, cap)?,
            _ => return Err(Error::Type),
        },
        Schema::Bytes => match value {
            Value::Bytes(data) => write_bytes(out, data, cap)?,
            _ => return Err(Error::Type),
        },
        Schema::List(elem) => write_list(out, elem, value, cap)?,
        Schema::Opt(inner) => match value {
            Value::Opt(None) => write_u8(out, 0),
            Value::Opt(Some(inner_value)) => {
                write_u8(out, 1);
                write_value(out, inner, inner_value, cap)?;
            }
            _ => return Err(Error::Type),
        },
        Schema::Dict(inner) => write_dict(out, inner, value, cap)?,
        Schema::Struct(fields) => {
            let Value::Struct(values) = value else {
                return Err(Error::Type);
            };
            if values.len() != fields.len() {
                return Err(Error::Type);
            }
            for (value, (_, field)) in values.iter().zip(fields.iter()) {
                write_value(out, field, value, cap)?;
            }
        }
        Schema::Tensors => {
            let Value::Tensors(dir) = value else {
                return Err(Error::Type);
            };
            encode_tensors_into(&dir.tensors, out, cap)?;
        }
    }
    Ok(())
}

fn write_list(out: &mut Vec<u8>, elem: &Schema, value: &Value<'_>, cap: u64) -> Result<(), Error> {
    match elem {
        Schema::U8 => match value {
            Value::Bytes(data) => write_bytes(out, data, cap)?,
            Value::List(items) => {
                write_count(out, items.len(), cap)?;
                for item in items {
                    let Value::U8(byte) = item else {
                        return Err(Error::Type);
                    };
                    write_u8(out, *byte);
                }
            }
            _ => return Err(Error::Type),
        },
        Schema::I8 => match value {
            Value::I8s(data) => write_bytes(out, data, cap)?,
            Value::List(items) => {
                write_count(out, items.len(), cap)?;
                for item in items {
                    let Value::I8(byte) = item else {
                        return Err(Error::Type);
                    };
                    write_i8(out, *byte);
                }
            }
            _ => return Err(Error::Type),
        },
        Schema::F32 => {
            let Value::List(items) = value else {
                return Err(Error::Type);
            };
            write_count(out, items.len(), cap)?;
            for item in items {
                let Value::F32(bits) = item else {
                    return Err(Error::Type);
                };
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
        _ => {
            let Value::List(items) = value else {
                return Err(Error::Type);
            };
            write_count(out, items.len(), cap)?;
            for item in items {
                write_value(out, elem, item, cap)?;
            }
        }
    }
    Ok(())
}

fn write_dict(out: &mut Vec<u8>, elem: &Schema, value: &Value<'_>, cap: u64) -> Result<(), Error> {
    let Value::Dict(pairs) = value else {
        return Err(Error::Type);
    };
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by(|&a, &b| pairs[a].0.as_bytes().cmp(pairs[b].0.as_bytes()));
    for window in order.windows(2) {
        if pairs[window[0]].0.as_bytes() == pairs[window[1]].0.as_bytes() {
            return Err(Error::Type);
        }
    }
    write_count(out, pairs.len(), cap)?;
    for index in order {
        write_str(out, &pairs[index].0, cap)?;
        write_value(out, elem, &pairs[index].1, cap)?;
    }
    Ok(())
}

#[cfg_attr(not(debug_assertions), allow(dead_code))]
fn value_len(schema: &Schema, value: &Value<'_>, cap: u64) -> Result<usize, Error> {
    let n = match schema {
        Schema::Bool | Schema::I8 | Schema::U8 => 1,
        Schema::I16 | Schema::U16 => 2,
        Schema::I32 | Schema::U32 | Schema::F32 => 4,
        Schema::I64 | Schema::Int | Schema::U64 | Schema::Uint | Schema::F64 => 8,
        Schema::Str => match value {
            Value::Str(text) => prefixed(text.len(), cap)?,
            _ => return Err(Error::Type),
        },
        Schema::Bytes => match value {
            Value::Bytes(data) => prefixed(data.len(), cap)?,
            _ => return Err(Error::Type),
        },
        Schema::List(elem) => list_len(elem, value, cap)?,
        Schema::Opt(inner) => match value {
            Value::Opt(None) => 1,
            Value::Opt(Some(inner_value)) => 1 + value_len(inner, inner_value, cap)?,
            _ => return Err(Error::Type),
        },
        Schema::Dict(inner) => dict_len(inner, value, cap)?,
        Schema::Struct(fields) => {
            let Value::Struct(values) = value else {
                return Err(Error::Type);
            };
            if values.len() != fields.len() {
                return Err(Error::Type);
            }
            let mut n = 0usize;
            for (value, (_, field)) in values.iter().zip(fields.iter()) {
                n += value_len(field, value, cap)?;
            }
            n
        }
        Schema::Tensors => match value {
            Value::Tensors(dir) => tensor_block_len(&dir.tensors, cap)?,
            _ => return Err(Error::Type),
        },
    };
    Ok(n)
}

#[cfg_attr(not(debug_assertions), allow(dead_code))]
fn prefixed(len: usize, cap: u64) -> Result<usize, Error> {
    if len > u32::MAX as usize || len as u64 > cap {
        return Err(Error::Limit);
    }
    Ok(4 + len)
}

#[cfg_attr(not(debug_assertions), allow(dead_code))]
fn list_len(elem: &Schema, value: &Value<'_>, cap: u64) -> Result<usize, Error> {
    match elem {
        Schema::U8 => match value {
            Value::Bytes(data) => prefixed(data.len(), cap),
            Value::List(items) => {
                if items.len() as u64 > cap {
                    return Err(Error::Limit);
                }
                Ok(4 + items.len())
            }
            _ => Err(Error::Type),
        },
        Schema::I8 => match value {
            Value::I8s(data) => prefixed(data.len(), cap),
            Value::List(items) => {
                if items.len() as u64 > cap {
                    return Err(Error::Limit);
                }
                Ok(4 + items.len())
            }
            _ => Err(Error::Type),
        },
        _ => {
            let Value::List(items) = value else {
                return Err(Error::Type);
            };
            if items.len() as u64 > cap || items.len() > u32::MAX as usize {
                return Err(Error::Limit);
            }
            let mut n = 4usize;
            for item in items {
                n += value_len(elem, item, cap)?;
            }
            Ok(n)
        }
    }
}

#[cfg_attr(not(debug_assertions), allow(dead_code))]
fn dict_len(elem: &Schema, value: &Value<'_>, cap: u64) -> Result<usize, Error> {
    let Value::Dict(pairs) = value else {
        return Err(Error::Type);
    };
    if pairs.len() as u64 > cap || pairs.len() > u32::MAX as usize {
        return Err(Error::Limit);
    }
    let mut n = 4usize;
    for (key, value) in pairs {
        n += prefixed(key.len(), cap)?;
        n += value_len(elem, value, cap)?;
    }
    Ok(n)
}

/// Encode one value, appending to `out`.
pub fn encode_into(
    schema: &Schema,
    value: &Value<'_>,
    out: &mut Vec<u8>,
    cap: u64,
) -> Result<(), Error> {
    // Debug builds check the size walk against the bytes actually written.
    // Release builds write once so a struct batch does not walk the tree twice.
    #[cfg(debug_assertions)]
    let expect = value_len(schema, value, cap)?;
    #[cfg(debug_assertions)]
    out.reserve(expect);
    #[cfg(not(debug_assertions))]
    out.reserve(32);
    #[cfg(debug_assertions)]
    let start = out.len();
    write_value(out, schema, value, cap)?;
    #[cfg(debug_assertions)]
    if out.len() - start != expect {
        return Err(Error::Type);
    }
    Ok(())
}

pub fn encode(schema: &Schema, value: &Value<'_>) -> Result<Vec<u8>, Error> {
    encode_with(schema, value, DEFAULT_CAP)
}

pub fn encode_with(schema: &Schema, value: &Value<'_>, cap: u64) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    encode_into(schema, value, &mut out, cap)?;
    Ok(out)
}

pub fn encode_framed(schema: &Schema, value: &Value<'_>) -> Result<Vec<u8>, Error> {
    encode_framed_with(schema, value, DEFAULT_CAP)
}

pub fn encode_framed_with(schema: &Schema, value: &Value<'_>, cap: u64) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    out.extend_from_slice(b"KLN0");
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&type_hash(schema).to_le_bytes());
    encode_into(schema, value, &mut out, cap)?;
    Ok(out)
}

pub fn decode<'a>(schema: &Schema, buf: &'a [u8]) -> Result<Value<'a>, Error> {
    decode_with(schema, buf, DecodeOpts::default())
}

pub fn decode_with<'a>(
    schema: &Schema,
    buf: &'a [u8],
    opts: DecodeOpts,
) -> Result<Value<'a>, Error> {
    if matches!(schema, Schema::Tensors) {
        let dir = decode_tensors(buf, opts)?;
        if opts.standalone && dir.block_len != buf.len() {
            return Err(Error::Trailing);
        }
        return Ok(Value::Tensors(dir));
    }
    let mut reader = Reader::with_opts(buf, opts);
    let value = reader.read_value(schema)?;
    if opts.standalone {
        reader.finish()?;
    }
    Ok(value)
}

pub fn decode_framed<'a>(schema: &Schema, buf: &'a [u8]) -> Result<Value<'a>, Error> {
    decode_framed_with(schema, buf, DecodeOpts::default())
}

pub fn decode_framed_with<'a>(
    schema: &Schema,
    buf: &'a [u8],
    opts: DecodeOpts,
) -> Result<Value<'a>, Error> {
    if buf.len() < 4 {
        return Err(Error::Eof);
    }
    if &buf[0..4] != b"KLN0" {
        return Err(Error::Frame);
    }
    if buf.len() < 16 {
        return Err(Error::Eof);
    }
    if buf[4..8] != [0, 0, 0, 0] {
        return Err(Error::Frame);
    }
    let mut hash_bytes = [0u8; 8];
    hash_bytes.copy_from_slice(&buf[8..16]);
    let hash = u64::from_le_bytes(hash_bytes);
    if hash != type_hash(schema) {
        return Err(Error::Schema);
    }
    let nested = DecodeOpts {
        standalone: false,
        ..opts
    };
    let mut reader = Reader::with_opts(&buf[16..], nested);
    let value = reader.read_value(schema)?;
    if opts.standalone && reader.position() != buf.len() - 16 {
        return Err(Error::Trailing);
    }
    Ok(value)
}
