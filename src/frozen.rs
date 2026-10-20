//! Frozen positional values. Scalars are little-endian.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};

use crate::error::{DecodeOpts, Error, DEFAULT_CAP};
use crate::schema::{min_size, type_hash, Schema};
use crate::tensor::{decode_tensors, encode_tensors_into, tensor_block_len};
use crate::value::Value;

const MAX_DEPTH: u32 = 64;

fn is_container(schema: &Schema) -> bool {
    matches!(
        schema,
        Schema::List(_) | Schema::Opt(_) | Schema::Dict(_) | Schema::Struct(_) | Schema::Tensors
    )
}

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

    #[inline(always)]
    fn need(&self, n: usize) -> Result<usize, Error> {
        let at = self.i;
        if self.buf.len().saturating_sub(at) < n {
            return Err(Error::Eof);
        }
        Ok(at)
    }

    #[inline]
    pub fn u8(&mut self) -> Result<u8, Error> {
        let at = self.need(1)?;
        self.i = at + 1;
        Ok(self.buf[at])
    }

    /// Bounds-checked byte read. The cap does not apply.
    #[inline]
    pub fn u8_fast(&mut self) -> Result<u8, Error> {
        self.u8()
    }

    pub fn i8(&mut self) -> Result<i8, Error> {
        Ok(self.u8()? as i8)
    }

    pub fn u16(&mut self) -> Result<u16, Error> {
        let at = self.need(2)?;
        self.i = at + 2;
        Ok(load_u16(self.buf, at))
    }

    pub fn i16(&mut self) -> Result<i16, Error> {
        Ok(self.u16()? as i16)
    }

    #[inline]
    pub fn u32(&mut self) -> Result<u32, Error> {
        let at = self.need(4)?;
        self.i = at + 4;
        Ok(load_u32(self.buf, at))
    }

    /// Bounds-checked `u32` read. The cap does not apply.
    #[inline]
    pub fn u32_fast(&mut self) -> Result<u32, Error> {
        self.u32()
    }

    #[inline]
    pub fn i32(&mut self) -> Result<i32, Error> {
        Ok(self.u32()? as i32)
    }

    pub fn u64(&mut self) -> Result<u64, Error> {
        let at = self.need(8)?;
        self.i = at + 8;
        Ok(load_u64(self.buf, at))
    }

    /// Bounds-checked `u64` read. The cap does not apply.
    #[inline]
    pub fn u64_fast(&mut self) -> Result<u64, Error> {
        self.u64()
    }

    pub fn i64(&mut self) -> Result<i64, Error> {
        Ok(self.u64()? as i64)
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
        self.read_str(Some(self.cap))
    }

    /// String read that ignores the count cap.
    /// A length past the buffer is `EOF`. Invalid UTF-8 is `UTF8`.
    #[inline]
    pub fn str_fast(&mut self) -> Result<&'a str, Error> {
        self.read_str(None)
    }

    #[inline(always)]
    fn read_str(&mut self, cap: Option<u64>) -> Result<&'a str, Error> {
        let at = self.need(4)?;
        let len = load_u32(self.buf, at);
        if let Some(cap) = cap {
            if u64::from(len) > cap {
                return Err(Error::Limit);
            }
        }
        let start = at + 4;
        let end = start.checked_add(len as usize).ok_or(Error::Limit)?;
        if end > self.buf.len() {
            return Err(Error::Eof);
        }
        self.i = end;
        bytes_as_str(&self.buf[start..end], self.trusted)
    }

    pub fn string_owned(&mut self) -> Result<String, Error> {
        self.string().map(str::to_owned)
    }

    pub(crate) fn read_value(&mut self, schema: &Schema) -> Result<Value<'a>, Error> {
        let buf = self.buf;
        let cap = self.cap;
        let trusted = self.trusted;
        read_value_at(buf, &mut self.i, schema, cap, trusted, 0)
    }
}

fn read_value_at<'a>(
    buf: &'a [u8],
    i: &mut usize,
    schema: &Schema,
    cap: u64,
    trusted: bool,
    depth: u32,
) -> Result<Value<'a>, Error> {
    if depth >= MAX_DEPTH && is_container(schema) {
        return Err(Error::Limit);
    }
    let child = depth.saturating_add(1);
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
        Schema::Str => Ok(Value::Str(Cow::Borrowed(read_string(
            buf, i, cap, trusted,
        )?))),
        Schema::Bytes => Ok(Value::Bytes(Cow::Borrowed(read_bytes(buf, i, cap)?))),
        Schema::List(elem) => read_list(buf, i, elem, cap, trusted, child),
        Schema::Opt(inner) => {
            if read_tag(buf, i)? {
                Ok(Value::Opt(Some(Box::new(read_value_at(
                    buf, i, inner, cap, trusted, child,
                )?))))
            } else {
                Ok(Value::Opt(None))
            }
        }
        Schema::Dict(inner) => read_dict(buf, i, inner, cap, trusted, child),
        Schema::Struct(fields) => {
            if let Some((kinds, nfields)) = scalar_kinds(fields) {
                let mut values = take_vec(nfields);
                read_struct_scalars(buf, i, &kinds[..nfields], cap, trusted, &mut values)?;
                return Ok(Value::Struct(values));
            }
            let mut values = take_vec(fields.len());
            for (_, field) in fields {
                values.push(read_value_at(buf, i, field, cap, trusted, child)?);
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
    depth: u32,
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
        Schema::F32 => Ok(Value::F32s(read_pod_list(buf, i, 4, cap)?)),
        Schema::F64 => Ok(Value::F64s(read_pod_list(buf, i, 8, cap)?)),
        Schema::I64 | Schema::Int => Ok(Value::I64s(read_pod_list(buf, i, 8, cap)?)),
        Schema::Struct(fields) => {
            let n = list_count_at(buf, i, min_size(elem), cap)?;
            if n > 0 && depth >= MAX_DEPTH {
                return Err(Error::Limit);
            }
            let mut items = take_vec(n as usize);
            if let Some((kinds, nfields)) = scalar_kinds(fields) {
                let mut bufs = take_small(n as usize, nfields.max(1));
                if nfields == 3
                    && matches!(
                        (kinds[0], kinds[1], kinds[2]),
                        (Kind::I32, Kind::I32, Kind::Str)
                    )
                {
                    read_i32_i32_str_list(buf, i, n, cap, trusted, &mut items, &mut bufs)?;
                } else {
                    for _ in 0..n {
                        let mut values = bufs.pop().unwrap();
                        values.clear();
                        read_struct_scalars(buf, i, &kinds[..nfields], cap, trusted, &mut values)?;
                        items.push(Value::Struct(values));
                    }
                }
                return Ok(Value::List(items));
            }
            for _ in 0..n {
                items.push(read_value_at(buf, i, elem, cap, trusted, depth)?);
            }
            Ok(Value::List(items))
        }
        _ => {
            let n = list_count_at(buf, i, min_size(elem), cap)?;
            if n > 0 && depth >= MAX_DEPTH && is_container(elem) {
                return Err(Error::Limit);
            }
            let mut items = take_vec(n as usize);
            for _ in 0..n {
                items.push(read_value_at(buf, i, elem, cap, trusted, depth)?);
            }
            Ok(Value::List(items))
        }
    }
}

fn read_i32_i32_str_list<'a>(
    buf: &'a [u8],
    i: &mut usize,
    n: u32,
    cap: u64,
    trusted: bool,
    items: &mut Vec<Value<'a>>,
    bufs: &mut Vec<Vec<Value<'a>>>,
) -> Result<(), Error> {
    let mut p = *i;
    let end = buf.len();
    for _ in 0..n {
        if end.saturating_sub(p) < 12 {
            *i = p;
            return Err(Error::Eof);
        }
        let x = load_u32(buf, p) as i32;
        let y = load_u32(buf, p + 4) as i32;
        let len = load_u32(buf, p + 8);
        if u64::from(len) > cap {
            *i = p;
            return Err(Error::Limit);
        }
        let start = p + 12;
        let Some(stop) = start.checked_add(len as usize) else {
            *i = p;
            return Err(Error::Eof);
        };
        if stop > end {
            *i = p;
            return Err(Error::Eof);
        }
        let text = bytes_as_str(&buf[start..stop], trusted)?;
        p = stop;
        let mut values = bufs.pop().unwrap();
        if !values.is_empty() {
            values.clear();
        }
        if values.capacity() < 3 {
            values.reserve(3);
        }
        // SAFETY: `values` is empty and its capacity is at least 3. The three
        // writes initialize those slots before `set_len` exposes them.
        unsafe {
            let dst = values.as_mut_ptr();
            std::ptr::write(dst, Value::I32(x));
            std::ptr::write(dst.add(1), Value::I32(y));
            std::ptr::write(dst.add(2), Value::Str(Cow::Borrowed(text)));
            values.set_len(3);
        }
        items.push(Value::Struct(values));
    }
    *i = p;
    Ok(())
}

fn read_struct_scalars<'a>(
    buf: &'a [u8],
    i: &mut usize,
    kinds: &[Kind],
    cap: u64,
    trusted: bool,
    values: &mut Vec<Value<'a>>,
) -> Result<(), Error> {
    match kinds {
        [k0, k1, k2] => {
            values.push(read_kind(buf, i, *k0, cap, trusted)?);
            values.push(read_kind(buf, i, *k1, cap, trusted)?);
            values.push(read_kind(buf, i, *k2, cap, trusted)?);
        }
        _ => {
            for kind in kinds {
                values.push(read_kind(buf, i, *kind, cap, trusted)?);
            }
        }
    }
    Ok(())
}

fn read_dict<'a>(
    buf: &'a [u8],
    i: &mut usize,
    elem: &Schema,
    cap: u64,
    trusted: bool,
    depth: u32,
) -> Result<Value<'a>, Error> {
    let n = list_count_at(
        buf,
        i,
        4u64.checked_add(min_size(elem)).ok_or(Error::Limit)?,
        cap,
    )?;
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
        let value = read_value_at(buf, i, elem, cap, trusted, depth)?;
        let key = &buf[at..at + len];
        let key = bytes_as_str(key, trusted)?;
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

/// Count plus a borrowed body. `width` is the element size in bytes.
/// The count is rejected before any element storage is allocated.
fn read_pod_list<'a>(
    buf: &'a [u8],
    i: &mut usize,
    width: u64,
    cap: u64,
) -> Result<Cow<'a, [u8]>, Error> {
    let n = list_count_at(buf, i, width, cap)?;
    let nbytes = u64::from(n).checked_mul(width).ok_or(Error::Limit)?;
    Ok(Cow::Borrowed(read_raw(buf, i, nbytes as usize)?))
}

fn write_aligned_bytes(
    out: &mut Vec<u8>,
    data: &[u8],
    width: usize,
    cap: u64,
) -> Result<(), Error> {
    if width == 0 || data.len() % width != 0 {
        return Err(Error::Type);
    }
    write_count(out, data.len() / width, cap)?;
    push_bytes(out, data);
    Ok(())
}

fn pod_list_len(data: &[u8], width: usize, cap: u64) -> Result<usize, Error> {
    if width == 0 || data.len() % width != 0 {
        return Err(Error::Type);
    }
    counted_width(data.len() / width, width, cap)
}

fn counted_width(count: usize, width: usize, cap: u64) -> Result<usize, Error> {
    if count > u32::MAX as usize || count as u64 > cap {
        return Err(Error::Limit);
    }
    let body = count.checked_mul(width).ok_or(Error::Limit)?;
    4usize.checked_add(body).ok_or(Error::Limit)
}

fn read_string<'a>(
    buf: &'a [u8],
    i: &mut usize,
    cap: u64,
    trusted: bool,
) -> Result<&'a str, Error> {
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
    bytes_as_str(bytes, trusted)
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
    let at = *i;
    if buf.len().saturating_sub(at) < 2 {
        return Err(Error::Eof);
    }
    *i = at + 2;
    Ok(load_u16(buf, at))
}

fn read_i16(buf: &[u8], i: &mut usize) -> Result<i16, Error> {
    Ok(read_u16(buf, i)? as i16)
}

fn read_u32(buf: &[u8], i: &mut usize) -> Result<u32, Error> {
    let at = *i;
    if buf.len().saturating_sub(at) < 4 {
        return Err(Error::Eof);
    }
    *i = at + 4;
    Ok(load_u32(buf, at))
}

fn read_i32(buf: &[u8], i: &mut usize) -> Result<i32, Error> {
    Ok(read_u32(buf, i)? as i32)
}

fn read_u64(buf: &[u8], i: &mut usize) -> Result<u64, Error> {
    let at = *i;
    if buf.len().saturating_sub(at) < 8 {
        return Err(Error::Eof);
    }
    *i = at + 8;
    Ok(load_u64(buf, at))
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
    write_u8(out, v as u8);
}

#[inline]
pub fn write_u16(out: &mut Vec<u8>, v: u16) {
    push_bytes(out, &v.to_le_bytes());
}

#[inline]
pub fn write_i16(out: &mut Vec<u8>, v: i16) {
    push_bytes(out, &v.to_le_bytes());
}

#[inline]
pub fn write_u32(out: &mut Vec<u8>, v: u32) {
    push_bytes(out, &v.to_le_bytes());
}

#[inline]
pub fn write_i32(out: &mut Vec<u8>, v: i32) {
    push_bytes(out, &v.to_le_bytes());
}

#[inline]
pub fn write_u64(out: &mut Vec<u8>, v: u64) {
    push_bytes(out, &v.to_le_bytes());
}

#[inline]
pub fn write_i64(out: &mut Vec<u8>, v: i64) {
    push_bytes(out, &v.to_le_bytes());
}

#[inline]
pub fn write_f32(out: &mut Vec<u8>, v: f32) {
    push_bytes(out, &v.to_bits().to_le_bytes());
}

#[inline]
pub fn write_f64(out: &mut Vec<u8>, v: f64) {
    push_bytes(out, &v.to_bits().to_le_bytes());
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
    push_bytes(out, data);
    Ok(())
}

#[inline]
pub fn write_str(out: &mut Vec<u8>, text: &str, cap: u64) -> Result<(), Error> {
    write_bytes(out, text.as_bytes(), cap)
}

fn write_value(
    out: &mut Vec<u8>,
    schema: &Schema,
    value: &Value<'_>,
    cap: u64,
    depth: u32,
) -> Result<(), Error> {
    if depth >= MAX_DEPTH && is_container(schema) {
        return Err(Error::Limit);
    }
    let child = depth.saturating_add(1);
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
            Value::F32(bits) => push_bytes(out, &bits.to_le_bytes()),
            _ => return Err(Error::Type),
        },
        Schema::F64 => match value {
            Value::F64(bits) => push_bytes(out, &bits.to_le_bytes()),
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
        Schema::List(elem) => write_list(out, elem, value, cap, child)?,
        Schema::Opt(inner) => match value {
            Value::Opt(None) => write_u8(out, 0),
            Value::Opt(Some(inner_value)) => {
                write_u8(out, 1);
                write_value(out, inner, inner_value, cap, child)?;
            }
            _ => return Err(Error::Type),
        },
        Schema::Dict(inner) => write_dict(out, inner, value, cap, child)?,
        Schema::Struct(fields) => {
            let Value::Struct(values) = value else {
                return Err(Error::Type);
            };
            if values.len() != fields.len() {
                return Err(Error::Type);
            }
            if let Some((kinds, nfields)) = scalar_kinds(fields) {
                write_struct_scalars(out, &kinds[..nfields], values, cap)?;
            } else {
                for (value, (_, field)) in values.iter().zip(fields.iter()) {
                    write_value(out, field, value, cap, child)?;
                }
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

fn write_list(
    out: &mut Vec<u8>,
    elem: &Schema,
    value: &Value<'_>,
    cap: u64,
    depth: u32,
) -> Result<(), Error> {
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
        Schema::F32 => match value {
            Value::F32s(data) => write_aligned_bytes(out, data, 4, cap)?,
            Value::List(items) => {
                write_count(out, items.len(), cap)?;
                out.reserve(items.len().saturating_mul(4));
                for item in items {
                    let Value::F32(bits) = item else {
                        return Err(Error::Type);
                    };
                    push_bytes(out, &bits.to_le_bytes());
                }
            }
            _ => return Err(Error::Type),
        },
        Schema::F64 => match value {
            Value::F64s(data) => write_aligned_bytes(out, data, 8, cap)?,
            Value::List(items) => {
                write_count(out, items.len(), cap)?;
                out.reserve(items.len().saturating_mul(8));
                for item in items {
                    let Value::F64(bits) = item else {
                        return Err(Error::Type);
                    };
                    push_bytes(out, &bits.to_le_bytes());
                }
            }
            _ => return Err(Error::Type),
        },
        Schema::I64 | Schema::Int => match value {
            Value::I64s(data) => write_aligned_bytes(out, data, 8, cap)?,
            Value::List(items) => {
                write_count(out, items.len(), cap)?;
                out.reserve(items.len().saturating_mul(8));
                for item in items {
                    let Value::I64(bits) = item else {
                        return Err(Error::Type);
                    };
                    push_bytes(out, &bits.to_le_bytes());
                }
            }
            _ => return Err(Error::Type),
        },
        Schema::Struct(fields) => {
            let Value::List(items) = value else {
                return Err(Error::Type);
            };
            if !items.is_empty() && depth >= MAX_DEPTH {
                return Err(Error::Limit);
            }
            write_count(out, items.len(), cap)?;
            if let Some((kinds, nfields)) = scalar_kinds(fields) {
                let each = min_size(elem) as usize + 8;
                out.reserve(items.len().saturating_mul(each));
                if nfields == 3
                    && matches!(
                        (kinds[0], kinds[1], kinds[2]),
                        (Kind::I32, Kind::I32, Kind::Str)
                    )
                {
                    write_i32_i32_str_list(out, items, cap)?;
                } else {
                    let kinds = &kinds[..nfields];
                    for item in items {
                        let Value::Struct(values) = item else {
                            return Err(Error::Type);
                        };
                        if values.len() != nfields {
                            return Err(Error::Type);
                        }
                        write_struct_scalars(out, kinds, values, cap)?;
                    }
                }
            } else {
                for item in items {
                    write_value(out, elem, item, cap, depth)?;
                }
            }
        }
        _ => {
            let Value::List(items) = value else {
                return Err(Error::Type);
            };
            if !items.is_empty() && depth >= MAX_DEPTH && is_container(elem) {
                return Err(Error::Limit);
            }
            write_count(out, items.len(), cap)?;
            let slack: usize = if matches!(elem, Schema::Str | Schema::Bytes) {
                32
            } else if matches!(elem, Schema::Struct(_)) {
                8
            } else {
                0
            };
            out.reserve(
                items
                    .len()
                    .saturating_mul((min_size(elem) as usize).saturating_add(slack)),
            );
            for item in items {
                write_value(out, elem, item, cap, depth)?;
            }
        }
    }
    Ok(())
}

fn write_dict(
    out: &mut Vec<u8>,
    elem: &Schema,
    value: &Value<'_>,
    cap: u64,
    depth: u32,
) -> Result<(), Error> {
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
        write_value(out, elem, &pairs[index].1, cap, depth)?;
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
        Schema::F32 => match value {
            Value::F32s(data) => pod_list_len(data, 4, cap),
            Value::List(items) => counted_width(items.len(), 4, cap),
            _ => Err(Error::Type),
        },
        Schema::F64 => match value {
            Value::F64s(data) => pod_list_len(data, 8, cap),
            Value::List(items) => counted_width(items.len(), 8, cap),
            _ => Err(Error::Type),
        },
        Schema::I64 | Schema::Int => match value {
            Value::I64s(data) => pod_list_len(data, 8, cap),
            Value::List(items) => counted_width(items.len(), 8, cap),
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
    write_value(out, schema, value, cap, 0)?;
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

#[derive(Clone, Copy)]
enum Kind {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Str,
    Bytes,
}

fn kind_of(schema: &Schema) -> Option<Kind> {
    Some(match schema {
        Schema::Bool => Kind::Bool,
        Schema::I8 => Kind::I8,
        Schema::I16 => Kind::I16,
        Schema::I32 => Kind::I32,
        Schema::I64 | Schema::Int => Kind::I64,
        Schema::U8 => Kind::U8,
        Schema::U16 => Kind::U16,
        Schema::U32 => Kind::U32,
        Schema::U64 | Schema::Uint => Kind::U64,
        Schema::F32 => Kind::F32,
        Schema::F64 => Kind::F64,
        Schema::Str => Kind::Str,
        Schema::Bytes => Kind::Bytes,
        _ => return None,
    })
}

fn scalar_kinds(fields: &[(String, Schema)]) -> Option<([Kind; 32], usize)> {
    if fields.len() > 32 {
        return None;
    }
    let mut kinds = [Kind::Bool; 32];
    for (index, (_, field)) in fields.iter().enumerate() {
        kinds[index] = kind_of(field)?;
    }
    Some((kinds, fields.len()))
}

#[inline(always)]
fn read_kind<'a>(
    buf: &'a [u8],
    i: &mut usize,
    kind: Kind,
    cap: u64,
    trusted: bool,
) -> Result<Value<'a>, Error> {
    Ok(match kind {
        Kind::Bool => Value::Bool(read_bool(buf, i)?),
        Kind::I8 => Value::I8(read_i8(buf, i)?),
        Kind::I16 => Value::I16(read_i16(buf, i)?),
        Kind::I32 => Value::I32(read_i32(buf, i)?),
        Kind::I64 => Value::I64(read_i64(buf, i)?),
        Kind::U8 => Value::U8(read_u8(buf, i)?),
        Kind::U16 => Value::U16(read_u16(buf, i)?),
        Kind::U32 => Value::U32(read_u32(buf, i)?),
        Kind::U64 => Value::U64(read_u64(buf, i)?),
        Kind::F32 => Value::F32(read_u32(buf, i)?),
        Kind::F64 => Value::F64(read_u64(buf, i)?),
        Kind::Str => Value::Str(Cow::Borrowed(read_string(buf, i, cap, trusted)?)),
        Kind::Bytes => Value::Bytes(Cow::Borrowed(read_bytes(buf, i, cap)?)),
    })
}

fn write_i32_i32_str_list(out: &mut Vec<u8>, items: &[Value<'_>], cap: u64) -> Result<(), Error> {
    for item in items {
        let Value::Struct(values) = item else {
            return Err(Error::Type);
        };
        let (x, y, text) = match values.as_slice() {
            [Value::I32(x), Value::I32(y), Value::Str(text)] => (*x, *y, text.as_ref()),
            _ => return Err(Error::Type),
        };
        if text.len() as u64 > cap || text.len() > u32::MAX as usize {
            return Err(Error::Limit);
        }
        let len = out.len();
        let need = 12 + text.len();
        if out.capacity() - len < need {
            out.reserve(need);
        }
        // SAFETY: the reserve above leaves `need` bytes of capacity. The stores
        // initialize that range before `set_len`.
        unsafe {
            let dst = out.as_mut_ptr().add(len);
            std::ptr::copy_nonoverlapping(x.to_le_bytes().as_ptr(), dst, 4);
            std::ptr::copy_nonoverlapping(y.to_le_bytes().as_ptr(), dst.add(4), 4);
            std::ptr::copy_nonoverlapping(
                (text.len() as u32).to_le_bytes().as_ptr(),
                dst.add(8),
                4,
            );
            std::ptr::copy_nonoverlapping(text.as_ptr(), dst.add(12), text.len());
            out.set_len(len + need);
        }
    }
    Ok(())
}

fn write_struct_scalars(
    out: &mut Vec<u8>,
    kinds: &[Kind],
    values: &[Value<'_>],
    cap: u64,
) -> Result<(), Error> {
    if values.len() != kinds.len() {
        return Err(Error::Type);
    }
    match kinds {
        [k0, k1, k2] => {
            write_kind(out, *k0, &values[0], cap)?;
            write_kind(out, *k1, &values[1], cap)?;
            write_kind(out, *k2, &values[2], cap)?;
        }
        _ => {
            for (kind, value) in kinds.iter().zip(values.iter()) {
                write_kind(out, *kind, value, cap)?;
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn write_kind(out: &mut Vec<u8>, kind: Kind, value: &Value<'_>, cap: u64) -> Result<(), Error> {
    match (kind, value) {
        (Kind::Bool, Value::Bool(v)) => write_bool(out, *v),
        (Kind::I8, Value::I8(v)) => write_i8(out, *v),
        (Kind::I16, Value::I16(v)) => write_i16(out, *v),
        (Kind::I32, Value::I32(v)) => write_i32(out, *v),
        (Kind::I64, Value::I64(v)) => write_i64(out, *v),
        (Kind::U8, Value::U8(v)) => write_u8(out, *v),
        (Kind::U16, Value::U16(v)) => write_u16(out, *v),
        (Kind::U32, Value::U32(v)) => write_u32(out, *v),
        (Kind::U64, Value::U64(v)) => write_u64(out, *v),
        (Kind::F32, Value::F32(bits)) => push_bytes(out, &bits.to_le_bytes()),
        (Kind::F64, Value::F64(bits)) => push_bytes(out, &bits.to_le_bytes()),
        (Kind::Str, Value::Str(text)) => write_str(out, text, cap)?,
        (Kind::Bytes, Value::Bytes(data)) => write_bytes(out, data, cap)?,
        _ => return Err(Error::Type),
    }
    Ok(())
}

#[inline(always)]
fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    let len = out.len();
    if out.capacity() - len >= bytes.len() {
        // SAFETY: `len + bytes.len()` fits in capacity. Those bytes are
        // initialized before `set_len` makes them visible.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().add(len), bytes.len());
            out.set_len(len + bytes.len());
        }
    } else {
        out.extend_from_slice(bytes);
    }
}

#[inline(always)]
fn load_u16(buf: &[u8], at: usize) -> u16 {
    let mut raw = [0u8; 2];
    // SAFETY: callers check `at + 2 <= buf.len()` before this load.
    unsafe {
        std::ptr::copy_nonoverlapping(buf.as_ptr().add(at), raw.as_mut_ptr(), 2);
    }
    u16::from_le_bytes(raw)
}

#[inline(always)]
fn load_u32(buf: &[u8], at: usize) -> u32 {
    let mut raw = [0u8; 4];
    // SAFETY: callers check `at + 4 <= buf.len()` before this load.
    unsafe {
        std::ptr::copy_nonoverlapping(buf.as_ptr().add(at), raw.as_mut_ptr(), 4);
    }
    u32::from_le_bytes(raw)
}

#[inline(always)]
fn load_u64(buf: &[u8], at: usize) -> u64 {
    let mut raw = [0u8; 8];
    // SAFETY: callers check `at + 8 <= buf.len()` before this load.
    unsafe {
        std::ptr::copy_nonoverlapping(buf.as_ptr().add(at), raw.as_mut_ptr(), 8);
    }
    u64::from_le_bytes(raw)
}

#[inline(always)]
fn bytes_as_str(bytes: &[u8], trusted: bool) -> Result<&str, Error> {
    if trusted || bytes.iter().all(|byte| *byte < 0x80) {
        // SAFETY: trusted mode promises UTF-8. Bytes below 0x80 are ASCII.
        Ok(unsafe { std::str::from_utf8_unchecked(bytes) })
    } else {
        std::str::from_utf8(bytes).map_err(|_| Error::Utf8)
    }
}

struct RawVec {
    ptr: *mut u8,
    cap: usize,
}

struct ValuePool {
    small: Vec<RawVec>,
    large: Vec<RawVec>,
}

impl Drop for ValuePool {
    fn drop(&mut self) {
        for raw in self.small.drain(..).chain(self.large.drain(..)) {
            free_raw(raw);
        }
    }
}

thread_local! {
    static POOL: RefCell<ValuePool> = const {
        RefCell::new(ValuePool {
            small: Vec::new(),
            large: Vec::new(),
        })
    };
    static SKIP_DROP: Cell<u32> = const { Cell::new(0) };
}

struct SkipDrop;

impl SkipDrop {
    fn new() -> Self {
        SKIP_DROP.with(|flag| flag.set(flag.get().saturating_add(1)));
        SkipDrop
    }
}

impl Drop for SkipDrop {
    fn drop(&mut self) {
        SKIP_DROP.with(|flag| flag.set(flag.get().saturating_sub(1)));
    }
}

fn from_raw<'a>(raw: RawVec) -> Vec<Value<'a>> {
    // SAFETY: `ptr` and `cap` came from an empty `Vec<Value>` forgotten by
    // `into_raw`. No element is live, so the lifetime can be reborrowed.
    // The block still belongs to the global allocator.
    unsafe { Vec::from_raw_parts(raw.ptr.cast::<Value<'a>>(), 0, raw.cap) }
}

fn free_raw(raw: RawVec) {
    if raw.cap > 0 {
        drop(from_raw::<'_>(raw));
    }
}

fn into_raw(mut values: Vec<Value<'_>>) -> Option<RawVec> {
    values.clear();
    let cap = values.capacity();
    if cap == 0 || cap > 65_536 {
        return None;
    }
    let ptr = values.as_mut_ptr().cast::<u8>();
    std::mem::forget(values);
    Some(RawVec { ptr, cap })
}

fn take_small<'a>(n: usize, min_cap: usize) -> Vec<Vec<Value<'a>>> {
    let mut got = Vec::with_capacity(n);
    if n == 0 {
        return got;
    }
    let min_cap = min_cap.max(1);
    let _ = POOL.try_with(|pool| {
        let mut pool = pool.borrow_mut();
        while got.len() < n {
            let Some(raw) = pool.small.pop() else {
                break;
            };
            if raw.cap >= min_cap {
                got.push(from_raw(raw));
            } else {
                free_raw(raw);
            }
        }
    });
    let cap = min_cap.clamp(8, 32);
    while got.len() < n {
        got.push(Vec::with_capacity(cap));
    }
    got
}

fn take_vec<'a>(min_cap: usize) -> Vec<Value<'a>> {
    if min_cap == 0 {
        return Vec::new();
    }
    if min_cap <= 32 {
        return take_small(1, min_cap).pop().unwrap();
    }
    let found = POOL
        .try_with(|pool| {
            let mut pool = pool.borrow_mut();
            let pos = pool.large.iter().rposition(|raw| raw.cap >= min_cap)?;
            Some(pool.large.swap_remove(pos))
        })
        .ok()
        .flatten();
    match found {
        Some(raw) => from_raw(raw),
        None => Vec::with_capacity(min_cap),
    }
}

fn store_all(mut small: Vec<RawVec>, mut large: Option<RawVec>) {
    let ok = POOL.try_with(|pool| {
        let mut pool = pool.borrow_mut();
        if let Some(raw) = large.take() {
            if (33..=65_536).contains(&raw.cap) && pool.large.len() < 8 {
                pool.large.push(raw);
            } else if (1..=32).contains(&raw.cap) {
                small.push(raw);
            } else {
                free_raw(raw);
            }
        }
        for raw in small.drain(..) {
            if (1..=32).contains(&raw.cap) && pool.small.len() < 8192 {
                pool.small.push(raw);
            } else {
                free_raw(raw);
            }
        }
    });
    if ok.is_err() {
        if let Some(raw) = large {
            free_raw(raw);
        }
        for raw in small {
            free_raw(raw);
        }
    }
}

fn recycle_into(mut fields: Vec<Value<'_>>, small: &mut Vec<RawVec>) {
    for item in &mut fields {
        match item {
            Value::Struct(inner) => {
                let inner = std::mem::take(inner);
                recycle_into(inner, small);
            }
            Value::List(inner) => {
                let inner = std::mem::take(inner);
                recycle_list(inner);
            }
            _ => {}
        }
    }
    if let Some(raw) = into_raw(fields) {
        if raw.cap <= 32 {
            small.push(raw);
        } else {
            store_all(Vec::new(), Some(raw));
        }
    }
}

fn is_plain(value: &Value<'_>) -> bool {
    !matches!(
        value,
        Value::Str(Cow::Owned(_))
            | Value::Bytes(Cow::Owned(_))
            | Value::I8s(Cow::Owned(_))
            | Value::F32s(Cow::Owned(_))
            | Value::F64s(Cow::Owned(_))
            | Value::I64s(Cow::Owned(_))
            | Value::List(_)
            | Value::Struct(_)
            | Value::Dict(_)
            | Value::Opt(Some(_))
            | Value::Tensors(_)
    )
}

fn forget_plain(mut values: Vec<Value<'_>>) -> Option<RawVec> {
    // SAFETY: every element is plain (no owned heap), so skipping Drop leaks nothing.
    unsafe { values.set_len(0) }
    let cap = values.capacity();
    if cap == 0 || cap > 65_536 {
        return None;
    }
    let ptr = values.as_mut_ptr().cast::<u8>();
    std::mem::forget(values);
    Some(RawVec { ptr, cap })
}

fn recycle_list(mut items: Vec<Value<'_>>) {
    let plain = items.iter().all(|item| match item {
        Value::Struct(fields) => fields.iter().all(is_plain),
        _ => false,
    });
    if plain {
        let mut small = Vec::with_capacity(items.len());
        for item in &mut items {
            if let Value::Struct(fields) = item {
                let fields = std::mem::take(fields);
                if let Some(raw) = forget_plain(fields) {
                    small.push(raw);
                }
            }
        }
        // SAFETY: field buffers were moved out. Each shell is an empty `Vec`
        // (capacity 0), so forgetting the struct values does not leak.
        unsafe { items.set_len(0) }
        store_all(small, forget_plain(items));
        return;
    }
    let mut small = Vec::new();
    for item in &mut items {
        match item {
            Value::Struct(fields) => {
                let fields = std::mem::take(fields);
                recycle_into(fields, &mut small);
            }
            Value::List(inner) => {
                let inner = std::mem::take(inner);
                recycle_list(inner);
            }
            _ => {}
        }
    }
    {
        let _skip = SkipDrop::new();
        items.clear();
    }
    let list_raw = into_raw(items);
    store_all(small, list_raw);
}

impl Drop for Value<'_> {
    fn drop(&mut self) {
        if SKIP_DROP.with(|flag| flag.get()) > 0 {
            return;
        }
        match self {
            Value::List(items) => {
                let items = std::mem::take(items);
                recycle_list(items);
            }
            Value::Struct(items) => {
                let items = std::mem::take(items);
                let mut small = Vec::new();
                recycle_into(items, &mut small);
                store_all(small, None);
            }
            _ => {}
        }
    }
}
