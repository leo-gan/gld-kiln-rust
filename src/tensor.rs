//! Version-2 tensor directory. Check order matches the reference decoder.

use crate::error::{DecodeOpts, Error};
use crate::schema::{add_u64, align64, mul_u64};
use crate::value::{Tensor, TensorDir};

pub const DTYPE_F64: u16 = 1;
pub const DTYPE_F32: u16 = 2;
pub const DTYPE_F16: u16 = 3;
pub const DTYPE_BF16: u16 = 4;
pub const DTYPE_F8E4M3: u16 = 5;
pub const DTYPE_F8E5M2: u16 = 6;
pub const DTYPE_BOOL: u16 = 7;
pub const DTYPE_I8: u16 = 8;
pub const DTYPE_U8: u16 = 9;
pub const DTYPE_I16: u16 = 10;
pub const DTYPE_U16: u16 = 11;
pub const DTYPE_I32: u16 = 12;
pub const DTYPE_U32: u16 = 13;
pub const DTYPE_I64: u16 = 14;
pub const DTYPE_U64: u16 = 15;
pub const DTYPE_I4: u16 = 16;
pub const DTYPE_U4: u16 = 17;
pub const DTYPE_E2M1: u16 = 18;
pub const DTYPE_E8M0: u16 = 19;

const HEADER: usize = 64;
const DESC: usize = 128;
const FLAG_NAME: u8 = 0x01;
const FLAG_LE: u8 = 0x02;
const OPAQUE_MIN: u16 = 4096;
const OPAQUE_MAX: u16 = 32767;

pub fn dtype_name(code: u16) -> Option<&'static str> {
    Some(match code {
        1 => "f64",
        2 => "f32",
        3 => "f16",
        4 => "bf16",
        5 => "f8e4m3",
        6 => "f8e5m2",
        7 => "bool",
        8 => "i8",
        9 => "u8",
        10 => "i16",
        11 => "u16",
        12 => "i32",
        13 => "u32",
        14 => "i64",
        15 => "u64",
        16 => "i4",
        17 => "u4",
        18 => "e2m1",
        19 => "e8m0",
        _ => return None,
    })
}

fn whole_size(dtype: u16) -> Option<u64> {
    Some(match dtype {
        1 | 14 | 15 => 8,
        2 | 12 | 13 => 4,
        3 | 4 | 10 | 11 => 2,
        5 | 6 | 7 | 8 | 9 | 19 => 1,
        _ => return None,
    })
}

fn packed(dtype: u16) -> bool {
    matches!(dtype, 16..=18)
}

fn lu16(buf: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([buf[i], buf[i + 1]])
}

fn lu32(buf: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]])
}

fn lu64(buf: &[u8], i: usize) -> u64 {
    u64::from_le_bytes([
        buf[i],
        buf[i + 1],
        buf[i + 2],
        buf[i + 3],
        buf[i + 4],
        buf[i + 5],
        buf[i + 6],
        buf[i + 7],
    ])
}

fn zeros(buf: &[u8]) -> bool {
    buf.iter().all(|&b| b == 0)
}

struct Raw<'a> {
    dtype: u16,
    rank: u8,
    name_off: u16,
    name_len: u16,
    layout_off: u16,
    layout_len: u16,
    block_size: u32,
    quant_scheme: u16,
    quant_axis: u16,
    group_size: u32,
    scale_index: u32,
    outer_index: u32,
    byte_offset: u64,
    byte_length: u64,
    shape: [u64; 8],
    name: Option<&'a str>,
    layout: Option<&'a str>,
}

fn element_count(rank: u8, shape: &[u64; 8]) -> Result<u64, Error> {
    if rank == 0 {
        return Ok(1);
    }
    let mut count = 1u64;
    for dim in shape.iter().take(rank as usize) {
        count = mul_u64(u128::from(count), u128::from(*dim))?;
    }
    Ok(count)
}

fn expected_bytes(dtype: u16, count: u64) -> Result<u64, Error> {
    if let Some(width) = whole_size(dtype) {
        return mul_u64(count as u128, width as u128);
    }
    if packed(dtype) {
        return Ok(((count as u128 + 1) / 2) as u64);
    }
    Err(Error::Type)
}

fn check_tensor(item: &Raw<'_>) -> Result<(), Error> {
    if item.rank > 8 {
        return Err(Error::Type);
    }
    if item.shape[item.rank as usize..].iter().any(|&dim| dim != 0) {
        return Err(Error::Type);
    }
    let dtype = item.dtype;
    if dtype == 0 || (20..OPAQUE_MIN).contains(&dtype) || dtype > OPAQUE_MAX {
        return Err(Error::Type);
    }
    let count = element_count(item.rank, &item.shape)?;
    if dtype >= OPAQUE_MIN {
        if item.layout.is_none() || item.block_size == 0 {
            return Err(Error::Type);
        }
        if item.byte_length % u64::from(item.block_size) != 0 {
            return Err(Error::Type);
        }
        return Ok(());
    }
    if item.layout.is_some() || item.block_size != 0 {
        return Err(Error::Type);
    }
    let expect = expected_bytes(dtype, count)?;
    if item.byte_length != expect {
        return Err(Error::Type);
    }
    Ok(())
}

fn check_quant(items: &[Raw<'_>], trusted: bool) -> Result<(), Error> {
    if trusted {
        return Ok(());
    }
    for (index, item) in items.iter().enumerate() {
        let scheme = item.quant_scheme;
        if scheme == 0 {
            if item.group_size != 0
                || item.quant_axis != 0xffff
                || item.scale_index != 0xffff_ffff
                || item.outer_index != 0xffff_ffff
            {
                return Err(Error::Type);
            }
            continue;
        }
        if scheme != 1 && scheme != 2 {
            return Err(Error::Type);
        }
        if item.rank == 0 || u32::from(item.quant_axis) >= u32::from(item.rank) {
            return Err(Error::Type);
        }
        if item.group_size == 0 {
            return Err(Error::Type);
        }
        let scale = item.scale_index;
        if scale as usize == index || scale as usize >= items.len() {
            return Err(Error::Type);
        }
        let other = &items[scale as usize];
        if other.quant_scheme != 0 {
            return Err(Error::Type);
        }
        let axis = item.quant_axis as usize;
        let groups = (item.shape[axis] as u128 + u128::from(item.group_size) - 1)
            / u128::from(item.group_size);
        let mut other_dims = 1u64;
        for dim in 0..item.rank as usize {
            if dim != axis {
                other_dims = mul_u64(other_dims as u128, item.shape[dim] as u128)?;
            }
        }
        let expect = mul_u64(groups, other_dims as u128)?;
        let scale_count = element_count(other.rank, &other.shape)?;
        if scale_count != expect {
            return Err(Error::Type);
        }
        if scheme == 1 {
            if item.outer_index != 0xffff_ffff {
                return Err(Error::Type);
            }
            continue;
        }
        let outer = item.outer_index;
        if outer as usize == index || outer == scale || outer as usize >= items.len() {
            return Err(Error::Type);
        }
        let outer_item = &items[outer as usize];
        let outer_count = element_count(outer_item.rank, &outer_item.shape)?;
        if outer_item.quant_scheme != 0 || outer_item.dtype != DTYPE_F32 || outer_count != 1 {
            return Err(Error::Type);
        }
    }
    Ok(())
}

fn utf8(bytes: &[u8], trusted: bool) -> Result<&str, Error> {
    if trusted {
        // SAFETY: a trusted reader promises that names and layouts are UTF-8.
        Ok(unsafe { std::str::from_utf8_unchecked(bytes) })
    } else {
        std::str::from_utf8(bytes).map_err(|_| Error::Utf8)
    }
}

/// Decode one version-2 directory. `Tensor::data` aliases `buf`.
pub fn decode_tensors<'a>(buf: &'a [u8], opts: DecodeOpts) -> Result<TensorDir<'a>, Error> {
    if buf.len() < 4 {
        return Err(Error::Eof);
    }
    if &buf[0..4] != b"KLNT" {
        return Err(Error::Frame);
    }
    if buf.len() < HEADER {
        return Err(Error::Eof);
    }
    let version = buf[4];
    if version != 2 {
        return Err(Error::Frame);
    }
    let flags = buf[5];
    if flags & !0x03 != 0 {
        return Err(Error::Frame);
    }
    if flags & FLAG_LE == 0 {
        return Err(Error::Frame);
    }
    if lu16(buf, 6) != 0 || !zeros(&buf[32..HEADER]) {
        return Err(Error::Frame);
    }
    let count = lu32(buf, 8);
    let name_bytes = lu32(buf, 12);
    let payload_bytes = lu64(buf, 16);
    let directory_bytes = lu64(buf, 24);
    if u64::from(count) > opts.cap || u64::from(name_bytes) > opts.cap {
        return Err(Error::Limit);
    }
    let expect_dir = add_u64(mul_u64(u128::from(count), DESC as u128)? as u128, u128::from(name_bytes))?;
    if directory_bytes != expect_dir {
        return Err(Error::Type);
    }
    let directory_end = add_u64(HEADER as u128, u128::from(directory_bytes))?;
    if u128::from(directory_end) > buf.len() as u128 {
        return Err(Error::Eof);
    }
    let directory_end = directory_end as usize;
    let payload_base = align64(directory_end as u128);
    let block_end = add_u64(payload_base, u128::from(payload_bytes))?;
    if u128::from(block_end) > buf.len() as u128 {
        return Err(Error::Eof);
    }
    let payload_base = payload_base as usize;
    let block_end = block_end as usize;
    if !zeros(&buf[directory_end..payload_base]) {
        return Err(Error::Type);
    }

    let mut raw: Vec<Raw<'a>> = Vec::new();
    if count > 1 {
        raw.reserve(count as usize);
    }
    for index in 0..count as usize {
        let base = HEADER + index * DESC;
        if !zeros(&buf[base + 112..base + DESC]) {
            return Err(Error::Frame);
        }
        let mut shape = [0u64; 8];
        for (dim, slot) in shape.iter_mut().enumerate() {
            *slot = lu64(buf, base + 48 + 8 * dim);
        }
        raw.push(Raw {
            dtype: lu16(buf, base),
            rank: buf[base + 2],
            name_off: lu16(buf, base + 4),
            name_len: lu16(buf, base + 6),
            layout_off: lu16(buf, base + 8),
            layout_len: lu16(buf, base + 10),
            block_size: lu32(buf, base + 12),
            quant_scheme: lu16(buf, base + 16),
            quant_axis: lu16(buf, base + 18),
            group_size: lu32(buf, base + 20),
            scale_index: lu32(buf, base + 24),
            outer_index: lu32(buf, base + 28),
            byte_offset: lu64(buf, base + 32),
            byte_length: lu64(buf, base + 40),
            shape,
            name: None,
            layout: None,
        });
    }

    let heap = &buf[HEADER + count as usize * DESC..directory_end];
    let names_flag = flags & FLAG_NAME != 0;
    let has_name = raw.iter().any(|item| item.name_len != 0);
    if names_flag != has_name {
        return Err(Error::Type);
    }
    if names_flag && raw.iter().any(|item| item.name_len == 0) {
        return Err(Error::Type);
    }

    let mut name_parts: Vec<&[u8]> = Vec::new();
    let mut layout_parts: Vec<&[u8]> = Vec::new();
    let mut seen: Vec<&[u8]> = Vec::new();
    for (index, item) in raw.iter_mut().enumerate() {
        let base = HEADER + index * DESC;
        if buf[base + 3] != 0 {
            return Err(Error::Frame);
        }
        if item.name_len == 0 {
            if item.name_off != 0xffff {
                return Err(Error::Type);
            }
        } else {
            if item.name_off == 0xffff {
                return Err(Error::Type);
            }
            let end = add_u64(u128::from(item.name_off), u128::from(item.name_len))?;
            if end > u64::from(name_bytes) {
                return Err(Error::Type);
            }
            let name = &heap[item.name_off as usize..end as usize];
            let text = utf8(name, opts.trusted)?;
            if seen.contains(&name) {
                return Err(Error::Type);
            }
            seen.push(name);
            name_parts.push(name);
            item.name = Some(text);
        }
        if item.layout_len == 0 {
            if item.layout_off != 0xffff {
                return Err(Error::Type);
            }
            item.layout = None;
        } else {
            let end = add_u64(u128::from(item.layout_off), u128::from(item.layout_len))?;
            if end > u64::from(name_bytes) {
                return Err(Error::Type);
            }
            let layout = &heap[item.layout_off as usize..end as usize];
            let text = utf8(layout, opts.trusted)?;
            layout_parts.push(layout);
            item.layout = Some(text);
        }
    }
    if !heap_matches(heap, &name_parts, &layout_parts) {
        return Err(Error::Type);
    }

    if raw.len() > 1 {
        let mut ranges: Vec<(u64, u64)> = Vec::with_capacity(raw.len());
        for item in &raw {
            let item_end = add_u64(u128::from(item.byte_offset), u128::from(item.byte_length))?;
            if item_end > payload_bytes {
                return Err(Error::Type);
            }
            ranges.push((item.byte_offset, item_end));
            check_tensor(item)?;
        }
        for left in 0..ranges.len() {
            for right in (left + 1)..ranges.len() {
                let (a0, a1) = ranges[left];
                let (b0, b1) = ranges[right];
                if a0 == a1 || b0 == b1 {
                    continue;
                }
                if a0 < b1 && b0 < a1 {
                    return Err(Error::Type);
                }
            }
        }
    } else if let Some(item) = raw.first() {
        let item_end = add_u64(u128::from(item.byte_offset), u128::from(item.byte_length))?;
        if item_end > payload_bytes {
            return Err(Error::Type);
        }
        check_tensor(item)?;
    }
    let mut end = 0u64;
    for item in &raw {
        let aligned = align64(u128::from(end));
        if u128::from(item.byte_offset) != aligned {
            return Err(Error::Type);
        }
        end = item.byte_offset + item.byte_length;
    }
    if end != payload_bytes {
        return Err(Error::Type);
    }
    let payload = &buf[payload_base..block_end];
    for item in &raw {
        let start = item.byte_offset as usize;
        let stop = start + item.byte_length as usize;
        let region = &payload[start..stop];
        if !opts.trusted && packed(item.dtype) {
            let count_elems = element_count(item.rank, &item.shape)?;
            if count_elems % 2 == 1 {
                if let Some((&last, _)) = region.split_last() {
                    if last >> 4 != 0 {
                        return Err(Error::Type);
                    }
                }
            }
        }
        if !opts.trusted && item.dtype == DTYPE_BOOL && region.iter().any(|&byte| byte > 1) {
            return Err(Error::Type);
        }
    }
    let mut cursor = 0u64;
    for item in &raw {
        if item.byte_offset > cursor && !zeros(&payload[cursor as usize..item.byte_offset as usize]) {
            return Err(Error::Type);
        }
        cursor = item.byte_offset + item.byte_length;
    }
    check_quant(&raw, opts.trusted)?;
    if opts.standalone && block_end != buf.len() {
        return Err(Error::Trailing);
    }

    let mut tensors = Vec::with_capacity(raw.len());
    for item in &raw {
        let start = item.byte_offset as usize;
        let stop = start + item.byte_length as usize;
        tensors.push(Tensor {
            name: item.name,
            dtype: item.dtype,
            rank: item.rank,
            shape: item.shape,
            quant_scheme: item.quant_scheme,
            quant_axis: item.quant_axis,
            group_size: item.group_size,
            scale_index: item.scale_index,
            outer_index: item.outer_index,
            block_size: item.block_size,
            layout: item.layout,
            data: &payload[start..stop],
        });
    }
    Ok(TensorDir {
        tensors,
        block_len: block_end,
        names: names_flag,
    })
}

fn heap_matches(heap: &[u8], names: &[&[u8]], layouts: &[&[u8]]) -> bool {
    let mut pos = 0usize;
    for part in names.iter().chain(layouts.iter()) {
        let end = pos + part.len();
        if end > heap.len() || &heap[pos..end] != *part {
            return false;
        }
        pos = end;
    }
    pos == heap.len()
}

fn put_u16(buf: &mut [u8], i: usize, v: u16) {
    buf[i..i + 2].copy_from_slice(&v.to_le_bytes());
}

fn put_u32(buf: &mut [u8], i: usize, v: u32) {
    buf[i..i + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(buf: &mut [u8], i: usize, v: u64) {
    buf[i..i + 8].copy_from_slice(&v.to_le_bytes());
}

struct Layout {
    flags: u8,
    heap: Vec<u8>,
    name_off: Vec<u16>,
    name_len: Vec<u16>,
    layout_off: Vec<u16>,
    layout_len: Vec<u16>,
    offsets: Vec<u64>,
    payload_len: u64,
    directory: u64,
    payload_base: usize,
}

fn layout_tensors(tensors: &[Tensor<'_>], cap: u64) -> Result<Layout, Error> {
    let n = tensors.len();
    if n > u32::MAX as usize || n as u64 > cap {
        return Err(Error::Limit);
    }
    let mut heap = Vec::new();
    let mut name_off = vec![0xffffu16; n];
    let mut name_len = vec![0u16; n];
    let mut layout_off = vec![0xffffu16; n];
    let mut layout_len = vec![0u16; n];
    let mut flags = FLAG_LE;
    for (i, tensor) in tensors.iter().enumerate() {
        if let Some(name) = tensor.name {
            if heap.len() > u16::MAX as usize || name.len() > u16::MAX as usize {
                return Err(Error::Limit);
            }
            name_off[i] = heap.len() as u16;
            name_len[i] = name.len() as u16;
            heap.extend_from_slice(name.as_bytes());
            flags |= FLAG_NAME;
        }
    }
    for (i, tensor) in tensors.iter().enumerate() {
        if let Some(layout) = tensor.layout {
            if heap.len() > u16::MAX as usize || layout.len() > u16::MAX as usize {
                return Err(Error::Limit);
            }
            layout_off[i] = heap.len() as u16;
            layout_len[i] = layout.len() as u16;
            heap.extend_from_slice(layout.as_bytes());
        }
    }
    if heap.len() as u64 > cap {
        return Err(Error::Limit);
    }
    let mut end = 0u64;
    let mut offsets = Vec::with_capacity(n);
    for tensor in tensors {
        let off = align64(u128::from(end));
        if off > u64::MAX as u128 {
            return Err(Error::Limit);
        }
        let off = off as u64;
        end = add_u64(u128::from(off), tensor.data.len() as u128)?;
        offsets.push(off);
    }
    let directory = add_u64(mul_u64(n as u128, DESC as u128)? as u128, heap.len() as u128)?;
    let payload_base = align64(add_u64(HEADER as u128, u128::from(directory))? as u128);
    if payload_base > u64::MAX as u128 {
        return Err(Error::Limit);
    }
    let payload_base = usize::try_from(payload_base).map_err(|_| Error::Limit)?;
    let payload_len = usize::try_from(end).map_err(|_| Error::Limit)?;
    if payload_base.checked_add(payload_len).is_none() {
        return Err(Error::Limit);
    }
    Ok(Layout {
        flags,
        heap,
        name_off,
        name_len,
        layout_off,
        layout_len,
        offsets,
        payload_len: end,
        directory,
        payload_base,
    })
}

#[cfg_attr(not(debug_assertions), allow(dead_code))]
pub(crate) fn tensor_block_len(tensors: &[Tensor<'_>], cap: u64) -> Result<usize, Error> {
    let laid = layout_tensors(tensors, cap)?;
    laid.payload_base
        .checked_add(usize::try_from(laid.payload_len).map_err(|_| Error::Limit)?)
        .ok_or(Error::Limit)
}

fn write_desc(
    desc: &mut [u8],
    tensor: &Tensor<'_>,
    name_off: u16,
    name_len: u16,
    layout_off: u16,
    layout_len: u16,
    byte_offset: u64,
) {
    put_u16(desc, 0, tensor.dtype);
    desc[2] = tensor.rank;
    put_u16(desc, 4, name_off);
    put_u16(desc, 6, name_len);
    put_u16(desc, 8, layout_off);
    put_u16(desc, 10, layout_len);
    put_u32(desc, 12, tensor.block_size);
    put_u16(desc, 16, tensor.quant_scheme);
    put_u16(desc, 18, tensor.quant_axis);
    put_u32(desc, 20, tensor.group_size);
    put_u32(desc, 24, tensor.scale_index);
    put_u32(desc, 28, tensor.outer_index);
    put_u64(desc, 32, byte_offset);
    put_u64(desc, 40, tensor.data.len() as u64);
    for (dim, axis) in tensor.shape.iter().enumerate() {
        put_u64(desc, 48 + 8 * dim, *axis);
    }
}

fn encode_plain_one(tensor: &Tensor<'_>, out: &mut Vec<u8>, cap: u64) -> Result<(), Error> {
    if cap == 0 {
        return Err(Error::Limit);
    }
    let payload = tensor.data.len();
    let total = 192usize.checked_add(payload).ok_or(Error::Limit)?;
    let mut head = [0u8; 192];
    head[0..4].copy_from_slice(b"KLNT");
    head[4] = 2;
    head[5] = FLAG_LE;
    put_u32(&mut head, 8, 1);
    put_u64(&mut head, 16, payload as u64);
    put_u64(&mut head, 24, DESC as u64);
    write_desc(&mut head[HEADER..], tensor, 0xffff, 0, 0xffff, 0, 0);
    out.reserve(total);
    out.extend_from_slice(&head);
    out.extend_from_slice(tensor.data);
    Ok(())
}

/// Append one canonical version-2 directory.
pub fn encode_tensors_into(tensors: &[Tensor<'_>], out: &mut Vec<u8>, cap: u64) -> Result<(), Error> {
    if let [tensor] = tensors {
        if tensor.name.is_none() && tensor.layout.is_none() {
            return encode_plain_one(tensor, out, cap);
        }
    }
    let laid = layout_tensors(tensors, cap)?;
    let start = out.len();
    // Zero only the header, descriptors, name heap, and alignment pad.
    // Payload bytes are copied below, so they are not cleared first.
    out.resize(start + laid.payload_base, 0);
    {
        let block = &mut out[start..];
        block[0..4].copy_from_slice(b"KLNT");
        block[4] = 2;
        block[5] = laid.flags;
        put_u32(block, 8, tensors.len() as u32);
        put_u32(block, 12, laid.heap.len() as u32);
        put_u64(block, 16, laid.payload_len);
        put_u64(block, 24, laid.directory);
        for (index, tensor) in tensors.iter().enumerate() {
            let base = HEADER + index * DESC;
            write_desc(
                &mut block[base..base + DESC],
                tensor,
                laid.name_off[index],
                laid.name_len[index],
                laid.layout_off[index],
                laid.layout_len[index],
                laid.offsets[index],
            );
        }
        if !laid.heap.is_empty() {
            let at = HEADER + tensors.len() * DESC;
            block[at..at + laid.heap.len()].copy_from_slice(&laid.heap);
        }
    }
    let mut cursor = 0u64;
    for (tensor, offset) in tensors.iter().zip(laid.offsets.iter()) {
        if *offset > cursor {
            out.resize(out.len() + (*offset - cursor) as usize, 0);
        }
        out.extend_from_slice(tensor.data);
        cursor = offset + tensor.data.len() as u64;
    }
    if laid.payload_len > cursor {
        out.resize(out.len() + (laid.payload_len - cursor) as usize, 0);
    }
    Ok(())
}

pub fn encode_tensors(tensors: &[Tensor<'_>], cap: u64) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    encode_tensors_into(tensors, &mut out, cap)?;
    Ok(out)
}
