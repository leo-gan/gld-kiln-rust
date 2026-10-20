# Techniques

The hot path is ordinary Rust. Nothing in encode or decode calls a Kiln
library written in another language.

## Borrow the bytes you already have

`Reader` is an index into the input slice. A string decode returns `&str`
after the UTF-8 check, not a new `String`, unless the caller asks for an owned
value. A tensor payload is `&[u8]` inside that same buffer. Re-encoding copies
once, from that slice into the output `Vec`.

`encode_into` appends to a buffer the caller can clear and reuse. Release
builds write the value in one walk. Debug builds also compute the encoded size
and reject a drift between the two, which keeps the size helper honest.

## Bulk numeric copies

`list<u8>` and `list<i8>` copy one byte range. An unnamed tensor with no layout
string is a fixed 192-byte head plus one `extend_from_slice` of the payload.
The head is the 64-byte directory header and one 128-byte descriptor. The
payload is not cleared and then overwritten. Multi-tensor blocks clear only
the header, the name heap, and the alignment gaps.

## No element storage for a hostile count

`list_count` reads the `u32` count, compares it with the cap, and compares
`count * min_size` with the unread tail. Only after that returns does a list
call `Vec::with_capacity`. The same check gates dicts. Tensor counts above the
cap return `LIMIT` before a descriptor `Vec` is reserved. A count inside the
cap whose bytes are absent returns `EOF` and also does not reserve.

Field names of a struct are not allocated on decode. They stay in the `Schema`
value the caller already owns.

## Frame hash

`type_hash` is FNV-1a 64 with the offset basis and the usual prime. Each
schema token is mixed one byte at a time, then an extra multiply stands in for
a trailing `0x00`. Struct fields contribute the name, a colon, the field
schema, and a comma. The tensor token is `tensors`.

## Errors

The library returns `Error`, a `Copy` enum. `Error::name` is the stable wire
name: `FRAME`, `LIMIT`, `EOF`, `TYPE`, `UTF8`, `TRAILING`, or `SCHEMA`. Decode
does not build a heap string to report those failures.
