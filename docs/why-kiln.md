# Why Kiln

Kiln is a binary encoding for records and tensors. A Rust struct is the schema.
Field names exist in the type, not on the wire. The encoder writes fields in
declaration order and the decoder reads them back in that same order. There is
no per-field tag and no padding between scalars.

This library is a Rust implementation of that idea. It is not the Kiln
specification. The generator command is `gld-kilngen-rust`.

---

## The wire

Normative layout, dtype codes, and the directory rules live in the Kiln spec:

<https://leo-gan.github.io/gld-kiln/spec/>

Frozen scalars are little-endian. A string is a `u32` byte length plus UTF-8.
`list<u8>` and `list<i8>` are a count plus raw bytes. Other lists concatenate
their elements. An optional value is a `0` or `1` byte. Dict keys sort by raw
UTF-8 bytes. A struct is only its fields.

A tensor directory is a separate block inside a value. The numeric payload is
not copied into a new array on decode. The returned slice points at the input
buffer, so an `f32` tensor stays `f32` bytes. A frame puts a schema hash in
front of one Frozen value. The hash token for a tensor field is the ASCII
bytes `tensors`. If the hash does not match the schema the caller claimed, the
error is `SCHEMA`.

---

## Counts before allocation

A list count is trusted only after a size check. The decoder multiplies the
count by the minimum size of one element and requires that product to fit in
the unread bytes. The count is also capped, by default at 16,777,216. When the
product does not fit, the error is `EOF` or `LIMIT` and the element `Vec` is
never created. A claimed length past the buffer is rejected even by a trusted
reader that skips UTF-8 and quantization-shape checks.

That rule is why the Rust API can borrow tensor payloads. The library does not
allocate an element for a count it has not proved the buffer can hold.

---

## Generated structs

JSON Schema is a convenient way to name fields. `gld-kilngen-rust` turns an
object into a Rust struct whose `encode` and `decode` use the same Frozen
rules. Optional properties become `Option<T>`. The struct is still not the
specification. It is one schema, compiled into Rust, for this crate.
