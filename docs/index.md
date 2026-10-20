# gld-kiln-rust

gld-kiln-rust is a [Kiln](https://leo-gan.github.io/gld-kiln/spec/) codec
written in Rust. The runtime and the code generator are Rust. They do not wrap
another language's Kiln library. This crate is not the specification.

<div class="grid cards" markdown="1">

-   __Why Kiln__

    ---

    Why a Rust struct is the schema, how tensor bytes stay in the input array,
    and why a hostile count never allocates.

    [:octicons-arrow-right-24: Read Why Kiln](why-kiln.md)

-   __Instructions__

    ---

    Install the crate, decode a value, generate Rust from a schema, and run
    the tests.

    [:octicons-arrow-right-24: Open Instructions](instructions.md)

-   __Examples__

    ---

    Encode a dynamic Point and a generated struct.

    [:octicons-arrow-right-24: See Examples](examples.md)

-   __Techniques__

    ---

    Borrowed payloads, the allocation check, and the frame hash.

    [:octicons-arrow-right-24: Read Techniques](techniques.md)

-   __Test data__

    ---

    The 46 wire cases, the schema files, and the Python oracle.

    [:octicons-arrow-right-24: Open Test data](test-data.md)

</div>
