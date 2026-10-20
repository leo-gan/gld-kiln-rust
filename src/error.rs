//! Decode and encode failures. Names match the Kiln wire checks.

/// Default cap for counts and for one string or byte run: 16,777,216.
pub const DEFAULT_CAP: u64 = 16_777_216;

/// Failure from a wire check. The name is stable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Magic, version, flags, or reserved bytes are wrong.
    Frame,
    /// A count, product, or sum exceeds the cap or 2^64-1.
    Limit,
    /// The buffer ended before the value did.
    Eof,
    /// A tag, dtype, layout, or canonical rule failed.
    Type,
    /// A length-prefixed string or a name was not UTF-8.
    Utf8,
    /// Bytes remained after a top-level value.
    Trailing,
    /// A frame hash does not match the schema.
    Schema,
}

impl Error {
    /// Stable error name used by the golden cases.
    pub fn name(self) -> &'static str {
        match self {
            Error::Frame => "FRAME",
            Error::Limit => "LIMIT",
            Error::Eof => "EOF",
            Error::Type => "TYPE",
            Error::Utf8 => "UTF8",
            Error::Trailing => "TRAILING",
            Error::Schema => "SCHEMA",
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::error::Error for Error {}

/// Knobs for a decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeOpts {
    /// Maximum count, and maximum length of one string or byte run.
    pub cap: u64,
    /// Skip UTF-8 checks and quantization-shape checks.
    /// Lengths past the buffer are still rejected.
    pub trusted: bool,
    /// Reject bytes that follow the value.
    pub standalone: bool,
}

impl Default for DecodeOpts {
    fn default() -> Self {
        Self {
            cap: DEFAULT_CAP,
            trusted: false,
            standalone: true,
        }
    }
}

impl DecodeOpts {
    pub fn new() -> Self {
        Self::default()
    }
}
