//! This module contains the implementation of a 'histogram'.
//!
//! A histogram is simply a counter of how many times an individual item has appeared
//! in a given dataset.
//!
//! For example, given the bytes `[0, 1, 2, 0, 1]`, the histogram would be `[2, 2, 1]`,
//! as bytes `0` and `1` appear two times, but the byte `2` appears once.
//!
//! The histogram code in this module is built around calculating occurrences of bytes, the amount
//! of times a byte has been met is stored.

pub mod histogram32;
pub use histogram32::*;

/// The implementation of a generic histogram, storing the for each byte using type `T`.
/// `T` should be a type that can be incremented.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Histogram<T> {
    pub counter: [T; 256],
}

#[cfg(any(test, feature = "bench"))]
pub mod histogram32_private;
#[cfg(any(test, feature = "bench"))]
pub use histogram32_private::*;

/// Benchmark only re-exports.
#[cfg(feature = "bench")]
pub mod bench {
    use super::Histogram32;

    pub fn histogram32_generic_batched_unroll_4_u32(bytes: &[u8]) -> Histogram32 {
        super::histogram32_generic_batched_unroll_4_u32(bytes)
    }

    pub fn histogram32_reference(bytes: &[u8]) -> Histogram32 {
        super::histogram32_reference(bytes)
    }
}
