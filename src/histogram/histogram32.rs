use super::Histogram;
use core::ops::{Deref, DerefMut};

/// Implementation of a histogram using unsigned 32 bit integers as the counter.
///
/// Max safe array size to pass is 4,294,967,295, naturally, as a result, though in practice
/// it can be a bit bigger.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Histogram32 {
    pub inner: Histogram<u32>,
}

impl Default for Histogram<u32> {
    // Defaults to a zero'd array.
    fn default() -> Self {
        Histogram { counter: [0; 256] }
    }
}

impl Deref for Histogram32 {
    type Target = Histogram<u32>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Histogram32 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Histogram32 {
    /// Calculates a new histogram given
    pub fn from_bytes(bytes: &[u8]) -> Self {
        histogram32_from_bytes(bytes)
    }
}

#[inline]
pub fn histogram32_from_bytes(bytes: &[u8]) -> Histogram32 {
    // TODO: Benchmark exact crossover point
    #[allow(clippy::if_same_then_else)]
    if bytes.len() < 256 {
        histogram32_from_bytes_generic_batched(bytes)
    } else {
        histogram32_from_bytes_generic_batched(bytes)
    }
}

/// Generic, version of [`Histogram32`] generation that batches reads by reading [`usize`] bytes
/// at any given time.
///
/// This function is used when [`histogram32_from_bytes`] determines that using a
/// specialized version isn't worth it (for example, if the input is very small).
pub fn histogram32_from_bytes_generic_batched(bytes: &[u8]) -> Histogram32 {
    // 1K on stack, should be good.
    let mut histogram = Histogram32 {
        inner: Histogram { counter: [0; 256] },
    };

    unsafe {
        let histo_ptr = histogram.inner.counter.as_mut_ptr();
        let mut current_ptr = bytes.as_ptr() as *const usize;
        let ptr_end = bytes.as_ptr().add(bytes.len());

        // Unroll the loop by fetching `usize` elements at once, then doing a shift.
        // Although there is a data dependency in the shift, this is still generally faster.
        let ptr_end_unroll =
            bytes.as_ptr().add(bytes.len() & !(size_of::<usize>() - 1)) as *const usize;

        while current_ptr < ptr_end_unroll {
            #[cfg(target_pointer_width = "64")]
            {
                let value = *current_ptr;
                current_ptr = current_ptr.add(1);
                *histo_ptr.add(value & 0xFF) += 1;
                *histo_ptr.add((value >> 8) & 0xFF) += 1;
                *histo_ptr.add((value >> 16) & 0xFF) += 1;
                *histo_ptr.add((value >> 24) & 0xFF) += 1;
                *histo_ptr.add((value >> 32) & 0xFF) += 1;
                *histo_ptr.add((value >> 40) & 0xFF) += 1;
                *histo_ptr.add((value >> 48) & 0xFF) += 1;
                *histo_ptr.add((value >> 56) & 0xFF) += 1;
            }

            #[cfg(target_pointer_width = "32")]
            {
                let value = *current_ptr;
                current_ptr = current_ptr.add(1);
                *histo_ptr.add(value & 0xFF) += 1;
                *histo_ptr.add((value >> 8) & 0xFF) += 1;
                *histo_ptr.add((value >> 16) & 0xFF) += 1;
                *histo_ptr.add((value >> 24) & 0xFF) += 1;
            }

            #[cfg(not(any(target_pointer_width = "32", target_pointer_width = "64")))]
            panic!("Unsupported word size. histogram32_from_bytes_generic_batched only supports 32/64 bit architectures.")
        }

        // Handle any remaining bytes.
        let mut current_ptr = current_ptr as *const u8;
        while current_ptr < ptr_end {
            let byte = *current_ptr;
            current_ptr = current_ptr.add(1);
            *histo_ptr.add(byte as usize) += 1;
        }
    }

    histogram
}

/// Generic, slower version of [`Histogram32`] generation that doesn't assume anything.
/// This is the Rust fallback, reference implementation to run other tests against.
pub fn histogram32_from_bytes_reference(bytes: &[u8]) -> Histogram32 {
    // 1K on stack, should be good.
    let mut histogram = Histogram32 {
        inner: Histogram { counter: [0; 256] },
    };

    let histo_ptr = histogram.inner.counter.as_mut_ptr();
    let mut current_ptr = bytes.as_ptr();
    let ptr_end = unsafe { current_ptr.add(bytes.len()) };

    // Unroll the loop by fetching `usize` elements at once, then doing a shift.
    // Although there is a data dependency in the shift.
    unsafe {
        while current_ptr < ptr_end {
            let byte = *current_ptr;
            current_ptr = current_ptr.add(1);
            *histo_ptr.add(byte as usize) += 1;
        }
    }

    histogram
}

#[cfg(test)]
mod reference_tests {
    use super::*;
    use std::vec::Vec;

    // Creates bytes 0..255, to verify we reach the full range.
    // This should be sufficient for unrolled impl.
    #[test]
    fn verify_full_range_in_reference_impl() {
        let input: Vec<u8> = (0..=255).collect();
        let histogram = histogram32_from_bytes_reference(&input);

        // Every value should appear exactly once
        for count in histogram.inner.counter.iter() {
            assert_eq!(*count, 1);
        }
    }
}

#[cfg(test)]
mod batched_tests {
    use super::*;
    use rstest::rstest;
    use std::vec;
    use std::vec::Vec;

    #[rstest]
    #[case(&[1, 1, 2, 3, 1], &[0, 3, 1, 1, 0, 0, 0, 0])] // Simple case with repeated numbers
    #[case(&[], &[0; 256])] // Empty input
    fn batched_implementation(#[case] input: &[u8], #[case] expected_counts: &[u32]) {
        let histogram = histogram32_from_bytes_generic_batched(input);

        // For the test cases where we provided shortened expected arrays,
        // we need to verify the full histogram
        let mut full_expected = [0u32; 256];
        full_expected[..expected_counts.len()].copy_from_slice(expected_counts);

        assert_eq!(histogram.inner.counter, full_expected);
    }

    #[test]
    fn verify_full_range_in_batched_impl() {
        let input: Vec<u8> = (0..=255).collect();
        let histogram = histogram32_from_bytes_generic_batched(&input);

        // Every value should appear exactly once
        for count in histogram.inner.counter.iter() {
            assert_eq!(*count, 1);
        }
    }

    #[test]
    fn partial_batch() {
        // Test with a number of bytes that isn't divisible by size_of::<usize>()
        let input: Vec<u8> = vec![1; 9]; // 9 bytes (partial batch on both 32 and 64 bit systems)
        let histogram = histogram32_from_bytes_generic_batched(&input);

        // First (up to) 8 bytes were handled in one batch, rest byte by byte
        assert_eq!(histogram.inner.counter[1], 9); // Should count all 9 ones
        assert_eq!(histogram.inner.counter.iter().sum::<u32>(), 9); // Total should be 9
    }

    #[test]
    fn compare_with_reference_impl() {
        // Generate a varied test input
        let mut input = Vec::with_capacity(1024);
        for i in 0..1024 {
            input.push((i % 256) as u8);
        }

        let batched_result = histogram32_from_bytes_generic_batched(&input);
        let reference_result = histogram32_from_bytes_reference(&input);

        assert_eq!(batched_result.inner.counter, reference_result.inner.counter);
    }
}
