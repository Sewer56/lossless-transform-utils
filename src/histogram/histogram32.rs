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
    // Obtained by benching on a 5900X. May vary with different hardware.
    #[allow(clippy::if_same_then_else)]
    if bytes.len() < 64 {
        histogram32_from_bytes_reference(bytes)
    } else {
        histogram32_from_bytes_generic_batched(bytes)
    }
}

const NUM_SLICES: usize = 4;
const SLICE_SIZE_U32S: usize = 256;

/// Based on `histo_asm_scalar8_var5_core` by fabian 'ryg' giesen
/// https://gist.github.com/rygorous/a86a5cf348922cdea357c928e32fc7e0
///
/// # Safety
///
/// This function is safe with any input.
///
/// # Remarks
///
/// For some reason on my AMD 5900X machine this is slower than the `batched` implementation.
/// When experimenting with implementations, I don't (in general) seem to be getting benefits
/// from preventing aliasing.
///
/// The reason may be something related to https://www.agner.org/forum/viewtopic.php?t=41 .
/// I did check the assembly, it's comparable (near identical) to ryg's original.
pub fn histogram_nonaliased_withruns_core(data: &[u8]) -> Histogram32 {
    // 1K on stack, should be good.
    let mut histogram = [Histogram32::default(); NUM_SLICES];

    unsafe {
        let mut ptr = data.as_ptr();
        let end = ptr.add(data.len());
        let current_ptr = histogram[0].inner.counter.as_mut_ptr();

        if data.len() > 24 {
            let aligned_end = end.sub(24);
            let mut current = (ptr as *const u64).read_unaligned();

            while ptr < aligned_end {
                // Prefetch next 1 iteration.
                let next = (ptr.add(8) as *const u64).read_unaligned();

                if current == next {
                    // Check if all bytes are the same within 'current'.

                    // With a XOR, we can check every byte (except byte 0)
                    // with its predecessor. If our value is <256,
                    // then all bytes are the same value.
                    let shifted = current << 8;
                    if (shifted ^ current) < 256 {
                        // All bytes same - increment single bucket by 16
                        // (current is all same byte and current equals next)
                        *current_ptr.add((current & 0xFF) as usize) += 16;
                    } else {
                        // Same 8 bytes twice - sum with INC2
                        sum8(current_ptr, current, 2);
                    }
                } else {
                    // Process both 8-byte chunks with INC1
                    sum8(current_ptr, current, 1);
                    sum8(current_ptr, next, 1);
                }

                current = ((ptr.add(16)) as *const u64).read_unaligned();
                ptr = ptr.add(16);
            }
        }

        while ptr < end {
            let byte = *ptr;
            *current_ptr.add(byte as usize) += 1;
            ptr = ptr.add(1);
        }

        // Sum up all bytes
        // Vectorization-friendly summation
        if NUM_SLICES <= 1 {
            histogram[0]
        } else {
            let mut result = histogram[0];
            for x in (0..256).step_by(4) {
                let mut sum0 = 0_u32;
                let mut sum1 = 0_u32;
                let mut sum2 = 0_u32;
                let mut sum3 = 0_u32;

                // Changing to suggested code breaks.
                #[allow(clippy::needless_range_loop)]
                for slice in 0..NUM_SLICES {
                    sum0 += histogram[slice].inner.counter[x];
                    sum1 += histogram[slice].inner.counter[x + 1];
                    sum2 += histogram[slice].inner.counter[x + 2];
                    sum3 += histogram[slice].inner.counter[x + 3];
                }

                result.inner.counter[x] = sum0;
                result.inner.counter[x + 1] = sum1;
                result.inner.counter[x + 2] = sum2;
                result.inner.counter[x + 3] = sum3;
            }

            result
        }
    }
}

#[inline(always)]
unsafe fn sum8(current_ptr: *mut u32, mut value: u64, increment: u32) {
    for index in 0..8 {
        let byte = (value & 0xFF) as usize;
        let slice_offset = (index % NUM_SLICES) * SLICE_SIZE_U32S;
        let write_ptr = current_ptr.add(slice_offset + byte);
        let current = (write_ptr as *const u32).read_unaligned();
        (write_ptr).write_unaligned(current + increment);
        value >>= 8;
    }
}

/// Generic, version of [`Histogram32`] generation that batches reads by reading [`usize`] bytes
/// at any given time.
///
/// This function is used when [`histogram32_from_bytes`] determines that using a
/// specialized version isn't worth it (for example, if the input is very small).
#[inline(never)]
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

#[cfg(test)]
mod nonaliased_tests {
    use super::*;
    use std::vec::Vec;

    #[test]
    fn empty_input() {
        let input = [];
        let histogram = histogram_nonaliased_withruns_core(&input);
        assert_eq!(histogram.inner.counter, [0; 256]);
    }

    #[test]
    fn small_input() {
        let input = [1, 2, 3, 4];
        let histogram = histogram_nonaliased_withruns_core(&input);

        let mut expected = [0; 256];
        expected[1] = 1;
        expected[2] = 1;
        expected[3] = 1;
        expected[4] = 1;

        assert_eq!(histogram.inner.counter, expected);
    }

    #[test]
    fn repeated_bytes() {
        // Create input with repeated bytes to test the optimization for identical 8-byte chunks
        let input = [42; 32]; // 32 bytes of value 42, which will trigger the optimization
        let histogram = histogram_nonaliased_withruns_core(&input);

        let mut expected = [0; 256];
        expected[42] = 32; // Should count all 32 occurrences

        assert_eq!(histogram.inner.counter, expected);
    }

    #[test]
    fn alignment_boundaries() {
        // Test with different buffer sizes around the 24-byte boundary
        // mentioned in the implementation
        let sizes = [23, 24, 25];

        for size in sizes {
            let input: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let histogram = histogram_nonaliased_withruns_core(&input);

            // Verify against reference implementation
            let reference = histogram32_from_bytes_reference(&input);
            assert_eq!(
                histogram.inner.counter, reference.inner.counter,
                "Failed for size {}",
                size
            );
        }
    }

    #[test]
    fn repeated_chunks_different_bytes() {
        // Create input where we have identical 8-byte chunks that contain different bytes
        // This specifically tests the sum8(counter_ptr, current, true) branch
        let chunk = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut input = Vec::with_capacity(32);

        // Add the same 8-byte chunk twice, followed by another set
        input.extend_from_slice(&chunk); // First chunk
        input.extend_from_slice(&chunk); // Same chunk again - this should trigger INC2
        input.extend_from_slice(&[9, 9, 9, 9, 9, 9, 9, 9]); // Different chunk
        input.extend_from_slice(&[9, 9, 9, 9, 9, 9, 9, 9]); // Different chunk

        let histogram = histogram_nonaliased_withruns_core(&input);

        let mut expected = [0; 256];
        // First two chunks counted twice each (4 total)
        expected[1] = 2;
        expected[2] = 2;
        expected[3] = 2;
        expected[4] = 2;
        expected[5] = 2;
        expected[6] = 2;
        expected[7] = 2;
        expected[8] = 2;
        // Last chunk counted once
        expected[9] = 16;

        assert_eq!(histogram.inner.counter, expected);
    }

    #[test]
    fn compare_against_reference() {
        // Generate a varied test input
        let mut input = Vec::with_capacity(1024);
        for i in 0..1024 {
            input.push((i % 256) as u8);
        }

        let asm_result = histogram_nonaliased_withruns_core(&input);
        let reference_result = histogram32_from_bytes_reference(&input);
        assert_eq!(asm_result.inner.counter, reference_result.inner.counter);
    }
}
