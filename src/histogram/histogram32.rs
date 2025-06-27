//! Implementation of a histogram using 32-bit unsigned integers as counters.
//!
//! This module provides an efficient for creating a histogram which supplies you
//! with the byte occurrences in data. It's particularly optimized for
//! processing large amounts of data quickly.
//!
//! # Key Features
//!
//! - Fast histogram generation from byte slices
//! - Optimized for different input sizes and hardware capabilities
//! - Supports x86 and x86_64 specific optimizations (with nightly Rust)
//!
//! # Main Types
//!
//! - [Histogram32]: The primary struct representing a histogram with 32-bit counters.
//!
//! # Main Functions
//!
//! - [`histogram32_from_bytes`]: Efficiently creates a histogram from a byte slice.
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```
//! use lossless_transform_utils::histogram::histogram32_from_bytes;
//! use lossless_transform_utils::histogram::Histogram32;
//!
//! let data = [1, 2, 3, 1, 2, 1];
//! let mut histogram = Histogram32::default();
//! histogram32_from_bytes(&data, &mut histogram);
//!
//! assert_eq!(histogram.inner.counter[1], 3); // Byte value 1 appears 3 times
//! assert_eq!(histogram.inner.counter[2], 2); // Byte value 2 appears 2 times
//! assert_eq!(histogram.inner.counter[3], 1); // Byte value 3 appears 1 time
//! ```
//!
//! # Performance Considerations
//!
//! The implementations in this module are optimized for different input sizes:
//!
//! - Small inputs (< 64 bytes) use a simple, efficient implementation.
//! - Larger inputs use batched processing with loop unrolling for better performance.
//! - On x86_64 and x86 platforms (with nightly Rust), BMI1 instructions are utilized if available.
//!
//! Not optimized for non-x86 platforms, as I (Sewer) don't own any hardware.
//!
//! # Safety
//!
//! While some functions in this module use unsafe code internally for performance reasons,
//! all public interfaces are safe to use from safe Rust code.

use super::Histogram;
use core::ops::{Deref, DerefMut};

/// Implementation of a histogram using unsigned 32 bit integers as the counter.
///
/// Max safe array size to pass is 4,294,967,295, naturally, as a result, though in practice
/// it can be a bit bigger.
#[repr(C)]
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
    /// This is a shortcut for [`histogram32_from_bytes`]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut histogram = Histogram32::default();
        histogram32_from_bytes(bytes, &mut histogram);
        histogram
    }
}

/// Calculates a new histogram given a byte slice.
///
/// This function computes a histogram of byte occurrences in the input slice.
/// It automatically selects the most efficient implementation based on the
/// input size and available hardware features.
///
/// # Performance
///
/// - For small inputs (less than 64 bytes), it uses a simple reference implementation.
/// - For larger inputs, it uses an optimized implementation with batched processing and loop unrolling.
/// - On x86_64 and x86 (with nightly feature) platforms, it can utilize BMI1 instructions if available.
///
/// Not optimized for non-x86 platforms, as I (Sewer) don't own any hardware.
///
/// # Arguments
///
/// * `bytes` - A slice of bytes to process.
///
/// # Returns
///
/// Returns a [Histogram32] struct containing the computed histogram.
/// Each element in the histogram represents the count of occurrences for a byte value (0-255).
///
/// # Example
///
/// ```
/// use lossless_transform_utils::histogram::histogram32_from_bytes;
/// use lossless_transform_utils::histogram::Histogram32;
///
/// let data = [1, 2, 3, 1, 2, 1];
/// let mut histogram = Histogram32::default();
/// histogram32_from_bytes(&data, &mut histogram);
///
/// assert_eq!(histogram.inner.counter[1], 3); // Byte value 1 appears 3 times
/// assert_eq!(histogram.inner.counter[2], 2); // Byte value 2 appears 2 times
/// assert_eq!(histogram.inner.counter[3], 1); // Byte value 3 appears 1 time
/// ```
///
/// # Notes
///
/// - The function is optimized for different input sizes and hardware capabilities.
/// - The threshold for switching between implementations (64 bytes) is based on
///   benchmarks performed on an AMD Ryzen 9 5900X processor. This may vary on different hardware.
///
/// # Safety
///
/// While this function uses unsafe code internally for performance optimization,
/// it is safe to call and use from safe Rust code.
pub fn histogram32_from_bytes(bytes: &[u8], hist: &mut Histogram32) {
    // Obtained by benching on a 5900X. May vary with different hardware.
    if bytes.len() < 64 {
        histogram32_reference(bytes, hist)
    } else {
        histogram32_generic_batched_unroll_4_u32(bytes, hist)
    }
}

pub(crate) fn histogram32_generic_batched_unroll_4_u32(bytes: &[u8], histogram: &mut Histogram32) {
    if bytes.is_empty() {
        return;
    }

    unsafe {
        let histo_ptr = histogram.inner.counter.as_mut_ptr();
        let mut current_ptr = bytes.as_ptr() as *const u32;
        let ptr_end = bytes.as_ptr().add(bytes.len());

        // We'll read 4 u32 values at a time, so adjust alignment accordingly
        let ptr_end_unroll = bytes
            .as_ptr()
            .add(bytes.len() & !(4 * size_of::<u32>() - 1))
            as *const u32;

        #[cfg(all(target_arch = "x86_64", feature = "std"))]
        if std::is_x86_feature_detected!("bmi1") {
            if current_ptr < ptr_end_unroll {
                process_four_u32_bmi(histo_ptr, &mut current_ptr, ptr_end_unroll);
            }
        } else if current_ptr < ptr_end_unroll {
            process_four_u32_generic(histo_ptr, &mut current_ptr, ptr_end_unroll);
        }

        #[cfg(all(target_arch = "x86", feature = "nightly", feature = "std"))]
        if std::is_x86_feature_detected!("bmi1") {
            if current_ptr < ptr_end_unroll {
                process_four_u32_bmi(histo_ptr, &mut current_ptr, ptr_end_unroll);
            }
        } else if current_ptr < ptr_end_unroll {
            process_four_u32_generic(histo_ptr, &mut current_ptr, ptr_end_unroll);
        }

        #[cfg(not(any(
            all(target_arch = "x86_64", feature = "std"),
            all(target_arch = "x86", feature = "nightly", feature = "std")
        )))]
        if current_ptr < ptr_end_unroll {
            process_four_u32_generic(histo_ptr, &mut current_ptr, ptr_end_unroll);
        }

        // Handle remaining bytes that didn't fit in the unrolled loop
        let mut current_ptr = current_ptr as *const u8;
        while current_ptr < ptr_end {
            let byte = *current_ptr;
            current_ptr = current_ptr.add(1);
            *histo_ptr.add(byte as usize) += 1;
        }
    }
}

#[inline(never)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi1")]
unsafe extern "sysv64" fn process_four_u32_bmi(
    histo_ptr: *mut u32,
    values_ptr: &mut *const u32,
    ptr_end_unroll: *const u32,
) {
    core::arch::asm!(
        // Main loop
        "push rbp",
        "2:",
        "mov {eax:e}, [{cur_ptr}]",      // Load first value
        "mov {ebx:e}, [{cur_ptr} + 4]",  // Load second value
        "mov {ecx:e}, [{cur_ptr} + 8]",  // Load third value
        "mov {edx:e}, [{cur_ptr} + 12]", // Load fourth value
        "add {cur_ptr}, 16",               // Advance pointer by 16 bytes

        // Process first value
        "movzx {tmp_e:e}, {eax:l}",
        "movzx ebp, {eax:h}",
        "inc dword ptr [{hist_ptr} + 4*{tmp_e:r}]",
        "bextr {tmp_e:e}, {eax:e}, {bextr_pat:e}",
        "shr {eax:e}, 24",
        "inc dword ptr [{hist_ptr} + 4*rbp]",
        "inc dword ptr [{hist_ptr} + 4*{tmp_e:r}]",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",

        // Process second value
        "movzx {eax:e}, {ebx:l}",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "movzx {eax:e}, {ebx:h}",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "bextr {eax:e}, {ebx:e}, {bextr_pat:e}",
        "shr {ebx:e}, 24",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "inc dword ptr [{hist_ptr} + 4*{ebx:r}]",

        // Process third value
        "movzx {eax:e}, {ecx:l}",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "movzx {eax:e}, {ecx:h}",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "bextr {eax:e}, {ecx:e}, {bextr_pat:e}",
        "shr {ecx:e}, 24",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "inc dword ptr [{hist_ptr} + 4*{ecx:r}]",

        // Process fourth value
        "movzx {eax:e}, {edx:l}",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "movzx {eax:e}, {edx:h}",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "bextr {eax:e}, {edx:e}, {bextr_pat:e}",
        "shr {edx:e}, 24",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "inc dword ptr [{hist_ptr} + 4*{edx:r}]",

        // Loop condition
        "cmp {cur_ptr}, {end_ptr}",
        "jb 2b",
        "pop rbp",

        cur_ptr = inout(reg) *values_ptr,
        hist_ptr = in(reg) histo_ptr,
        end_ptr = in(reg) ptr_end_unroll,
        bextr_pat = in(reg) 2064u32,
        eax = out(reg_abcd) _,
        ebx = out(reg_abcd) _,
        ecx = out(reg_abcd) _,
        edx = out(reg_abcd) _,
        tmp_e = out(reg) _,
        options(nostack)
    );
}

#[cfg(feature = "nightly")]
#[unsafe(naked)]
#[cfg(target_arch = "x86")]
#[target_feature(enable = "bmi1")]
/// From a i686 linux machine with native zen3 target.
unsafe extern "stdcall" fn process_four_u32_bmi(
    histo_ptr: *mut u32,
    values_ptr: &mut *const u32,
    ptr_end_unroll: *const u32,
) {
    core::arch::naked_asm!(
        // Prologue - save registers
        "push ebp",
        "push ebx",
        "push edi",
        "push esi",
        "push eax", // Extra push for temporary storage
        // Initial setup - load pointers
        "mov eax, dword ptr [esp + 28]", // Load values_ptr
        "mov esi, dword ptr [esp + 24]", // Load histo_ptr
        "mov edx, dword ptr [eax]",      // Load current pointer value
        // Ensure 16-byte alignment for the loop
        ".p2align 4, 0x90",
        // Main processing loop
        "2:",
        // Load four 32-bit values
        "mov eax, dword ptr [edx]",      // Load first value
        "mov edi, dword ptr [edx + 12]", // Load fourth value
        "mov ecx, dword ptr [edx + 4]",  // Load second value
        "mov ebx, dword ptr [edx + 8]",  // Load third value
        "add edx, 16",                   // Advance pointer
        // Process first value (in eax)
        "movzx ebp, al",               // Extract low byte
        "mov dword ptr [esp], edi",    // Save fourth value
        "mov edi, 2064",               // bextr pattern
        "inc dword ptr [esi + 4*ebp]", // Update histogram
        "movzx ebp, ah",
        "inc dword ptr [esi + 4*ebp]",
        "bextr ebp, eax, edi",
        "shr eax, 24",
        "inc dword ptr [esi + 4*ebp]",
        "inc dword ptr [esi + 4*eax]",
        // Process second value (in ecx)
        "movzx eax, cl",
        "inc dword ptr [esi + 4*eax]",
        "movzx eax, ch",
        "inc dword ptr [esi + 4*eax]",
        "bextr eax, ecx, edi",
        "shr ecx, 24",
        "inc dword ptr [esi + 4*eax]",
        "inc dword ptr [esi + 4*ecx]",
        // Process third value (in ebx)
        "mov ecx, dword ptr [esp]", // Restore fourth value
        "movzx eax, bl",
        "inc dword ptr [esi + 4*eax]",
        "movzx eax, bh",
        "inc dword ptr [esi + 4*eax]",
        "bextr eax, ebx, edi",
        "shr ebx, 24",
        "inc dword ptr [esi + 4*eax]",
        "inc dword ptr [esi + 4*ebx]",
        // Process fourth value (in ecx)
        "movzx eax, cl",
        "inc dword ptr [esi + 4*eax]",
        "movzx eax, ch",
        "inc dword ptr [esi + 4*eax]",
        "bextr eax, ecx, edi",
        "shr ecx, 24",
        "inc dword ptr [esi + 4*eax]",
        "inc dword ptr [esi + 4*ecx]",
        // Loop control
        "cmp edx, dword ptr [esp + 32]", // Compare with end pointer
        "jb 2b",                         // Loop if not at end
        // Store final pointer
        "mov eax, dword ptr [esp + 28]", // Load values_ptr
        "mov dword ptr [eax], edx",      // Store back final position
        // Epilogue - restore registers and return
        "add esp, 4", // Clean up temporary storage
        "pop esi",
        "pop edi",
        "pop ebx",
        "pop ebp",
        "ret 12", // stdcall return - clean up 12 bytes (3 params * 4 bytes)
    );
}

#[inline(never)] // extern "C" == cdecl
unsafe extern "C" fn process_four_u32_generic(
    histo_ptr: *mut u32,
    values_ptr: &mut *const u32,
    ptr_end_unroll: *const u32,
) {
    while {
        // Read four 32-bit values at once
        let value1 = **values_ptr;
        let value2 = *values_ptr.add(1);
        let value3 = *values_ptr.add(2);
        let value4 = *values_ptr.add(3);

        // Process first value
        *histo_ptr.add((value1 & 0xFF) as usize) += 1;
        *histo_ptr.add(((value1 >> 8) & 0xFF) as usize) += 1;
        *histo_ptr.add(((value1 >> 16) & 0xFF) as usize) += 1;
        *histo_ptr.add((value1 >> 24) as usize) += 1;

        // Process second value
        *histo_ptr.add((value2 & 0xFF) as usize) += 1;
        *histo_ptr.add(((value2 >> 8) & 0xFF) as usize) += 1;
        *histo_ptr.add(((value2 >> 16) & 0xFF) as usize) += 1;
        *histo_ptr.add((value2 >> 24) as usize) += 1;

        // Process third value
        *histo_ptr.add((value3 & 0xFF) as usize) += 1;
        *histo_ptr.add(((value3 >> 8) & 0xFF) as usize) += 1;
        *histo_ptr.add(((value3 >> 16) & 0xFF) as usize) += 1;
        *histo_ptr.add((value3 >> 24) as usize) += 1;

        // Process fourth value
        *histo_ptr.add((value4 & 0xFF) as usize) += 1;
        *histo_ptr.add(((value4 >> 8) & 0xFF) as usize) += 1;
        *histo_ptr.add(((value4 >> 16) & 0xFF) as usize) += 1;
        *histo_ptr.add((value4 >> 24) as usize) += 1;

        *values_ptr = values_ptr.add(4);
        *values_ptr < ptr_end_unroll
    } {}
}

/// Generic, slower version of [`Histogram32`] generation that doesn't assume anything.
/// This is the Rust fallback, reference implementation to run other tests against.
pub(crate) fn histogram32_reference(bytes: &[u8], histogram: &mut Histogram32) {
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
        let mut histogram = Histogram32::default();
        histogram32_reference(&input, &mut histogram);

        // Every value should appear exactly once
        for count in histogram.inner.counter.iter() {
            assert_eq!(*count, 1);
        }
    }
}

#[cfg(test)]
mod alternative_implementation_tests {
    use super::*;
    use crate::histogram::histogram32_private::*;
    use rstest::rstest;
    use std::vec::Vec;

    // Helper function to generate test data
    fn generate_test_data(size: usize) -> Vec<u8> {
        (0..size).map(|i| (i % 256) as u8).collect()
    }

    #[rstest]
    #[case::batched_u32(histogram32_generic_batched_u32)]
    #[case::batched_u64(histogram32_generic_batched_u64)]
    #[case::batched_unroll2_u32(histogram32_generic_batched_unroll_2_u32)]
    #[case::batched_unroll2_u64(histogram32_generic_batched_unroll_2_u64)]
    #[case::batched_unroll4_u32(histogram32_generic_batched_unroll_4_u32)]
    #[case::batched_unroll4_u64(histogram32_generic_batched_unroll_4_u64)]
    #[case::nonaliased_withruns(histogram_nonaliased_withruns_core)]
    fn test_against_reference(#[case] implementation: fn(&[u8], &mut Histogram32)) {
        // Test sizes from 0 to 767 bytes
        for size in 0..=767 {
            let test_data = generate_test_data(size);

            // Get results from both implementations
            let mut implementation_result = Histogram32::default();
            let mut reference_result = Histogram32::default();
            implementation(&test_data, &mut implementation_result);
            histogram32_reference(&test_data, &mut reference_result);

            assert_eq!(
                implementation_result.inner.counter, reference_result.inner.counter,
                "Implementation failed for size {size}"
            );
        }
    }
}
