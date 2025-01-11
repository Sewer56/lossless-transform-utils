use crate::histogram::Histogram32;
use std::slice;

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
/// * [data] - Address to the first element
/// * [length] - Length of the array
///
/// # Returns
///
/// Returns a [Histogram32] struct containing the computed histogram.
/// Each element in the histogram represents the count of occurrences for a byte value (0-255).
///
/// # Notes
///
/// - The function is optimized for different input sizes and hardware capabilities.
/// - The threshold for switching between implementations (64 bytes) is based on
///   benchmarks performed on an AMD Ryzen 9 5900X processor. This may vary on different hardware.
///
/// # Safety
///
/// This function assumes the provided address and length are valid.
#[no_mangle]
pub unsafe extern "C" fn histogram32_from_bytes(
    data: *const u8,
    length: usize,
    hist: *mut Histogram32,
) {
    if !data.is_null() && !hist.is_null() {
        let bytes = slice::from_raw_parts(data, length);
        let rust_hist = crate::histogram::histogram32_from_bytes(bytes);
        (*hist).counter = rust_hist.inner.counter;
    }
}

/// Gets the count for a specific byte value from the histogram.
///
/// # Safety
///
/// The caller must ensure `hist` points to a valid `Histogram32` struct.
///
/// # Remarks
///
/// Provided only for completeness, you can access this through the field of [`Histogram32`] directly
/// too.
#[no_mangle]
pub unsafe extern "C" fn histogram32_get_count(hist: *const Histogram32, byte: u8) -> u32 {
    if hist.is_null() {
        return 0;
    }
    (*hist).counter[byte as usize]
}

/// Gets a pointer to the array of counts.
///
/// # Safety
///
/// The caller must ensure `hist` points to a valid `Histogram32` struct.
/// The returned pointer is valid as long as the histogram exists and is not modified.
///
/// # Remarks
///
/// Provided only for completeness, you can access this through the field of [`Histogram32`] directly
/// too.
#[no_mangle]
pub unsafe extern "C" fn histogram32_get_counts(hist: *const Histogram32) -> *const u32 {
    if hist.is_null() {
        return core::ptr::null();
    }
    (*hist).counter.as_ptr()
}
