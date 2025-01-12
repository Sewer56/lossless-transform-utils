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
/// This function assumes the provided address, length and histogram are valid.
#[no_mangle]
pub unsafe extern "C" fn histogram32_from_bytes(
    data: *const u8,
    length: usize,
    hist: *mut Histogram32,
) {
    let bytes = slice::from_raw_parts(data, length);
    let rust_hist = crate::histogram::histogram32_from_bytes(bytes);
    (*hist).counter = rust_hist.inner.counter;
}

/// Gets the count for a specific byte value from the histogram.
///
/// # Safety
///
/// The caller must ensure `hist` points to a valid [`Histogram32`] struct.
///
/// # Remarks
///
/// Provided only for completeness, you can access this through the field of [`Histogram32`] directly
/// too.
#[no_mangle]
pub unsafe extern "C" fn histogram32_get_count(hist: *const Histogram32, byte: u8) -> u32 {
    (*hist).counter[byte as usize]
}

/// Gets a pointer to the array of counts.
///
/// # Safety
///
/// The caller must ensure `hist` points to a valid [`Histogram32`] struct.
/// The returned pointer is valid as long as the histogram exists and is not modified.
///
/// # Remarks
///
/// Provided only for completeness, you can access this through the field of [`Histogram32`] directly
/// too.
#[no_mangle]
pub unsafe extern "C" fn histogram32_get_counts(hist: *const Histogram32) -> *const u32 {
    (*hist).counter.as_ptr()
}

/// Calculates the Shannon entropy of a histogram using floating point arithmetic.
/// The entropy is the average number of bits needed to represent each symbol.
///
/// This lets us estimate how compressible the data is during 'entropy coding' steps.
///
/// # Arguments
///
/// * `hist` - A pointer to a [Histogram32] containing symbol counts
/// * `total` - The total count of all symbols
///
/// # Returns
///
/// The Shannon entropy in bits. i.e. the average number of bits needed to represent each symbol
///
/// # Safety
///
/// The caller must ensure `hist` points to a valid [`Histogram32`] struct.
/// This API does not validate this, passing a null pointer will crash the program.
///
/// # Notes
///
/// - This implementation prioritizes accuracy over performance for small histograms (256 elements).
/// - For high-throughput scenarios, consider using more optimized methods if performance is critical.
#[no_mangle]
pub unsafe extern "C" fn shannon_entropy_of_histogram32(
    hist: *const Histogram32,
    total: u64,
) -> f64 {
    crate::entropy::shannon_entropy_of_histogram32(&(*hist).counter, total)
}

/// Calculates the ideal code length in bits for a given histogram.
/// This lets us estimate how compressible the data is during 'entropy coding' steps.
///
/// # Arguments
///
/// * `hist` - A pointer to a [Histogram32] containing symbol counts
/// * `total` - The total count of all symbols (total bytes in input data fed to histogram)
///
/// # Returns
///
/// The ideal code length in bits
///
/// # Safety
///
/// The caller must ensure `hist` points to a valid [`Histogram32`] struct.
/// This API does not validate this, passing a null pointer will crash the program.
#[no_mangle]
pub unsafe extern "C" fn code_length_of_histogram32(hist: *const Histogram32, total: u64) -> f64 {
    crate::entropy::code_length_of_histogram32(&(*hist), total)
}

/// Calculates the ideal code length in bits for a given histogram.
/// This lets us estimate how compressible the data is during 'entropy coding' steps.
///
/// Unlike [code_length_of_histogram32], this function calculates the total internally.
///
/// # Arguments
///
/// * `hist` - A pointer to a [Histogram32] containing symbol counts
///
/// # Returns
///
/// The ideal code length in bits
///
/// # Safety
///
/// The caller must ensure `hist` points to a valid [`Histogram32`] struct.
/// This API does not validate this, passing a null pointer will crash the program.
#[no_mangle]
pub unsafe extern "C" fn code_length_of_histogram32_no_size(hist: *const Histogram32) -> f64 {
    crate::entropy::code_length_of_histogram32_no_size(&(*hist))
}
