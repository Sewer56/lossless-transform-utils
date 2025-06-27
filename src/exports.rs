use crate::{histogram::Histogram32, match_estimator};
use core::slice;

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
/// * `data` - Pointer to the first byte of the input data array
/// * `length` - Number of bytes in the input data array
/// * `hist` - Pointer to a [`Histogram32`] struct that will be populated with the results
///
/// # Returns
///
/// This function does not return a value. The histogram results are written to the
/// [`Histogram32`] struct pointed to by `hist`. Each element in the histogram represents
/// the count of occurrences for a byte value (0-255).
///
/// # Example
///
/// ```c
/// // C code example
/// uint8_t data[] = {1, 2, 3, 1, 2, 1};
/// Histogram32 hist = {0}; // Initialize to zero
/// histogram32_from_bytes(data, sizeof(data), &hist);
/// // hist.inner.counter[1] will now be 3 (byte 1 appears 3 times)
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
/// This function assumes the provided pointers and length are valid:
/// - `data` must point to a valid memory region of at least `length` bytes
/// - `hist` must point to a valid, writable [`Histogram32`] struct
/// - The caller is responsible for ensuring the memory regions don't overlap in undefined ways
#[no_mangle]
pub unsafe extern "C" fn histogram32_from_bytes(
    data: *const u8,
    length: usize,
    hist: *mut Histogram32,
) {
    crate::histogram::histogram32_from_bytes(slice::from_raw_parts(data, length), &mut *hist);
}

/// Gets the count for a specific byte value from the histogram.
///
/// # Arguments
///
/// * `hist` - Pointer to a [`Histogram32`] struct containing the histogram data
/// * `byte` - The byte value (0-255) to get the count for
///
/// # Returns
///
/// The number of times the specified byte value appears in the histogram.
/// Returns 0 if the byte value has not been seen.
///
/// # Example
///
/// ```c
/// // C code example
/// uint8_t data[] = {1, 2, 3, 1, 2, 1};
/// Histogram32 hist = {0};
/// histogram32_from_bytes(data, sizeof(data), &hist);
/// uint32_t count = histogram32_get_count(&hist, 1); // Returns 3
/// ```
///
/// # Safety
///
/// The caller must ensure `hist` points to a valid [`Histogram32`] struct.
/// Passing a null pointer or invalid pointer will result in undefined behavior.
///
/// # Remarks
///
/// Provided only for completeness, you can access this through the field of [`Histogram32`] directly
/// too.
#[no_mangle]
pub unsafe extern "C" fn histogram32_get_count(hist: *const Histogram32, byte: u8) -> u32 {
    (&(*hist)).counter[byte as usize]
}

/// Gets a pointer to the array of counts.
///
/// # Arguments
///
/// * `hist` - Pointer to a [`Histogram32`] struct containing the histogram data
///
/// # Returns
///
/// A pointer to the internal array of 256 `uint32_t` values representing byte occurrence counts.
/// The array is indexed by byte value (0-255), so `counts[42]` gives the count for byte value 42.
/// The returned pointer is valid as long as the histogram exists and is not modified.
///
/// # Example
///
/// ```c
/// // C code example
/// uint8_t data[] = {1, 2, 3, 1, 2, 1};
/// Histogram32 hist = {0};
/// histogram32_from_bytes(data, sizeof(data), &hist);
/// const uint32_t* counts = histogram32_get_counts(&hist);
/// uint32_t count_for_byte_1 = counts[1]; // Returns 3
/// ```
///
/// # Safety
///
/// The caller must ensure `hist` points to a valid [`Histogram32`] struct.
/// Passing a null pointer or invalid pointer will result in undefined behavior.
/// The returned pointer is valid as long as the histogram exists and is not modified.
///
/// # Remarks
///
/// Provided only for completeness, you can access this through the field of [`Histogram32`] directly
/// too.
#[no_mangle]
pub unsafe extern "C" fn histogram32_get_counts(hist: *const Histogram32) -> *const u32 {
    (&(*hist)).counter.as_ptr()
}

/// Calculates the Shannon entropy of a histogram using floating point arithmetic.
/// The entropy is the average number of bits needed to represent each symbol.
///
/// This lets us estimate how compressible the data is during 'entropy coding' steps.
///
/// # Arguments
///
/// * `hist` - A pointer to a [`Histogram32`] containing symbol counts
/// * `total` - The total count of all symbols (should equal the sum of all histogram counts)
///
/// # Returns
///
/// The Shannon entropy in bits. i.e. the average number of bits needed to represent each symbol.
/// Values range from 0.0 (perfectly compressible, single symbol) to 8.0 (maximum entropy, uniform distribution).
///
/// # Example
///
/// ```c
/// // C code example
/// uint8_t data[] = {1, 2, 3, 1, 2, 1}; // 6 bytes total
/// Histogram32 hist = {0};
/// histogram32_from_bytes(data, sizeof(data), &hist);
/// double entropy = shannon_entropy_of_histogram32(&hist, 6);
/// // Returns entropy value, lower means more compressible
/// ```
///
/// # Safety
///
/// The caller must ensure `hist` points to a valid [`Histogram32`] struct.
/// This API does not validate input parameters, passing a null pointer will result in undefined behavior.
/// The `total` parameter should accurately represent the sum of all counts in the histogram.
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
    crate::entropy::shannon_entropy_of_histogram32(&(&(*hist)).counter, total)
}

/// Calculates the ideal code length in bits for a given histogram.
/// This lets us estimate how compressible the data is during 'entropy coding' steps.
///
/// # Arguments
///
/// * `hist` - A pointer to a [`Histogram32`] containing symbol counts
/// * `total` - The total count of all symbols (should equal the sum of all histogram counts)
///
/// # Returns
///
/// The ideal code length in bits. This represents the theoretical minimum number of bits
/// needed to encode the data using optimal entropy coding.
///
/// # Example
///
/// ```c
/// // C code example
/// uint8_t data[] = {1, 2, 3, 1, 2, 1}; // 6 bytes total
/// Histogram32 hist = {0};
/// histogram32_from_bytes(data, sizeof(data), &hist);
/// double code_length = code_length_of_histogram32(&hist, 6);
/// // Returns theoretical minimum bits needed to encode this data
/// ```
///
/// # Safety
///
/// The caller must ensure `hist` points to a valid [`Histogram32`] struct.
/// This API does not validate input parameters, passing a null pointer will result in undefined behavior.
/// The `total` parameter should accurately represent the sum of all counts in the histogram.
#[no_mangle]
pub unsafe extern "C" fn code_length_of_histogram32(hist: *const Histogram32, total: u64) -> f64 {
    crate::entropy::code_length_of_histogram32(&(*hist), total)
}

/// Calculates the ideal code length in bits for a given histogram.
/// This lets us estimate how compressible the data is during 'entropy coding' steps.
///
/// Unlike [`code_length_of_histogram32`], this function calculates the total internally
/// by summing all the counts in the histogram.
///
/// # Arguments
///
/// * `hist` - A pointer to a [`Histogram32`] containing symbol counts
///
/// # Returns
///
/// The ideal code length in bits. This represents the theoretical minimum number of bits
/// needed to encode the data using optimal entropy coding.
///
/// # Example
///
/// ```c
/// // C code example
/// uint8_t data[] = {1, 2, 3, 1, 2, 1};
/// Histogram32 hist = {0};
/// histogram32_from_bytes(data, sizeof(data), &hist);
/// double code_length = code_length_of_histogram32_no_size(&hist);
/// // Automatically calculates total from histogram and returns code length
/// ```
///
/// # Safety
///
/// The caller must ensure `hist` points to a valid [`Histogram32`] struct.
/// This API does not validate input parameters, passing a null pointer will result in undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn code_length_of_histogram32_no_size(hist: *const Histogram32) -> f64 {
    crate::entropy::code_length_of_histogram32_no_size(&(*hist))
}

/// Estimates the number of >=3 byte LZ matches in a given input data stream.
/// This implementation sacrifices a bit of accuracy for speed, i.e. it focuses more on shorter
/// range matches.
///
/// # Arguments
///
/// * `data` - Pointer to the input data stream to analyze
/// * `len` - Length of the input data stream in bytes
///
/// # Returns
///
/// The estimated number of >=3 byte LZ matches that could be found in the data.
/// This number is an estimate, not an exact count.
///
/// # Example
///
/// ```c
/// // C code example
/// uint8_t data[] = "hello world hello world hello";
/// size_t matches = estimate_num_lz_matches_fast(data, strlen((char*)data));
/// // Returns an estimate of repeated sequences of 3+ bytes
/// ```
///
/// # Notes
///
/// This function is optimized around more modern speedy LZ compressors; namely, those which
/// match 3 or more bytes at a time.
///
/// Do note that this is an estimator; it is not an exact number; but the number should be accurate-ish
/// given that we use 32-bit hashes (longer than 24-bit source). Think of this as equivalent to a
/// 'fast mode'/low compression level mode.
///
/// # Safety
///
/// The caller must ensure `data` points to a valid region of memory of at least `len` bytes.
/// This API does not validate input parameters, passing a null pointer or invalid length will
/// result in undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn estimate_num_lz_matches_fast(data: *const u8, len: usize) -> usize {
    match_estimator::estimate_num_lz_matches_fast(slice::from_raw_parts(data, len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::histogram::Histogram32;

    #[test]
    fn test_histogram32_from_bytes() {
        let test_data = [1u8, 2, 3, 1, 2, 1, 0, 255];
        let mut c_histogram = Histogram32::default();
        let mut rust_histogram = Histogram32::default();

        unsafe {
            histogram32_from_bytes(test_data.as_ptr(), test_data.len(), &mut c_histogram);
        }
        crate::histogram::histogram32_from_bytes(&test_data, &mut rust_histogram);

        // Both histograms should be identical
        assert_eq!(c_histogram.counter, rust_histogram.counter);

        // Verify specific counts
        assert_eq!(c_histogram.counter[0], 1); // byte 0 appears once
        assert_eq!(c_histogram.counter[1], 3); // byte 1 appears three times
        assert_eq!(c_histogram.counter[2], 2); // byte 2 appears twice
        assert_eq!(c_histogram.counter[3], 1); // byte 3 appears once
        assert_eq!(c_histogram.counter[255], 1); // byte 255 appears once
    }

    #[test]
    fn test_histogram32_get_count() {
        let test_data = [1u8, 2, 3, 1, 2, 1];
        let mut histogram = Histogram32::default();
        crate::histogram::histogram32_from_bytes(&test_data, &mut histogram);

        unsafe {
            assert_eq!(histogram32_get_count(&histogram, 0), 0);
            assert_eq!(histogram32_get_count(&histogram, 1), 3);
            assert_eq!(histogram32_get_count(&histogram, 2), 2);
            assert_eq!(histogram32_get_count(&histogram, 3), 1);
            assert_eq!(histogram32_get_count(&histogram, 4), 0);
        }
    }

    #[test]
    fn test_histogram32_get_counts() {
        let test_data = [1u8, 2, 3, 1, 2, 1];
        let mut histogram = Histogram32::default();
        crate::histogram::histogram32_from_bytes(&test_data, &mut histogram);

        unsafe {
            let counts_ptr = histogram32_get_counts(&histogram);
            assert!(!counts_ptr.is_null());

            // Verify we can access the counts through the pointer
            assert_eq!(*counts_ptr.add(0), 0);
            assert_eq!(*counts_ptr.add(1), 3);
            assert_eq!(*counts_ptr.add(2), 2);
            assert_eq!(*counts_ptr.add(3), 1);
            assert_eq!(*counts_ptr.add(4), 0);

            // Verify the pointer points to the same data as the original histogram
            assert_eq!(counts_ptr, histogram.counter.as_ptr());
        }
    }

    #[test]
    fn test_shannon_entropy_of_histogram32() {
        let test_data = [1u8, 2, 3, 1, 2, 1]; // 3 ones, 2 twos, 1 three
        let mut histogram = Histogram32::default();
        crate::histogram::histogram32_from_bytes(&test_data, &mut histogram);
        let total = test_data.len() as u64;

        let rust_entropy =
            crate::entropy::shannon_entropy_of_histogram32(&histogram.counter, total);
        let c_entropy = unsafe { shannon_entropy_of_histogram32(&histogram, total) };

        // Should be exactly equal since they use the same implementation
        assert_eq!(rust_entropy, c_entropy);
        assert!(c_entropy > 0.0);
    }

    #[test]
    fn test_code_length_of_histogram32() {
        let test_data = [1u8, 2, 3, 1, 2, 1];
        let mut histogram = Histogram32::default();
        crate::histogram::histogram32_from_bytes(&test_data, &mut histogram);
        let total = test_data.len() as u64;

        let rust_code_length = crate::entropy::code_length_of_histogram32(&histogram, total);
        let c_code_length = unsafe { code_length_of_histogram32(&histogram, total) };

        assert_eq!(rust_code_length, c_code_length);
        assert!(c_code_length > 0.0);
    }

    #[test]
    fn test_code_length_of_histogram32_no_size() {
        let test_data = [1u8, 2, 3, 1, 2, 1];
        let mut histogram = Histogram32::default();
        crate::histogram::histogram32_from_bytes(&test_data, &mut histogram);

        let rust_code_length = crate::entropy::code_length_of_histogram32_no_size(&histogram);
        let c_code_length = unsafe { code_length_of_histogram32_no_size(&histogram) };

        assert_eq!(rust_code_length, c_code_length);
        assert!(c_code_length > 0.0);
    }

    #[test]
    fn test_estimate_num_lz_matches_fast() {
        // Test with data that has some repetition
        let test_data = b"hello world hello world hello";

        let rust_estimate = match_estimator::estimate_num_lz_matches_fast(test_data);
        let c_estimate =
            unsafe { estimate_num_lz_matches_fast(test_data.as_ptr(), test_data.len()) };

        assert_eq!(rust_estimate, c_estimate);
    }

    #[test]
    fn test_estimate_num_lz_matches_fast_empty() {
        let test_data = b"";

        let rust_estimate = match_estimator::estimate_num_lz_matches_fast(test_data);
        let c_estimate =
            unsafe { estimate_num_lz_matches_fast(test_data.as_ptr(), test_data.len()) };

        assert_eq!(rust_estimate, c_estimate);
        assert_eq!(c_estimate, 0);
    }

    #[test]
    fn test_estimate_num_lz_matches_fast_small() {
        let test_data = b"ab";

        let rust_estimate = match_estimator::estimate_num_lz_matches_fast(test_data);
        let c_estimate =
            unsafe { estimate_num_lz_matches_fast(test_data.as_ptr(), test_data.len()) };

        assert_eq!(rust_estimate, c_estimate);
    }

    #[test]
    fn test_histogram_with_empty_data() {
        let test_data: &[u8] = &[];
        let mut c_histogram = Histogram32::default();
        let mut rust_histogram = Histogram32::default();

        unsafe {
            histogram32_from_bytes(test_data.as_ptr(), test_data.len(), &mut c_histogram);
        }
        crate::histogram::histogram32_from_bytes(test_data, &mut rust_histogram);

        assert_eq!(c_histogram.counter, rust_histogram.counter);

        // All counts should be zero
        for count in c_histogram.counter.iter() {
            assert_eq!(*count, 0);
        }
    }

    #[test]
    fn test_histogram_with_all_same_byte() {
        let test_data = [42u8; 100];
        let mut c_histogram = Histogram32::default();
        let mut rust_histogram = Histogram32::default();

        unsafe {
            histogram32_from_bytes(test_data.as_ptr(), test_data.len(), &mut c_histogram);
        }
        crate::histogram::histogram32_from_bytes(&test_data, &mut rust_histogram);

        assert_eq!(c_histogram.counter, rust_histogram.counter);
        assert_eq!(c_histogram.counter[42], 100);

        // All other counts should be zero
        for (i, count) in c_histogram.counter.iter().enumerate() {
            if i == 42 {
                assert_eq!(*count, 100);
            } else {
                assert_eq!(*count, 0);
            }
        }
    }

    #[test]
    fn test_histogram_with_full_byte_range() {
        let test_data: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut c_histogram = Histogram32::default();
        let mut rust_histogram = Histogram32::default();

        unsafe {
            histogram32_from_bytes(test_data.as_ptr(), test_data.len(), &mut c_histogram);
        }
        crate::histogram::histogram32_from_bytes(&test_data, &mut rust_histogram);

        assert_eq!(c_histogram.counter, rust_histogram.counter);

        // Every byte should appear exactly once
        for count in c_histogram.counter.iter() {
            assert_eq!(*count, 1);
        }
    }

    #[test]
    fn test_entropy_edge_cases() {
        // Test with uniform distribution
        let uniform_data: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut histogram = Histogram32::default();
        crate::histogram::histogram32_from_bytes(&uniform_data, &mut histogram);
        let total = uniform_data.len() as u64;

        let entropy = unsafe { shannon_entropy_of_histogram32(&histogram, total) };

        // Uniform distribution should have entropy close to 8 bits
        assert!((entropy - 8.0).abs() < 0.001);

        // Test with single symbol (minimum entropy)
        let single_symbol_data = [42u8; 100];
        let mut single_histogram = Histogram32::default();
        crate::histogram::histogram32_from_bytes(&single_symbol_data, &mut single_histogram);
        let single_total = single_symbol_data.len() as u64;

        let single_entropy =
            unsafe { shannon_entropy_of_histogram32(&single_histogram, single_total) };

        // Single symbol should have entropy of 0
        assert_eq!(single_entropy, 0.0);
    }
}
