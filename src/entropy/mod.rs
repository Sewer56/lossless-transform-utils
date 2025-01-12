//! Entropy calculation module for histograms.
//!
//! This module provides functions to calculate the entropy (average code length) of data
//! represented by histograms. The implementations are tuned for accuracy rather than raw
//! performance, because the histogram only has 256 elements.
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```
//! use lossless_transform_utils::histogram::Histogram32;
//! use lossless_transform_utils::entropy::shannon_entropy;
//!
//! let mut histogram = Histogram32::default();
//! histogram.inner.counter[0] = 3;
//! histogram.inner.counter[1] = 2;
//! histogram.inner.counter[2] = 1;
//!
//! let entropy = shannon_entropy(&histogram, 6);
//! println!("Entropy: {}", entropy);
//! ```
//!
//! # Performance Note
//!
//! While these functions are optimized for small histogram sizes, they may not
//! be the fastest option for high-throughput scenarios. This code was made for a very specific
//! usa case. If performance is critical, consider using another implementation.

use crate::histogram::Histogram32;

/// Calculates the Shannon entropy of a histogram using floating point arithmetic.
/// The entropy is the average number of bits needed to represent each symbol.
///
/// This lets us estimate how compressible the data is during 'entropy coding' steps.
///
/// # Arguments
///
/// * `histogram` - A [Histogram32] containing symbol counts
/// * `total` - The total count of all symbols
///
/// # Returns
///
/// The Shannon entropy in bits. i.e. the average number of bits needed to represent each symbol
///
/// # Example
///
/// ```
/// use lossless_transform_utils::histogram::Histogram32;
/// use lossless_transform_utils::entropy::shannon_entropy;
///
/// let mut histogram = Histogram32::default();
/// histogram.inner.counter[0] = 3;
/// histogram.inner.counter[1] = 2;
/// histogram.inner.counter[2] = 1;
///
/// let entropy = shannon_entropy(&histogram, 6);
/// println!("Entropy: {}", entropy);
/// ```
///
/// # Notes
///
/// - This implementation prioritizes accuracy over performance for small histograms (256 elements).
/// - For high-throughput scenarios, consider using more optimized methods if performance is critical.
#[no_mangle]
pub fn shannon_entropy(histogram: &Histogram32, total: u64) -> f64 {
    // Pseudocode for Shannon Entropy 'the proper way':
    //
    // double entropy = 0.0;
    // for (int i = 0; i < MAX_VALUE; i++) {
    //    if (frequencies[i] > 0) {
    //        double probability = (double)frequencies[i] / length;
    //        entropy -= probability * log2(probability);
    //    }
    // }

    let total = total as f64;
    if histogram.counter.iter().all(|&x| x > 0) {
        shannon_entropy_fast(&histogram.counter, total)
    } else {
        shannon_entropy_slow(&histogram.counter, total)
    }
}

#[inline(never)]
fn shannon_entropy_fast(counter: &[u32; 256], total: f64) -> f64 {
    let mut entropy0 = 0.0;
    let mut entropy1 = 0.0;
    let mut entropy2 = 0.0;
    let mut entropy3 = 0.0;

    for chunk in counter.chunks(4) {
        let p0 = chunk[0] as f64 / total;
        let p1 = chunk[1] as f64 / total;
        let p2 = chunk[2] as f64 / total;
        let p3 = chunk[3] as f64 / total;

        entropy0 -= p0 * p0.log2();
        entropy1 -= p1 * p1.log2();
        entropy2 -= p2 * p2.log2();
        entropy3 -= p3 * p3.log2();
    }

    entropy0 + entropy1 + entropy2 + entropy3
}

#[inline(always)]
fn shannon_entropy_slow(counter: &[u32; 256], total: f64) -> f64 {
    let mut entropy = 0.0;
    for count in counter {
        if *count == 0 {
            continue;
        }
        let probability = *count as f64 / total;
        let entropy_value = probability * probability.log2();
        entropy -= entropy_value;
    }
    entropy
}

/// Calculates the ideal code length in bits for a given histogram.
/// This lets us estimate how compressible the data is during 'entropy coding' steps.
///
/// See [`shannon_entropy`] for more details; this is just a wrapper around that function.
pub fn code_length_no_size(histogram: &Histogram32) -> f64 {
    let total: u64 = histogram.counter.iter().map(|&x| x as u64).sum();
    code_length(histogram, total)
}

/// Calculates the ideal code length in bits for a given histogram.
/// This lets us estimate how compressible the data is during 'entropy coding' steps.
///
/// See [`shannon_entropy`] for more details; this is just a wrapper around that function.
pub fn code_length(histogram: &Histogram32, total: u64) -> f64 {
    shannon_entropy(histogram, total)
}

#[cfg(test)]
mod tests {
    use std::vec::Vec;

    use super::*;
    use crate::histogram::Histogram32;

    #[test]
    fn with_uniform_distribution() {
        // Create a sequence with equal counts of 0,1,2,3: [0,1,2,3]
        let hist = Histogram32::from_bytes(&[0, 1, 2, 3]);
        let total = 4;

        // For uniform distribution of 4 different values, entropy should be 2.0 bits
        // Because log2(4) = 2 bits needed to encode 4 equally likely possibilities
        assert!((shannon_entropy(&hist, total) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn with_single_value() {
        let hist = Histogram32::from_bytes(&[1, 1, 1, 1]);
        let total = 4;

        // For single value distribution, entropy should be 0 bits
        assert!(shannon_entropy(&hist, total).abs() < 1e-10);
    }

    #[test]
    fn with_binary_distribution() {
        let hist = Histogram32::from_bytes(&[0, 0, 1, 1]);
        let total = 4;

        // For 50-50 binary distribution, entropy should be 1.0 bit
        assert!((shannon_entropy(&hist, total) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn with_skewed_distribution() {
        let hist = Histogram32::from_bytes(&[0, 0, 0, 1]);
        let total = 4;

        // For p=[0.75, 0.25] distribution:
        // -0.75*log2(0.75) - 0.25*log2(0.25) ≈ 0.811278124459
        let expected = 0.811278124459;
        assert!((shannon_entropy(&hist, total) - expected).abs() < 1e-10);
    }

    #[test]
    fn with_empty_histogram() {
        let hist = Histogram32::from_bytes(&[]);
        assert!((shannon_entropy(&hist, 0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn code_length_matches_entropy() {
        let hist = Histogram32::from_bytes(&[0, 0, 0, 1]);
        let total = 4;

        assert!((code_length(&hist, total) - shannon_entropy(&hist, total)).abs() < 1e-10);
    }

    #[test]
    fn code_length_no_size_equals_with_size() {
        let hist = Histogram32::from_bytes(&[0, 0, 0, 1]);
        let total = 4;

        assert!((code_length_no_size(&hist) - code_length(&hist, total)).abs() < 1e-10);
    }

    #[test]
    fn fast_path_matches_slow_path() {
        // Generate a large array of non-zero random bytes
        let data: Vec<u8> = (0..10_000_u32)
            .map(|x| (x * 33) as u8) // 1-255 ensures non-zero
            .collect();

        let total = data.len() as u64;
        let hist = Histogram32::from_bytes(&data);

        // Test non-zero case
        let fast = shannon_entropy_fast(&hist.counter, total as f64);
        let slow = shannon_entropy_slow(&hist.counter, total as f64);

        assert!(
            (fast - slow).abs() < 1e-10,
            "Non-zero case mismatch: fast={} slow={}",
            fast,
            slow
        );
    }
}
