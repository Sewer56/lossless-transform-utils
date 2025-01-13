//! Match Estimator
//!
//! This module provides functions for estimating the number of matches in the data, once LZ
//! compression is applied to a given byte array.

/// # Golden Ratio constant used for better hash scattering
/// https://softwareengineering.stackexchange.com/a/402543
/// It's a very 'irrational' number, the most, dare I say.
const GOLDEN_RATIO: u32 = 0x9E3779B1_u32;

// These values are based around CPU cache data sizes:
//
// - AMD Ryzen 1000/2000/3000/5000/7000: 32K L1
// - Intel 4790k/6700k/7700k/9900k: 32K L1
//
// Newer CPUs can have slightly larger L1:
// - Intel 11700k: 48K
// - Intel 12700k, 13700k, 14700k, 265k (2024): 48K (PCore), 32K (ECore)
// - AMD Ryzen 9000: 48K
//
// Since this workload can last a longer time, we're going to completely fill up our L1
// cache; hoping that increasing accuracy will give us better results.
//
// # Hardware prefetching of data
//
// Exact info on 'how much to prefetch' is hard to come by, but in the 'Intel Optimization Manual'
// I found this for L3.
//
// > Automatic hardware prefetch can bring cache lines into the unified last-level cache
// > based on prior data misses. It will attempt to prefetch ***two cache lines ahead**** of
// > the prefetch stream.
//
// For L1, this quote may be relevant:
//
// > This prefetcher, also known as the streaming prefetcher, is
// > triggered by an ascending access to very recently loaded data. The processor assumes that this
// > access is part of a streaming algorithm and automatically fetches ***the next line***.
//
// In this case it says (1 line).
// In other words, space contention between read in data and the hash table is not going to be too
// big, perf shouldn't change much.
//
// # What does testing say?
//
// On my 5900X with 32K of L1, using a HASH_BITS == 14 == 32K table (of u16s) yields optimal results
// for the u16 type. However, I decided to extend to the `u32` type, meaning that we use up 64K
// of L1.
//
// When table fits in L1 cache, doing compares with u32(s) is faster (any data):
// - u16 -> 1.45GiB/s
// - u32 -> 1.77GiB/s
//
// This is because we can reuse hash as value, no shift is needed, less data dependency.
//
// However, because we overrun the L1 cache, we do suffer a speed penalty (HASH_BITS == 14):
// - u32, Random Data -> 1.48GiB/s
// - u32, Repeating Data -> 1.77GiB/s
//
// When data is compressible, penalty is less. When not compressible, it is more.
// This should be fairly natural, as repeating numbers means accessing same slots in hash table,
// making parts of the table rarely accessed.
// For exact numbers, run benchmark.
// In any case, it is still faster, AND more accurate, because we compare more bits; it's a win-win.
//
// # When to change this value?
//
// When *common* CPUs hit the next power of 2 on L1, we'll lift this, but this probably wouldn't
// happen for quite a while.
//
// Fun, semi-related reading: https://en.algorithmica.org/hpc/cpu-cache/associativity/#hardware-caches
// And I found this after writing all this code: https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
const HASH_BITS: usize = 14; // 2^14 = 16k (of u32s) == 64KBytes
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: u32 = (HASH_SIZE - 1) as u32;

/// Estimates the number of >=3 byte LZ matches in a given input data stream.
/// This implementation sacrifices a bit of accuracy for speed, i.e. it focuses more on shorter
/// range matches.
///
/// # Arguments
///
/// * `bytes` - The input data stream.
///
/// # Returns
///
/// The estimate number of >=3 byte LZ matches.
/// This number is an estimate, it is not an exact amount.
///
/// # Remarks
///
/// This function is optimized around more modern speedy LZ compressors; namely, those which
/// match 3 or more bytes at a time. This logic can however be adjusted for 2 and 4 byte matches.
/// For those, you would make separate methods; as to keep the mask at read time known at compile time.
///
/// Do note that this is an estimator; it is not an exact number; but the number should be accurate-ish
/// given that we use 32-bit hashes (longer than 24-bit source).
pub fn estimate_num_lz_matches_fast(bytes: &[u8]) -> usize {
    // This table stores 3 byte hashes, each hash is transformed
    // via the hash3 function.
    let mut hash_table = [0u32; HASH_SIZE];
    let mut matches = 0;

    let mut begin_ptr = bytes.as_ptr();
    unsafe {
        // 7 == (4) u32 match (4 bytes), using hash
        //      +3 bytes for offset
        // We're dropping it, this is an estimation, after all.
        let end_ptr = begin_ptr.add(bytes.len().saturating_sub(7)); // min 0

        // We're doing a little 'trick' here.
        // Because doing a lookup earlier in the buffer is a bit expensive, cache wise, and because
        // this is an estimate, rather than an accurate lookup.

        // So we hash the bytes
        while begin_ptr < end_ptr {
            // Get the hashes at x+0, x+1, x+2, x+3
            let h0 = hash_u32(read_3_byte_le_unaligned(begin_ptr, 0));
            let h1 = hash_u32(read_3_byte_le_unaligned(begin_ptr, 1));
            let h2 = hash_u32(read_3_byte_le_unaligned(begin_ptr, 2));
            let h3 = hash_u32(read_3_byte_le_unaligned(begin_ptr, 3));
            begin_ptr = begin_ptr.add(4);

            // Good reading: https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
            // In this case, I create a multiplicative hash which creates a large 'sea of red'
            // as I call it https://probablydance.com/wp-content/uploads/2018/06/avalanche_fibonacci1.png
            // near the top bits.

            // We get rid of this 'sea of red' by shifting right `32 - HASH_BITS`, and then AND-ing
            // to fit our value in the mask. I'm not sure if it's the article not specifying bit
            // order here, or whether my results are 'backwards',

            // Use HASH_MASK bits for index into HASH_SIZE table
            let index0 = ((h0 >> (32 - HASH_BITS)) & HASH_MASK) as usize;
            let index1 = ((h1 >> (32 - HASH_BITS)) & HASH_MASK) as usize;
            let index2 = ((h2 >> (32 - HASH_BITS)) & HASH_MASK) as usize;
            let index3 = ((h3 >> (32 - HASH_BITS)) & HASH_MASK) as usize;

            // Increment matches if the 32-bit data at the table matches
            // (which indicates a very likely LZ match)
            matches += (hash_table[index0] == h0) as usize;
            matches += (hash_table[index1] == h1) as usize;
            matches += (hash_table[index2] == h2) as usize;
            matches += (hash_table[index3] == h3) as usize;

            // Update the data at the given index.
            hash_table[index0] = h0;
            hash_table[index1] = h1;
            hash_table[index2] = h2;
            hash_table[index3] = h3;
        }
    }

    matches
}

/// Hashes a 32-bit value by multiplying it with the golden ratio,
/// ensuring that it
#[inline(always)]
pub fn hash_u32(value: u32) -> u32 {
    value.wrapping_mul(GOLDEN_RATIO)
}

/// Reads a 3 byte value from a 32-bit unaligned pointer.
///
/// # Safety
///
/// This function dereferences a raw pointer
#[inline(always)]
pub unsafe fn read_3_byte_le_unaligned(ptr: *const u8, offset: usize) -> u32 {
    (ptr.add(offset) as *const u32).read_unaligned().to_le() & 0xFFFFFF
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use core::slice;
    use std::vec::Vec;
    use std::{println, vec};

    #[test]
    fn can_hash_u32() {
        // Test that different inputs produce different hashes
        assert_ne!(
            hash_u32(1),
            hash_u32(2),
            "Different inputs should produce different hashes"
        );
    }

    #[test]
    fn can_read_3_byte_le_unaligned() {
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        unsafe {
            let ptr = data.as_ptr();
            assert_eq!(read_3_byte_le_unaligned(ptr, 0), 0x030201);
            assert_eq!(read_3_byte_le_unaligned(ptr, 1), 0x040302);
            assert_eq!(read_3_byte_le_unaligned(ptr, 2), 0x050403);
        }
    }

    #[test]
    fn is_zero_on_empty_input() {
        let empty: Vec<u8> = vec![];
        assert_eq!(
            estimate_num_lz_matches_fast(&empty),
            0,
            "Empty input should return 0 matches"
        );
    }

    #[test]
    fn is_0_on_small_input() {
        // Test input smaller than minimum required length (7 bytes)
        let small = vec![1, 2, 3, 4, 5, 6];
        assert_eq!(
            estimate_num_lz_matches_fast(&small),
            0,
            "Input smaller than 7 bytes should return 0 matches"
        );
    }

    /// A test to assert the quality of the estimate.
    /// This tests the 'error' of our estimation.
    #[test]
    fn with_no_repetition_should_have_no_matches_128k() {
        // Misc note: 128K is a nice range to test, as this is the zstd block size when not in long mode,
        // and a bit larger than Deflate.
        const EXPECTED: usize = ((1 << 17) as f32 * 0.001f32) as usize; // 0.1% margin of error

        // Create sequence with no repetitions
        // 128K of test data. Across 16K (currently) sized hash table of u32s
        let unique: Vec<u16> = (0..u16::MAX).collect();

        // There should actually be 0 matches, but there's a tiny chance for error.
        // If this test trips with 1/2 matches, this is still ok.
        let matches = estimate_num_lz_matches_fast(cast_u16_slice_to_u8_slice(&unique));
        assert!(
            matches < EXPECTED,
            "Sequence with no repetitions should have very few matches"
        );
        println!(
            "[res:no_matches_128k] matches: {}, expected: < {}",
            matches, EXPECTED
        ); // cargo test -- --nocapture | grep -i "^\[res:"
    }

    /// A test to assert the quality of the estimate.
    /// This tests the 'error' of our estimation for false positives.
    #[test]
    fn with_no_repetition_should_have_no_matches_long_distance() {
        const TARGET_BYTES: usize = 16777215; // 16.8MiB
        const EXPECTED: usize = (TARGET_BYTES as f32 * 0.001f32) as usize; // 0.1% margin of error

        // Generate a sequence of all unique 3-byte integers
        // Since the estimator matches for >= 3 bytes, this should ideally return
        // a number as close to 0 as possible.
        let unique = generate_unique_3byte_sequence(16777215 / 3);

        // There should actually be 0 matches, but there's always going to be a bit of
        // error with hash collisions.
        let matches = estimate_num_lz_matches_fast(&unique);
        assert!(
            matches < EXPECTED,
            "Sequence with no repetitions should have very few matches"
        );
        println!(
            "[res:no_matches_long_distance] matches: {}, expected: < {}",
            matches, EXPECTED
        ); // cargo test -- --nocapture | grep -i "^\[res:"
    }

    fn generate_unique_3byte_sequence(length: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(length * 3);

        // Generate sequence of unique 3-byte integers
        for x in 0..length {
            // Extract individual bytes from a 24-bit number
            let b0 = (x & 0xFF) as u8; // Least significant byte
            let b1 = ((x >> 8) & 0xFF) as u8; // Middle byte
            let b2 = ((x >> 16) & 0xFF) as u8; // Most significant byte

            result.push(b0);
            result.push(b1);
            result.push(b2);
        }

        result
    }

    #[rstest]
    #[case(1 << 17, 1 << 13, 95000)] // 128K size, 8K offset, expect at least 90K matches (~21.6% error)
    #[case(1 << 17, 1 << 14, 60000)] // 128K size, 16K offset, expect at least 60K matches (~46.2% error)
    #[case(1 << 17, 1 << 15, 13000)] // 128K size, 32K offset, expect at least 13K matches (~86.3% error)
    #[case(1 << 17, 1 << 16, 450)] // 128K size, 64K offset, expect at least 450 matches (~99.3% error)
    fn estimate_num_lz_matches_at_various_offsets(
        #[case] test_size: usize,
        #[case] match_interval: usize,
        #[case] min_matches: usize,
    ) {
        assert!(
            match_interval <= 1 << 16,
            "Match interval must be <= 64K due to u16 limits"
        );
        assert!(match_interval > 0, "Match interval must be positive");
        assert!(
            test_size >= match_interval,
            "Test size must be >= match interval"
        );
        assert!(
            test_size % 2 == 0 && match_interval % 2 == 0,
            "Test size and match interval must be even due to u16 alignment"
        );

        // Create sequence that repeats every match_interval bytes
        let mut unique: Vec<u16> = Vec::with_capacity(test_size / 2);
        for x in 0..test_size / 2 {
            let val = (x % (match_interval / 2)) as u16;
            unique.push(val);
        }

        // Example of pattern:
        // Bytes 0-16383:     [00 00, 01 00, 02 00, ..., FF 3F]  First cycle
        // Bytes 16384-32767: [00 00, 01 00, 02 00, ..., FF 3F]  Second cycle (matches first)
        // Bytes 32768-49151: [00 00, 01 00, 02 00, ..., FF 3F]  Third cycle
        // And so on...

        let matches = estimate_num_lz_matches_fast(cast_u16_slice_to_u8_slice(&unique));

        // After first match_interval bytes, every position matches with match_interval bytes before it
        let expected = test_size - match_interval;

        // Calculate percentage of matches found vs expected
        let percentage = (matches as f32 / expected as f32) * 100.0;

        // Assert against minimum matches based on empirical results
        assert!(
            matches >= min_matches,
            "Got {} matches, which is below minimum threshold of {}",
            matches,
            min_matches
        );

        println!(
            "[res:matches_{}_intervals_{}] matches: {}, expected: < {}, minimum: {}, found: {:.1}%",
            match_interval, test_size, matches, expected, min_matches, percentage
        ); // cargo test -- --nocapture | grep -i "^\[res:"
    }

    fn cast_u16_slice_to_u8_slice(u16_slice: &[u16]) -> &[u8] {
        let ptr = u16_slice.as_ptr() as *const u8;
        let len = u16_slice.len() * 2; // Each u16 is 2 bytes
        unsafe { slice::from_raw_parts(ptr, len) }
    }
}
