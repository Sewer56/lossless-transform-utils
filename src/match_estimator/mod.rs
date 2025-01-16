//! Match Estimator
//!
//! This module provides functions for estimating the number of matches in the data, once LZ
//! compression is applied to a given byte array.
use core::alloc::Layout;
use safe_allocator_api::RawAlloc;

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
// when using `u16` as the value type. However, I decided to extend to the `u32` type, to speed up
// computation a little as we can skip a shift and zero extend.
//
// When table fits in L1 cache, doing compares with u32(s) is faster (any data):
// - u16 -> 1.45GiB/s
// - u32 -> 1.77GiB/s
//
// However, because we overrun the L1 cache, we do suffer a speed penalty (HASH_BITS == 14):
// - u32, Random Data -> 1.48GiB/s
// - u32, Repeating Data -> 1.77GiB/s
//
// When data is compressible, penalty is less. When not compressible, it is more.
// This should be fairly natural, as repeating numbers means accessing same slots in hash table,
// making parts of the table rarely accessed.
//
// For exact numbers, run benchmark.
// In any case, it is still faster, AND more accurate, because we compare more bits; it's a win-win.
//
// # When to change this value?
//
// When *common* CPUs hit the next power of 2 on L1, we'll lift this, but this probably wouldn't
// happen for quite a while. That said, the lift to next power of 2 only brings marginal improvements
// accuracy wise.
//
// Fun, semi-related reading: https://en.algorithmica.org/hpc/cpu-cache/associativity/#hardware-caches
// And I found this after writing all this code: https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
const HASH_BITS: usize = 15; // 2^15 = 32k (of u32s) == 128KBytes
const HASH_SIZE: usize = 1 << HASH_BITS;
#[allow(dead_code)]
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
/// match 3 or more bytes at a time.
///
/// Do note that this is an estimator; it is not an exact number; but the number should be accurate-ish
/// given that we use 32-bit hashes (longer than 24-bit source). Think of this as equivalent to a
/// 'fast mode'/low compression level mode.
pub fn estimate_num_lz_matches_fast(bytes: &[u8]) -> usize {
    // This table stores 3 byte hashes, each hash is transformed
    let layout = unsafe { Layout::from_size_align_unchecked(size_of::<u32>() * HASH_SIZE, 64) };
    let mut alloc = RawAlloc::new_zeroed(layout).unwrap();
    let hash_table = unsafe { &mut *(alloc.as_mut_ptr() as *mut [u32; HASH_SIZE]) };

    let mut matches = 0;
    let begin_ptr = bytes.as_ptr();
    unsafe {
        // 7 == (4) u32 match (4 bytes), using hash
        //      +3 bytes for offset
        // We're dropping it, this is an estimation, after all.
        let end_ptr = begin_ptr.add(bytes.len().saturating_sub(7)); // min 0

        // We're doing a little 'trick' here.
        // Because doing a lookup earlier in the buffer is a bit expensive, cache wise, and because
        // this is an estimate, rather than an accurate lookup.

        #[cfg(not(target_arch = "x86_64"))]
        calculate_matches_generic(hash_table, &mut matches, begin_ptr, end_ptr);

        #[cfg(target_arch = "x86_64")]
        calculate_matches_x86_64(hash_table, &mut matches, begin_ptr, end_ptr);
    }

    matches
}

// Generic, for any CPU.
#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
unsafe fn calculate_matches_generic(
    hash_table: &mut [u32; HASH_SIZE],
    matches: &mut usize,
    mut begin_ptr: *const u8,
    end_ptr: *const u8,
) {
    // So we hash the bytes
    while begin_ptr < end_ptr {
        // Note: I had this in nicer form
        // - hash_u32(read_3_byte_le_unaligned(begin_ptr, ofs))
        // But LLVM failed to optimize properly when `target-cpu != native`

        // Get the values at x+0, x+1, x+2, x+3
        let d0 = read_4_byte_le_unaligned(begin_ptr, 0);
        let d1 = read_4_byte_le_unaligned(begin_ptr, 1);
        let d2 = read_4_byte_le_unaligned(begin_ptr, 2);
        let d3 = read_4_byte_le_unaligned(begin_ptr, 3);
        begin_ptr = begin_ptr.add(4);

        // Drop the byte we're not accounting for
        let d0 = reduce_to_3byte(d0);
        let d1 = reduce_to_3byte(d1);
        let d2 = reduce_to_3byte(d2);
        let d3 = reduce_to_3byte(d3);

        // Convert to hashes.
        let h0 = hash_u32(d0);
        let h1 = hash_u32(d1);
        let h2 = hash_u32(d2);
        let h3 = hash_u32(d3);

        // Good reading: https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
        // In this case, I create a multiplicative hash which creates a large 'sea of red'
        // as I call it https://probablydance.com/wp-content/uploads/2018/06/avalanche_fibonacci1.png
        // near the top bits.

        // We get rid of this 'sea of red' by shifting right `32 - HASH_BITS`, and then AND-ing
        // to fit our value in the mask. I'm not sure if it's the article not specifying bit
        // order here, or whether my results are 'backwards',

        // Use HASH_MASK bits for index into HASH_SIZE table
        // Note: We don't need to AND with HASH_MASK because we're only taking upper bits.
        let index0 = (h0 >> (32 - HASH_BITS)) as usize;
        let index1 = (h1 >> (32 - HASH_BITS)) as usize;
        let index2 = (h2 >> (32 - HASH_BITS)) as usize;
        let index3 = (h3 >> (32 - HASH_BITS)) as usize;

        // Increment matches if the 32-bit data at the table matches
        // (which indicates a very likely LZ match)
        *matches += (hash_table[index0] == h0) as usize;
        *matches += (hash_table[index1] == h1) as usize;
        *matches += (hash_table[index2] == h2) as usize;
        *matches += (hash_table[index3] == h3) as usize;

        // Update the data at the given index.
        hash_table[index0] = h0;
        hash_table[index1] = h1;
        hash_table[index2] = h2;
        hash_table[index3] = h3;
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unused_assignments)]
#[inline(never)]
unsafe extern "win64" fn calculate_matches_x86_64(
    hash_table: &mut [u32; HASH_SIZE],
    matches: &mut usize,
    mut begin_ptr: *const u8,
    end_ptr: *const u8,
) {
    // Taken from building `calculate_matches_generic` with `target-cpu = native`
    // and cleaning up the output.
    unsafe {
        std::arch::asm!(
            // Initial comparison
            "cmp {begin_ptr}, {end_ptr}",
            "jae 4f",

            // Load initial values
            "mov {match_count}, qword ptr [{matches}]",

            // Main loop
            ".p2align 5, 0x90",
            "2:",
            // Load values
            "mov {first_hash:e}, dword ptr [{begin_ptr}]",
            "mov {second_hash:e}, dword ptr [{begin_ptr} + 1]",
            "mov {third_hash:e}, dword ptr [{begin_ptr} + 2]",
            "mov {fourth_hash:e}, dword ptr [{begin_ptr} + 3]",
            "add {begin_ptr}, 4",

            // Clear counter and apply masks
            "xor {temp_count}, {temp_count}",
            "and {first_hash:e}, {mask:e}",
            "and {second_hash:e}, {mask:e}",
            "and {third_hash:e}, {mask:e}",
            "and {fourth_hash:e}, {mask:e}",

            // Hash calculations
            "imul {first_hash:e}, {first_hash:e}, -1640531535",
            "imul {second_hash:e}, {second_hash:e}, -1640531535",
            "imul {third_hash:e}, {third_hash:e}, -1640531535",
            "imul {fourth_hash:e}, {fourth_hash:e}, -1640531535",

            // Index calculations
            "mov {first_index:e}, {first_hash:e}",
            "mov {second_index:e}, {second_hash:e}",
            "mov {third_index:e}, {third_hash:e}",
            "mov {fourth_index:e}, {fourth_hash:e}",
            "shr {first_index:e}, 17",
            "shr {second_index:e}, 17",
            "shr {third_index:e}, 17",
            "shr {fourth_index:e}, 17",

            // Compare and count matches
            "cmp dword ptr [{hash_table} + 4*{first_index}], {first_hash:e}",
            "sete {temp_count:l}",
            "add {match_count}, {temp_count}",
            "xor {temp_count}, {temp_count}",

            "cmp dword ptr [{hash_table} + 4*{second_index}], {second_hash:e}",
            "sete {temp_count:l}",
            "add {match_count}, {temp_count}",
            "xor {temp_count}, {temp_count}",

            "cmp dword ptr [{hash_table} + 4*{third_index}], {third_hash:e}",
            "sete {temp_count:l}",
            "add {match_count}, {temp_count}",
            "xor {temp_count}, {temp_count}",

            "cmp dword ptr [{hash_table} + 4*{fourth_index}], {fourth_hash:e}",
            "sete {temp_count:l}",
            "add {match_count}, {temp_count}",

            // Update hash table
            "mov dword ptr [{hash_table} + 4*{first_index}], {first_hash:e}",
            "mov dword ptr [{hash_table} + 4*{second_index}], {second_hash:e}",
            "mov dword ptr [{hash_table} + 4*{third_index}], {third_hash:e}",
            "mov dword ptr [{hash_table} + 4*{fourth_index}], {fourth_hash:e}",

            // Loop condition
            "cmp {begin_ptr}, {end_ptr}",
            "jb 2b",

            // Store final count
            "mov qword ptr [{matches}], {match_count}",
            "4:",

            begin_ptr = inout(reg) begin_ptr,
            end_ptr = in(reg) end_ptr,
            hash_table = in(reg) hash_table.as_mut_ptr(),
            matches = in(reg) matches,
            match_count = out(reg) _,
            first_hash = out(reg) _,
            second_hash = out(reg) _,
            third_hash = out(reg) _,
            fourth_hash = out(reg) _,
            first_index = out(reg) _,
            second_index = out(reg) _,
            third_index = out(reg) _,
            fourth_index = out(reg) _,
            temp_count = out(reg) _,
            mask = in(reg) 16777215,
            options(nostack, pure, readonly)
        );
    }
}

/// Hashes a 32-bit value by multiplying it with the golden ratio,
/// ensuring that it
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn hash_u32(value: u32) -> u32 {
    value.wrapping_mul(GOLDEN_RATIO)
}

/// Reads a 3 byte value from a 32-bit unaligned pointer.
///
/// # Safety
///
/// This function dereferences a raw pointer
#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn read_4_byte_le_unaligned(ptr: *const u8, offset: usize) -> u32 {
    (ptr.add(offset) as *const u32).read_unaligned().to_le()
}

/// Drops the upper 8 bits of a 32-bit value.
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn reduce_to_3byte(value: u32) -> u32 {
    value & 0xFFFFFF
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use core::slice;
    use std::borrow::ToOwned;
    use std::format;
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

    #[rstest]
    #[case(1 << 17, 0.001)] // 128K data (zstd block size), 0.1% error margin
    #[case(16777215, 0.001)] // 16.8MiB data, 0.1% error margin
    fn with_no_repetition_should_have_no_matches(
        #[case] test_size: usize,
        #[case] allowed_error: f32,
    ) {
        let expected = (test_size as f32 * allowed_error) as usize;

        // Use u16 sequence for 128K test, 3-byte sequence for larger tests
        let data = if test_size == 1 << 17 {
            // Create sequence with no repetitions using u16
            let unique: Vec<u16> = (0..u16::MAX).collect();
            cast_u16_slice_to_u8_slice(&unique).to_vec()
        } else {
            // Generate a sequence of all unique 3-byte integers
            // Since the estimator matches for >= 3 bytes, this should ideally return
            // a number as close to 0 as possible.
            generate_unique_3byte_sequence(test_size / 3)
        };

        // There should actually be 0 matches, but there's always going to be a bit of
        // error with hash collisions.
        let matches = estimate_num_lz_matches_fast(&data);
        assert!(
            matches < expected,
            "Sequence with no repetitions should have very few matches, \
             but got {} matches, expected at most {}",
            matches,
            expected
        );
        println!(
            "[res:no_matches_{}] matches: {}, expected: < {}, allowed_error: {:.1}%, actual_error: {:.3}%",
            if test_size == 1 << 17 {
                "128k".to_owned()
            } else {
                format!("long_distance_{}", test_size)
            },
            matches,
            expected,
            allowed_error * 100.0,
            (matches as f32 / test_size as f32) * 100.0
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
    #[case(1 << 17, 1 << 12, 113000)] // 128K size, 4K offset, expect at least 90K matches (found 89.4%)
    #[case(1 << 17, 1 << 13, 95000)] // 128K size, 8K offset, expect at least 90K matches (found 78.3%)
    #[case(1 << 17, 1 << 14, 60000)] // 128K size, 16K offset, expect at least 60K matches (found 53.8%)
    #[case(1 << 17, 1 << 15, 13000)] // 128K size, 32K offset, expect at least 13K matches (found 13.7%)
    #[case(1 << 17, 1 << 16, 450)] // 128K size, 64K offset, expect at least 450 matches (found 0.7%)
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
