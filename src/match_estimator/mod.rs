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
// On my 5900X with 32K of L1, using a HASH_BITS == 14 == 32K table yields optimal results.
// Oddly, using smaller tables yields in a slowdown; until you get to the tiny tables (<= 8 bits).
//
// # When to change this value?
//
// When *common* CPUs hit the next power of 2 on L1, we'll lift this, but this probably wouldn't
// happen for quite a while.
//
// Fun, semi-related reading: https://en.algorithmica.org/hpc/cpu-cache/associativity/#hardware-caches
const HASH_BITS: usize = 14; // 2^14 = 16k (of u16s) == 32KBytes
const SHIFT_BITS: usize = max(16, HASH_BITS); // Shift at least 16, because it's cheaper.
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
/// match 3 or more bytes at a time. This logic does not detect for 2 byte matches, so certain
/// patterns are not detected.
///
/// For a 2 byte implementation, could bootstrap this:
/// https://github.com/Sewer56/prs-rs/blob/1b20890655ba9a0c652d2095a463e5cd4c35a3f9/src/impls/comp/comp_dict.rs#L20
/// original implementation of mine.
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

            // Use HASH_MASK bits for index into HASH_SIZE table
            let index0 = (h0 & HASH_MASK) as usize;
            let index1 = (h1 & HASH_MASK) as usize;
            let index2 = (h2 & HASH_MASK) as usize;
            let index3 = (h3 & HASH_MASK) as usize;

            // Use next 16 bits for the stored value
            // Because this distribution is fairly uniform (it's a hash),
            // we can really use any bits, just happens that last 16 is easiest.
            let value0 = (h0 >> SHIFT_BITS) as u32;
            let value1 = (h1 >> SHIFT_BITS) as u32;
            let value2 = (h2 >> SHIFT_BITS) as u32;
            let value3 = (h3 >> SHIFT_BITS) as u32;

            // Increment matches if the 16-bit data at the table matches
            // (which indicates a very likely LZ match)
            matches += (hash_table[index0] == value0) as usize;
            matches += (hash_table[index1] == value1) as usize;
            matches += (hash_table[index2] == value2) as usize;
            matches += (hash_table[index3] == value3) as usize;

            // Update the data at the given index.
            hash_table[index0] = value0;
            hash_table[index1] = value1;
            hash_table[index2] = value2;
            hash_table[index3] = value3;

            begin_ptr = begin_ptr.add(4);
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

const fn max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}
