use super::{calculate_matches_generic, GOLDEN_RATIO, HASH_BITS, HASH_SIZE};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;

#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vl")]
#[inline(never)]
pub(crate) unsafe fn calculate_matches_avx512(
    hash_table: &mut [u32; HASH_SIZE],
    matches: &mut usize,
    mut begin_ptr: *const u8,
    end_ptr: *const u8,
) {
    let mask_24bit = _mm256_set1_epi32(0x00FFFFFF);
    let golden_ratio = _mm256_set1_epi32(GOLDEN_RATIO as i32);

    const SHIFT_RIGHT: i32 = 32 - HASH_BITS as i32;
    let mut matches_accumulator = _mm256_setzero_si256();

    // Process 8 positions at once using AVX2
    while begin_ptr.add(35) <= end_ptr {
        // Load 32 bytes to process 8 positions with unaligned loads
        let bytes0 = _mm256_loadu_si256(begin_ptr as *const __m256i);
        let bytes1 = _mm256_loadu_si256(begin_ptr.add(1) as *const __m256i);
        let bytes2 = _mm256_loadu_si256(begin_ptr.add(2) as *const __m256i);
        let bytes3 = _mm256_loadu_si256(begin_ptr.add(3) as *const __m256i);

        // Mask to 24 bits
        let d0 = _mm256_and_si256(bytes0, mask_24bit);
        let d1 = _mm256_and_si256(bytes1, mask_24bit);
        let d2 = _mm256_and_si256(bytes2, mask_24bit);
        let d3 = _mm256_and_si256(bytes3, mask_24bit);

        // Hash values
        let h0 = _mm256_mullo_epi32(d0, golden_ratio);
        let h1 = _mm256_mullo_epi32(d1, golden_ratio);
        let h2 = _mm256_mullo_epi32(d2, golden_ratio);
        let h3 = _mm256_mullo_epi32(d3, golden_ratio);

        // Calculate hash table indices
        let idx0 = _mm256_srli_epi32(h0, SHIFT_RIGHT);
        let idx1 = _mm256_srli_epi32(h1, SHIFT_RIGHT);
        let idx2 = _mm256_srli_epi32(h2, SHIFT_RIGHT);
        let idx3 = _mm256_srli_epi32(h3, SHIFT_RIGHT);

        // Gather values from hash table using computed indices
        // 4 = stride in bytes
        let table_vals0 = _mm256_i32gather_epi32(hash_table.as_ptr() as *const i32, idx0, 4);
        let table_vals1 = _mm256_i32gather_epi32(hash_table.as_ptr() as *const i32, idx1, 4);
        let table_vals2 = _mm256_i32gather_epi32(hash_table.as_ptr() as *const i32, idx2, 4);
        let table_vals3 = _mm256_i32gather_epi32(hash_table.as_ptr() as *const i32, idx3, 4);

        // Compare values with hash table entries
        let eq0 = _mm256_cmpeq_epi32(d0, table_vals0);
        let eq1 = _mm256_cmpeq_epi32(d1, table_vals1);
        let eq2 = _mm256_cmpeq_epi32(d2, table_vals2);
        let eq3 = _mm256_cmpeq_epi32(d3, table_vals3);

        // Add matches to accumulator
        matches_accumulator = _mm256_sub_epi32(matches_accumulator, eq0);
        matches_accumulator = _mm256_sub_epi32(matches_accumulator, eq1);
        matches_accumulator = _mm256_sub_epi32(matches_accumulator, eq2);
        matches_accumulator = _mm256_sub_epi32(matches_accumulator, eq3);

        // Update hash table entries
        // Unfortunately we still need to do this one by one as there's no scatter in AVX2
        // (only in AVX512)
        _mm256_i32scatter_epi32(hash_table.as_mut_ptr().cast(), idx0, d0, 4);
        _mm256_i32scatter_epi32(hash_table.as_mut_ptr().cast(), idx1, d1, 4);
        _mm256_i32scatter_epi32(hash_table.as_mut_ptr().cast(), idx2, d2, 4);
        _mm256_i32scatter_epi32(hash_table.as_mut_ptr().cast(), idx3, d3, 4);

        begin_ptr = begin_ptr.add(35);
    }

    // Add matches from accumulator to total matches
    let mut match_counts = [0u32; 32];
    _mm256_storeu_si256(
        match_counts.as_mut_ptr() as *mut __m256i,
        matches_accumulator,
    );
    for m in match_counts {
        *matches += m as usize;
    }

    // Handle remaining bytes with scalar code
    calculate_matches_generic(hash_table, matches, begin_ptr, end_ptr);
}
