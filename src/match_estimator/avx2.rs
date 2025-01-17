use super::{calculate_matches_generic, GOLDEN_RATIO, HASH_BITS, HASH_SIZE};
use core::arch::x86_64::*;

#[target_feature(enable = "avx2")]
#[inline(never)]
pub(crate) unsafe fn calculate_matches_avx2(
    hash_table: &mut [u32; HASH_SIZE],
    matches: &mut usize,
    mut begin_ptr: *const u8,
    end_ptr: *const u8,
) {
    let mask_24bit = _mm256_set1_epi32(0x00FFFFFF);
    let golden_ratio = _mm256_set1_epi32(GOLDEN_RATIO as i32);
    let mut indices = [0u32; 32];
    let mut data = [0u32; 32];

    const SHIFT_RIGHT: i32 = 32 - HASH_BITS as i32;

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

        // Count matches using movemask and popcnt
        let mask0 = _mm256_movemask_ps(_mm256_castsi256_ps(eq0)) as u32;
        let mask1 = _mm256_movemask_ps(_mm256_castsi256_ps(eq1)) as u32;
        let mask2 = _mm256_movemask_ps(_mm256_castsi256_ps(eq2)) as u32;
        let mask3 = _mm256_movemask_ps(_mm256_castsi256_ps(eq3)) as u32;

        let match_count =
            mask0.count_ones() + mask1.count_ones() + mask2.count_ones() + mask3.count_ones();
        *matches += match_count as usize;

        // Update hash table entries
        // Unfortunately we still need to do this one by one as there's no scatter in AVX2
        // (only in AVX512)
        _mm256_storeu_si256(indices.as_mut_ptr() as *mut __m256i, idx0);
        _mm256_storeu_si256((indices.as_mut_ptr() as *mut __m256i).add(1), idx1);
        _mm256_storeu_si256((indices.as_mut_ptr() as *mut __m256i).add(2), idx2);
        _mm256_storeu_si256((indices.as_mut_ptr() as *mut __m256i).add(3), idx3);
        _mm256_storeu_si256(data.as_mut_ptr() as *mut __m256i, d0);
        _mm256_storeu_si256((data.as_mut_ptr() as *mut __m256i).add(1), d1);
        _mm256_storeu_si256((data.as_mut_ptr() as *mut __m256i).add(2), d2);
        _mm256_storeu_si256((data.as_mut_ptr() as *mut __m256i).add(3), d3);

        // Update for d0/idx0
        hash_table[indices[0] as usize] = data[0];
        hash_table[indices[1] as usize] = data[1];
        hash_table[indices[2] as usize] = data[2];
        hash_table[indices[3] as usize] = data[3];
        hash_table[indices[4] as usize] = data[4];
        hash_table[indices[5] as usize] = data[5];
        hash_table[indices[6] as usize] = data[6];
        hash_table[indices[7] as usize] = data[7];

        // Update for d1/idx1
        hash_table[indices[8] as usize] = data[8];
        hash_table[indices[9] as usize] = data[9];
        hash_table[indices[10] as usize] = data[10];
        hash_table[indices[11] as usize] = data[11];
        hash_table[indices[12] as usize] = data[12];
        hash_table[indices[13] as usize] = data[13];
        hash_table[indices[14] as usize] = data[14];
        hash_table[indices[15] as usize] = data[15];

        // Update for d2/idx2
        hash_table[indices[16] as usize] = data[16];
        hash_table[indices[17] as usize] = data[17];
        hash_table[indices[18] as usize] = data[18];
        hash_table[indices[19] as usize] = data[19];
        hash_table[indices[20] as usize] = data[20];
        hash_table[indices[21] as usize] = data[21];
        hash_table[indices[22] as usize] = data[22];
        hash_table[indices[23] as usize] = data[23];

        // Update for d3/idx3
        hash_table[indices[24] as usize] = data[24];
        hash_table[indices[25] as usize] = data[25];
        hash_table[indices[26] as usize] = data[26];
        hash_table[indices[27] as usize] = data[27];
        hash_table[indices[28] as usize] = data[28];
        hash_table[indices[29] as usize] = data[29];
        hash_table[indices[30] as usize] = data[30];
        hash_table[indices[31] as usize] = data[31];

        begin_ptr = begin_ptr.add(35);
    }

    // Handle remaining bytes with scalar code
    calculate_matches_generic(hash_table, matches, begin_ptr, end_ptr);
}
