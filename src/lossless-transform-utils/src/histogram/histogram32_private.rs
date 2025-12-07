//! This contains all implementations that don't ship to the public API, for testing and
//! benchmarking.

use super::*;

const NUM_SLICES: usize = 4;
const SLICE_SIZE_U32S: usize = 256;

/// Based on `histo_asm_scalar8_var5_core` by fabian 'ryg' giesen
/// https://gist.github.com/rygorous/a86a5cf348922cdea357c928e32fc7e0
///
/// # Safety
///
/// This function is safe with any input.
///
/// # Remarks
///
/// For some reason on my AMD 5900X machine this is slower than the `batched` implementation.
/// When experimenting with implementations, I don't (in general) seem to be getting benefits
/// from preventing aliasing.
///
/// The reason may be something related to https://www.agner.org/forum/viewtopic.php?t=41 .
/// I did check the assembly, it's comparable (near identical) to ryg's original.
pub fn histogram_nonaliased_withruns_core(data: &[u8], histogram_result: &mut Histogram32) {
    // 1K on stack, should be good.
    let mut histogram = [Histogram32::default(); NUM_SLICES];

    unsafe {
        let mut ptr = data.as_ptr();
        let end = ptr.add(data.len());
        let current_ptr = histogram[0].inner.counter.as_mut_ptr();

        if data.len() > 24 {
            let aligned_end = end.sub(24);
            let mut current = (ptr as *const u64).read_unaligned();

            while ptr < aligned_end {
                // Prefetch next 1 iteration.
                let next = (ptr.add(8) as *const u64).read_unaligned();

                if current == next {
                    // Check if all bytes are the same within 'current'.

                    // With a XOR, we can check every byte (except byte 0)
                    // with its predecessor. If our value is <256,
                    // then all bytes are the same value.
                    let shifted = current << 8;
                    if (shifted ^ current) < 256 {
                        // All bytes same - increment single bucket by 16
                        // (current is all same byte and current equals next)
                        *current_ptr.add((current & 0xFF) as usize) += 16;
                    } else {
                        // Same 8 bytes twice - sum with INC2
                        sum8(current_ptr, current, 2);
                    }
                } else {
                    // Process both 8-byte chunks with INC1
                    sum8(current_ptr, current, 1);
                    sum8(current_ptr, next, 1);
                }

                current = ((ptr.add(16)) as *const u64).read_unaligned();
                ptr = ptr.add(16);
            }
        }

        while ptr < end {
            let byte = *ptr;
            *current_ptr.add(byte as usize) += 1;
            ptr = ptr.add(1);
        }

        // Sum up all bytes
        // Vectorization-friendly summation, LLVM is good at vectorizing this, so there's no need
        // to write this by hand.
        if NUM_SLICES <= 1 {
            // Copy bytes.
            *histogram_result = histogram[0]
        } else {
            for x in (0..256).step_by(4) {
                let mut sum0 = 0_u32;
                let mut sum1 = 0_u32;
                let mut sum2 = 0_u32;
                let mut sum3 = 0_u32;

                // Changing to suggested code breaks.
                #[allow(clippy::needless_range_loop)]
                for slice in 0..NUM_SLICES {
                    sum0 += histogram[slice].inner.counter[x];
                    sum1 += histogram[slice].inner.counter[x + 1];
                    sum2 += histogram[slice].inner.counter[x + 2];
                    sum3 += histogram[slice].inner.counter[x + 3];
                }

                histogram_result.inner.counter[x] = sum0;
                histogram_result.inner.counter[x + 1] = sum1;
                histogram_result.inner.counter[x + 2] = sum2;
                histogram_result.inner.counter[x + 3] = sum3;
            }
        }
    }
}

#[inline(always)]
unsafe fn sum8(current_ptr: *mut u32, mut value: u64, increment: u32) {
    for index in 0..8 {
        let byte = (value & 0xFF) as usize;
        let slice_offset = (index % NUM_SLICES) * SLICE_SIZE_U32S;
        let write_ptr = current_ptr.add(slice_offset + byte);
        let current = (write_ptr as *const u32).read_unaligned();
        (write_ptr).write_unaligned(current + increment);
        value >>= 8;
    }
}

pub fn histogram32_generic_batched_u32(bytes: &[u8], histogram: &mut Histogram32) {
    unsafe {
        let histo_ptr = histogram.inner.counter.as_mut_ptr();
        let mut current_ptr = bytes.as_ptr() as *const u32;
        let ptr_end = bytes.as_ptr().add(bytes.len());

        // Unroll the loop by fetching `usize` elements at once, then doing a shift.
        // Although there is a data dependency in the shift, this is still generally faster.
        let ptr_end_unroll =
            bytes.as_ptr().add(bytes.len() & !(size_of::<u32>() - 1)) as *const u32;

        while current_ptr < ptr_end_unroll {
            let value = current_ptr.read_unaligned();
            current_ptr = current_ptr.add(1);
            *histo_ptr.add((value & 0xFF) as usize) += 1;
            *histo_ptr.add(((value >> 8) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value >> 16) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value >> 24) & 0xFF) as usize) += 1;
        }

        // Handle any remaining bytes.
        let mut current_ptr = current_ptr as *const u8;
        while current_ptr < ptr_end {
            let byte = *current_ptr;
            current_ptr = current_ptr.add(1);
            *histo_ptr.add(byte as usize) += 1;
        }
    }
}

pub fn histogram32_generic_batched_u64(bytes: &[u8], histogram: &mut Histogram32) {
    // 1K on stack, should be good.
    unsafe {
        let histo_ptr = histogram.inner.counter.as_mut_ptr();
        let mut current_ptr = bytes.as_ptr() as *const u64;
        let ptr_end = bytes.as_ptr().add(bytes.len());

        // Unroll the loop by fetching `usize` elements at once, then doing a shift.
        // Although there is a data dependency in the shift, this is still generally faster.
        let ptr_end_unroll =
            bytes.as_ptr().add(bytes.len() & !(size_of::<u64>() - 1)) as *const u64;

        while current_ptr < ptr_end_unroll {
            let value = current_ptr.read_unaligned();
            current_ptr = current_ptr.add(1);
            *histo_ptr.add((value & 0xFF) as usize) += 1;
            *histo_ptr.add(((value >> 8) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value >> 16) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value >> 24) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value >> 32) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value >> 40) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value >> 48) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value >> 56) & 0xFF) as usize) += 1;
        }

        // Handle any remaining bytes.
        let mut current_ptr = current_ptr as *const u8;
        while current_ptr < ptr_end {
            let byte = *current_ptr;
            current_ptr = current_ptr.add(1);
            *histo_ptr.add(byte as usize) += 1;
        }
    }
}

pub fn histogram32_generic_batched_unroll_2_u64(bytes: &[u8], histogram: &mut Histogram32) {
    unsafe {
        let histo_ptr = histogram.inner.counter.as_mut_ptr();
        let mut current_ptr = bytes.as_ptr() as *const u64;
        let ptr_end = bytes.as_ptr().add(bytes.len());

        // We'll read 2 usize values at a time, so adjust alignment accordingly
        let ptr_end_unroll = bytes
            .as_ptr()
            .add(bytes.len() & !(2 * size_of::<u64>() - 1))
            as *const u64;

        while current_ptr < ptr_end_unroll {
            // Read two 64-bit values at once
            let value1 = current_ptr.read_unaligned();
            let value2 = current_ptr.add(1).read_unaligned();
            current_ptr = current_ptr.add(2);

            // Process first value
            *histo_ptr.add((value1 & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 8) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 16) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 24) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 32) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 40) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 48) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 56) & 0xFF) as usize) += 1;

            // Process second value
            *histo_ptr.add((value2 & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 8) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 16) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 24) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 32) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 40) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 48) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 56) & 0xFF) as usize) += 1;
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

pub fn histogram32_generic_batched_unroll_2_u32(bytes: &[u8], histogram: &mut Histogram32) {
    unsafe {
        let histo_ptr = histogram.inner.counter.as_mut_ptr();
        let mut current_ptr = bytes.as_ptr() as *const u32;
        let ptr_end = bytes.as_ptr().add(bytes.len());

        // We'll read 2 usize values at a time, so adjust alignment accordingly
        let ptr_end_unroll = bytes
            .as_ptr()
            .add(bytes.len() & !(2 * size_of::<u32>() - 1))
            as *const u32;

        while current_ptr < ptr_end_unroll {
            // Read two 32-bit values at once
            let value1 = current_ptr.read_unaligned();
            let value2 = current_ptr.add(1).read_unaligned();
            current_ptr = current_ptr.add(2);

            // Process first value
            *histo_ptr.add((value1 & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 8) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 16) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 24) & 0xFF) as usize) += 1;

            // Process second value
            *histo_ptr.add((value2 & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 8) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 16) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 24) & 0xFF) as usize) += 1;
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

pub fn histogram32_generic_batched_unroll_4_u64(bytes: &[u8], histogram: &mut Histogram32) {
    unsafe {
        let histo_ptr = histogram.inner.counter.as_mut_ptr();
        let mut current_ptr = bytes.as_ptr() as *const u64;
        let ptr_end = bytes.as_ptr().add(bytes.len());

        // We'll read 4 u64 values at a time, so adjust alignment accordingly
        let ptr_end_unroll = bytes
            .as_ptr()
            .add(bytes.len() & !(4 * size_of::<u64>() - 1))
            as *const u64;

        while current_ptr < ptr_end_unroll {
            // Read four 64-bit values at once
            let value1 = current_ptr.read_unaligned();
            let value2 = current_ptr.add(1).read_unaligned();
            let value3 = current_ptr.add(2).read_unaligned();
            let value4 = current_ptr.add(3).read_unaligned();
            current_ptr = current_ptr.add(4);

            // Process first value
            *histo_ptr.add((value1 & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 8) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 16) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 24) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 32) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 40) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 48) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value1 >> 56) & 0xFF) as usize) += 1;

            // Process second value
            *histo_ptr.add((value2 & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 8) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 16) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 24) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 32) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 40) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 48) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value2 >> 56) & 0xFF) as usize) += 1;

            // Process third value
            *histo_ptr.add((value3 & 0xFF) as usize) += 1;
            *histo_ptr.add(((value3 >> 8) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value3 >> 16) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value3 >> 24) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value3 >> 32) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value3 >> 40) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value3 >> 48) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value3 >> 56) & 0xFF) as usize) += 1;

            // Process fourth value
            *histo_ptr.add((value4 & 0xFF) as usize) += 1;
            *histo_ptr.add(((value4 >> 8) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value4 >> 16) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value4 >> 24) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value4 >> 32) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value4 >> 40) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value4 >> 48) & 0xFF) as usize) += 1;
            *histo_ptr.add(((value4 >> 56) & 0xFF) as usize) += 1;
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
