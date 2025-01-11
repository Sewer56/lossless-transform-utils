use super::Histogram;
use core::ops::{Deref, DerefMut};

/// Implementation of a histogram using unsigned 32 bit integers as the counter.
///
/// Max safe array size to pass is 4,294,967,295, naturally, as a result, though in practice
/// it can be a bit bigger.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Histogram32 {
    pub inner: Histogram<u32>,
}

impl Default for Histogram<u32> {
    // Defaults to a zero'd array.
    fn default() -> Self {
        Histogram { counter: [0; 256] }
    }
}

impl Deref for Histogram32 {
    type Target = Histogram<u32>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Histogram32 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Histogram32 {
    /// Calculates a new histogram given
    pub fn from_bytes(bytes: &[u8]) -> Self {
        histogram32_from_bytes(bytes)
    }
}

#[inline]
pub fn histogram32_from_bytes(bytes: &[u8]) -> Histogram32 {
    // Obtained by benching on a 5900X. May vary with different hardware.
    #[allow(clippy::if_same_then_else)]
    if bytes.len() < 64 {
        histogram32_reference(bytes)
    } else {
        histogram32_generic_batched_unroll_4_u32(bytes)
    }
}

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
pub fn histogram_nonaliased_withruns_core(data: &[u8]) -> Histogram32 {
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
            histogram[0]
        } else {
            let mut result = histogram[0];
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

                result.inner.counter[x] = sum0;
                result.inner.counter[x + 1] = sum1;
                result.inner.counter[x + 2] = sum2;
                result.inner.counter[x + 3] = sum3;
            }

            result
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

pub fn histogram32_generic_batched_u32(bytes: &[u8]) -> Histogram32 {
    // 1K on stack, should be good.
    let mut histogram = Histogram32 {
        inner: Histogram { counter: [0; 256] },
    };

    unsafe {
        let histo_ptr = histogram.inner.counter.as_mut_ptr();
        let mut current_ptr = bytes.as_ptr() as *const u32;
        let ptr_end = bytes.as_ptr().add(bytes.len());

        // Unroll the loop by fetching `usize` elements at once, then doing a shift.
        // Although there is a data dependency in the shift, this is still generally faster.
        let ptr_end_unroll =
            bytes.as_ptr().add(bytes.len() & !(size_of::<u32>() - 1)) as *const u32;

        while current_ptr < ptr_end_unroll {
            let value = *current_ptr;
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

    histogram
}

pub fn histogram32_generic_batched_u64(bytes: &[u8]) -> Histogram32 {
    // 1K on stack, should be good.
    let mut histogram = Histogram32 {
        inner: Histogram { counter: [0; 256] },
    };

    unsafe {
        let histo_ptr = histogram.inner.counter.as_mut_ptr();
        let mut current_ptr = bytes.as_ptr() as *const u64;
        let ptr_end = bytes.as_ptr().add(bytes.len());

        // Unroll the loop by fetching `usize` elements at once, then doing a shift.
        // Although there is a data dependency in the shift, this is still generally faster.
        let ptr_end_unroll =
            bytes.as_ptr().add(bytes.len() & !(size_of::<u64>() - 1)) as *const u64;

        while current_ptr < ptr_end_unroll {
            let value = *current_ptr;
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

    histogram
}

pub fn histogram32_generic_batched_unroll_2_u64(bytes: &[u8]) -> Histogram32 {
    let mut histogram = Histogram32 {
        inner: Histogram { counter: [0; 256] },
    };

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
            let value1 = *current_ptr;
            let value2 = *current_ptr.add(1);
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

    histogram
}

pub fn histogram32_generic_batched_unroll_2_u32(bytes: &[u8]) -> Histogram32 {
    let mut histogram = Histogram32 {
        inner: Histogram { counter: [0; 256] },
    };

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
            let value1 = *current_ptr;
            let value2 = *current_ptr.add(1);
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

    histogram
}

#[no_mangle]
pub fn histogram32_generic_batched_unroll_4_u32(bytes: &[u8]) -> Histogram32 {
    let mut histogram = Histogram32 {
        inner: Histogram { counter: [0; 256] },
    };

    if bytes.is_empty() {
        return histogram;
    }

    unsafe {
        let histo_ptr = histogram.inner.counter.as_mut_ptr();
        let mut current_ptr = bytes.as_ptr() as *const u32;
        let ptr_end = bytes.as_ptr().add(bytes.len());

        // We'll read 4 u32 values at a time, so adjust alignment accordingly
        let ptr_end_unroll = bytes
            .as_ptr()
            .add(bytes.len() & !(4 * size_of::<u32>() - 1))
            as *const u32;

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("bmi1") {
            if current_ptr < ptr_end_unroll {
                process_four_u32_bmi(histo_ptr, &mut current_ptr, ptr_end_unroll);
            }
        } else if current_ptr < ptr_end_unroll {
            process_four_u32_generic(histo_ptr, &mut current_ptr, ptr_end_unroll);
        }

        #[cfg(all(target_arch = "x86", feature = "nightly"))]
        if std::is_x86_feature_detected!("bmi1") {
            if current_ptr < ptr_end_unroll {
                process_four_u32_bmi(histo_ptr, &mut current_ptr, ptr_end_unroll);
            }
        } else if current_ptr < ptr_end_unroll {
            process_four_u32_generic(histo_ptr, &mut current_ptr, ptr_end_unroll);
        }

        #[cfg(not(any(target_arch = "x86_64", all(target_arch = "x86", feature = "nightly"))))]
        if current_ptr < ptr_end_unroll {
            process_four_u32_generic(histo_ptr, &mut current_ptr, ptr_end_unroll);
        }

        // Handle remaining bytes that didn't fit in the unrolled loop
        let mut current_ptr = current_ptr as *const u8;
        while current_ptr < ptr_end {
            let byte = *current_ptr;
            current_ptr = current_ptr.add(1);
            *histo_ptr.add(byte as usize) += 1;
        }
    }

    histogram
}

#[inline(never)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi1")]
unsafe fn process_four_u32_bmi(
    histo_ptr: *mut u32,
    values_ptr: &mut *const u32,
    ptr_end_unroll: *const u32,
) {
    std::arch::asm!(
        // Main loop
        "push rbp",
        "2:",
        "mov {eax:e}, [{cur_ptr}]",      // Load first value
        "mov {ebx:e}, [{cur_ptr} + 4]",  // Load second value
        "mov {ecx:e}, [{cur_ptr} + 8]",  // Load third value
        "mov {edx:e}, [{cur_ptr} + 12]", // Load fourth value
        "add {cur_ptr}, 16",               // Advance pointer by 16 bytes

        // Process first value
        "movzx {tmp_e:e}, {eax:l}",
        "movzx ebp, {eax:h}",
        "inc dword ptr [{hist_ptr} + 4*{tmp_e:r}]",
        "bextr {tmp_e:e}, {eax:e}, {bextr_pat:e}",
        "shr {eax:e}, 24",
        "inc dword ptr [{hist_ptr} + 4*rbp]",
        "inc dword ptr [{hist_ptr} + 4*{tmp_e:r}]",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",

        // Process second value
        "movzx {eax:e}, {ebx:l}",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "movzx {eax:e}, {ebx:h}",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "bextr {eax:e}, {ebx:e}, {bextr_pat:e}",
        "shr {ebx:e}, 24",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "inc dword ptr [{hist_ptr} + 4*{ebx:r}]",

        // Process third value
        "movzx {eax:e}, {ecx:l}",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "movzx {eax:e}, {ecx:h}",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "bextr {eax:e}, {ecx:e}, {bextr_pat:e}",
        "shr {ecx:e}, 24",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "inc dword ptr [{hist_ptr} + 4*{ecx:r}]",

        // Process fourth value
        "movzx {eax:e}, {edx:l}",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "movzx {eax:e}, {edx:h}",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "bextr {eax:e}, {edx:e}, {bextr_pat:e}",
        "shr {edx:e}, 24",
        "inc dword ptr [{hist_ptr} + 4*{eax:r}]",
        "inc dword ptr [{hist_ptr} + 4*{edx:r}]",

        // Loop condition
        "cmp {cur_ptr}, {end_ptr}",
        "jb 2b",
        "pop rbp",

        cur_ptr = inout(reg) *values_ptr,
        hist_ptr = in(reg) histo_ptr,
        end_ptr = in(reg) ptr_end_unroll,
        bextr_pat = in(reg) 2064u32,
        eax = out(reg_abcd) _,
        ebx = out(reg_abcd) _,
        ecx = out(reg_abcd) _,
        edx = out(reg_abcd) _,
        tmp_e = out(reg) _,
        options(nostack)
    );
}

#[cfg(feature = "nightly")]
#[naked]
#[cfg(target_arch = "x86")]
#[target_feature(enable = "bmi1")]
/// From a i686 linux machine with native zen3 target.
unsafe extern "stdcall" fn process_four_u32_bmi(
    histo_ptr: *mut u32,
    values_ptr: &mut *const u32,
    ptr_end_unroll: *const u32,
) {
    std::arch::naked_asm!(
        // Prologue - save registers
        "push ebp",
        "push ebx",
        "push edi",
        "push esi",
        "push eax", // Extra push for temporary storage
        // Initial setup - load pointers
        "mov eax, dword ptr [esp + 28]", // Load values_ptr
        "mov esi, dword ptr [esp + 24]", // Load histo_ptr
        "mov edx, dword ptr [eax]",      // Load current pointer value
        // Ensure 16-byte alignment for the loop
        ".p2align 4, 0x90",
        // Main processing loop
        "2:",
        // Load four 32-bit values
        "mov eax, dword ptr [edx]",      // Load first value
        "mov edi, dword ptr [edx + 12]", // Load fourth value
        "mov ecx, dword ptr [edx + 4]",  // Load second value
        "mov ebx, dword ptr [edx + 8]",  // Load third value
        "add edx, 16",                   // Advance pointer
        // Process first value (in eax)
        "movzx ebp, al",               // Extract low byte
        "mov dword ptr [esp], edi",    // Save fourth value
        "mov edi, 2064",               // bextr pattern
        "inc dword ptr [esi + 4*ebp]", // Update histogram
        "movzx ebp, ah",
        "inc dword ptr [esi + 4*ebp]",
        "bextr ebp, eax, edi",
        "shr eax, 24",
        "inc dword ptr [esi + 4*ebp]",
        "inc dword ptr [esi + 4*eax]",
        // Process second value (in ecx)
        "movzx eax, cl",
        "inc dword ptr [esi + 4*eax]",
        "movzx eax, ch",
        "inc dword ptr [esi + 4*eax]",
        "bextr eax, ecx, edi",
        "shr ecx, 24",
        "inc dword ptr [esi + 4*eax]",
        "inc dword ptr [esi + 4*ecx]",
        // Process third value (in ebx)
        "mov ecx, dword ptr [esp]", // Restore fourth value
        "movzx eax, bl",
        "inc dword ptr [esi + 4*eax]",
        "movzx eax, bh",
        "inc dword ptr [esi + 4*eax]",
        "bextr eax, ebx, edi",
        "shr ebx, 24",
        "inc dword ptr [esi + 4*eax]",
        "inc dword ptr [esi + 4*ebx]",
        // Process fourth value (in ecx)
        "movzx eax, cl",
        "inc dword ptr [esi + 4*eax]",
        "movzx eax, ch",
        "inc dword ptr [esi + 4*eax]",
        "bextr eax, ecx, edi",
        "shr ecx, 24",
        "inc dword ptr [esi + 4*eax]",
        "inc dword ptr [esi + 4*ecx]",
        // Loop control
        "cmp edx, dword ptr [esp + 32]", // Compare with end pointer
        "jb 2b",                         // Loop if not at end
        // Store final pointer
        "mov eax, dword ptr [esp + 28]", // Load values_ptr
        "mov dword ptr [eax], edx",      // Store back final position
        // Epilogue - restore registers and return
        "add esp, 4", // Clean up temporary storage
        "pop esi",
        "pop edi",
        "pop ebx",
        "pop ebp",
        "ret 12", // stdcall return - clean up 12 bytes (3 params * 4 bytes)
    );
}

#[inline(never)]
unsafe extern "cdecl" fn process_four_u32_generic(
    histo_ptr: *mut u32,
    values_ptr: &mut *const u32,
    ptr_end_unroll: *const u32,
) {
    while {
        // Read four 32-bit values at once
        let value1 = **values_ptr;
        let value2 = *values_ptr.add(1);
        let value3 = *values_ptr.add(2);
        let value4 = *values_ptr.add(3);

        // Process first value
        *histo_ptr.add((value1 & 0xFF) as usize) += 1;
        *histo_ptr.add(((value1 >> 8) & 0xFF) as usize) += 1;
        *histo_ptr.add(((value1 >> 16) & 0xFF) as usize) += 1;
        *histo_ptr.add((value1 >> 24) as usize) += 1;

        // Process second value
        *histo_ptr.add((value2 & 0xFF) as usize) += 1;
        *histo_ptr.add(((value2 >> 8) & 0xFF) as usize) += 1;
        *histo_ptr.add(((value2 >> 16) & 0xFF) as usize) += 1;
        *histo_ptr.add((value2 >> 24) as usize) += 1;

        // Process third value
        *histo_ptr.add((value3 & 0xFF) as usize) += 1;
        *histo_ptr.add(((value3 >> 8) & 0xFF) as usize) += 1;
        *histo_ptr.add(((value3 >> 16) & 0xFF) as usize) += 1;
        *histo_ptr.add((value3 >> 24) as usize) += 1;

        // Process fourth value
        *histo_ptr.add((value4 & 0xFF) as usize) += 1;
        *histo_ptr.add(((value4 >> 8) & 0xFF) as usize) += 1;
        *histo_ptr.add(((value4 >> 16) & 0xFF) as usize) += 1;
        *histo_ptr.add((value4 >> 24) as usize) += 1;

        *values_ptr = values_ptr.add(4);
        *values_ptr < ptr_end_unroll
    } {}
}

pub fn histogram32_generic_batched_unroll_4_u64(bytes: &[u8]) -> Histogram32 {
    let mut histogram = Histogram32 {
        inner: Histogram { counter: [0; 256] },
    };

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
            let value1 = *current_ptr;
            let value2 = *current_ptr.add(1);
            let value3 = *current_ptr.add(2);
            let value4 = *current_ptr.add(3);
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

    histogram
}

/// Generic, slower version of [`Histogram32`] generation that doesn't assume anything.
/// This is the Rust fallback, reference implementation to run other tests against.
pub fn histogram32_reference(bytes: &[u8]) -> Histogram32 {
    // 1K on stack, should be good.
    let mut histogram = Histogram32 {
        inner: Histogram { counter: [0; 256] },
    };

    let histo_ptr = histogram.inner.counter.as_mut_ptr();
    let mut current_ptr = bytes.as_ptr();
    let ptr_end = unsafe { current_ptr.add(bytes.len()) };

    // Unroll the loop by fetching `usize` elements at once, then doing a shift.
    // Although there is a data dependency in the shift.
    unsafe {
        while current_ptr < ptr_end {
            let byte = *current_ptr;
            current_ptr = current_ptr.add(1);
            *histo_ptr.add(byte as usize) += 1;
        }
    }

    histogram
}

#[cfg(test)]
mod reference_tests {
    use super::*;
    use std::vec::Vec;

    // Creates bytes 0..255, to verify we reach the full range.
    // This should be sufficient for unrolled impl.
    #[test]
    fn verify_full_range_in_reference_impl() {
        let input: Vec<u8> = (0..=255).collect();
        let histogram = histogram32_reference(&input);

        // Every value should appear exactly once
        for count in histogram.inner.counter.iter() {
            assert_eq!(*count, 1);
        }
    }
}

#[cfg(test)]
mod alternative_implementation_tests {
    use super::*;
    use rstest::rstest;
    use std::vec::Vec;

    // Helper function to generate test data
    fn generate_test_data(size: usize) -> Vec<u8> {
        (0..size).map(|i| (i % 256) as u8).collect()
    }

    #[rstest]
    #[case::batched_u32(histogram32_generic_batched_u32)]
    #[case::batched_u64(histogram32_generic_batched_u64)]
    #[case::batched_unroll2_u32(histogram32_generic_batched_unroll_2_u32)]
    #[case::batched_unroll2_u64(histogram32_generic_batched_unroll_2_u64)]
    #[case::batched_unroll4_u32(histogram32_generic_batched_unroll_4_u32)]
    #[case::batched_unroll4_u64(histogram32_generic_batched_unroll_4_u64)]
    #[case::nonaliased_withruns(histogram_nonaliased_withruns_core)]
    fn test_against_reference(#[case] implementation: fn(&[u8]) -> Histogram32) {
        // Test sizes from 0 to 767 bytes
        for size in 0..=767 {
            let test_data = generate_test_data(size);

            // Get results from both implementations
            let implementation_result = implementation(&test_data);
            let reference_result = histogram32_reference(&test_data);

            assert_eq!(
                implementation_result.inner.counter, reference_result.inner.counter,
                "Implementation failed for size {}",
                size
            );
        }
    }
}
