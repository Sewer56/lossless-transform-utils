# lossless-transform-utils

[![Crates.io](https://img.shields.io/crates/v/lossless-transform-utils.svg)](https://crates.io/crates/lossless-transform-utils)
[![Docs.rs](https://docs.rs/lossless-transform-utils/badge.svg)](https://docs.rs/lossless-transform-utils)
[![CI](https://github.com/Sewer56/lossless-transform-utils/actions/workflows/rust.yml/badge.svg)](https://github.com/Sewer56/lossless-transform-utils/actions)

## About

General purpose utility methods for creating lossless transforms for various file formats.

## Utilities

This covers the API, a usage example and explanation for how to use the crate can be found
at the bottom of the README file. Additional info for each method is attached to each method
in the source code.

### Histogram

Calculate the amount of times each byte appears in a dataset.

```rust
use lossless_transform_utils::histogram::histogram32_from_bytes;
use lossless_transform_utils::histogram::Histogram32;

let data = [1, 2, 3, 1, 2, 1];
let mut histogram = Histogram32::default();
histogram32_from_bytes(&data, &mut histogram);

assert_eq!(histogram.inner.counter[1], 3); // Byte value 1 appears 3 times
assert_eq!(histogram.inner.counter[2], 2); // Byte value 2 appears 2 times
assert_eq!(histogram.inner.counter[3], 1); // Byte value 3 appears 1 time
```

### Entropy

Calculates the entropy (ideal code length) for a given histogram.

```rust
use lossless_transform_utils::histogram::Histogram32;
use lossless_transform_utils::entropy::code_length_of_histogram32;
use lossless_transform_utils::histogram::histogram32_from_bytes;

let data = [1, 2, 3, 1, 2, 1];
let mut histogram = Histogram32::default();
histogram32_from_bytes(&data, &mut histogram);

let entropy = code_length_of_histogram32(&histogram, data.len() as u64);
println!("Entropy: {}", entropy); // 'entropy' bits per byte
```

This allows you to estimate how compressible data is by entropy coding step of a compressor.
The returned result is average number of bits needed to represent each symbol (1 byte).

i.e. `1.0` == 1 bit.

The code length in this module favours accuracy rather than performance, so it does
proper `f64` arithmetic, with `log2`; which is slow in terms of CPU time.

However, because the input histograms only have 256 elements, the accuracy tradeoff for performance
is considered worthwhile here. The runtime is constant ~900ns per histogram on a 5900X as a point
of reference.

### Match Estimator

Estimates the number of `>=3 byte` LZ matches in a given input array.

```rust
use lossless_transform_utils::match_estimator::estimate_num_lz_matches_fast;

let data = [1, 2, 3, 1, 2, 1];
let num_lz_matches = estimate_num_lz_matches_fast(&data);
println!("Number of LZ matches: {}", num_lz_matches);
```

## Crate Features

- `std` [default]: Enables x86 CPU feature detection.
    - Because x86 feature detection is implemented via CPU instruction, you can use
      the `std` feature in a `no_std` environment. It's just that the API needed
      here isn't available in `no_std` environments.
- `c-exports`: Builds the library with C exports for the public APIs.
- `nightly`: Enables x86 acceleration for `histogram32` creation (requires naked ASM).
- `bench`: Enable benchmarks for non-public API items.

***These exist but are currently unused***:

- `pgo`: Additional workloads for profile guided optimization in the `all` benchmark project.

***These features should not be used***:

- `estimator-avx2`: Uses AVX2 instructions for the match estimator.
    - Don't use this, it's not fast enough, due to a lack of an `AVX2 'scatter'` instruction.
    - Loses a bit of accuracy; at around same speed as scalar.
    - If you're a SIMD wizard with better ideas, feel free to contribute.

***These features are not fully tested***:

- `estimator-avx512`: Uses AVX512 instructions for the match estimator.
    - I (Sewer) don't own AVX512 capable hardware.
    - So this hasn't been fully benchmarked/optimized.
    - It (should) be faster than scalar, however.
    - Tested only in CI.

(If you have an AVX512 machine, please reach out with what perf. results you get!!)

## Reference Performance Numbers

Tested with regular `cargo build`.
R9 5900X, single core, CL16 3200MHz DDR4 RAM.

Histogram Creation:

```ignore
entropy/code_length_of_histogram32/8388608
                        time:   [885.25 ns 886.12 ns 887.10 ns]
                        thrpt:  [1.1273 Melem/s 1.1285 Melem/s 1.1296 Melem/s]

histogram/portable/public-api/8388608
                        time:   [1.1709 ms 1.1720 ms 1.1733 ms]
                        thrpt:  [6.6587 GiB/s 6.6661 GiB/s 6.6723 GiB/s]
```

Match Estimator:

```ignore
match_estimator/random_data/8388608
                        time:   [5.8430 ms 5.8499 ms 5.8586 ms]
                        thrpt:  [1.3335 GiB/s 1.3355 GiB/s 1.3371 GiB/s]
match_estimator/repeated_data_len_1/8388608
                        time:   [5.2753 ms 5.2781 ms 5.2816 ms]
                        thrpt:  [1.4792 GiB/s 1.4802 GiB/s 1.4810 GiB/s]
match_estimator/repeated_data_len_2/8388608
                        time:   [5.3962 ms 5.4068 ms 5.4204 ms]
                        thrpt:  [1.4413 GiB/s 1.4449 GiB/s 1.4478 GiB/s]
match_estimator/repeated_data_len_3/8388608
                        time:   [5.6629 ms 5.6675 ms 5.6726 ms]
                        thrpt:  [1.3772 GiB/s 1.3785 GiB/s 1.3796 GiB/s]
match_estimator/repeated_data_len_4/8388608
                        time:   [6.0370 ms 6.0420 ms 6.0478 ms]
                        thrpt:  [1.2918 GiB/s 1.2930 GiB/s 1.2941 GiB/s]
match_estimator/repeated_data_len_5/8388608
                        time:   [5.5378 ms 5.5467 ms 5.5567 ms]
                        thrpt:  [1.4060 GiB/s 1.4085 GiB/s 1.4108 GiB/s]
match_estimator/repeated_data_len_6/8388608
                        time:   [5.4477 ms 5.4510 ms 5.4547 ms]
                        thrpt:  [1.4323 GiB/s 1.4332 GiB/s 1.4341 GiB/s]
match_estimator/repeated_data_len_7/8388608
                        time:   [5.4951 ms 5.5006 ms 5.5071 ms]
                        thrpt:  [1.4186 GiB/s 1.4203 GiB/s 1.4217 GiB/s]
```

Note: Building with `-C target-cpu=native` may sometimes yield performance improvements on some CPUs,
I've tried to rearrange the code to minimize that as much as possible however.

The combined speed/rate of the operations [e.g. 'splitting data'](#example-splitting-data) for an
8MiB file on a single thread is `1.22GB/s`.

Calculation:

- `histogram`: 1.1720 ms (6.6661 GiB/s)
- `match_estimator`: 5.6675 ms (1.3785GiB/s) 
- `code_length_of_histogram32`: constant `886.12 ns`

8388608 bytes / (1.1720ms + 5.6675ms + 886.12ns) == `1169.525 MiB/s` == `1226.335 MB/s`

To put this into perspective, if you have 1GiB of data, and for each file you test 4 different
splits/variants, your rate (per thread) is 305MB/s. Performance on multi threads has not been
tested, but is expected to scale well, given the workload appears to be CPU bound.

## Implementation Accuracy of the Match Estimator

Can be tested with `cargo test -- --nocapture | grep -i "^\[res:"`

Found Matches at (4K - 64K offsets):

```ignore
[res:matches_4096_intervals_131072] matches: 126298, expected: < 126976, minimum: 113000, found: 99.5%
[res:matches_8192_intervals_131072] matches: 121123, expected: < 122880, minimum: 95000, found: 98.6%
[res:matches_16384_intervals_131072] matches: 112196, expected: < 114688, minimum: 60000, found: 97.8%
[res:matches_32768_intervals_131072] matches: 59506, expected: < 98304, minimum: 13000, found: 60.5%
[res:matches_65536_intervals_131072] matches: 3737, expected: < 65536, minimum: 450, found: 5.7%
```

False Positives Testing:

```ignore
[res:no_matches_128k] matches: 68, expected: < 131, allowed_error: 0.1%, actual_error: 0.052%
[res:no_matches_long_distance_16777215] matches: 11949, expected: < 16777, allowed_error: 0.1%, actual_error: 0.071%
```

Parameters such as hash table sized were tuned for best tradeoff between performance and accuracy.

## How to use this crate

This guide explains how to effectively use the crate to estimate and compare the
compressibility of different data.

[Please note: I (Sewer) am not a compression expert; I just deliver high perf, bleeding edge
libraries for modding games. Consult experts in the field]

### Overview

To determine if one piece of data is more compressible than another, you'll want to look at two key metrics:

1. [Entropy differences](#entropy) 
2. [LZ match count differences](#match-estimator)

These are provided by this crate.

A significant difference in either metric suggests a likely improvement in compression ratio.

### Example: Notes

The parameters `MATCH_RATIO_THRESHOLD` and `ENTROPY_THRESHOLD` below are provided as reference
only, appropriate numbers may vary depending on the nature of the input data.

If you are writing performance-focused transforms, you may want to factor in the number of clock
cycles into the calculations of whether it's worthwhile so transform too.

### Example: Reordering Data

In this case, you have a file composed of structures, for example 16-byte structs with bit
packed values. You want to rearrange the order of the bits to improve the compression ratio.

In this case, LZ match differences are most relevant.

```rust
use lossless_transform_utils::match_estimator::*;

fn should_reorder_data(original: &[u8], reordered: &[u8]) -> bool {
    // Constants for determining significance
    const MATCH_RATIO_THRESHOLD: f64 = 2.0;  // needs 2x more matches

    // Calculate match ratios
    let matches_orig = estimate_num_lz_matches_fast(original) as f64 / original.len() as f64;
    let matches_reordered = estimate_num_lz_matches_fast(reordered) as f64 / reordered.len() as f64;

    matches_reordered > (matches_orig * MATCH_RATIO_THRESHOLD)
}
```

Tip: If you have a struct with bitfields, you can often find compression wins by rearranging
bits such that fields likely to repeat are byte aligned. This is what we're testing for here.

### Example: Splitting Data

When you have different types of data mixed together (e.g. 🍎 mixed with 🍐), you might want to 
split them into separate streams for compression.

You can measure if this is beneficial by comparing both matches and entropy.

```rust
use lossless_transform_utils::{
    entropy::*,
    histogram::*,
    match_estimator::*,
};

// original data: 🍎🍐🍎🍐🍎🍐🍎🍐
// split:
//     part1: 🍎🍎🍎🍎
//     part2: 🍐🍐🍐🍐
fn should_split(part1: &[u8], part2: &[u8]) -> bool {
    // Constants for determining significance
    const MATCH_RATIO_THRESHOLD: f64 = 2.0;  // One input needs 2x more matches
    const ENTROPY_THRESHOLD: f64 = 1.0;      // Or entropy difference >1 bit per byte

    // Calculate match ratios
    let matches1 = estimate_num_lz_matches_fast(part1) as f64 / part1.len() as f64;
    let matches2 = estimate_num_lz_matches_fast(part2) as f64 / part2.len() as f64;

    // Calculate entropies
    let mut hist1 = Histogram32::default();
    let mut hist2 = Histogram32::default();
    histogram32_from_bytes(part1, &mut hist1);
    histogram32_from_bytes(part2, &mut hist2);
    
    let entropy1 = code_length_of_histogram32(&hist1, part1.len() as u64);
    let entropy2 = code_length_of_histogram32(&hist2, part2.len() as u64);

    // Check for significant differences
    let significant_matches = matches1 > (matches2 * MATCH_RATIO_THRESHOLD) || 
                              matches2 > (matches1 * MATCH_RATIO_THRESHOLD);
    let significant_entropy = (entropy1 - entropy2).abs() > ENTROPY_THRESHOLD;

    significant_matches || significant_entropy
}
```

This is similar to what I do for BC7 blocks in my own [dxt-lossless-transform] project (soon).

Tip: Compressors dynamically adapt the entropy state of the data being compressed
periodically as the data is being compressed (by resetting entropy table, etc.). Making smaller
chunks of data within a file more similar therefore improves compression. (Even if entropy of the
file as a whole is unchanged)

### Example: Transforming Data

In this case, you have data that is not being rearranged, but instead being transformed (mutated)
in a reversible way.

```rust
fn encode_delta_diff<T>(data: &mut [T]) 
where 
    T: Copy + std::ops::Sub<Output = T> + Default
{
    let mut prev = T::default(); // Equivalent to 0 for numeric types
    for item in data.iter_mut() {
        let v = *item;
        *item = v - prev;
        prev = v;
    }
}

fn decode_delta_diff<T>(data: &mut [T]) 
where 
    T: Copy + std::ops::Add<Output = T> + Default
{
    let mut prev = T::default();
    for item in data.iter_mut() {
        let v = *item;
        let new_v = prev + v;
        *item = new_v;
        prev = new_v;
    }
}
```

[An example from "Reorder floats + Delta" from Aras' excellent blog][aras-blog], translated to Rust.

More details on linked blog, but essentially, if you're encoding 'differences', then more patterns
may emerge in the data. Longer runs of bytes, or more repeated bytes.

In this case, when you have a 'transformed' and 'non-transformed' variant, you want to follow
the same steps as when [splitting data](#example-splitting-data).

### Example: Estimate File Size

This is a very 'rough', makeshift estimate.

Don't use as estimate of actual file size, but do use with other results from this function.
i.e. You can use this to know if file will be smaller than before; after compression.

```rust
use lossless_transform_utils::{
    entropy::*,
    histogram::*,
    match_estimator::*,
};

/// Estimates the compressed size of data in bytes
/// 
/// # Arguments
/// 
/// * `data` - The input data to estimate compressed size for
/// 
/// # Returns
/// 
/// Estimated size in bytes after compression
pub fn size_estimate(data: &[u8]) -> usize {
    // Estimate number of LZ matches
    let num_matches = estimate_num_lz_matches_fast(data);

    // Calculate expected bytes after LZ
    // Not entirely accurate since we don't factor in the length of each match
    // but it's good enough for comparing against other files. In practice, many
    // matches are short, so this wouldn't even be that far off.
    let bytes_after_lz = data.len() - num_matches;

    // Calculate entropy using histogram
    let mut histogram = Histogram32::default();
    histogram32_from_bytes(data, &mut histogram);
    let bits_per_byte = code_length_of_histogram32(&histogram, data.len() as u64);

    // Calculate expected bits. Result from LZ, 
    // and now with entropy coding (multiply by bits per byte)
    let expected_bits = (bytes_after_lz) as f64 * bits_per_byte;

    // Convert to bytes, rounding up
    expected_bits.ceil() as usize / 8
}
```

## Developer Manual

For step-by-step development guidance, see the [Developer Manual](https://reloaded-project.github.io/reloaded-templates-rust/manual/).

## Contributing

We welcome contributions! See the [Contributing Guide](https://reloaded-project.github.io/reloaded-templates-rust/manual/#contributing) for details.

## License

Licensed under [MIT](./LICENSE).

[dxt-lossless-transform]: https://github.com/Sewer56/dxt-lossless-transform
[aras-blog]: https://aras-p.info/blog/2023/02/01/Float-Compression-3-Filters/