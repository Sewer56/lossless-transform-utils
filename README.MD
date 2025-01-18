# lossless-transform-utils

[![Crates.io](https://img.shields.io/crates/v/lossless-transform-utils.svg)](https://crates.io/crates/lossless-transform-utils)
[![Docs.rs](https://docs.rs/lossless-transform-utils/badge.svg)](https://docs.rs/lossless-transform-utils)
[![CI](https://github.com/Sewer56/lossless-transform-utils/actions/workflows/rust.yml/badge.svg)](https://github.com/Sewer56/lossless-transform-utils/actions)

## About

General purpose utility methods for creating lossless transforms for various file formats.

## Utilities

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

let entropy = code_length_of_histogram32(&histogram, 6);
println!("Entropy: {}", entropy);
```

This allows you to estimate how compressible data is by entropy coding step of a compressor;
more details in the module readme.

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

Note: Building with `-C target-cpu=native` may sometimes yield performance improvements, I've tried
to rearrange the code to minimize that as much as possible however.

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

## Development

For information on how to work with this codebase, see [README-DEV.MD](README-DEV.MD).

## License

Licensed under [MIT](./LICENSE).  

[codecov]: https://about.codecov.io/
[crates-io-key]: https://crates.io/settings/tokens
[nuget-key]: https://www.nuget.org/account/apikeys