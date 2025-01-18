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

## Development

For information on how to work with this codebase, see [README-DEV.MD](README-DEV.MD).

## License

Licensed under [MIT](./LICENSE).  

[codecov]: https://about.codecov.io/
[crates-io-key]: https://crates.io/settings/tokens
[nuget-key]: https://www.nuget.org/account/apikeys