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

## Development

For information on how to work with this codebase, see [README-DEV.MD](README-DEV.MD).

## License

Licensed under [MIT](./LICENSE).  

[codecov]: https://about.codecov.io/
[crates-io-key]: https://crates.io/settings/tokens
[nuget-key]: https://www.nuget.org/account/apikeys