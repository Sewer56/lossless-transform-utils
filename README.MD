# lossless-transform-utils

[![Crates.io](https://img.shields.io/crates/v/lossless-transform-utils.svg)](https://crates.io/crates/lossless-transform-utils)
[![Docs.rs](https://docs.rs/lossless-transform-utils/badge.svg)](https://docs.rs/lossless-transform-utils)
[![CI](https://github.com/Sewer56/lossless-transform-utils/actions/workflows/rust.yml/badge.svg)](https://github.com/Sewer56/lossless-transform-utils/actions)

General purpose utility methods for creating lossless transforms for various file formats.

This crate provides tools to estimate data compressibility:
- **Histogram**: Count byte frequencies in data
- **Entropy**: Calculate ideal bits-per-byte (how compressible by entropy coders)
- **Match Estimator**: Estimate LZ match count (how compressible by LZ-based compressors)

Use these to compare different data transformations and determine which yields better compression.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
lossless-transform-utils = "0.1.0"
```

```rust
use lossless_transform_utils::histogram::{histogram32_from_bytes, Histogram32};
use lossless_transform_utils::entropy::code_length_of_histogram32;
use lossless_transform_utils::match_estimator::estimate_num_lz_matches_fast;

// Calculate byte histogram
let data = [1, 2, 3, 1, 2, 1];
let mut histogram = Histogram32::default();
histogram32_from_bytes(&data, &mut histogram);

// Calculate entropy (bits per byte)
let entropy = code_length_of_histogram32(&histogram, data.len() as u64);

// Estimate LZ matches
let num_matches = estimate_num_lz_matches_fast(&data);
```

See the [lossless-transform-utils crate documentation](./src/lossless-transform-utils/README.MD) for detailed usage, feature flags, performance numbers, and comprehensive examples.

## Crates

- [lossless-transform-utils](./src/lossless-transform-utils/README.MD): Core library with histogram, entropy, and LZ match estimation utilities

## Developer Manual

For step-by-step development guidance, see the [Developer Manual](https://reloaded-project.github.io/reloaded-templates-rust/manual/).

## Contributing

We welcome contributions! See the [Contributing Guide](https://reloaded-project.github.io/reloaded-templates-rust/manual/#contributing) for details.

## License

Licensed under [MIT](./LICENSE).
