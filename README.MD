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

let data = [1, 2, 3, 1, 2, 1];
let histogram = histogram32_from_bytes(&data);

assert_eq!(histogram.inner.counter[1], 3); // Byte value 1 appears 3 times
assert_eq!(histogram.inner.counter[2], 2); // Byte value 2 appears 2 times
assert_eq!(histogram.inner.counter[3], 1); // Byte value 3 appears 1 time
```

## Development

For information on how to work with this codebase, see [README-DEV.MD](README-DEV.MD).

## License

Licensed under [MIT](./LICENSE).  

[codecov]: https://about.codecov.io/
[crates-io-key]: https://crates.io/settings/tokens
[nuget-key]: https://www.nuget.org/account/apikeys