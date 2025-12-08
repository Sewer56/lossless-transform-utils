# lossless-transform-utils

General purpose utility methods for creating lossless transforms for various file formats.

# Project Structure

- `lossless-transform-utils/` - Main library crate
  - `src/` - Library source code
  - `src/exports.rs` - C FFI exports
  - `benches/` - Benchmarks

# Code Guidelines

- Optimize for performance; use zero-cost abstractions, avoid allocations.
- Keep modules under 500 lines (excluding tests); split if larger.
- Place `use` inside functions only for `#[cfg]` conditional compilation.
- Prefer `core` over `std` where possible (`core::mem` over `std::mem`).

# Documentation Standards

- Document public items with `///`
- Add examples in docs where helpful
- Use `//!` for module-level docs
- Focus comments on "why" not "what"
- Use [`TypeName`] rustdoc links, not backticks.

# Post-Change Verification

```bash
cargo test --workspace --all-features
cargo clippy --workspace --all-features -- -D warnings
cargo doc --workspace --all-features
cargo fmt --all
cargo publish --dry-run -p lossless-transform-utils
```

All must pass before submitting.
