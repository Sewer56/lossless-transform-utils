// benches/histogram_benchmark/mod.rs
pub mod portable;
use criterion::*;

// Payload sizes for benchmarking
pub const PAYLOAD_SIZES: &[usize] = &[
    64,        // 64 bytes, this seems to be crossover point between reference and batched
    128,       // 128 bytes
    192,       // 192 bytes
    256,       // 256 bytes
    65536,     // 64 KiB
    1048576,   // 1 MiB
    8388608,   // 8 MiB
    178957156, // 170.7MiB
];

// Generate test data of specified size
pub fn generate_test_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

// Benchmark group configuration
#[cfg(not(target_os = "windows"))]
use pprof::criterion::{Output, PProfProfiler};

#[cfg(not(target_os = "windows"))]
pub fn get_benchmark_config() -> Criterion {
    Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)))
}

#[cfg(target_os = "windows")]
pub fn get_benchmark_config() -> Criterion {
    Criterion::default()
}

// Main benchmark function
pub fn run_histogram_benchmarks(c: &mut Criterion) {
    for &size in PAYLOAD_SIZES {
        let mut group = c.benchmark_group("histogram");
        group.throughput(Throughput::Bytes(size as u64));
        let mut memcpy_buf = vec![0u8; size];

        // Prepare test data
        let data = generate_test_data(size);

        // Benchmark portable implementation
        // Reference impl.
        group.bench_with_input(
            BenchmarkId::new("portable/reference", size),
            &data,
            |b, data| b.iter(|| portable::histogram32_from_bytes_reference(black_box(data))),
        );

        // Batched impl.
        group.bench_with_input(
            BenchmarkId::new("portable/batched", size),
            &data,
            |b, data| b.iter(|| portable::histogram32_from_bytes_generic_batched(black_box(data))),
        );

        // NonAliased impl.
        group.bench_with_input(
            BenchmarkId::new("portable/nonaliased", size),
            &data,
            |b, data| b.iter(|| portable::histogram_nonaliased_withruns_core(black_box(data))),
        );

        // Memcpy
        group.bench_with_input(
            BenchmarkId::new("portable/memcpy", size),
            &data,
            |b, data| b.iter(|| memcpy_buf.copy_from_slice(data)),
        );

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = get_benchmark_config();
    targets = run_histogram_benchmarks
}

criterion_main!(benches);
