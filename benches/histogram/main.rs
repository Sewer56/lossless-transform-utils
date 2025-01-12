use core::time::Duration;
use criterion::*;
#[cfg(feature = "bench")]
use lossless_transform_utils::histogram::bench::*;
pub use lossless_transform_utils::histogram::*;
use std::fs;

// Payload sizes for benchmarking
pub const PAYLOAD_SIZES: &[usize] = &[
    /*
    64,        // 64 bytes, this seems to be crossover point between reference and batched
    128,       // 128 bytes
    192,       // 192 bytes
    256,       // 256 bytes
    65536,     // 64 KiB
    1048576,   // 1 MiB
    */
    8388608,   // 8 MiB
    178957156, // 170.7MiB
];

// Optional file path for benchmark data
const INPUT_FILE: Option<&str> = None;
// Change this to Some("path/to/file") to use file

// Generate test data of specified size, either from file or synthetic data
pub fn generate_test_data(size: usize) -> Vec<u8> {
    if let Some(path) = INPUT_FILE {
        let file_content = fs::read(path).expect("Failed to read input file");

        if file_content.len() >= size {
            // If file is large enough, take a slice
            file_content[..size].to_vec()
        } else {
            // If file is too small, duplicate its content
            let mut result = Vec::with_capacity(size);
            while result.len() < size {
                let remaining = size - result.len();
                let chunk_size = file_content.len().min(remaining);
                result.extend_from_slice(&file_content[..chunk_size]);
            }
            result
        }
    } else {
        // Original synthetic data generation
        (0..size).map(|i| (i % 256) as u8).collect()
    }
}

// Benchmark group configuration
#[cfg(not(target_os = "windows"))]
use pprof::criterion::{Output, PProfProfiler};

#[cfg(not(target_os = "windows"))]
#[allow(dead_code)]
pub fn get_benchmark_config() -> Criterion {
    Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)))
}

#[cfg(target_os = "windows")]
#[allow(dead_code)]
pub fn get_benchmark_config() -> Criterion {
    Criterion::default()
}

// Main benchmark function
pub fn run_histogram_benchmarks(c: &mut Criterion) {
    #[cfg(not(feature = "bench"))]
    println!("Note: Use the 'bench' feature to enable additional benchmarks");
    println!("Note: You can edit the 'INPUT_FILE' variable to test on a real file.");

    for &size in PAYLOAD_SIZES {
        let mut group = c.benchmark_group("histogram");
        group.throughput(Throughput::Bytes(size as u64));
        group.warm_up_time(Duration::from_secs(30));
        group.measurement_time(Duration::from_secs(30));
        let mut memcpy_buf = vec![0u8; size];

        // Prepare test data
        let data = generate_test_data(size);

        // Public API
        group.bench_with_input(
            BenchmarkId::new("portable/public-api", size),
            &data,
            |b, data| b.iter(|| histogram32_from_bytes(black_box(data))),
        );

        // Benchmark portable implementation
        // Reference impl.
        #[cfg(feature = "bench")]
        group.bench_with_input(
            BenchmarkId::new("portable/reference", size),
            &data,
            |b, data| b.iter(|| histogram32_reference(black_box(data))),
        );

        // Finished warmup, now do shorter tests.
        group.warm_up_time(Duration::from_secs(5));
        group.measurement_time(Duration::from_secs(10));

        // Batched impl.
        #[cfg(feature = "bench")]
        group.bench_with_input(
            BenchmarkId::new("portable/batched_u32", size),
            &data,
            |b, data| b.iter(|| histogram32_generic_batched_u32(black_box(data))),
        );

        #[cfg(feature = "bench")]
        group.bench_with_input(
            BenchmarkId::new("portable/batched_u64", size),
            &data,
            |b, data| b.iter(|| histogram32_generic_batched_u64(black_box(data))),
        );

        // Batched impl (unroll 2)
        #[cfg(feature = "bench")]
        group.bench_with_input(
            BenchmarkId::new("portable/batched/unroll2_u32", size),
            &data,
            |b, data| b.iter(|| histogram32_generic_batched_unroll_2_u32(black_box(data))),
        );

        // Batched impl (unroll 2)
        #[cfg(feature = "bench")]
        group.bench_with_input(
            BenchmarkId::new("portable/batched/unroll2_u64", size),
            &data,
            |b, data| b.iter(|| histogram32_generic_batched_unroll_2_u64(black_box(data))),
        );
        // Batched impl (unroll 4)
        #[cfg(feature = "bench")]
        group.bench_with_input(
            BenchmarkId::new("portable/batched/unroll4_u32", size),
            &data,
            |b, data| b.iter(|| histogram32_generic_batched_unroll_4_u32(black_box(data))),
        );

        // Batched impl (unroll 4)
        #[cfg(feature = "bench")]
        group.bench_with_input(
            BenchmarkId::new("portable/batched/unroll4_u64", size),
            &data,
            |b, data| b.iter(|| histogram32_generic_batched_unroll_4_u64(black_box(data))),
        );

        // NonAliased impl.
        #[cfg(feature = "bench")]
        group.bench_with_input(
            BenchmarkId::new("portable/nonaliased", size),
            &data,
            |b, data| b.iter(|| histogram_nonaliased_withruns_core(black_box(data))),
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
