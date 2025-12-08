use criterion::*;
pub use lossless_transform_utils::entropy::*;
pub use lossless_transform_utils::histogram::*;
use std::hint::black_box;

// Main benchmark function
pub fn run_entropy_benchmarks(c: &mut Criterion) {
    const SIZE: usize = 8388608;

    let mut group = c.benchmark_group("entropy");
    group.throughput(Throughput::Elements(1));

    // Generate test data - incrementing integers
    let data: Vec<u8> = (0..SIZE).map(|i| i as u8).collect();

    // Create histogram for the test data
    let histogram = Histogram32::from_bytes(&data);

    // The speed of this should be constant; because the input histogram size is constant.
    group.bench_with_input(
        BenchmarkId::new("code_length_of_histogram32", SIZE),
        &histogram,
        |b, hist| {
            b.iter(|| code_length_of_histogram32(black_box(hist), SIZE as u64));
        },
    );

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = run_entropy_benchmarks
}

criterion_main!(benches);
