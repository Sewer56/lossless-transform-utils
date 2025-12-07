use criterion::*;
use lossless_transform_utils::match_estimator::estimate_num_lz_matches_fast;

pub(crate) fn get_benchmark_config() -> Criterion {
    Criterion::default()
}

// Main benchmark function
pub fn run_match_estimator_benchmarks(c: &mut Criterion) {
    const SIZE: usize = 8388608;

    let mut group = c.benchmark_group("match_estimator");
    group.throughput(Throughput::Bytes(SIZE as u64));

    {
        // Generate test data - random integers
        let random_data: Vec<u8> = {
            let mut state: u32 = 12345; // seed
            (0..SIZE)
                .map(|_| {
                    // LCG parameters from numerical recipes
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    (state >> 24) as u8 // Take the highest 8 bits
                })
                .collect()
        };

        // Pseudorandom data, with equal distribution. Least possible number of matches.
        group.bench_with_input(
            BenchmarkId::new("random_data", SIZE),
            &random_data,
            |b, random_data| {
                b.iter(|| estimate_num_lz_matches_fast(black_box(random_data)));
            },
        );
    }

    {
        for repeat in 1..8 {
            // Generate test data - repeated integers.
            // 0,1,2,3,4,5,6,7,8,9 etc. [repeat 1] (match every 256)
            // 0,0,1,1,2,2,3,3,4,4 etc. [repeat 2] (match every 512)
            // 0,0,0,0,1,1,1,1,2,2,2,2 etc. [repeat 4] (match every 1024)
            // etc.
            let repeated_bytes: Vec<u8> = (0..SIZE).map(|x| (x / repeat) as u8).collect();
            group.bench_with_input(
                BenchmarkId::new(format!("repeated_data_len_{repeat}"), SIZE),
                &repeated_bytes,
                |b, repeated_data| {
                    b.iter(|| estimate_num_lz_matches_fast(black_box(repeated_data)));
                },
            );
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = get_benchmark_config();
    targets = run_match_estimator_benchmarks
}

criterion_main!(benches);
