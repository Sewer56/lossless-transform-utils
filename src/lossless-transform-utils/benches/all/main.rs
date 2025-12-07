// benches/histogram_benchmark/mod.rs
use criterion::*;
#[path = "../entropy/main.rs"]
mod entropy;

#[path = "../histogram/main.rs"]
mod histogram;

#[path = "../match_estimator/main.rs"]
mod match_estimator;

pub fn get_benchmark_config() -> Criterion {
    Criterion::default()
}

// Main benchmark function
pub fn run_all_benchmarks(c: &mut Criterion) {
    entropy::run_entropy_benchmarks(c);
    histogram::run_histogram_benchmarks(c);
    match_estimator::run_match_estimator_benchmarks(c);
}

criterion_group! {
    name = benches;
    config = get_benchmark_config();
    targets = run_all_benchmarks
}

criterion_main!(benches);
