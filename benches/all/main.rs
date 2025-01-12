// benches/histogram_benchmark/mod.rs
use criterion::*;
#[path = "../entropy/main.rs"]
mod entropy;

#[path = "../histogram/main.rs"]
mod histogram;

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
pub fn run_all_benchmarks(c: &mut Criterion) {
    entropy::run_entropy_benchmarks(c);
    histogram::run_histogram_benchmarks(c);
}

criterion_group! {
    name = benches;
    config = get_benchmark_config();
    targets = run_all_benchmarks
}

criterion_main!(benches);