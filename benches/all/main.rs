// benches/histogram_benchmark/mod.rs
use criterion::*;
#[path = "../entropy/main.rs"]
mod entropy;

#[path = "../histogram/main.rs"]
mod histogram;

#[path = "../match_estimator/main.rs"]
mod match_estimator;

// Benchmark group configuration
#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
))]
use pprof::criterion::{Output, PProfProfiler};

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
))]
pub fn get_benchmark_config() -> Criterion {
    Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)))
}

#[cfg(not(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
)))]
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
