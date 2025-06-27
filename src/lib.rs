#![doc = include_str!("../README.MD")]
#![no_std]
#![cfg_attr(feature = "nightly", feature(naked_functions))]
#![cfg_attr(feature = "estimator-avx512", feature(stdarch_x86_avx512))]
#![cfg_attr(feature = "estimator-avx512", feature(avx512_target_feature))]
#![allow(stable_features)]
#![cfg_attr(
    all(target_arch = "x86", feature = "nightly"),
    feature(naked_functions_target_feature)
)]

#[cfg(feature = "c-exports")]
pub mod exports;

#[cfg(feature = "std")]
extern crate std;

pub mod entropy;
pub mod histogram;
pub mod match_estimator;
