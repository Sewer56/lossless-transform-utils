#![doc = include_str!("../README.MD")]
#![no_std]
#![cfg_attr(feature = "nightly", feature(naked_functions))]

#[cfg(feature = "c-exports")]
pub mod exports;

#[cfg(feature = "std")]
extern crate std;

pub mod entropy;
pub mod histogram;
