#![doc = include_str!("../README.MD")]
#![no_std]
#[cfg(feature = "c-exports")]
pub mod exports;

#[cfg(feature = "std")]
extern crate std;

pub mod histogram;
