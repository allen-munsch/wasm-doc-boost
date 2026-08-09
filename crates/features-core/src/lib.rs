#![no_std]

extern crate alloc;

use alloc::vec::Vec;

/// Convert RGB pixels to grayscale f64 values (BT.601 luminance).
/// This is THE single grayscale conversion — all modules share it.
pub fn to_grayscale(pixels: &[u8], width: usize, height: usize) -> Vec<f64> {
    let n = width * height;
    let mut gray = Vec::with_capacity(n);
    for i in 0..n {
        let r = pixels[i * 3] as f64;
        let g = pixels[i * 3 + 1] as f64;
        let b = pixels[i * 3 + 2] as f64;
        gray.push(0.299 * r + 0.587 * g + 0.114 * b);
    }
    gray
}

/// Get grayscale pixel value at (x, y), clamping to border.
#[inline]
pub fn pixel_at(gray: &[f64], width: usize, _height: usize, x: isize, y: isize) -> f64 {
    let x = x.clamp(0, width as isize - 1) as usize;
    let y = y.clamp(0, _height as isize - 1) as usize;
    gray[y * width + x]
}

pub mod color;
pub mod crumple;
pub mod document;
pub mod edges;
pub mod ink;
pub mod noise;
pub mod projection;
pub mod shadow;
pub mod texture;
