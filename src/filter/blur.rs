// ══════════════════════════════════════════════════════════════════════════
//  EFV · Gaussian Blur  –  2-pass separable kernel, rayon-parallelised
// ══════════════════════════════════════════════════════════════════════════

use pyo3::{prelude::*, exceptions::PyValueError};
use rayon::prelude::*;
use crate::frame::Frame;

/// Apply a separable Gaussian blur with the given radius (sigma ≈ radius/3).
///
/// `radius` is the half-kernel size; kernel width = 2*radius+1.
#[pyfunction]
#[pyo3(signature = (frame, radius))]
pub fn py_gaussian_blur(frame: &Frame, radius: u32) -> PyResult<Frame> {
    if radius == 0 {
        return Ok(frame.copy());
    }
    if radius > 64 {
        return Err(PyValueError::new_err("blur radius must be ≤ 64"));
    }
    Ok(gaussian_blur(frame, radius))
}

pub fn gaussian_blur(src: &Frame, radius: u32) -> Frame {
    let kernel = build_gaussian_kernel(radius);
    let h_pass = blur_horizontal(src, &kernel, radius);
    blur_vertical(&h_pass, &kernel, radius)
}

// ── Kernel ────────────────────────────────────────────────────────────────

fn build_gaussian_kernel(radius: u32) -> Vec<f32> {
    let size   = (radius * 2 + 1) as usize;
    let sigma  = radius as f32 / 3.0_f32.max(1.0);
    let s2     = 2.0 * sigma * sigma;
    let mut k: Vec<f32> = (0..size)
        .map(|i| {
            let x = i as f32 - radius as f32;
            (-x * x / s2).exp()
        })
        .collect();
    let sum: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    k
}

// ── Horizontal pass ───────────────────────────────────────────────────────

fn blur_horizontal(src: &Frame, kernel: &[f32], radius: u32) -> Frame {
    let w = src.width  as usize;
    let h = src.height as usize;
    let r = radius as isize;

    let mut buf = vec![0u8; w * h * 4];

    buf.par_chunks_mut(w * 4)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                let mut acc = [0f32; 4];
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sx = (x as isize + ki as isize - r).clamp(0, w as isize - 1) as usize;
                    let si = (y * w + sx) * 4;
                    for c in 0..4 {
                        acc[c] += src.buf[si + c] as f32 * kv;
                    }
                }
                let di = x * 4;
                for c in 0..4 {
                    row[di + c] = acc[c].clamp(0.0, 255.0) as u8;
                }
            }
        });

    Frame { width: src.width, height: src.height, buf }
}

// ── Vertical pass ─────────────────────────────────────────────────────────

fn blur_vertical(src: &Frame, kernel: &[f32], radius: u32) -> Frame {
    let w = src.width  as usize;
    let h = src.height as usize;
    let r = radius as isize;

    let mut buf = vec![0u8; w * h * 4];

    // We process column-striped chunks to maintain cache efficiency.
    // Each output pixel (x, y) reads a column of src.
    buf.par_chunks_mut(w * 4)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                let mut acc = [0f32; 4];
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sy = (y as isize + ki as isize - r).clamp(0, h as isize - 1) as usize;
                    let si = (sy * w + x) * 4;
                    for c in 0..4 {
                        acc[c] += src.buf[si + c] as f32 * kv;
                    }
                }
                let di = x * 4;
                for c in 0..4 {
                    row[di + c] = acc[c].clamp(0.0, 255.0) as u8;
                }
            }
        });

    Frame { width: src.width, height: src.height, buf }
}
