// ══════════════════════════════════════════════════════════════════════════
//  EFV · Color Filters  –  brightness, contrast, saturation, LUT
//  All operations work in-place and are rayon-parallelised.
// ══════════════════════════════════════════════════════════════════════════

use pyo3::{prelude::*, exceptions::PyValueError};
use rayon::prelude::*;
use crate::frame::Frame;

// ── Brightness ────────────────────────────────────────────────────────────

/// Adjust brightness.  `factor` 0.0 = black, 1.0 = unchanged, 2.0 = double.
#[pyfunction]
#[pyo3(signature = (frame, factor))]
pub fn py_adjust_brightness(frame: &Frame, factor: f32) -> Frame {
    adjust_brightness(frame, factor)
}

pub fn adjust_brightness(src: &Frame, factor: f32) -> Frame {
    let fi = (factor.clamp(0.0, 4.0) * 256.0) as i32;
    let mut buf = src.buf.clone();

    buf.par_chunks_mut(4).for_each(|px| {
        for c in 0..3 {
            px[c] = ((px[c] as i32 * fi) >> 8).clamp(0, 255) as u8;
        }
    });

    Frame { width: src.width, height: src.height, buf }
}

// ── Contrast ──────────────────────────────────────────────────────────────

/// Adjust contrast about mid-grey (128).
/// `factor` 0.0 = grey, 1.0 = unchanged, 2.0 = double.
#[pyfunction]
pub fn py_adjust_contrast(frame: &Frame, factor: f32) -> Frame {
    adjust_contrast(frame, factor)
}

pub fn adjust_contrast(src: &Frame, factor: f32) -> Frame {
    let fi = (factor.clamp(0.0, 4.0) * 256.0) as i32;
    let mut buf = src.buf.clone();

    buf.par_chunks_mut(4).for_each(|px| {
        for c in 0..3 {
            let v = px[c] as i32 - 128;
            px[c] = ((v * fi >> 8) + 128).clamp(0, 255) as u8;
        }
    });

    Frame { width: src.width, height: src.height, buf }
}

// ── Saturation ────────────────────────────────────────────────────────────

/// Adjust colour saturation.
/// `factor` 0.0 = greyscale, 1.0 = unchanged, 2.0 = highly saturated.
#[pyfunction]
pub fn py_adjust_saturation(frame: &Frame, factor: f32) -> Frame {
    adjust_saturation(frame, factor)
}

pub fn adjust_saturation(src: &Frame, factor: f32) -> Frame {
    let fi = (factor.clamp(0.0, 4.0) * 256.0) as i32;
    let mut buf = src.buf.clone();

    buf.par_chunks_mut(4).for_each(|px| {
        let r = px[0] as i32;
        let g = px[1] as i32;
        let b = px[2] as i32;
        // Luma (BT.601)
        let luma = (r * 77 + g * 150 + b * 29) >> 8;
        for (c, orig) in [(0, r), (1, g), (2, b)] {
            px[c] = ((luma + (((orig - luma) * fi) >> 8)).clamp(0, 255)) as u8;
        }
    });

    Frame { width: src.width, height: src.height, buf }
}

// ── LUT (Look-Up Table) ───────────────────────────────────────────────────

/// Apply an RGB Look-Up Table.
///
/// `lut` must be a flat `bytes` object of exactly 768 bytes (3 × 256 entries).
/// Layout: [R_lut[0..256], G_lut[0..256], B_lut[0..256]].
#[pyfunction]
pub fn py_apply_lut(frame: &Frame, lut: &[u8]) -> PyResult<Frame> {
    if lut.len() != 768 {
        return Err(PyValueError::new_err(
            "LUT must be exactly 768 bytes (3 × 256 entries)",
        ));
    }
    Ok(apply_lut(frame, lut))
}

pub fn apply_lut(src: &Frame, lut: &[u8]) -> Frame {
    let mut buf = src.buf.clone();

    buf.par_chunks_mut(4).for_each(|px| {
        px[0] = lut[px[0] as usize];           // R
        px[1] = lut[256 + px[1] as usize];     // G
        px[2] = lut[512 + px[2] as usize];     // B
    });

    Frame { width: src.width, height: src.height, buf }
}

/// Build a simple linear LUT: `[gamma_r, gamma_g, gamma_b]`.
/// gamma 1.0 = identity.  Useful to pre-compute before calling `apply_lut`.
#[pyfunction]
#[pyo3(signature = (gamma_r=1.0, gamma_g=1.0, gamma_b=1.0))]
pub fn py_build_gamma_lut<'py>(
    py:      Python<'py>,
    gamma_r: f32,
    gamma_g: f32,
    gamma_b: f32,
) -> pyo3::Bound<'py, pyo3::types::PyBytes> {
    let mut lut = [0u8; 768];
    for i in 0u32..256 {
        let v = i as f32 / 255.0;
        lut[i as usize]       = (v.powf(1.0 / gamma_r) * 255.0).clamp(0.0, 255.0) as u8;
        lut[256 + i as usize] = (v.powf(1.0 / gamma_g) * 255.0).clamp(0.0, 255.0) as u8;
        lut[512 + i as usize] = (v.powf(1.0 / gamma_b) * 255.0).clamp(0.0, 255.0) as u8;
    }
    pyo3::types::PyBytes::new_bound(py, &lut)
}
