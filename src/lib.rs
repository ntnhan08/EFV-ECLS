// ╔══════════════════════════════════════════════════════════════════════╗
// ║           EFV · Extreme Fast Video  –  Rust Core Engine             ║
// ║  PyO3 entry-point: registers every Python-visible class/function.   ║
// ╚══════════════════════════════════════════════════════════════════════╝
#![allow(clippy::missing_errors_doc)]

mod audio;
mod codec;
mod container;
mod filter;
mod frame;

use pyo3::prelude::*;

// Re-export the public surface
use audio::pcm::{
    py_change_volume, py_mix_pcm, py_mute_pcm, py_pad_or_trim_pcm, py_trim_pcm,
};
use codec::{
    jpeg::{py_decode_jpeg, py_encode_jpeg},
    png::py_decode_png,
};
use container::avi::{AviReader, AviWriter};
use filter::{
    blur::py_gaussian_blur,
    color::{py_adjust_brightness, py_adjust_contrast, py_adjust_saturation, py_apply_lut},
    composite::{py_blend_frames, py_composite_over, py_overlay_image},
    resize::{py_resize_bilinear, py_resize_nearest, py_zoom_frame},
};
use frame::Frame;

/// EFV core engine — compiled Rust extension module.
///
/// Import via `import efv_core` after building with maturin.
#[pymodule]
fn efv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── Frame ───────────────────────────────────────────────────────────
    m.add_class::<Frame>()?;

    // ── JPEG / PNG codec ────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(py_encode_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_png, m)?)?;

    // ── Resize / zoom ───────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(py_resize_bilinear, m)?)?;
    m.add_function(wrap_pyfunction!(py_resize_nearest, m)?)?;
    m.add_function(wrap_pyfunction!(py_zoom_frame, m)?)?;

    // ── Compositing ─────────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(py_composite_over, m)?)?;
    m.add_function(wrap_pyfunction!(py_overlay_image, m)?)?;
    m.add_function(wrap_pyfunction!(py_blend_frames, m)?)?;

    // ── Color filters ───────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(py_adjust_brightness, m)?)?;
    m.add_function(wrap_pyfunction!(py_adjust_contrast, m)?)?;
    m.add_function(wrap_pyfunction!(py_adjust_saturation, m)?)?;
    m.add_function(wrap_pyfunction!(py_apply_lut, m)?)?;
    m.add_function(wrap_pyfunction!(py_gaussian_blur, m)?)?;

    // ── Audio PCM ───────────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(py_change_volume, m)?)?;
    m.add_function(wrap_pyfunction!(py_mix_pcm, m)?)?;
    m.add_function(wrap_pyfunction!(py_mute_pcm, m)?)?;
    m.add_function(wrap_pyfunction!(py_trim_pcm, m)?)?;
    m.add_function(wrap_pyfunction!(py_pad_or_trim_pcm, m)?)?;

    // ── AVI container ───────────────────────────────────────────────────
    m.add_class::<AviReader>()?;
    m.add_class::<AviWriter>()?;

    // ── Module metadata ─────────────────────────────────────────────────
    m.add("__version__", "1.0.0")?;
    m.add("__author__", "EFV Team")?;

    Ok(())
}
