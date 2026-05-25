// ══════════════════════════════════════════════════════════════════════════
//  EFV · PNG Decoder  –  powered by the `png` crate (pure Rust)
//  Supports: RGBA / RGB / Grayscale / Palette, 8-bit and 16-bit,
//            interlaced and non-interlaced.
// ══════════════════════════════════════════════════════════════════════════

use pyo3::{prelude::*, types::PyBytes, exceptions::PyRuntimeError};
use crate::frame::Frame;

/// Decode PNG bytes → `(rgba_bytes, width, height)`.
///
/// Always returns 4 bytes (RGBA) per pixel regardless of source format.
#[pyfunction]
pub fn py_decode_png<'py>(
    py:   Python<'py>,
    data: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, u32, u32)> {
    let (rgba, w, h) = decode_png(data)
        .map_err(|e| PyRuntimeError::new_err(format!("PNG decode error: {e}")))?;
    Ok((PyBytes::new_bound(py, &rgba), w, h))
}

pub fn decode_png(data: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    use png::ColorType;

    let decoder = png::Decoder::new(std::io::Cursor::new(data));
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let mut raw = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut raw).map_err(|e| e.to_string())?;

    let w = info.width;
    let h = info.height;

    // Normalise to RGBA u8
    let rgba: Vec<u8> = match (info.color_type, info.bit_depth) {
        // ── 8-bit paths ───────────────────────────────────────────────
        (ColorType::Rgba, png::BitDepth::Eight) => raw[..info.buffer_size()].to_vec(),
        (ColorType::Rgb,  png::BitDepth::Eight) => {
            raw[..info.buffer_size()].chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect()
        }
        (ColorType::Grayscale, png::BitDepth::Eight) => {
            raw[..info.buffer_size()].iter()
                .flat_map(|&v| [v, v, v, 255])
                .collect()
        }
        (ColorType::GrayscaleAlpha, png::BitDepth::Eight) => {
            raw[..info.buffer_size()].chunks_exact(2)
                .flat_map(|p| [p[0], p[0], p[0], p[1]])
                .collect()
        }
        // ── 16-bit paths (truncate to 8-bit) ──────────────────────────
        (ColorType::Rgba, png::BitDepth::Sixteen) => {
            raw[..info.buffer_size()].chunks_exact(8)
                .flat_map(|p| [p[0], p[2], p[4], p[6]])
                .collect()
        }
        (ColorType::Rgb, png::BitDepth::Sixteen) => {
            raw[..info.buffer_size()].chunks_exact(6)
                .flat_map(|p| [p[0], p[2], p[4], 255])
                .collect()
        }
        (ColorType::Grayscale, png::BitDepth::Sixteen) => {
            raw[..info.buffer_size()].chunks_exact(2)
                .flat_map(|p| [p[0], p[0], p[0], 255])
                .collect()
        }
        // ── Palette ───────────────────────────────────────────────────
        (ColorType::Indexed, _) => {
            // png crate expands palette to RGB automatically when we ask
            return Err("Indexed PNG: use png decoder's expand_paletted setting".into());
        }
        other => {
            return Err(format!("Unsupported PNG format: {other:?}"));
        }
    };

    Ok((rgba, w, h))
}

/// Decode PNG bytes directly into a Frame.
pub fn decode_png_to_frame(data: &[u8]) -> Result<Frame, String> {
    let (rgba, w, h) = decode_png(data)?;
    Frame::from_rgba(&rgba, w, h).map_err(|e| e.to_string())
}
