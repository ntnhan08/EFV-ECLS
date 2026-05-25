// ══════════════════════════════════════════════════════════════════════════
//  EFV · JPEG Codec  –  encode & decode via pure-Rust crates
//  Crates: jpeg-decoder 0.3, jpeg-encoder 0.6
// ══════════════════════════════════════════════════════════════════════════

use pyo3::{prelude::*, types::PyBytes, exceptions::PyRuntimeError};
use crate::frame::Frame;

// ── Decode ────────────────────────────────────────────────────────────────

/// Decode JPEG bytes → `(rgb_bytes, width, height)`.
///
/// Returns packed RGB (24-bit, no alpha).
#[pyfunction]
pub fn py_decode_jpeg<'py>(
    py:   Python<'py>,
    data: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, u32, u32)> {
    let (rgb, w, h) = decode_jpeg(data)
        .map_err(|e| PyRuntimeError::new_err(format!("JPEG decode error: {e}")))?;
    Ok((PyBytes::new_bound(py, &rgb), w, h))
}

pub fn decode_jpeg(data: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    let pixels      = decoder.decode().map_err(|e| e.to_string())?;
    let info        = decoder.info().ok_or("No JPEG info")?;
    let w           = info.width  as u32;
    let h           = info.height as u32;

    // jpeg-decoder can return Grayscale, RGB, CMYK …
    let rgb = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => pixels,
        jpeg_decoder::PixelFormat::L8    => {
            // Grayscale → RGB
            pixels.iter().flat_map(|&v| [v, v, v]).collect()
        }
        jpeg_decoder::PixelFormat::CMYK32 => {
            // CMYK → RGB (approximate)
            pixels.chunks_exact(4).flat_map(|c| {
                let (cy, m, ye, k) = (c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32);
                let r = (255.0 - cy) * (255.0 - k) / 255.0;
                let g = (255.0 - m)  * (255.0 - k) / 255.0;
                let b = (255.0 - ye) * (255.0 - k) / 255.0;
                [r as u8, g as u8, b as u8]
            }).collect()
        }
        _ => return Err("Unsupported JPEG pixel format".to_string()),
    };

    Ok((rgb, w, h))
}

// ── Encode ────────────────────────────────────────────────────────────────

/// Encode RGB bytes → JPEG bytes at the given quality (1–100).
#[pyfunction]
#[pyo3(signature = (rgb, width, height, quality=85))]
pub fn py_encode_jpeg<'py>(
    py:     Python<'py>,
    rgb:    &[u8],
    width:  u32,
    height: u32,
    quality: u8,
) -> PyResult<Bound<'py, PyBytes>> {
    let encoded = encode_jpeg(rgb, width, height, quality)
        .map_err(|e| PyRuntimeError::new_err(format!("JPEG encode error: {e}")))?;
    Ok(PyBytes::new_bound(py, &encoded))
}

pub fn encode_jpeg(rgb: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>, String> {
    let quality = quality.clamp(1, 100);
    let mut buf = Vec::new();
    let encoder = jpeg_encoder::Encoder::new(&mut buf, quality);
    encoder
        .encode(rgb, width as u16, height as u16, jpeg_encoder::ColorType::Rgb)
        .map_err(|e| e.to_string())?;
    Ok(buf)
}

// ── Frame helpers ─────────────────────────────────────────────────────────

/// Decode JPEG bytes directly into a Frame (RGBA, with A=255).
pub fn decode_jpeg_to_frame(data: &[u8]) -> Result<Frame, String> {
    let (rgb, w, h) = decode_jpeg(data)?;
    Frame::from_rgb(&rgb, w, h).map_err(|e| e.to_string())
}

/// Encode a Frame's RGB data to JPEG.
pub fn encode_frame_jpeg(frame: &Frame, quality: u8) -> Result<Vec<u8>, String> {
    let rgb = {
        let n = (frame.width * frame.height) as usize;
        let mut out = vec![0u8; n * 3];
        for i in 0..n {
            out[i * 3]     = frame.buf[i * 4];
            out[i * 3 + 1] = frame.buf[i * 4 + 1];
            out[i * 3 + 2] = frame.buf[i * 4 + 2];
        }
        out
    };
    encode_jpeg(&rgb, frame.width, frame.height, quality)
}
