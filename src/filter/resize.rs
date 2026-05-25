// ══════════════════════════════════════════════════════════════════════════
//  EFV · Resize  –  bilinear & nearest-neighbour, rayon-parallelised
// ══════════════════════════════════════════════════════════════════════════

use pyo3::{prelude::*, exceptions::PyValueError};
use rayon::prelude::*;
use crate::frame::Frame;

// ── Bilinear ──────────────────────────────────────────────────────────────

/// High-quality bilinear resize.  Each output row is computed in parallel.
#[pyfunction]
pub fn py_resize_bilinear(src: &Frame, dst_w: u32, dst_h: u32) -> PyResult<Frame> {
    if dst_w == 0 || dst_h == 0 {
        return Err(PyValueError::new_err("Target dimensions must be > 0"));
    }
    Ok(resize_bilinear(src, dst_w, dst_h))
}

pub fn resize_bilinear(src: &Frame, dst_w: u32, dst_h: u32) -> Frame {
    let sw = src.width  as usize;
    let sh = src.height as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;

    let x_ratio = sw as f32 / dw as f32;
    let y_ratio = sh as f32 / dh as f32;

    // Allocate destination buffer
    let mut buf = vec![0u8; dw * dh * 4];

    // Process rows in parallel via rayon
    buf.par_chunks_mut(dw * 4)
        .enumerate()
        .for_each(|(dy, row)| {
            let sy_f = (dy as f32 + 0.5) * y_ratio - 0.5;
            let sy0  = (sy_f as isize).clamp(0, sh as isize - 1) as usize;
            let sy1  = (sy0 + 1).min(sh - 1);
            let wy1  = sy_f - sy_f.floor();
            let wy0  = 1.0 - wy1;

            for dx in 0..dw {
                let sx_f = (dx as f32 + 0.5) * x_ratio - 0.5;
                let sx0  = (sx_f as isize).clamp(0, sw as isize - 1) as usize;
                let sx1  = (sx0 + 1).min(sw - 1);
                let wx1  = sx_f - sx_f.floor();
                let wx0  = 1.0 - wx1;

                let i00 = (sy0 * sw + sx0) * 4;
                let i10 = (sy0 * sw + sx1) * 4;
                let i01 = (sy1 * sw + sx0) * 4;
                let i11 = (sy1 * sw + sx1) * 4;

                let di = dx * 4;
                for c in 0..4 {
                    let v = src.buf[i00 + c] as f32 * wx0 * wy0
                          + src.buf[i10 + c] as f32 * wx1 * wy0
                          + src.buf[i01 + c] as f32 * wx0 * wy1
                          + src.buf[i11 + c] as f32 * wx1 * wy1;
                    row[di + c] = (v + 0.5) as u8;
                }
            }
        });

    Frame { width: dst_w, height: dst_h, buf }
}

// ── Nearest-neighbour ─────────────────────────────────────────────────────

/// Fast nearest-neighbour resize (lower quality, used for thumbnails / previews).
#[pyfunction]
pub fn py_resize_nearest(src: &Frame, dst_w: u32, dst_h: u32) -> PyResult<Frame> {
    if dst_w == 0 || dst_h == 0 {
        return Err(PyValueError::new_err("Target dimensions must be > 0"));
    }
    Ok(resize_nearest(src, dst_w, dst_h))
}

pub fn resize_nearest(src: &Frame, dst_w: u32, dst_h: u32) -> Frame {
    let sw = src.width  as usize;
    let sh = src.height as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    let x_ratio = sw as f32 / dw as f32;
    let y_ratio = sh as f32 / dh as f32;

    let mut buf = vec![0u8; dw * dh * 4];

    buf.par_chunks_mut(dw * 4)
        .enumerate()
        .for_each(|(dy, row)| {
            let sy = ((dy as f32 * y_ratio) as usize).min(sh - 1);
            for dx in 0..dw {
                let sx = ((dx as f32 * x_ratio) as usize).min(sw - 1);
                let si = (sy * sw + sx) * 4;
                let di = dx * 4;
                row[di..di + 4].copy_from_slice(&src.buf[si..si + 4]);
            }
        });

    Frame { width: dst_w, height: dst_h, buf }
}

// ── Zoom helper ───────────────────────────────────────────────────────────

/// Zoom a frame by `factor`, ensuring output dimensions are even (codec-safe).
#[pyfunction]
#[pyo3(signature = (frame, factor, smooth=true))]
pub fn py_zoom_frame(frame: &Frame, factor: f32, smooth: bool) -> PyResult<Frame> {
    if factor <= 0.0 {
        return Err(PyValueError::new_err("zoom factor must be > 0"));
    }
    let mut w = (frame.width  as f32 * factor) as u32;
    let mut h = (frame.height as f32 * factor) as u32;
    w += w % 2;  // keep even
    h += h % 2;
    w = w.max(2);
    h = h.max(2);

    if smooth {
        Ok(resize_bilinear(frame, w, h))
    } else {
        Ok(resize_nearest(frame, w, h))
    }
}
