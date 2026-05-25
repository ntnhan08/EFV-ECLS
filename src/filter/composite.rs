// ══════════════════════════════════════════════════════════════════════════
//  EFV · Compositing  –  Porter-Duff "over", rayon-parallelised
// ══════════════════════════════════════════════════════════════════════════

use pyo3::{prelude::*, exceptions::PyValueError};
use rayon::prelude::*;
use crate::filter::resize::{resize_bilinear, resize_nearest};
use crate::frame::Frame;

// ── composite_over ────────────────────────────────────────────────────────

/// Composite `src` over `dst` in-place at pixel offset (x_off, y_off).
///
/// Uses Porter-Duff "over" blend with `opacity` applied to source alpha.
#[pyfunction]
#[pyo3(signature = (dst, src, x_off, y_off, opacity=1.0))]
pub fn py_composite_over(
    dst:     &mut Frame,
    src:     &Frame,
    x_off:   i32,
    y_off:   i32,
    opacity: f32,
) -> PyResult<()> {
    composite_over(dst, src, x_off, y_off, opacity);
    Ok(())
}

pub fn composite_over(dst: &mut Frame, src: &Frame, x_off: i32, y_off: i32, opacity: f32) {
    let dw = dst.width  as i32;
    let dh = dst.height as i32;
    let sw = src.width  as i32;
    let sh = src.height as i32;

    let x0 = x_off.max(0);
    let y0 = y_off.max(0);
    let x1 = (x_off + sw).min(dw);
    let y1 = (y_off + sh).min(dh);

    if x1 <= x0 || y1 <= y0 { return; }

    let opacity_i = (opacity.clamp(0.0, 1.0) * 255.0) as u32;
    let dw_u = dst.width as usize;
    let sw_u = src.width as usize;

    for dy in y0..y1 {
        let sy = (dy - y_off) as usize;
        for dx in x0..x1 {
            let sx = (dx - x_off) as usize;
            let si = (sy * sw_u + sx) * 4;
            let di = (dy as usize * dw_u + dx as usize) * 4;

            let sa = ((src.buf[si + 3] as u32 * opacity_i + 127) >> 8).min(255) as u32;
            if sa == 0 { continue; }

            if sa >= 254 {
                dst.buf[di]     = src.buf[si];
                dst.buf[di + 1] = src.buf[si + 1];
                dst.buf[di + 2] = src.buf[si + 2];
                dst.buf[di + 3] = 255;
                continue;
            }

            let inv = 255 - sa;
            let da  = dst.buf[di + 3] as u32;
            for c in 0..3 {
                dst.buf[di + c] =
                    ((src.buf[si + c] as u32 * sa + dst.buf[di + c] as u32 * inv) >> 8) as u8;
            }
            dst.buf[di + 3] = (sa + ((da * inv) >> 8)) as u8;
        }
    }
}

// ── overlay_image ─────────────────────────────────────────────────────────

/// Overlay an image onto a video frame.  Returns a new composite Frame.
#[pyfunction]
#[pyo3(signature = (video_frame, overlay, position="topright", scale=1.0,
                   opacity=1.0, margin=10, x=None, y=None))]
pub fn py_overlay_image(
    video_frame: &Frame,
    overlay:     &Frame,
    position:    &str,
    scale:       f32,
    opacity:     f32,
    margin:      i32,
    x:           Option<i32>,
    y:           Option<i32>,
) -> PyResult<Frame> {
    Ok(overlay_image(video_frame, overlay, position, scale, opacity, margin, x, y))
}

pub fn overlay_image(
    video_frame: &Frame,
    overlay:     &Frame,
    position:    &str,
    scale:       f32,
    opacity:     f32,
    margin:      i32,
    x:           Option<i32>,
    y:           Option<i32>,
) -> Frame {
    let mut dst = video_frame.copy();

    // Scale overlay if needed
    let ovl: Frame = if (scale - 1.0).abs() > 1e-4 {
        let ow = ((overlay.width  as f32 * scale) as u32).max(1);
        let oh = ((overlay.height as f32 * scale) as u32).max(1);
        resize_bilinear(overlay, ow, oh)
    } else {
        overlay.copy()
    };

    let vw = dst.width  as i32;
    let vh = dst.height as i32;
    let ow = ovl.width  as i32;
    let oh = ovl.height as i32;

    let (px, py) = match (x, y) {
        (Some(xi), Some(yi)) => (xi, yi),
        _ => position_to_coords(position, vw, vh, ow, oh, margin),
    };

    composite_over(&mut dst, &ovl, px, py, opacity);
    dst
}

fn position_to_coords(
    pos: &str, vw: i32, vh: i32, ow: i32, oh: i32, margin: i32,
) -> (i32, i32) {
    match pos {
        "center"      => ((vw - ow) / 2,      (vh - oh) / 2),
        "topright"    => (vw - ow - margin,    margin),
        "topleft"     => (margin,              margin),
        "bottomright" => (vw - ow - margin,    vh - oh - margin),
        "bottomleft"  => (margin,              vh - oh - margin),
        "top"         => ((vw - ow) / 2,       margin),
        "bottom"      => ((vw - ow) / 2,       vh - oh - margin),
        "left"        => (margin,              (vh - oh) / 2),
        "right"       => (vw - ow - margin,    (vh - oh) / 2),
        _             => (vw - ow - margin,    margin),  // default: topright
    }
}

// ── blend_frames ──────────────────────────────────────────────────────────

/// Linear blend: t=0 → frame A, t=1 → frame B.  Both must be same size.
#[pyfunction]
pub fn py_blend_frames(a: &Frame, b: &Frame, t: f32) -> PyResult<Frame> {
    if a.width != b.width || a.height != b.height {
        return Err(PyValueError::new_err(
            "blend_frames: both frames must have the same dimensions",
        ));
    }
    Ok(blend_frames(a, b, t))
}

pub fn blend_frames(a: &Frame, b: &Frame, t: f32) -> Frame {
    let t  = t.clamp(0.0, 1.0);
    let it = (t * 256.0) as u32;
    let ia = 256 - it;
    let n  = a.buf.len();
    let mut buf = vec![0u8; n];

    buf.par_chunks_mut(64)
        .zip(a.buf.par_chunks(64))
        .zip(b.buf.par_chunks(64))
        .for_each(|((out, ca), cb)| {
            for i in 0..out.len() {
                out[i] = ((ca[i] as u32 * ia + cb[i] as u32 * it) >> 8) as u8;
            }
        });

    Frame { width: a.width, height: a.height, buf }
}
