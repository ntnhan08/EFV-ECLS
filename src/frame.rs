// ══════════════════════════════════════════════════════════════════════════
//  EFV · Frame  –  RGBA pixel buffer
//  Layout : interleaved RGBA, row-major.
//  Index  : (y * width + x) * 4 + channel   [0=R 1=G 2=B 3=A]
// ══════════════════════════════════════════════════════════════════════════

use pyo3::{prelude::*, types::PyBytes};

// ── Internal Frame ────────────────────────────────────────────────────────

/// Raw RGBA pixel buffer.  Cheap to clone; heap-allocated via Vec<u8>.
#[pyclass]
#[derive(Clone)]
pub struct Frame {
    pub width:  u32,
    pub height: u32,
    pub buf:    Vec<u8>,   // length == width * height * 4
}

// Utility: clamp a value to [0, 255]
#[inline(always)]
pub fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[pymethods]
impl Frame {
    // ── Constructors ──────────────────────────────────────────────────────

    /// `Frame(width, height)` → black transparent canvas.
    #[new]
    #[pyo3(signature = (width, height, r=0, g=0, b=0, a=255))]
    pub fn new(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Self {
        let n = (width * height) as usize;
        let mut buf = Vec::with_capacity(n * 4);
        for _ in 0..n {
            buf.push(r); buf.push(g); buf.push(b); buf.push(a);
        }
        Frame { width, height, buf }
    }

    /// Create from packed RGB bytes (24-bit, no alpha).
    #[staticmethod]
    pub fn from_rgb(data: &[u8], width: u32, height: u32) -> PyResult<Frame> {
        let n = (width * height) as usize;
        if data.len() < n * 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "data too short for given width/height (RGB)",
            ));
        }
        let mut buf = vec![0u8; n * 4];
        for i in 0..n {
            buf[i * 4]     = data[i * 3];
            buf[i * 4 + 1] = data[i * 3 + 1];
            buf[i * 4 + 2] = data[i * 3 + 2];
            buf[i * 4 + 3] = 255;
        }
        Ok(Frame { width, height, buf })
    }

    /// Create from packed RGBA bytes (32-bit).
    #[staticmethod]
    pub fn from_rgba(data: &[u8], width: u32, height: u32) -> PyResult<Frame> {
        let n = (width * height * 4) as usize;
        if data.len() < n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "data too short for given width/height (RGBA)",
            ));
        }
        Ok(Frame { width, height, buf: data[..n].to_vec() })
    }

    // ── Pixel access ─────────────────────────────────────────────────────

    /// `get_pixel(x, y)` → `(r, g, b, a)`.
    pub fn get_pixel(&self, x: u32, y: u32) -> (u8, u8, u8, u8) {
        let i = ((y * self.width + x) * 4) as usize;
        (self.buf[i], self.buf[i + 1], self.buf[i + 2], self.buf[i + 3])
    }

    /// `set_pixel(x, y, r, g, b, a)`.
    pub fn set_pixel(&mut self, x: u32, y: u32, r: u8, g: u8, b: u8, a: u8) {
        let i = ((y * self.width + x) * 4) as usize;
        self.buf[i]     = r;
        self.buf[i + 1] = g;
        self.buf[i + 2] = b;
        self.buf[i + 3] = a;
    }

    // ── Export ───────────────────────────────────────────────────────────

    /// Export as packed RGB bytes (drops alpha).
    pub fn to_rgb<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let n = (self.width * self.height) as usize;
        let mut out = vec![0u8; n * 3];
        for i in 0..n {
            out[i * 3]     = self.buf[i * 4];
            out[i * 3 + 1] = self.buf[i * 4 + 1];
            out[i * 3 + 2] = self.buf[i * 4 + 2];
        }
        PyBytes::new_bound(py, &out)
    }

    /// Export as packed RGBA bytes.
    pub fn to_rgba<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.buf)
    }

    /// Export as bottom-up BGR bytes (AVI / BMP raw format).
    pub fn to_bgr<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut out = vec![0u8; w * h * 3];
        for y in 0..h {
            let src_row = y * w;
            let dst_row = (h - 1 - y) * w;
            for x in 0..w {
                let s = (src_row + x) * 4;
                let d = (dst_row + x) * 3;
                out[d]     = self.buf[s + 2];
                out[d + 1] = self.buf[s + 1];
                out[d + 2] = self.buf[s];
            }
        }
        PyBytes::new_bound(py, &out)
    }

    // ── Utilities ────────────────────────────────────────────────────────

    /// Deep copy.
    pub fn copy(&self) -> Frame {
        Frame { width: self.width, height: self.height, buf: self.buf.clone() }
    }

    /// Fill a rectangular region.
    #[pyo3(signature = (x0, y0, x1, y1, r, g, b, a=255))]
    pub fn fill_rect(&mut self,
        x0: u32, y0: u32, x1: u32, y1: u32,
        r: u8, g: u8, b: u8, a: u8,
    ) {
        let x0 = x0.min(self.width);
        let x1 = x1.min(self.width);
        let y0 = y0.min(self.height);
        let y1 = y1.min(self.height);
        let w  = self.width as usize;
        for y in y0..y1 {
            for x in x0..x1 {
                let i = (y as usize * w + x as usize) * 4;
                self.buf[i]     = r;
                self.buf[i + 1] = g;
                self.buf[i + 2] = b;
                self.buf[i + 3] = a;
            }
        }
    }

    #[getter]
    pub fn width(&self)  -> u32 { self.width  }
    #[getter]
    pub fn height(&self) -> u32 { self.height }

    pub fn __repr__(&self) -> String {
        format!("Frame({}x{})", self.width, self.height)
    }
}

// ── Helpers used inside Rust only ────────────────────────────────────────

impl Frame {
    /// Convert RGB bytes (r,g,b per pixel) to YCbCr planes.
    pub fn to_ycbcr(&self) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let n   = (self.width * self.height) as usize;
        let mut y_plane  = vec![0i32; n];
        let mut cb_plane = vec![0i32; n];
        let mut cr_plane = vec![0i32; n];
        for i in 0..n {
            let r = self.buf[i * 4]     as i32;
            let g = self.buf[i * 4 + 1] as i32;
            let b = self.buf[i * 4 + 2] as i32;
            y_plane[i]  = (( 66*r + 129*g +  25*b + 128) >> 8) + 16;
            cb_plane[i] = ((-38*r -  74*g + 112*b + 128) >> 8) + 128;
            cr_plane[i] = ((112*r -  94*g -  18*b + 128) >> 8) + 128;
        }
        (y_plane, cb_plane, cr_plane)
    }
}
