// ══════════════════════════════════════════════════════════════════════════
//  EFV · PCM Audio  –  mix, volume, trim, pad, silence
//  16-bit LE stereo (the dominant WAV format in AVI files).
// ══════════════════════════════════════════════════════════════════════════

use pyo3::{prelude::*, types::PyBytes, exceptions::PyValueError};

// ── Volume ────────────────────────────────────────────────────────────────

/// Scale 16-bit PCM amplitude.  factor=1.0 = no change.
#[pyfunction]
pub fn py_change_volume<'py>(
    py:         Python<'py>,
    pcm:        &[u8],
    factor:     f32,
    bit_depth:  u32,
) -> PyResult<pyo3::Bound<'py, PyBytes>> {
    if bit_depth != 16 {
        return Err(PyValueError::new_err("change_volume only supports 16-bit PCM"));
    }
    let out = change_volume(pcm, factor);
    Ok(PyBytes::new_bound(py, &out))
}

pub fn change_volume(pcm: &[u8], factor: f32) -> Vec<u8> {
    let fi  = (factor * 256.0) as i32;
    let n   = pcm.len() / 2;
    let mut out = vec![0u8; pcm.len()];
    for i in 0..n {
        let s = i16::from_le_bytes([pcm[i * 2], pcm[i * 2 + 1]]) as i32;
        let v = ((s * fi) >> 8).clamp(-32768, 32767) as i16;
        let b = v.to_le_bytes();
        out[i * 2]     = b[0];
        out[i * 2 + 1] = b[1];
    }
    out
}

// ── Mix ───────────────────────────────────────────────────────────────────

/// Mix two 16-bit PCM streams.  Output length = max(len(a), len(b)).
#[pyfunction]
#[pyo3(signature = (a, b, vol_a=1.0, vol_b=1.0))]
pub fn py_mix_pcm<'py>(
    py:   Python<'py>,
    a:    &[u8],
    b:    &[u8],
    vol_a: f32,
    vol_b: f32,
) -> pyo3::Bound<'py, PyBytes> {
    let out = mix_pcm(a, b, vol_a, vol_b);
    PyBytes::new_bound(py, &out)
}

pub fn mix_pcm(a: &[u8], b: &[u8], vol_a: f32, vol_b: f32) -> Vec<u8> {
    let ia = (vol_a * 256.0) as i32;
    let ib = (vol_b * 256.0) as i32;
    let na = a.len() / 2;
    let nb = b.len() / 2;
    let n  = na.max(nb);
    let mut out = vec![0u8; n * 2];

    for i in 0..n {
        let sa = if i < na {
            i16::from_le_bytes([a[i * 2], a[i * 2 + 1]]) as i32
        } else { 0 };
        let sb = if i < nb {
            i16::from_le_bytes([b[i * 2], b[i * 2 + 1]]) as i32
        } else { 0 };
        let v = ((sa * ia + sb * ib) >> 8).clamp(-32768, 32767) as i16;
        let bytes = v.to_le_bytes();
        out[i * 2]     = bytes[0];
        out[i * 2 + 1] = bytes[1];
    }
    out
}

// ── Mute ──────────────────────────────────────────────────────────────────

/// Return silence of the given number of samples.
#[pyfunction]
#[pyo3(signature = (n_samples, channels=2))]
pub fn py_mute_pcm<'py>(
    py:        Python<'py>,
    n_samples: usize,
    channels:  u32,
) -> pyo3::Bound<'py, PyBytes> {
    let out = vec![0u8; n_samples * channels as usize * 2];
    PyBytes::new_bound(py, &out)
}

// ── Trim ──────────────────────────────────────────────────────────────────

/// Trim 16-bit PCM to [start_sec, end_sec].
#[pyfunction]
pub fn py_trim_pcm<'py>(
    py:          Python<'py>,
    pcm:         &[u8],
    sample_rate: u32,
    channels:    u32,
    start_sec:   f64,
    end_sec:     f64,
) -> pyo3::Bound<'py, PyBytes> {
    let out = trim_pcm(pcm, sample_rate, channels, start_sec, end_sec);
    PyBytes::new_bound(py, &out)
}

pub fn trim_pcm(pcm: &[u8], sr: u32, ch: u32, start_sec: f64, end_sec: f64) -> Vec<u8> {
    let frame_bytes  = (ch * 2) as usize;
    let start_frame  = (start_sec * sr as f64) as usize;
    let end_frame    = (end_sec   * sr as f64) as usize;
    let total_frames = pcm.len() / frame_bytes;
    let s = (start_frame * frame_bytes).min(pcm.len());
    let e = (end_frame.min(total_frames) * frame_bytes).min(pcm.len());
    pcm[s..e].to_vec()
}

// ── Pad / trim to duration ────────────────────────────────────────────────

/// Trim or zero-pad PCM to exactly `target_sec` duration.
#[pyfunction]
pub fn py_pad_or_trim_pcm<'py>(
    py:          Python<'py>,
    pcm:         &[u8],
    sample_rate: u32,
    channels:    u32,
    target_sec:  f64,
) -> pyo3::Bound<'py, PyBytes> {
    let out = pad_or_trim_pcm(pcm, sample_rate, channels, target_sec);
    PyBytes::new_bound(py, &out)
}

pub fn pad_or_trim_pcm(pcm: &[u8], sr: u32, ch: u32, target_sec: f64) -> Vec<u8> {
    let frame_bytes   = (ch * 2) as usize;
    let target_bytes  = (target_sec * sr as f64) as usize * frame_bytes;
    if pcm.len() >= target_bytes {
        pcm[..target_bytes].to_vec()
    } else {
        let mut out = pcm.to_vec();
        out.resize(target_bytes, 0);
        out
    }
}
