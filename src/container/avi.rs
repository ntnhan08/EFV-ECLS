// ══════════════════════════════════════════════════════════════════════════
//  EFV · AVI Container  –  RIFF AVI reader + MJPEG-AVI writer
//  Mirrors the pure-Python implementation but runs in compiled Rust.
// ══════════════════════════════════════════════════════════════════════════

use pyo3::{prelude::*, exceptions::PyRuntimeError, types::PyBytes};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::PathBuf,
};

// ── Constants ─────────────────────────────────────────────────────────────

const RIFF: &[u8; 4] = b"RIFF";
const AVI_: &[u8; 4] = b"AVI ";
const LIST: &[u8; 4] = b"LIST";
const HDRL: &[u8; 4] = b"hdrl";
const AVIH: &[u8; 4] = b"avih";
const STRL: &[u8; 4] = b"strl";
const STRH: &[u8; 4] = b"strh";
const STRF: &[u8; 4] = b"strf";
const MOVI: &[u8; 4] = b"movi";
const IDX1: &[u8; 4] = b"idx1";
const VIDS: &[u8; 4] = b"vids";
const AUDS: &[u8; 4] = b"auds";
const MJPG: &[u8; 4] = b"MJPG";
const DC00: &[u8; 4] = b"00dc";
const WB01: &[u8; 4] = b"01wb";

// ── Metadata structs ──────────────────────────────────────────────────────

#[pyclass(get_all)]
#[derive(Clone, Debug, Default)]
pub struct VideoMeta {
    pub width:       u32,
    pub height:      u32,
    pub fps:         f64,
    pub frame_count: u32,
    pub codec:       String,
}

#[pyclass(get_all)]
#[derive(Clone, Debug, Default)]
pub struct AudioMeta {
    pub has_audio:   bool,
    pub sample_rate: u32,
    pub channels:    u16,
    pub bit_depth:   u16,
}

#[pyclass(get_all)]
#[derive(Clone, Debug, Default)]
pub struct AviMeta {
    pub video: VideoMeta,
    pub audio: AudioMeta,
    pub duration_sec: f64,
}

// ── AviReader ─────────────────────────────────────────────────────────────

/// Fast AVI reader.  Exposes frame iteration and audio extraction.
#[pyclass]
pub struct AviReader {
    path:         PathBuf,
    meta:         AviMeta,
    frame_index:  Vec<(u64, u32, bool)>,  // (offset, size, is_video)
    audio_offset: Option<u64>,
    audio_size:   u32,
}

#[pymethods]
impl AviReader {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let p = PathBuf::from(path);
        let f = File::open(&p)
            .map_err(|e| PyRuntimeError::new_err(format!("Cannot open {path}: {e}")))?;
        let mut r = BufReader::new(f);
        let (meta, frame_index, audio_offset, audio_size) = parse_avi(&mut r)
            .map_err(|e| PyRuntimeError::new_err(format!("AVI parse error: {e}")))?;
        Ok(AviReader { path: p, meta, frame_index, audio_offset, audio_size })
    }

    /// Return video/audio metadata.
    #[getter]
    pub fn info(&self) -> AviMeta {
        self.meta.clone()
    }

    /// Read all audio PCM bytes.  Returns empty bytes if no audio.
    pub fn read_audio<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyBytes>> {
        if let Some(off) = self.audio_offset {
            let mut f = File::open(&self.path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            f.seek(SeekFrom::Start(off))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let mut buf = vec![0u8; self.audio_size as usize];
            f.read_exact(&mut buf)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyBytes::new_bound(py, &buf))
        } else {
            Ok(PyBytes::new_bound(py, b""))
        }
    }

    /// Iterate frames: yields `(data_bytes, frame_type_str)` tuples.
    /// frame_type_str is "jpeg" or "rgb24".
    pub fn iter_frames<'py>(&self, py: Python<'py>) -> PyResult<Vec<(pyo3::Bound<'py, PyBytes>, String)>> {
        let mut f = File::open(&self.path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let codec = &self.meta.video.codec;
        let ftype = if codec.contains("MJPG") || codec.contains("JPEG") {
            "jpeg"
        } else {
            "rgb24"
        };

        let mut result = Vec::with_capacity(self.frame_index.len());
        for &(offset, size, is_video) in &self.frame_index {
            if !is_video { continue; }
            f.seek(SeekFrom::Start(offset))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let mut buf = vec![0u8; size as usize];
            f.read_exact(&mut buf)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            result.push((PyBytes::new_bound(py, &buf), ftype.to_string()));
        }
        Ok(result)
    }

    pub fn close(&self) { /* BufReader is dropped with the struct */ }
}

// ── AviWriter ─────────────────────────────────────────────────────────────

/// MJPEG-AVI writer.  Writes frames one at a time; finalises on `close()`.
#[pyclass]
pub struct AviWriter {
    path:             PathBuf,
    width:            u32,
    height:           u32,
    fps:              f64,
    has_audio:        bool,
    audio_sample_rate: u32,
    audio_channels:   u16,
    // runtime state
    writer:           Option<BufWriter<File>>,
    movi_start:       u64,
    frame_count:      u32,
    index:            Vec<IndexEntry>,
}

struct IndexEntry {
    fourcc: [u8; 4],
    flags:  u32,
    offset: u32,
    size:   u32,
}

#[pymethods]
impl AviWriter {
    #[new]
    #[pyo3(signature = (path, width, height, fps, has_audio=false,
                        audio_sample_rate=44100, audio_channels=2))]
    pub fn new(
        path:              &str,
        width:             u32,
        height:            u32,
        fps:               f64,
        has_audio:         bool,
        audio_sample_rate: u32,
        audio_channels:    u16,
    ) -> PyResult<Self> {
        let p = PathBuf::from(path);
        let f = OpenOptions::new().write(true).create(true).truncate(true).open(&p)
            .map_err(|e| PyRuntimeError::new_err(format!("Cannot create {path}: {e}")))?;
        let mut w = BufWriter::new(f);

        // Write a placeholder RIFF header (we'll patch it on close)
        let hdr_size = write_avi_header(
            &mut w, width, height, fps,
            has_audio, audio_sample_rate, audio_channels,
        ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let movi_start = hdr_size;

        // Write movi LIST header
        w.write_all(LIST).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        w.write_all(&[0u8; 4]).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;  // size placeholder
        w.write_all(MOVI).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(AviWriter {
            path:              p,
            width,
            height,
            fps,
            has_audio,
            audio_sample_rate,
            audio_channels,
            writer:            Some(w),
            movi_start,
            frame_count:       0,
            index:             Vec::new(),
        })
    }

    /// Write one JPEG video frame.
    pub fn write_video_frame(&mut self, data: &[u8]) -> PyResult<()> {
        let w = self.writer.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("AviWriter already closed"))?;
        let offset = current_pos(w).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let size   = data.len() as u32;
        w.write_all(DC00).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        w.write_all(&size.to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        w.write_all(data).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if size % 2 != 0 { w.write_all(&[0]).map_err(|e| PyRuntimeError::new_err(e.to_string()))?; }

        self.index.push(IndexEntry {
            fourcc: *DC00,
            flags:  0x10,  // AVIIF_KEYFRAME
            offset: offset as u32,
            size,
        });
        self.frame_count += 1;
        Ok(())
    }

    /// Write one audio chunk (raw PCM bytes).
    pub fn write_audio_chunk(&mut self, data: &[u8]) -> PyResult<()> {
        let w = self.writer.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("AviWriter already closed"))?;
        let offset = current_pos(w).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let size   = data.len() as u32;
        w.write_all(WB01).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        w.write_all(&size.to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        w.write_all(data).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if size % 2 != 0 { w.write_all(&[0]).map_err(|e| PyRuntimeError::new_err(e.to_string()))?; }

        self.index.push(IndexEntry {
            fourcc: *WB01,
            flags:  0,
            offset: offset as u32,
            size,
        });
        Ok(())
    }

    /// Finalise the file: patch headers, write idx1.
    pub fn close(&mut self) -> PyResult<()> {
        if self.writer.is_none() { return Ok(()); }

        let mut w = self.writer.take().unwrap();
        let movi_end = current_pos(&mut w).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Patch movi LIST size
        let movi_data_size = (movi_end - self.movi_start - 8) as u32;
        w.seek(SeekFrom::Start(self.movi_start + 4))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        w.write_all(&movi_data_size.to_le_bytes())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Write idx1
        w.seek(SeekFrom::Start(movi_end))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let idx_size = (self.index.len() * 16) as u32;
        w.write_all(IDX1).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        w.write_all(&idx_size.to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        for entry in &self.index {
            w.write_all(&entry.fourcc).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            w.write_all(&entry.flags.to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            w.write_all(&entry.offset.to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            w.write_all(&entry.size.to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }

        // Patch total RIFF size
        let total = current_pos(&mut w).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        w.seek(SeekFrom::Start(4)).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        w.write_all(&((total - 8) as u32).to_le_bytes())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Patch avih frame count
        w.seek(SeekFrom::Start(32)).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        w.write_all(&self.frame_count.to_le_bytes())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        w.flush().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }
}

// ── Private helpers ───────────────────────────────────────────────────────

fn current_pos<W: Seek>(w: &mut W) -> std::io::Result<u64> {
    w.seek(SeekFrom::Current(0))
}

fn u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off+1], buf[off+2], buf[off+3]])
}
fn u16_le(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off+1]])
}

/// Parse AVI RIFF structure and return metadata + frame index.
fn parse_avi<R: Read + Seek>(r: &mut R) -> Result<(AviMeta, Vec<(u64, u32, bool)>, Option<u64>, u32), String> {
    let mut hdr = [0u8; 12];
    r.read_exact(&mut hdr).map_err(|e| e.to_string())?;
    if &hdr[0..4] != RIFF { return Err("Not a RIFF file".into()); }
    if &hdr[8..12] != AVI_ { return Err("Not an AVI file".into()); }

    let mut meta   = AviMeta::default();
    let mut frames = Vec::new();
    let mut audio_blobs: Vec<(u64, u32)> = Vec::new();

    parse_chunks(r, 12, &mut meta, &mut frames, &mut audio_blobs)?;

    // Flatten audio blobs into a single contiguous view
    let audio_offset = audio_blobs.first().map(|&(off, _)| off);
    let audio_size   = audio_blobs.iter().map(|&(_, s)| s).sum();

    Ok((meta, frames, audio_offset, audio_size))
}

fn parse_chunks<R: Read + Seek>(
    r:           &mut R,
    start:       u64,
    meta:        &mut AviMeta,
    frames:      &mut Vec<(u64, u32, bool)>,
    audio_blobs: &mut Vec<(u64, u32)>,
) -> Result<(), String> {
    r.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;
    let file_len = r.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    r.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;

    let mut pos = start;
    while pos + 8 <= file_len {
        r.seek(SeekFrom::Start(pos)).map_err(|e| e.to_string())?;
        let mut tag = [0u8; 4];
        let mut sz  = [0u8; 4];
        r.read_exact(&mut tag).map_err(|e| e.to_string())?;
        r.read_exact(&mut sz).map_err(|e| e.to_string())?;
        let chunk_size = u32::from_le_bytes(sz);

        match &tag {
            b"LIST" => {
                let mut list_type = [0u8; 4];
                r.read_exact(&mut list_type).map_err(|e| e.to_string())?;
                parse_chunks(r, pos + 12, meta, frames, audio_blobs)?;
            }
            b"avih" => {
                let mut buf = vec![0u8; chunk_size as usize];
                r.read_exact(&mut buf).map_err(|e| e.to_string())?;
                let micro_per_frame = u32_le(&buf, 0);
                meta.video.fps         = if micro_per_frame > 0 { 1_000_000.0 / micro_per_frame as f64 } else { 25.0 };
                meta.video.frame_count = u32_le(&buf, 24);
                meta.video.width       = u32_le(&buf, 32);
                meta.video.height      = u32_le(&buf, 36);
            }
            b"strh" => {
                let mut buf = vec![0u8; chunk_size as usize];
                r.read_exact(&mut buf).map_err(|e| e.to_string())?;
                let stream_type = &buf[0..4];
                if stream_type == VIDS {
                    meta.video.codec = String::from_utf8_lossy(&buf[4..8]).to_string();
                } else if stream_type == AUDS {
                    meta.audio.has_audio = true;
                }
            }
            b"strf" => {
                let mut buf = vec![0u8; chunk_size as usize];
                r.read_exact(&mut buf).map_err(|e| e.to_string())?;
                if meta.audio.has_audio && buf.len() >= 16 {
                    meta.audio.channels    = u16_le(&buf, 2);
                    meta.audio.sample_rate = u32_le(&buf, 4);
                    meta.audio.bit_depth   = u16_le(&buf, 14);
                }
            }
            b"00dc" | b"00db" => {
                frames.push((pos + 8, chunk_size, true));
                r.seek(SeekFrom::Current(chunk_size as i64)).map_err(|e| e.to_string())?;
            }
            b"01wb" => {
                audio_blobs.push((pos + 8, chunk_size));
                r.seek(SeekFrom::Current(chunk_size as i64)).map_err(|e| e.to_string())?;
            }
            _ => {
                r.seek(SeekFrom::Current(chunk_size as i64 + (chunk_size % 2) as i64))
                    .map_err(|e| e.to_string())?;
            }
        }

        pos += 8 + chunk_size as u64 + (chunk_size % 2) as u64;
    }

    meta.duration_sec = if meta.video.fps > 0.0 {
        meta.video.frame_count as f64 / meta.video.fps
    } else { 0.0 };

    Ok(())
}

/// Write AVI header (avih + video strl + optional audio strl).
/// Returns the byte offset right after the header (= start of movi LIST).
fn write_avi_header<W: Write + Seek>(
    w:           &mut W,
    width:       u32,
    height:      u32,
    fps:         f64,
    has_audio:   bool,
    sample_rate: u32,
    channels:    u16,
) -> std::io::Result<u64> {
    let micro = (1_000_000.0 / fps.max(1.0)) as u32;

    // RIFF AVI  (size patched on close)
    w.write_all(RIFF)?;
    w.write_all(&0u32.to_le_bytes())?;
    w.write_all(AVI_)?;

    // LIST hdrl
    let hdrl_start = w.seek(SeekFrom::Current(0))?;
    w.write_all(LIST)?;
    w.write_all(&0u32.to_le_bytes())?;  // size placeholder
    w.write_all(HDRL)?;

    // avih (56 bytes)
    w.write_all(AVIH)?;
    w.write_all(&56u32.to_le_bytes())?;
    w.write_all(&micro.to_le_bytes())?;          // 0  micro_per_frame
    w.write_all(&0u32.to_le_bytes())?;            // 4  max_bytes_per_sec
    w.write_all(&0u32.to_le_bytes())?;            // 8  padding_granularity
    w.write_all(&0x10u32.to_le_bytes())?;         // 12 flags (AVIF_HASINDEX)
    w.write_all(&0u32.to_le_bytes())?;            // 16 total_frames  ← patched on close (offset 32)
    w.write_all(&0u32.to_le_bytes())?;            // 20 initial_frames
    let n_streams: u32 = if has_audio { 2 } else { 1 };
    w.write_all(&n_streams.to_le_bytes())?;       // 24 streams
    w.write_all(&0u32.to_le_bytes())?;            // 28 suggested_buffer_size
    w.write_all(&width.to_le_bytes())?;           // 32 width
    w.write_all(&height.to_le_bytes())?;          // 36 height
    w.write_all(&[0u8; 16])?;                     // 40-55 reserved

    // Video strl
    write_video_strl(w, width, height, fps)?;
    if has_audio {
        write_audio_strl(w, sample_rate, channels)?;
    }

    // Patch hdrl size
    let hdrl_end = w.seek(SeekFrom::Current(0))?;
    w.seek(SeekFrom::Start(hdrl_start + 4))?;
    w.write_all(&((hdrl_end - hdrl_start - 8) as u32).to_le_bytes())?;
    w.seek(SeekFrom::Start(hdrl_end))?;

    Ok(hdrl_end)
}

fn write_video_strl<W: Write + Seek>(w: &mut W, width: u32, height: u32, fps: f64) -> std::io::Result<()> {
    let micro = (1_000_000.0 / fps.max(1.0)) as u32;

    let strl_start = w.seek(SeekFrom::Current(0))?;
    w.write_all(LIST)?;
    w.write_all(&0u32.to_le_bytes())?;
    w.write_all(STRL)?;

    // strh (video) – 56 bytes
    w.write_all(STRH)?;
    w.write_all(&56u32.to_le_bytes())?;
    w.write_all(VIDS)?;           // fcc_type
    w.write_all(MJPG)?;           // fcc_handler
    w.write_all(&[0u8; 8])?;      // flags, priority, language, initial_frames
    w.write_all(&1u32.to_le_bytes())?;           // rate numerator  (scale=1, rate=fps)
    w.write_all(&(fps as u32).to_le_bytes())?;   // rate
    w.write_all(&0u32.to_le_bytes())?;           // start
    w.write_all(&0u32.to_le_bytes())?;           // length  (patched on close via avih)
    w.write_all(&0u32.to_le_bytes())?;           // suggested_buffer_size
    w.write_all(&(-1i32 as u32).to_le_bytes())?; // quality
    w.write_all(&0u32.to_le_bytes())?;           // sample_size
    w.write_all(&[0u8; 8])?;                     // rcFrame

    // strf (BITMAPINFOHEADER) – 40 bytes
    w.write_all(STRF)?;
    w.write_all(&40u32.to_le_bytes())?;
    w.write_all(&40u32.to_le_bytes())?;          // biSize
    w.write_all(&width.to_le_bytes())?;
    w.write_all(&height.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?;            // biPlanes
    w.write_all(&24u16.to_le_bytes())?;           // biBitCount
    w.write_all(MJPG)?;                           // biCompression
    w.write_all(&(width * height * 3).to_le_bytes())?;  // biSizeImage
    w.write_all(&[0u8; 16])?;                    // remaining BITMAPINFOHEADER fields

    let strl_end = w.seek(SeekFrom::Current(0))?;
    w.seek(SeekFrom::Start(strl_start + 4))?;
    w.write_all(&((strl_end - strl_start - 8) as u32).to_le_bytes())?;
    w.seek(SeekFrom::Start(strl_end))?;
    Ok(())
}

fn write_audio_strl<W: Write + Seek>(w: &mut W, sample_rate: u32, channels: u16) -> std::io::Result<()> {
    let bit_depth: u16 = 16;
    let block_align    = channels * (bit_depth / 8);
    let byte_rate      = sample_rate * block_align as u32;

    let strl_start = w.seek(SeekFrom::Current(0))?;
    w.write_all(LIST)?;
    w.write_all(&0u32.to_le_bytes())?;
    w.write_all(STRL)?;

    // strh (audio)
    w.write_all(STRH)?;
    w.write_all(&56u32.to_le_bytes())?;
    w.write_all(AUDS)?;
    w.write_all(&[0u8; 4])?;        // fcc_handler (none for PCM)
    w.write_all(&[0u8; 8])?;        // flags … initial_frames
    w.write_all(&block_align.to_le_bytes())?; // scale = block alignment
    w.write_all(&byte_rate.to_le_bytes())?;   // rate
    w.write_all(&[0u8; 16])?;       // start, length, buffer_size, quality
    w.write_all(&block_align.to_le_bytes())?; // sample_size
    w.write_all(&[0u8; 10])?;       // padding

    // strf (WAVEFORMATEX)
    w.write_all(STRF)?;
    w.write_all(&18u32.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?;             // PCM
    w.write_all(&channels.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&bit_depth.to_le_bytes())?;
    w.write_all(&0u16.to_le_bytes())?;             // cbSize

    let strl_end = w.seek(SeekFrom::Current(0))?;
    w.seek(SeekFrom::Start(strl_start + 4))?;
    w.write_all(&((strl_end - strl_start - 8) as u32).to_le_bytes())?;
    w.seek(SeekFrom::Start(strl_end))?;
    Ok(())
}
