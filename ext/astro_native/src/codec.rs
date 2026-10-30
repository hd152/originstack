//! Byte-shuffle + LZ4 frame codec for the Phase 1 compressed frame store.

use crate::*;

/// Shuffle bytes by element position, improving LZ4's compression of numeric arrays.
/// One sequential pass over `input`, `itemsize` sequential write streams. 4-byte items (every
/// frame the pipeline stores) go 16 at a time through SSSE3 when the CPU has it: the scalar
/// byte loop was ~35 ms of the 70 ms it took to code a 75 MB RGB frame, single thread.
pub(crate) fn byte_shuffle_into(input: &[u8], itemsize: usize, output: &mut [u8]) {
    let count = input.len() / itemsize;
    if itemsize == 4 {
        let (p0, rest) = output.split_at_mut(count);
        let (p1, rest) = rest.split_at_mut(count);
        let (p2, p3) = rest.split_at_mut(count);
        #[allow(unused_mut)]
        let mut done = 0;
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("ssse3") {
            done = count / 16 * 16;
            // SAFETY: SSSE3 checked above; every slice holds at least `done` items.
            unsafe { shuffle4_ssse3(&input[..4 * done], p0, p1, p2, p3) };
        }
        for ((((item, b0), b1), b2), b3) in input[4 * done..]
            .chunks_exact(4)
            .zip(p0[done..].iter_mut())
            .zip(p1[done..].iter_mut())
            .zip(p2[done..].iter_mut())
            .zip(p3[done..].iter_mut())
        {
            *b0 = item[0];
            *b1 = item[1];
            *b2 = item[2];
            *b3 = item[3];
        }
        return;
    }
    for (index, item) in input.chunks_exact(itemsize).enumerate() {
        for (byte, &value) in item.iter().enumerate() {
            output[byte * count + index] = value;
        }
    }
}

/// 16 items per step: `pshufb` groups each register's bytes by plane, a 4x4 transpose of
/// 32-bit lanes makes one register per plane. `input.len()` is a multiple of 64.
///
/// # Safety
/// The CPU must support SSSE3; `input.len()` is a multiple of 64 and each plane holds `input.len() / 4` bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn shuffle4_ssse3(input: &[u8], p0: &mut [u8], p1: &mut [u8], p2: &mut [u8], p3: &mut [u8]) {
    use std::arch::x86_64::*;
    let m = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    for (g, block) in input.chunks_exact(64).enumerate() {
        let src = block.as_ptr() as *const __m128i;
        let s0 = _mm_shuffle_epi8(_mm_loadu_si128(src), m);
        let s1 = _mm_shuffle_epi8(_mm_loadu_si128(src.add(1)), m);
        let s2 = _mm_shuffle_epi8(_mm_loadu_si128(src.add(2)), m);
        let s3 = _mm_shuffle_epi8(_mm_loadu_si128(src.add(3)), m);
        let t0 = _mm_unpacklo_epi32(s0, s1);
        let t1 = _mm_unpacklo_epi32(s2, s3);
        let t2 = _mm_unpackhi_epi32(s0, s1);
        let t3 = _mm_unpackhi_epi32(s2, s3);
        let at = 16 * g;
        _mm_storeu_si128(p0[at..at + 16].as_mut_ptr() as *mut __m128i, _mm_unpacklo_epi64(t0, t1));
        _mm_storeu_si128(p1[at..at + 16].as_mut_ptr() as *mut __m128i, _mm_unpackhi_epi64(t0, t1));
        _mm_storeu_si128(p2[at..at + 16].as_mut_ptr() as *mut __m128i, _mm_unpacklo_epi64(t2, t3));
        _mm_storeu_si128(p3[at..at + 16].as_mut_ptr() as *mut __m128i, _mm_unpackhi_epi64(t2, t3));
    }
}

/// Inverse of `byte_shuffle_into`. 4-byte items go 16 at a time through SSE2 byte and
/// 16-bit interleaves (baseline x86-64, so no run-time check).
pub(crate) fn byte_unshuffle_into(input: &[u8], itemsize: usize, output: &mut [u8]) {
    let count = input.len() / itemsize;
    if itemsize == 4 {
        let (p0, rest) = input.split_at(count);
        let (p1, rest) = rest.split_at(count);
        let (p2, p3) = rest.split_at(count);
        #[cfg(target_arch = "x86_64")]
        let done = {
            let done = count / 16 * 16;
            // SAFETY: SSE2 is part of the x86-64 baseline; every slice holds `done` items.
            unsafe { unshuffle4_sse2(p0, p1, p2, p3, &mut output[..4 * done]) };
            done
        };
        #[cfg(not(target_arch = "x86_64"))]
        let done = 0;
        for ((((item, &b0), &b1), &b2), &b3) in output[4 * done..]
            .chunks_exact_mut(4)
            .zip(p0[done..].iter())
            .zip(p1[done..].iter())
            .zip(p2[done..].iter())
            .zip(p3[done..].iter())
        {
            item.copy_from_slice(&[b0, b1, b2, b3]);
        }
        return;
    }
    for (index, item) in output.chunks_exact_mut(itemsize).enumerate() {
        for (byte, value) in item.iter_mut().enumerate() {
            *value = input[byte * count + index];
        }
    }
}

/// `output.len()` is a multiple of 64; 16 items per step.
///
/// # Safety
/// `output.len()` is a multiple of 64 and each plane holds `output.len() / 4` bytes (SSE2 is x86-64 baseline).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn unshuffle4_sse2(p0: &[u8], p1: &[u8], p2: &[u8], p3: &[u8], output: &mut [u8]) {
    use std::arch::x86_64::*;
    for (g, block) in output.chunks_exact_mut(64).enumerate() {
        let at = 16 * g;
        let a = _mm_loadu_si128(p0[at..at + 16].as_ptr() as *const __m128i);
        let b = _mm_loadu_si128(p1[at..at + 16].as_ptr() as *const __m128i);
        let c = _mm_loadu_si128(p2[at..at + 16].as_ptr() as *const __m128i);
        let d = _mm_loadu_si128(p3[at..at + 16].as_ptr() as *const __m128i);
        let ab_lo = _mm_unpacklo_epi8(a, b);
        let ab_hi = _mm_unpackhi_epi8(a, b);
        let cd_lo = _mm_unpacklo_epi8(c, d);
        let cd_hi = _mm_unpackhi_epi8(c, d);
        let dst = block.as_mut_ptr() as *mut __m128i;
        _mm_storeu_si128(dst, _mm_unpacklo_epi16(ab_lo, cd_lo));
        _mm_storeu_si128(dst.add(1), _mm_unpackhi_epi16(ab_lo, cd_lo));
        _mm_storeu_si128(dst.add(2), _mm_unpacklo_epi16(ab_hi, cd_hi));
        _mm_storeu_si128(dst.add(3), _mm_unpackhi_epi16(ab_hi, cd_hi));
    }
}

/// Losslessly compress a C-contiguous numeric array using byte shuffle + pure-Rust LZ4.
/// Works on the caller's buffer in place (no input copy); one shuffle scratch buffer.
#[pyfunction]
pub(crate) fn compress_frame(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, u8>,
    itemsize: usize,
) -> PyResult<Py<PyBytes>> {
    if itemsize == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "itemsize must be positive",
        ));
    }
    let input = data
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("frame bytes must be C-contiguous"))?;
    if input.len() % itemsize != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "frame byte count must be divisible by itemsize",
        ));
    }
    if input.len() > u32::MAX as usize {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "LZ4 frame input exceeds the 4 GiB block limit",
        ));
    }
    let compressed = py.detach(|| {
        let mut shuffled = vec![0u8; input.len()];
        byte_shuffle_into(input, itemsize, &mut shuffled);
        lz4_flex::block::compress_prepend_size(&shuffled)
    });
    Ok(PyBytes::new(py, &compressed).unbind())
}

/// Decompress a byte-shuffled LZ4 frame and verify its expected output length.
/// Returns a new, writable, numpy-aligned uint8 array (callers `.view()` it as
/// the frame dtype); the compressed bytes are read in place.
#[pyfunction]
pub(crate) fn decompress_frame<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyBytes>,
    expected_len: usize,
    itemsize: usize,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    if itemsize == 0 || expected_len % itemsize != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected length must be divisible by positive itemsize",
        ));
    }
    let compressed = data.as_bytes();
    let (stored_len, block) = lz4_flex::block::uncompressed_size(compressed)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    if stored_len != expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "decompressed frame has {stored_len} bytes; expected {expected_len}"
        )));
    }
    let out = PyArray1::<u8>::zeros(py, expected_len, false);
    {
        let mut out_rw = numpy::PyArrayMethods::readwrite(&out);
        let out_slice = out_rw.as_slice_mut().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("output array must be contiguous")
        })?;
        py.detach(|| {
            let mut shuffled = vec![0u8; expected_len];
            let written = lz4_flex::block::decompress_into(block, &mut shuffled)
                .map_err(|e| e.to_string())?;
            if written != expected_len {
                return Err(format!(
                    "decompressed frame has {written} bytes; expected {expected_len}"
                ));
            }
            byte_unshuffle_into(&shuffled, itemsize, out_slice);
            Ok(())
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    }
    Ok(out)
}

/// Chunked frame format written by `compress_frame_chunked`: "OSZ2", itemsize (u32),
/// uncompressed length (u64), chunk length (u32), chunk count (u32), then one u32 entry per
/// byte plane of every chunk (chunk-major; `FRAME_V2_RAW` set = stored uncompressed, the
/// rest is the stored length), then the planes' bytes in the same order. Every chunk is
/// byte-shuffled on its own, so the shuffle runs in cache (the whole-frame shuffle of
/// `compress_frame` was a 75 MB scratch pass), and chunks are coded in parallel. On a
/// calibrated float32 frame the two middle mantissa bytes do not compress at all (1.00x)
/// while LZ4 still spent ~30 ms compressing and ~18 ms decompressing each; a plane whose
/// first `FRAME_V2_PROBE` bytes do not shrink by 3% is copied instead.
pub(crate) const FRAME_V2_MAGIC: &[u8; 4] = b"OSZ2";

pub(crate) const FRAME_V2_HEADER: usize = 24;

/// LZ4's window is 64 KB, so 1 MB chunks lose nothing measurable to the chunk boundaries.
pub(crate) const FRAME_V2_CHUNK_BYTES: usize = 1 << 20;

pub(crate) const FRAME_V2_PROBE: usize = 16 << 10;

pub(crate) const FRAME_V2_RAW: u32 = 1 << 31;

/// One chunk's planes: (entries, payload). `shuffled` holds the chunk byte-shuffled.
pub(crate) fn frame_v2_encode_chunk(shuffled: &[u8], itemsize: usize) -> (Vec<u32>, Vec<u8>) {
    let plane = shuffled.len() / itemsize;
    let mut entries = Vec::with_capacity(itemsize);
    let mut payload = Vec::with_capacity(shuffled.len());
    for p in shuffled.chunks_exact(plane) {
        let probe = &p[..p.len().min(FRAME_V2_PROBE)];
        let worth = p.len() < 2 * FRAME_V2_PROBE
            || lz4_flex::block::compress(probe).len() * 100 < probe.len() * 97;
        let packed = if worth { Some(lz4_flex::block::compress(p)) } else { None };
        match packed {
            Some(c) if c.len() < p.len() => {
                entries.push(c.len() as u32);
                payload.extend_from_slice(&c);
            }
            _ => {
                entries.push(p.len() as u32 | FRAME_V2_RAW);
                payload.extend_from_slice(p);
            }
        }
    }
    (entries, payload)
}

/// Losslessly compress a C-contiguous numeric array in the chunked format (see
/// `FRAME_V2_MAGIC`); `decompress_frame_into` reads it back.
#[pyfunction]
pub(crate) fn compress_frame_chunked(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, u8>,
    itemsize: usize,
) -> PyResult<Py<PyBytes>> {
    if itemsize == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "itemsize must be positive",
        ));
    }
    let input = data
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("frame bytes must be C-contiguous"))?;
    if input.len() % itemsize != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "frame byte count must be divisible by itemsize",
        ));
    }
    let chunk = (FRAME_V2_CHUNK_BYTES / itemsize).max(1) * itemsize;
    if input.len().div_ceil(chunk) * itemsize > u32::MAX as usize {
        return Err(pyo3::exceptions::PyValueError::new_err("frame too large"));
    }
    let encoded = py.detach(|| {
        let parts: Vec<(Vec<u32>, Vec<u8>)> = input
            .par_chunks(chunk)
            .map_init(
                || vec![0u8; chunk],
                |scratch, src| {
                    let s = &mut scratch[..src.len()];
                    byte_shuffle_into(src, itemsize, s);
                    frame_v2_encode_chunk(s, itemsize)
                },
            )
            .collect();
        let body: usize = parts.iter().map(|(_, b)| b.len()).sum();
        let mut out = Vec::with_capacity(FRAME_V2_HEADER + 4 * itemsize * parts.len() + body);
        out.extend_from_slice(FRAME_V2_MAGIC);
        out.extend_from_slice(&(itemsize as u32).to_le_bytes());
        out.extend_from_slice(&(input.len() as u64).to_le_bytes());
        out.extend_from_slice(&(chunk as u32).to_le_bytes());
        out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
        for (entries, _) in &parts {
            for e in entries {
                out.extend_from_slice(&e.to_le_bytes());
            }
        }
        for (_, b) in &parts {
            out.extend_from_slice(b);
        }
        out
    });
    Ok(PyBytes::new(py, &encoded).unbind())
}

/// Chunk buffers reused across `compress_frame_chunked_to_file` calls. A Phase 1 worker
/// writes two frames per light; fresh ~1 MB allocations are zero pages the OS faults in on
/// every frame, which is memory traffic the other workers are short of.
pub(crate) static FRAME_V2_POOL: std::sync::OnceLock<std::sync::Mutex<Vec<Vec<u8>>>> =
    std::sync::OnceLock::new();

pub(crate) fn frame_v2_take(n: usize) -> Vec<u8> {
    let pool = FRAME_V2_POOL.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    if let Ok(mut p) = pool.lock() {
        if let Some(i) = p.iter().position(|v| v.len() >= n) {
            return p.swap_remove(i);
        }
    }
    vec![0u8; n]
}

pub(crate) fn frame_v2_give(v: Vec<u8>) {
    if let Some(pool) = FRAME_V2_POOL.get() {
        if let Ok(mut p) = pool.lock() {
            p.push(v);
        }
    }
}

/// `frame_v2_encode_chunk` writing into `out` (sized for the worst case) instead of fresh
/// vectors; the same planes, entries and bytes. Returns (entries, bytes used).
pub(crate) fn frame_v2_encode_chunk_into(shuffled: &[u8], itemsize: usize, out: &mut [u8]) -> (Vec<u32>, usize) {
    let plane = shuffled.len() / itemsize;
    let mut entries = Vec::with_capacity(itemsize);
    let mut probe_out = [0u8; lz4_flex::block::get_maximum_output_size(FRAME_V2_PROBE)];
    let mut used = 0usize;
    for p in shuffled.chunks_exact(plane) {
        let probe = &p[..p.len().min(FRAME_V2_PROBE)];
        let worth = p.len() < 2 * FRAME_V2_PROBE
            || lz4_flex::block::compress_into(probe, &mut probe_out).map_or(false, |n| {
                n * 100 < probe.len() * 97
            });
        let room = lz4_flex::block::get_maximum_output_size(p.len());
        let packed = if worth {
            lz4_flex::block::compress_into(p, &mut out[used..used + room]).ok()
        } else {
            None
        };
        match packed {
            Some(n) if n < p.len() => {
                entries.push(n as u32);
                used += n;
            }
            _ => {
                entries.push(p.len() as u32 | FRAME_V2_RAW);
                out[used..used + p.len()].copy_from_slice(p);
                used += p.len();
            }
        }
    }
    (entries, used)
}

/// `compress_frame_chunked` written straight to ``path`` (created or truncated); returns the
/// byte count. The same bytes as writing that function's result, without its copies: each
/// chunk is coded into a pooled buffer and written from there (the bytes path concatenated
/// the chunks, copied them into a Python bytes object, and the file write copied them again
/// -- three passes over ~55 MB per RGB frame while every Phase 1 worker competes for memory).
#[pyfunction]
pub(crate) fn compress_frame_chunked_to_file(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, u8>,
    itemsize: usize,
    path: std::path::PathBuf,
) -> PyResult<usize> {
    use std::io::Write;
    if itemsize == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "itemsize must be positive",
        ));
    }
    let input = data
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("frame bytes must be C-contiguous"))?;
    if input.len() % itemsize != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "frame byte count must be divisible by itemsize",
        ));
    }
    let chunk = (FRAME_V2_CHUNK_BYTES / itemsize).max(1) * itemsize;
    if input.len().div_ceil(chunk) * itemsize > u32::MAX as usize {
        return Err(pyo3::exceptions::PyValueError::new_err("frame too large"));
    }
    let room = itemsize * lz4_flex::block::get_maximum_output_size(chunk / itemsize);
    py.detach(|| -> std::io::Result<usize> {
        let parts: Vec<(Vec<u32>, Vec<u8>, usize)> = input
            .par_chunks(chunk)
            .map_init(
                || vec![0u8; chunk],
                |scratch, src| {
                    let s = &mut scratch[..src.len()];
                    byte_shuffle_into(src, itemsize, s);
                    let mut buf = frame_v2_take(room);
                    let (entries, used) = frame_v2_encode_chunk_into(s, itemsize, &mut buf);
                    (entries, buf, used)
                },
            )
            .collect();
        let mut head = Vec::with_capacity(FRAME_V2_HEADER + 4 * itemsize * parts.len());
        head.extend_from_slice(FRAME_V2_MAGIC);
        head.extend_from_slice(&(itemsize as u32).to_le_bytes());
        head.extend_from_slice(&(input.len() as u64).to_le_bytes());
        head.extend_from_slice(&(chunk as u32).to_le_bytes());
        head.extend_from_slice(&(parts.len() as u32).to_le_bytes());
        for (entries, _, _) in &parts {
            for e in entries {
                head.extend_from_slice(&e.to_le_bytes());
            }
        }
        let mut written = head.len();
        let result = (|| {
            let mut file = std::fs::File::create(&path)?;
            file.write_all(&head)?;
            for (_, buf, used) in &parts {
                file.write_all(&buf[..*used])?;
                written += used;
            }
            Ok(written)
        })();
        for (_, buf, _) in parts {
            frame_v2_give(buf);
        }
        result
    })
    .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
}

pub(crate) fn frame_v2_u32(src: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(src[at..at + 4].try_into().unwrap())
}

/// Decode a stored frame (chunked or whole-frame LZ4) into ``dst``.
pub(crate) fn frame_decode_into(src: &[u8], dst: &mut [u8], itemsize: usize) -> Result<(), String> {
    let expected = dst.len();
    if itemsize == 0 || expected % itemsize != 0 {
        return Err("expected length must be divisible by positive itemsize".into());
    }
    if src.len() >= FRAME_V2_HEADER && &src[..4] == FRAME_V2_MAGIC {
        let stored_item = frame_v2_u32(src, 4) as usize;
        let total = u64::from_le_bytes(src[8..16].try_into().unwrap()) as usize;
        let chunk = frame_v2_u32(src, 16) as usize;
        let n = frame_v2_u32(src, 20) as usize;
        if stored_item != itemsize {
            return Err(format!("frame itemsize {stored_item}; expected {itemsize}"));
        }
        if total != expected {
            return Err(format!("decompressed frame has {total} bytes; expected {expected}"));
        }
        if chunk == 0 || chunk % itemsize != 0 || n != total.div_ceil(chunk) {
            return Err("corrupt chunked frame header".into());
        }
        let table_end = FRAME_V2_HEADER + 4 * n * itemsize;
        if src.len() < table_end {
            return Err("truncated chunked frame".into());
        }
        // payload offset of every chunk, from the per-plane stored lengths
        let mut offsets = Vec::with_capacity(n + 1);
        let mut at = table_end;
        offsets.push(at);
        for i in 0..n {
            for p in 0..itemsize {
                at += (frame_v2_u32(src, FRAME_V2_HEADER + 4 * (i * itemsize + p)) & !FRAME_V2_RAW) as usize;
            }
            offsets.push(at);
        }
        if at != src.len() {
            return Err("truncated chunked frame".into());
        }
        return dst.par_chunks_mut(chunk).enumerate().try_for_each_init(
            || vec![0u8; chunk],
            |scratch, (i, d)| {
                let s = &mut scratch[..d.len()];
                let plane = d.len() / itemsize;
                let mut pos = offsets[i];
                for (p, sp) in s.chunks_exact_mut(plane).enumerate() {
                    let e = frame_v2_u32(src, FRAME_V2_HEADER + 4 * (i * itemsize + p));
                    let len = (e & !FRAME_V2_RAW) as usize;
                    let stored = &src[pos..pos + len];
                    pos += len;
                    if e & FRAME_V2_RAW != 0 {
                        if len != plane {
                            return Err(format!("chunk {i} plane {p}: {len} raw bytes; expected {plane}"));
                        }
                        sp.copy_from_slice(stored);
                    } else {
                        let written =
                            lz4_flex::block::decompress_into(stored, sp).map_err(|e| e.to_string())?;
                        if written != plane {
                            return Err(format!("chunk {i} plane {p} has {written} bytes; expected {plane}"));
                        }
                    }
                }
                byte_unshuffle_into(s, itemsize, d);
                Ok(())
            },
        );
    }
    let (stored_len, block) = lz4_flex::block::uncompressed_size(src).map_err(|e| e.to_string())?;
    if stored_len != expected {
        return Err(format!("decompressed frame has {stored_len} bytes; expected {expected}"));
    }
    let mut shuffled = vec![0u8; expected];
    let written = lz4_flex::block::decompress_into(block, &mut shuffled).map_err(|e| e.to_string())?;
    if written != expected {
        return Err(format!("decompressed frame has {written} bytes; expected {expected}"));
    }
    byte_unshuffle_into(&shuffled, itemsize, dst);
    Ok(())
}

/// Decompress a frame written by `compress_frame_chunked` (or `..._to_file`) *or*
/// `compress_frame` straight into ``out`` (a C-contiguous uint8 view of the destination
/// frame; its length is the expected uncompressed size). The caller allocates ``out``
/// (``np.empty``: no zero fill), and chunked frames are unshuffled chunk by chunk into it,
/// in parallel, with no frame-sized scratch buffer.
#[pyfunction]
pub(crate) fn decompress_frame_into(
    py: Python<'_>,
    data: &Bound<'_, PyBytes>,
    mut out: numpy::PyReadwriteArray1<'_, u8>,
    itemsize: usize,
) -> PyResult<()> {
    let src = data.as_bytes();
    let dst = out
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("output array must be contiguous"))?;
    py.detach(|| frame_decode_into(src, dst, itemsize))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// File read buffers reused across `decompress_frame_file_into` calls.
pub(crate) static FRAME_READ_POOL: std::sync::OnceLock<std::sync::Mutex<Vec<Vec<u8>>>> = std::sync::OnceLock::new();

/// `decompress_frame_into` reading the stored frame from ``path`` into a pooled buffer
/// (a Python ``read()`` allocated and filled a fresh frame-sized bytes object per read).
/// A missing file raises ``FileNotFoundError``.
#[pyfunction]
pub(crate) fn decompress_frame_file_into(
    py: Python<'_>,
    path: std::path::PathBuf,
    mut out: numpy::PyReadwriteArray1<'_, u8>,
    itemsize: usize,
) -> PyResult<()> {
    use std::io::Read;
    let dst = out
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("output array must be contiguous"))?;
    let pool = FRAME_READ_POOL.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut buf = pool.lock().ok().and_then(|mut p| p.pop()).unwrap_or_default();
    let result = py.detach(|| -> PyResult<()> {
        buf.clear();
        let mut file = std::fs::File::open(&path).map_err(|e| {
            let msg = format!("{}: {e}", path.display());
            if e.kind() == std::io::ErrorKind::NotFound {
                pyo3::exceptions::PyFileNotFoundError::new_err(msg)
            } else {
                pyo3::exceptions::PyOSError::new_err(msg)
            }
        })?;
        if let Ok(meta) = file.metadata() {
            buf.reserve(meta.len() as usize);
        }
        file.read_to_end(&mut buf)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{}: {e}", path.display())))?;
        frame_decode_into(&buf, dst, itemsize).map_err(pyo3::exceptions::PyValueError::new_err)
    });
    if let Ok(mut p) = pool.lock() {
        p.push(buf);
    }
    result
}
