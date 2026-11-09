//! Proper coaddition (src/proper_coadd.py, `--gpu-coadd`) on the GPU. Cargo feature `gpu`.
//!
//! Per frame: reject outliers against the reference stack, subtract the sky and zero-pad the
//! three channels, render the frame's PSF plane, 2-D FFT them (radix-4 style Stockham-free
//! Cooley-Tukey rows in workgroup memory, a tiled transpose, rows again: the spectrum stays
//! transposed), and accumulate `num += wn * F * conj(P)`, `den += wf * |P|^2` over the half
//! plane. At the end `R = num / sqrt(den)` is completed by Hermitian symmetry and
//! inverse-transformed. Channels 0 and 1 share one complex FFT plane (`ch0 + i ch1`, split again
//! by symmetry); channel 2 and the PSF each get their own plane.
//!
//! Numerics (the output is NOT bit-identical to the CPU path; see dev-notes/gpu-coadd.md):
//! - f32 throughout (WGSL has no f64); the twiddle table is built in f64 on the host, because
//!   GPU `sin`/`cos` are only accurate to ~2^-11;
//! - the PSF must not share an FFT plane with data: FFT error scales with the plane's largest
//!   component, and a sum-1 PSF next to values of ~1000 loses its spectrum's phase at high
//!   frequency (25-64% noise-like error, measured), which `R = F conj(P) / |P|` amplifies;
//! - planes are powers of two (rows and columns <= 4096), chosen by the caller.

use crate::gpu::{arg_err, bytes_of, check, dispatch, gpu, make_kernel, read_back, rt_err, storage_buf, uniform_buf, wait, Bind, Gpu, Kernel};
use crate::*;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

// ORIGINSTACK_GPU_PROFILE=1: every frame's device work is cut into stages that are submitted and
// waited for one by one, so the time of each can be read (`gpu_coadd_profile`). That serialises
// the pipeline: the total is slower than a normal run; only the proportions mean anything.
const STAGES: usize = 6; // upload, prep, row FFTs, transposes, column FFTs, accumulate
static PROFILE: Mutex<[f64; STAGES + 1]> = Mutex::new([0.0; STAGES + 1]);

fn profiling() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ORIGINSTACK_GPU_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

fn lap(g: &Gpu, enc: &mut wgpu::CommandEncoder, stage: usize, t: &mut Instant) -> PyResult<()> {
    let done = std::mem::replace(enc, g.device.create_command_encoder(&Default::default()));
    let sub = g.queue.submit(Some(done.finish()));
    wait(g, Some(sub))?;
    PROFILE.lock().unwrap_or_else(|p| p.into_inner())[stage] += t.elapsed().as_secs_f64();
    *t = Instant::now();
    Ok(())
}

/// Seconds spent per stage since the last call (upload, prep, row FFTs, transposes, column FFTs,
/// accumulate) and the number of frames, then reset. All zero unless ORIGINSTACK_GPU_PROFILE is set.
#[pyfunction]
pub(crate) fn gpu_coadd_profile() -> Vec<f64> {
    let mut p = PROFILE.lock().unwrap_or_else(|p| p.into_inner());
    let out = p.to_vec();
    *p = [0.0; STAGES + 1];
    out
}

const FFT_WGSL: &str = r#"
struct U { n: u32, log2n: u32, inverse: u32, skip: u32, scale: f32, ph: u32, _b: f32, _c: f32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> tw: array<vec2<f32>>;
var<workgroup> buf: array<vec2<f32>, 4096>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n = u.n;
    var row = wid.x;
    // skip > 0: only the PSF's rows (0..=skip and the last skip rows of ph) are nonzero
    if (u.skip > 0u && row > u.skip) { row = u.ph - 2u * u.skip - 1u + row; }
    let base = row * n;
    for (var i = lid; i < n; i += 256u) {
        let j = reverseBits(i) >> (32u - u.log2n);
        buf[j] = src[base + i];
    }
    workgroupBarrier();
    let inv = u.inverse == 1u;
    var s = 1u;
    if ((u.log2n & 1u) == 1u) {
        // odd stage count: one plain radix-2 stage first (twiddle 1)
        for (var b = lid; b < n / 2u; b += 256u) {
            let i0 = b << 1u;
            let x = buf[i0 + 1u];
            let y = buf[i0];
            buf[i0 + 1u] = y - x;
            buf[i0] = y + x;
        }
        workgroupBarrier();
        s = 2u;
    }
    // two radix-2 stages (s and s+1) per barrier, four elements per thread
    for (; s < u.log2n; s += 2u) {
        let q = 1u << (s - 1u);
        let step1 = 4096u >> s;
        let step2 = 4096u >> (s + 1u);
        for (var b = lid; b < n / 4u; b += 256u) {
            let k = b & (q - 1u);
            let a0 = ((b >> (s - 1u)) << (s + 1u)) + k;
            let a1 = a0 + q;
            let a2 = a1 + q;
            let a3 = a2 + q;
            var w1 = tw[k * step1];
            var wa = tw[k * step2];
            var wb = tw[(k + q) * step2];
            if (inv) { w1.y = -w1.y; wa.y = -wa.y; wb.y = -wb.y; }
            let x0 = buf[a0];
            let x1 = buf[a1];
            let x2 = buf[a2];
            let x3 = buf[a3];
            var t = vec2<f32>(x1.x * w1.x - x1.y * w1.y, x1.x * w1.y + x1.y * w1.x);
            let y0 = x0 + t;
            let y1 = x0 - t;
            t = vec2<f32>(x3.x * w1.x - x3.y * w1.y, x3.x * w1.y + x3.y * w1.x);
            let y2 = x2 + t;
            let y3 = x2 - t;
            t = vec2<f32>(y2.x * wa.x - y2.y * wa.y, y2.x * wa.y + y2.y * wa.x);
            buf[a0] = y0 + t;
            buf[a2] = y0 - t;
            t = vec2<f32>(y3.x * wb.x - y3.y * wb.y, y3.x * wb.y + y3.y * wb.x);
            buf[a1] = y1 + t;
            buf[a3] = y1 - t;
        }
        workgroupBarrier();
    }
    for (var i = lid; i < n; i += 256u) {
        dst[base + i] = buf[i] * u.scale;
    }
}
"#;

const TRANSPOSE_WGSL: &str = r#"
struct U { rows: u32, cols: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
var<workgroup> tile: array<vec2<f32>, 272>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let x = wid.x * 16u + lid.x;
    let y = wid.y * 16u + lid.y;
    if (x < u.cols && y < u.rows) { tile[lid.y * 17u + lid.x] = src[y * u.cols + x]; }
    workgroupBarrier();
    let ox = wid.y * 16u + lid.x;
    let oy = wid.x * 16u + lid.y;
    if (ox < u.rows && oy < u.cols) { dst[oy * u.rows + ox] = tile[lid.x * 17u + lid.y]; }
}
"#;

const PREP_WGSL: &str = r#"
struct U {
    h: u32, w: u32, ph: u32, pw: u32, pad: u32, r: u32, _a: u32, _b: u32,
    k: f32, frac: f32, f: f32, _c: f32,
    sky: vec4<f32>, s2: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> refsig: array<f32>;
@group(0) @binding(3) var<storage, read> stamp: array<f32>;
@group(0) @binding(4) var<storage, read_write> z1: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> z2: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> z3: array<vec2<f32>>;

fn rej(v: f32, e: f32, s2: f32) -> f32 {
    let resid = v - e;
    var t = select(0.0, e, e > 0.0);
    t = t * u.frac;
    t = t * t + s2;
    return select(v, e, abs(resid) > u.k * sqrt(t));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= u.pw || y >= u.ph) { return; }
    var a = vec3<f32>(0.0, 0.0, 0.0);
    if (y >= u.pad && y < u.pad + u.h && x >= u.pad && x < u.pad + u.w) {
        let pix = (y - u.pad) * u.w + (x - u.pad);
        let hw = u.h * u.w;
        a.x = rej(src[pix * 3u] - u.sky.x, refsig[pix] * u.f, u.s2.x);
        a.y = rej(src[pix * 3u + 1u] - u.sky.y, refsig[hw + pix] * u.f, u.s2.y);
        a.z = rej(src[pix * 3u + 2u] - u.sky.z, refsig[2u * hw + pix] * u.f, u.s2.z);
    }
    var dy: i32 = 100000;
    var dx: i32 = 100000;
    let r = i32(u.r);
    if (y <= u.r) { dy = i32(y); } else if (y + u.r >= u.ph) { dy = i32(y) - i32(u.ph); }
    if (x <= u.r) { dx = i32(x); } else if (x + u.r >= u.pw) { dx = i32(x) - i32(u.pw); }
    var psf = 0.0;
    if (abs(dy) <= r && abs(dx) <= r) { psf = stamp[u32((dy + r) * (2 * r + 1) + (dx + r))]; }
    z1[y * u.pw + x] = vec2<f32>(a.x, a.y);
    z2[y * u.pw + x] = vec2<f32>(a.z, 0.0);
    z3[y * u.pw + x] = vec2<f32>(psf, 0.0);
}
"#;

const ACCUM_WGSL: &str = r#"
struct U { ph: u32, pw: u32, _a: u32, _b: u32, wn: vec4<f32>, wf: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> z1: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> z2: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> z3: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> num: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> den: array<f32>;

fn cmul_conj(a: vec2<f32>, p: vec2<f32>) -> vec2<f32> {   // a * conj(p)
    return vec2<f32>(a.x * p.x + a.y * p.y, a.y * p.x - a.x * p.y);
}

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ky = gid.x;
    let kx = gid.y;
    let ph = u.ph;
    let pw = u.pw;
    if (ky >= ph || kx > pw / 2u) { return; }
    let mx = (pw - kx) % pw;
    let my = (ph - ky) % ph;
    let ik = kx * ph + ky;
    let im = mx * ph + my;
    let a = z1[ik];
    let am = z1[im];
    // z = u + i v  ->  U = (Z[k] + conj(Z[-k])) / 2, V = (Z[k] - conj(Z[-k])) / (2i)
    let a0 = 0.5 * vec2<f32>(a.x + am.x, a.y - am.y);
    let d1 = vec2<f32>(a.x - am.x, a.y + am.y);
    let a1 = 0.5 * vec2<f32>(d1.y, -d1.x);
    let a2 = z2[ik];
    let p = z3[ik];
    let nb = ph * (pw / 2u + 1u);
    let pp = p.x * p.x + p.y * p.y;
    num[ik] += u.wn.x * cmul_conj(a0, p);
    num[nb + ik] += u.wn.y * cmul_conj(a1, p);
    num[2u * nb + ik] += u.wn.z * cmul_conj(a2, p);
    den[ik] += u.wf.x * pp;
    den[nb + ik] += u.wf.y * pp;
    den[2u * nb + ik] += u.wf.z * pp;
}
"#;

const FINAL_WGSL: &str = r#"
struct U { ph: u32, pw: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> num: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> den: array<f32>;
@group(0) @binding(3) var<storage, read_write> y1: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> y2: array<vec2<f32>>;

fn rval(c: u32, idx: u32, conj: bool) -> vec2<f32> {
    let nb = u.ph * (u.pw / 2u + 1u);
    var r = num[c * nb + idx] / sqrt(max(den[c * nb + idx], 1e-30));
    if (conj) { r.y = -r.y; }
    return r;
}

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ky = gid.x;
    let kx = gid.y;
    let ph = u.ph;
    let pw = u.pw;
    if (ky >= ph || kx >= pw) { return; }
    var idx = kx * ph + ky;
    var conj = false;
    if (kx > pw / 2u) {
        idx = (pw - kx) * ph + (ph - ky) % ph;
        conj = true;
    }
    let r0 = rval(0u, idx, conj);
    let r1 = rval(1u, idx, conj);
    let r2 = rval(2u, idx, conj);
    y1[kx * ph + ky] = vec2<f32>(r0.x - r1.y, r0.y + r1.x);
    y2[kx * ph + ky] = r2;
}
"#;

struct Kernels {
    fft: Kernel,
    transpose: Kernel,
    prep: Kernel,
    accum: Kernel,
    fin: Kernel,
}

static KERNELS: OnceLock<Kernels> = OnceLock::new();

fn kernels(g: &Gpu) -> PyResult<&'static Kernels> {
    let k = KERNELS.get_or_init(|| {
        use Bind::*;
        Kernels {
            fft: make_kernel(g, "fft", FFT_WGSL, &[Uniform, Ro, Rw, Ro]),
            transpose: make_kernel(g, "transpose", TRANSPOSE_WGSL, &[Uniform, Ro, Rw]),
            prep: make_kernel(g, "prep", PREP_WGSL, &[Uniform, Ro, Ro, Ro, Rw, Rw, Rw]),
            accum: make_kernel(g, "accum", ACCUM_WGSL, &[Uniform, Ro, Ro, Ro, Rw, Rw]),
            fin: make_kernel(g, "final", FINAL_WGSL, &[Uniform, Ro, Ro, Rw, Rw]),
        }
    });
    check()?;
    Ok(k)
}

/// exp(-2 pi i j / 4096) for j in 0..2048 as interleaved (re, im), built in f64.
fn twiddles() -> Vec<f32> {
    let mut t = Vec::with_capacity(4096);
    for j in 0..2048 {
        let a = -2.0 * std::f64::consts::PI * f64::from(j) / 4096.0;
        t.push(a.cos() as f32);
        t.push(a.sin() as f32);
    }
    t
}

fn u32s(v: [u32; 4]) -> Vec<u8> {
    bytes_of(&v).to_vec()
}

fn check_dims(ph: usize, pw: usize) -> PyResult<()> {
    let ok = |n: usize| n.is_power_of_two() && (16..=4096).contains(&n);
    if !ok(ph) || !ok(pw) {
        return Err(arg_err("plane sizes must be powers of two in 16..=4096"));
    }
    Ok(())
}

fn gx(n: usize, d: usize) -> u32 {
    n.div_ceil(d) as u32
}

/// Encode a batched row FFT: `rows` rows of length `n`, src -> dst. With `skip = r > 0` only the
/// 2r+1 nonzero rows of a PSF plane (of `ph` rows) are transformed; the others are left as they
/// are in `dst`, which must therefore be a buffer that is zero outside them.
#[allow(clippy::too_many_arguments)]
fn enc_fft(g: &Gpu, ks: &Kernels, enc: &mut wgpu::CommandEncoder, tw: &wgpu::Buffer, src: &wgpu::Buffer,
           dst: &wgpu::Buffer, n: usize, rows: usize, inverse: bool, scale: f32, skip: usize, ph: usize) {
    let mut ub = u32s([n as u32, n.trailing_zeros(), u32::from(inverse), skip as u32]);
    ub.extend_from_slice(bytes_of(&[scale]));
    ub.extend_from_slice(&(ph as u32).to_ne_bytes());
    ub.extend_from_slice(bytes_of(&[0f32, 0f32]));
    let u = uniform_buf(g, &ub);
    let rows = if skip > 0 { 2 * skip + 1 } else { rows };
    dispatch(g, enc, &ks.fft, &[&u, src, dst, tw], (rows as u32, 1, 1));
}

fn enc_transpose(g: &Gpu, ks: &Kernels, enc: &mut wgpu::CommandEncoder, src: &wgpu::Buffer, dst: &wgpu::Buffer,
                 rows: usize, cols: usize) {
    let u = uniform_buf(g, &u32s([rows as u32, cols as u32, 0, 0]));
    dispatch(g, enc, &ks.transpose, &[&u, src, dst], (gx(cols, 16), gx(rows, 16), 1));
}

/// Test hook: 2-D FFT of an (ph, pw, 2) float32 array holding complex values, through the same
/// row-FFT / transpose kernels the coadd uses. Inverse transforms are scaled by 1/(ph*pw).
#[pyfunction]
pub(crate) fn gpu_fft2<'py>(py: Python<'py>, plane: PyReadonlyArray3<'py, f32>, inverse: bool) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let g = gpu()?;
    let ks = kernels(g)?;
    let arr = plane.as_array();
    let (ph, pw) = (arr.shape()[0], arr.shape()[1]);
    check_dims(ph, pw)?;
    if arr.shape()[2] != 2 {
        return Err(arg_err("plane must be (ph, pw, 2)"));
    }
    let flat = arr.as_slice().ok_or_else(|| arg_err("plane must be C-contiguous"))?;
    let bytes = (ph * pw * 8) as u64;
    let io = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let a = storage_buf(g, bytes, io)?;
    let b = storage_buf(g, bytes, io)?;
    let tw = storage_buf(g, 2048 * 8, wgpu::BufferUsages::COPY_DST)?;
    g.queue.write_buffer(&tw, 0, bytes_of(&twiddles()));
    g.queue.write_buffer(&a, 0, bytes_of(flat));
    let (sx, sy) = if inverse { (1.0 / pw as f32, 1.0 / ph as f32) } else { (1.0, 1.0) };
    let mut enc = g.device.create_command_encoder(&Default::default());
    enc_fft(g, ks, &mut enc, &tw, &a, &b, pw, ph, inverse, sx, 0, 0);
    enc_transpose(g, ks, &mut enc, &b, &a, ph, pw);
    enc_fft(g, ks, &mut enc, &tw, &a, &b, ph, pw, inverse, sy, 0, 0);
    enc_transpose(g, ks, &mut enc, &b, &a, pw, ph);
    g.queue.submit(Some(enc.finish()));
    check()?;
    let out = read_back(g, &a, bytes)?;
    Ok(numpy::ndarray::Array3::from_shape_vec((ph, pw, 2), out).map_err(|e| rt_err(e.to_string()))?.into_pyarray(py))
}

/// One accumulation in flight: the device buffers and the frame counter.
struct Session {
    h: usize,
    w: usize,
    ph: usize,
    pw: usize,
    pad: usize,
    r: usize,
    k: f32,
    frac: f32,
    tw: wgpu::Buffer,
    refbuf: wgpu::Buffer,
    srcs: [wgpu::Buffer; 2],
    stamp: wgpu::Buffer,
    z: [wgpu::Buffer; 3],
    t: [wgpu::Buffer; 3],
    t4: wgpu::Buffer,
    num: wgpu::Buffer,
    den: wgpu::Buffer,
    done: usize,
    prev: Option<wgpu::SubmissionIndex>,
}

static SESSION: Mutex<Option<Session>> = Mutex::new(None);

fn session() -> std::sync::MutexGuard<'static, Option<Session>> {
    SESSION.lock().unwrap_or_else(|p| p.into_inner())
}

/// Device bytes a session of this geometry allocates: seven (ph x pw) complex planes (three data,
/// three transposed, one PSF row-pass), the half-plane numerator and denominator, the reference,
/// two frame upload slots, the stamp and the twiddles.
fn session_bytes(h: usize, w: usize, ph: usize, pw: usize, r: usize) -> u64 {
    let plane = (ph * pw * 8) as u64;
    let half = (ph * (pw / 2 + 1)) as u64;
    7 * plane + 3 * half * 8 + 3 * half * 4 + (3 * h * w * 4) as u64 + 2 * (h * w * 3 * 4) as u64
        + ((2 * r + 1) * (2 * r + 1) * 4) as u64 + 2048 * 8
}

fn budget_bytes() -> u64 {
    let mb = std::env::var("ORIGINSTACK_GPU_MAX_MB").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(3072);
    mb << 20
}

/// Start an accumulation: `ref_sig` (3, H, W) is the reference stack minus its sky (already on
/// the frames' flux scale); planes are `ph` x `pw` (powers of two) with the frame placed at
/// offset (pad, pad); `r` is the PSF stamp radius. Replaces any session left over. Raises
/// RuntimeError when the device cannot hold it; nothing is left allocated then.
#[pyfunction]
pub(crate) fn gpu_coadd_begin(ref_sig: PyReadonlyArray3<'_, f32>, k: f32, frac: f32, pad: usize, r: usize, ph: usize, pw: usize) -> PyResult<()> {
    *session() = None;
    let g = gpu()?;
    kernels(g)?;
    check_dims(ph, pw)?;
    let rs = ref_sig.as_array();
    let (c, h, w) = (rs.shape()[0], rs.shape()[1], rs.shape()[2]);
    if c != 3 {
        return Err(arg_err("the GPU coadd handles 3-channel frames only"));
    }
    if ph < h + pad || pw < w + pad || r >= pad || 2 * r + 1 > ph.min(pw) {
        return Err(arg_err("plane too small for the frame, pad and PSF stamp"));
    }
    let need = session_bytes(h, w, ph, pw, r);
    if need > budget_bytes() {
        return Err(rt_err(format!(
            "the GPU coadd needs about {} MB of device memory, over the {} MB budget (ORIGINSTACK_GPU_MAX_MB)",
            need >> 20,
            budget_bytes() >> 20
        )));
    }
    let rsf = rs.as_slice().ok_or_else(|| arg_err("ref_sig must be C-contiguous"))?;
    let plane = (ph * pw * 8) as u64;
    let half = ph * (pw / 2 + 1);
    let dst = wgpu::BufferUsages::COPY_DST;
    let tw = storage_buf(g, 2048 * 8, dst)?;
    g.queue.write_buffer(&tw, 0, bytes_of(&twiddles()));
    let refbuf = storage_buf(g, (rsf.len() * 4) as u64, dst)?;
    g.queue.write_buffer(&refbuf, 0, bytes_of(rsf));
    let mkp = || storage_buf(g, plane, wgpu::BufferUsages::COPY_SRC);
    let sess = Session {
        h,
        w,
        ph,
        pw,
        pad,
        r,
        k,
        frac,
        tw,
        refbuf,
        srcs: [storage_buf(g, (h * w * 12) as u64, dst)?, storage_buf(g, (h * w * 12) as u64, dst)?],
        stamp: storage_buf(g, ((2 * r + 1) * (2 * r + 1) * 4) as u64, dst)?,
        z: [mkp()?, mkp()?, mkp()?],
        t: [mkp()?, mkp()?, mkp()?],
        t4: mkp()?,
        num: storage_buf(g, (3 * half * 8) as u64, wgpu::BufferUsages::empty())?,
        den: storage_buf(g, (3 * half * 4) as u64, wgpu::BufferUsages::empty())?,
        done: 0,
        prev: None,
    };
    check()?;
    *session() = Some(sess);
    Ok(())
}

/// Accumulate the frames `idx` of `aligned` (N, H, W, 3). The GIL is released while the device
/// works, so the caller can prepare the next chunk meanwhile. On any error the session is
/// dropped (never a half-finished accumulation) and RuntimeError is raised.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_coadd_add(
    py: Python<'_>,
    aligned: PyReadonlyArray4<'_, f32>,
    idx: Vec<usize>,
    f: Vec<f32>,
    sky: Vec<[f32; 3]>,
    sig2: Vec<[f32; 3]>,
    wn: Vec<[f32; 3]>,
    wf: Vec<[f32; 3]>,
    stamps: PyReadonlyArray3<'_, f32>,
) -> PyResult<()> {
    let g = gpu()?;
    let ks = kernels(g)?;
    let al = aligned.as_array();
    let alf = al.as_slice().ok_or_else(|| arg_err("aligned must be C-contiguous"))?;
    let st = stamps.as_array();
    let sd = st.shape().to_vec();
    let stf = st.as_slice().ok_or_else(|| arg_err("stamps must be C-contiguous"))?;
    let nf = idx.len();
    if [f.len(), sky.len(), sig2.len(), wn.len(), wf.len(), sd[0]].iter().any(|&n| n != nf) {
        return Err(arg_err("per-frame argument lengths differ"));
    }
    if idx.iter().any(|&j| j >= al.shape()[0]) {
        return Err(arg_err("frame index out of range"));
    }
    let dims = (al.shape()[1], al.shape()[2], al.shape()[3]);
    py.detach(|| {
        // The lock is taken here, after the GIL is released: another thread holding the lock
        // while waiting for the GIL would deadlock against this one.
        let mut guard = session();
        let s = guard.as_mut().ok_or_else(|| arg_err("gpu_coadd_begin was not called"))?;
        if dims != (s.h, s.w, 3) || sd[1] != 2 * s.r + 1 || sd[2] != 2 * s.r + 1 {
            return Err(arg_err("frame or stamp shape differs from the session"));
        }
        let hwc = s.h * s.w * 3;
        let frames: Vec<&[f32]> = idx.iter().map(|&j| &alf[j * hwc..(j + 1) * hwc]).collect();
        let res = add_frames(g, ks, s, &frames, &f, &sky, &sig2, &wn, &wf, stf, sd[1] * sd[2]);
        if res.is_err() {
            *guard = None;
        }
        res
    })
}

/// Like `gpu_coadd_add`, but the frames are separate (H, W, 3) arrays (a disk-backed stack's frames
/// already read into memory), so nothing has to be stacked into one array first.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_coadd_add_list(
    py: Python<'_>,
    frames: Vec<PyReadonlyArray3<'_, f32>>,
    f: Vec<f32>,
    sky: Vec<[f32; 3]>,
    sig2: Vec<[f32; 3]>,
    wn: Vec<[f32; 3]>,
    wf: Vec<[f32; 3]>,
    stamps: PyReadonlyArray3<'_, f32>,
) -> PyResult<()> {
    let g = gpu()?;
    let ks = kernels(g)?;
    let views: Vec<_> = frames.iter().map(|a| a.as_array()).collect();
    let mut slices: Vec<&[f32]> = Vec::with_capacity(views.len());
    for v in &views {
        slices.push(v.as_slice().ok_or_else(|| arg_err("each frame must be C-contiguous"))?);
    }
    let st = stamps.as_array();
    let sd = st.shape().to_vec();
    let stf = st.as_slice().ok_or_else(|| arg_err("stamps must be C-contiguous"))?;
    let nf = slices.len();
    if [f.len(), sky.len(), sig2.len(), wn.len(), wf.len(), sd[0]].iter().any(|&n| n != nf) {
        return Err(arg_err("per-frame argument lengths differ"));
    }
    let shape0 = views.first().map(|v| (v.shape()[0], v.shape()[1], v.shape()[2]));
    if views.iter().any(|v| Some((v.shape()[0], v.shape()[1], v.shape()[2])) != shape0) {
        return Err(arg_err("the frames differ in shape"));
    }
    py.detach(|| {
        let mut guard = session();
        let s = guard.as_mut().ok_or_else(|| arg_err("gpu_coadd_begin was not called"))?;
        if nf > 0 && (shape0 != Some((s.h, s.w, 3)) || sd[1] != 2 * s.r + 1 || sd[2] != 2 * s.r + 1) {
            return Err(arg_err("frame or stamp shape differs from the session"));
        }
        let res = add_frames(g, ks, s, &slices, &f, &sky, &sig2, &wn, &wf, stf, sd[1] * sd[2]);
        if res.is_err() {
            *guard = None;
        }
        res
    })
}

#[allow(clippy::too_many_arguments)]
fn add_frames(g: &Gpu, ks: &Kernels, s: &mut Session, frames: &[&[f32]], f: &[f32], sky: &[[f32; 3]],
              sig2: &[[f32; 3]], wn: &[[f32; 3]], wf: &[[f32; 3]], stf: &[f32], stamp_len: usize) -> PyResult<()> {
    let (h, w, ph, pw, pad, r) = (s.h, s.w, s.ph, s.pw, s.pad, s.r);
    let prof = profiling();
    for j in 0..frames.len() {
        let mut t = Instant::now();
        let src = &s.srcs[s.done % 2];
        g.queue.write_buffer(src, 0, bytes_of(frames[j]));
        g.queue.write_buffer(&s.stamp, 0, bytes_of(&stf[j * stamp_len..(j + 1) * stamp_len]));
        let mut ub = u32s([h as u32, w as u32, ph as u32, pw as u32]);
        ub.extend_from_slice(&u32s([pad as u32, r as u32, 0, 0]));
        ub.extend_from_slice(bytes_of(&[s.k, s.frac, f[j], 0f32]));
        ub.extend_from_slice(bytes_of(&[sky[j][0], sky[j][1], sky[j][2], 0f32]));
        ub.extend_from_slice(bytes_of(&[sig2[j][0], sig2[j][1], sig2[j][2], 0f32]));
        let pu = uniform_buf(g, &ub);
        let mut au = u32s([ph as u32, pw as u32, 0, 0]);
        au.extend_from_slice(bytes_of(&[wn[j][0], wn[j][1], wn[j][2], 0f32]));
        au.extend_from_slice(bytes_of(&[wf[j][0], wf[j][1], wf[j][2], 0f32]));
        let aub = uniform_buf(g, &au);
        let mut enc = g.device.create_command_encoder(&Default::default());
        let (z1, z2, z3) = (&s.z[0], &s.z[1], &s.z[2]);
        let (t1, t2, t3) = (&s.t[0], &s.t[1], &s.t[2]);
        if prof {
            lap(g, &mut enc, 0, &mut t)?; // the queued uploads are flushed by this submit
        }
        dispatch(g, &mut enc, &ks.prep, &[&pu, src, &s.refbuf, &s.stamp, z1, z2, z3], (gx(pw, 16), gx(ph, 16), 1));
        if prof {
            lap(g, &mut enc, 1, &mut t)?;
        }
        enc_fft(g, ks, &mut enc, &s.tw, z1, t1, pw, ph, false, 1.0, 0, 0);
        enc_fft(g, ks, &mut enc, &s.tw, z2, t2, pw, ph, false, 1.0, 0, 0);
        enc_fft(g, ks, &mut enc, &s.tw, z3, &s.t4, pw, ph, false, 1.0, r, ph);
        if prof {
            lap(g, &mut enc, 2, &mut t)?;
        }
        enc_transpose(g, ks, &mut enc, t1, z1, ph, pw);
        enc_transpose(g, ks, &mut enc, t2, z2, ph, pw);
        enc_transpose(g, ks, &mut enc, &s.t4, z3, ph, pw);
        if prof {
            lap(g, &mut enc, 3, &mut t)?;
        }
        enc_fft(g, ks, &mut enc, &s.tw, z1, t1, ph, pw, false, 1.0, 0, 0);
        enc_fft(g, ks, &mut enc, &s.tw, z2, t2, ph, pw, false, 1.0, 0, 0);
        enc_fft(g, ks, &mut enc, &s.tw, z3, t3, ph, pw, false, 1.0, 0, 0);
        if prof {
            lap(g, &mut enc, 4, &mut t)?;
        }
        dispatch(g, &mut enc, &ks.accum, &[&aub, t1, t2, t3, &s.num, &s.den], (gx(ph, 64), (pw / 2 + 1) as u32, 1));
        if prof {
            lap(g, &mut enc, 5, &mut t)?;
            PROFILE.lock().unwrap_or_else(|p| p.into_inner())[STAGES] += 1.0;
        }
        let sub = g.queue.submit(Some(enc.finish()));
        // Two upload slots: waiting for the previous frame makes the slot used next idle.
        if let Some(p) = s.prev.take() {
            wait(g, Some(p))?;
        }
        s.prev = Some(sub);
        s.done += 1;
        check()?;
    }
    Ok(())
}

/// Finish: `R = num / sqrt(den)`, inverse transforms, crop. Returns the raw (3, H, W) images
/// `irfft2(R)[pad:pad+H, pad:pad+W]` (before the caller's `/ sqrt(sum w) + sky` scaling) and
/// ends the session, also when it fails.
#[pyfunction]
pub(crate) fn gpu_coadd_finish(py: Python<'_>) -> PyResult<Bound<'_, PyArray3<f32>>> {
    let g = gpu()?;
    let ks = kernels(g)?;
    let (h, w, out) = py.detach(|| -> PyResult<(usize, usize, Vec<f32>)> {
        let s = session().take().ok_or_else(|| arg_err("gpu_coadd_begin was not called"))?;
        let (h, w, ph, pw, pad) = (s.h, s.w, s.ph, s.pw, s.pad);
        wait(g, None)?;
        let (z1, z2, t1, t2) = (&s.z[0], &s.z[1], &s.t[0], &s.t[1]);
        let fu = uniform_buf(g, &u32s([ph as u32, pw as u32, 0, 0]));
        let mut enc = g.device.create_command_encoder(&Default::default());
        dispatch(g, &mut enc, &ks.fin, &[&fu, &s.num, &s.den, t1, t2], (gx(ph, 64), pw as u32, 1));
        enc_fft(g, ks, &mut enc, &s.tw, t1, z1, ph, pw, true, 1.0 / ph as f32, 0, 0);
        enc_fft(g, ks, &mut enc, &s.tw, t2, z2, ph, pw, true, 1.0 / ph as f32, 0, 0);
        enc_transpose(g, ks, &mut enc, z1, t1, pw, ph);
        enc_transpose(g, ks, &mut enc, z2, t2, pw, ph);
        enc_fft(g, ks, &mut enc, &s.tw, t1, z1, pw, ph, true, 1.0 / pw as f32, 0, 0);
        enc_fft(g, ks, &mut enc, &s.tw, t2, z2, pw, ph, true, 1.0 / pw as f32, 0, 0);
        g.queue.submit(Some(enc.finish()));
        wait(g, None)?;
        let pbytes = (ph * pw * 8) as u64;
        let y1 = read_back(g, z1, pbytes)?;
        let y2 = read_back(g, z2, pbytes)?;
        let mut out = vec![0f32; 3 * h * w];
        let (c0, rest) = out.split_at_mut(h * w);
        let (c1, c2) = rest.split_at_mut(h * w);
        for y in 0..h {
            for x in 0..w {
                let o = ((y + pad) * pw + (x + pad)) * 2;
                c0[y * w + x] = y1[o];
                c1[y * w + x] = y1[o + 1];
                c2[y * w + x] = y2[o];
            }
        }
        Ok((h, w, out))
    })?;
    Ok(numpy::ndarray::Array3::from_shape_vec((3, h, w), out).map_err(|e| rt_err(e.to_string()))?.into_pyarray(py))
}

/// Drop the session (device buffers) if one exists. Safe to call at any time.
#[pyfunction]
pub(crate) fn gpu_coadd_abort() {
    *session() = None;
}
