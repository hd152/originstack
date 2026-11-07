//! GPU compute context (Cargo feature `gpu`): one wgpu device for the process, plus the small
//! helpers the GPU kernels share (pipelines, buffers, dispatch, read-back).
//!
//! Everything here is best-effort: no adapter, a lost device, an out-of-memory allocation or a
//! driver that panics all surface as a Python `RuntimeError` (never an abort), so the callers in
//! `src/` can fall back to the CPU path. wgpu's own default for an uncaptured error is to panic;
//! this module installs a handler that records the message instead and every entry point checks
//! it (`check`). `ORIGINSTACK_GPU_DISABLE=1` makes `gpu_available()` report False (tests, and a
//! way to force the CPU path); `ORIGINSTACK_GPU_ADAPTER=<substring>` picks an adapter by name.

use crate::*;
use std::sync::{Mutex, OnceLock};
use wgpu::util::DeviceExt;

/// The shaders keep one FFT row (4096 complex f32) in workgroup memory, with 256 threads.
const NEEDED_WORKGROUP_STORAGE: u32 = 4096 * 8;
const NEEDED_WORKGROUP_THREADS: u32 = 256;

pub(crate) struct Gpu {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) info: String,
    pub(crate) max_buffer: u64,
    pub(crate) max_binding: u64,
}

static GPU: OnceLock<Result<Gpu, String>> = OnceLock::new();
static ERROR: Mutex<Option<String>> = Mutex::new(None);

pub(crate) fn rt_err(msg: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(msg.into())
}

pub(crate) fn arg_err(msg: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.into())
}

fn record_error(msg: String) {
    let mut e = ERROR.lock().unwrap_or_else(|p| p.into_inner());
    if e.is_none() {
        *e = Some(msg);
    }
}

/// Fail if wgpu reported an error (device lost, out of memory, validation) since the last check.
pub(crate) fn check() -> PyResult<()> {
    let mut e = ERROR.lock().unwrap_or_else(|p| p.into_inner());
    match e.take() {
        Some(m) => Err(rt_err(format!("GPU error: {m}"))),
        None => Ok(()),
    }
}

fn init() -> Result<Gpu, String> {
    if std::env::var_os("ORIGINSTACK_GPU_DISABLE").is_some_and(|v| v != "0" && !v.is_empty()) {
        return Err("disabled by ORIGINSTACK_GPU_DISABLE".into());
    }
    let instance = wgpu::Instance::default();
    let want = std::env::var("ORIGINSTACK_GPU_ADAPTER").ok().map(|s| s.to_lowercase());
    let mut adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    // Prefer a discrete GPU, then an integrated one; never a software rasteriser.
    let rank = |t: wgpu::DeviceType| match t {
        wgpu::DeviceType::DiscreteGpu => Some(0),
        wgpu::DeviceType::IntegratedGpu => Some(1),
        wgpu::DeviceType::VirtualGpu => Some(2),
        _ => None,
    };
    adapters.retain(|a| {
        let i = a.get_info();
        rank(i.device_type).is_some()
            && match &want {
                Some(w) => i.name.to_lowercase().contains(w.as_str()),
                None => true,
            }
    });
    adapters.sort_by_key(|a| rank(a.get_info().device_type));
    let adapter = adapters.into_iter().next().ok_or("no usable GPU adapter")?;
    let ai = adapter.get_info();
    let info = format!("{} ({:?}, {:?})", ai.name, ai.backend, ai.device_type);
    let limits = adapter.limits();
    if limits.max_compute_workgroup_storage_size < NEEDED_WORKGROUP_STORAGE
        || limits.max_compute_invocations_per_workgroup < NEEDED_WORKGROUP_THREADS
        || limits.max_compute_workgroup_size_x < NEEDED_WORKGROUP_THREADS
    {
        return Err(format!("{info}: compute limits too small (need {NEEDED_WORKGROUP_STORAGE} bytes of workgroup memory, 256 threads)"));
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("originstack"),
        required_features: wgpu::Features::empty(),
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .map_err(|e| format!("{info}: request_device: {e}"))?;
    device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| record_error(e.to_string())));
    device.set_device_lost_callback(|reason, msg| record_error(format!("device lost ({reason:?}): {msg}")));
    Ok(Gpu {
        device,
        queue,
        info,
        max_buffer: limits.max_buffer_size,
        max_binding: limits.max_storage_buffer_binding_size,
    })
}

/// The process-wide device, created on first use. A panic inside wgpu or a driver becomes an
/// error string, and a failed initialisation is remembered (it is not retried).
pub(crate) fn gpu() -> PyResult<&'static Gpu> {
    let r = GPU.get_or_init(|| {
        std::panic::catch_unwind(init).unwrap_or_else(|_| Err("the GPU driver panicked during initialisation".into()))
    });
    match r {
        Ok(g) => Ok(g),
        Err(e) => Err(rt_err(format!("GPU unavailable: {e}"))),
    }
}

/// True when a usable GPU adapter exists (initialises the device on first call).
#[pyfunction]
pub(crate) fn gpu_available() -> bool {
    gpu().is_ok()
}

/// The adapter's name, backend and type, or None when no GPU is usable.
#[pyfunction]
pub(crate) fn gpu_info() -> Option<String> {
    gpu().ok().map(|g| g.info.clone())
}

// ---------------------------------------------------------------------------------------------
// Shared kernel helpers.
// ---------------------------------------------------------------------------------------------

pub(crate) fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices (f32 / u32) viewed as bytes, read-only, same lifetime.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[derive(Clone, Copy)]
pub(crate) enum Bind {
    Uniform,
    Ro,
    Rw,
}

pub(crate) struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// Compile `src` (entry point `main`) with one buffer binding per element of `binds`.
pub(crate) fn make_kernel(g: &Gpu, label: &str, src: &str, binds: &[Bind]) -> Kernel {
    let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let entries: Vec<wgpu::BindGroupLayoutEntry> = binds
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: match b {
                    Bind::Uniform => wgpu::BufferBindingType::Uniform,
                    Bind::Ro => wgpu::BufferBindingType::Storage { read_only: true },
                    Bind::Rw => wgpu::BufferBindingType::Storage { read_only: false },
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
    let pl = g.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        ..Default::default()
    });
    let pipeline = g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, layout }
}

/// Record one dispatch of `k` over `bufs` (binding i = bufs[i]).
pub(crate) fn dispatch(g: &Gpu, enc: &mut wgpu::CommandEncoder, k: &Kernel, bufs: &[&wgpu::Buffer], groups: (u32, u32, u32)) {
    let entries: Vec<wgpu::BindGroupEntry> = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();
    let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &k.layout, entries: &entries });
    let mut cp = enc.begin_compute_pass(&Default::default());
    cp.set_pipeline(&k.pipeline);
    cp.set_bind_group(0, &bg, &[]);
    cp.dispatch_workgroups(groups.0, groups.1, groups.2);
}

pub(crate) fn uniform_buf(g: &Gpu, bytes: &[u8]) -> wgpu::Buffer {
    g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

/// A storage buffer of `size` bytes (zero-initialised by wgpu). Fails, instead of letting wgpu
/// raise a device error, when the adapter cannot bind a buffer that large.
pub(crate) fn storage_buf(g: &Gpu, size: u64, extra: wgpu::BufferUsages) -> PyResult<wgpu::Buffer> {
    if size > g.max_buffer || size > g.max_binding {
        return Err(rt_err(format!(
            "a {} MB GPU buffer exceeds this adapter's limit ({} MB)",
            size >> 20,
            g.max_buffer.min(g.max_binding) >> 20
        )));
    }
    Ok(g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    }))
}

/// Block until submission `idx` (or, with None, everything submitted) has finished.
pub(crate) fn wait(g: &Gpu, idx: Option<wgpu::SubmissionIndex>) -> PyResult<()> {
    g.device
        .poll(wgpu::PollType::Wait { submission_index: idx, timeout: None })
        .map_err(|e| rt_err(format!("GPU wait failed: {e}")))?;
    check()
}

/// Copy `bytes` bytes of `src` (which needs COPY_SRC) back to the host as f32.
pub(crate) fn read_back(g: &Gpu, src: &wgpu::Buffer, bytes: u64) -> PyResult<Vec<f32>> {
    let rd = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(src, 0, &rd, 0, bytes);
    g.queue.submit(Some(enc.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    rd.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    wait(g, None)?;
    rx.recv_timeout(std::time::Duration::from_secs(60))
        .map_err(|e| rt_err(format!("GPU read-back did not complete: {e}")))?
        .map_err(|e| rt_err(format!("GPU read-back failed: {e}")))?;
    let mut out = vec![0f32; (bytes / 4) as usize];
    {
        let view = rd.slice(..).get_mapped_range().map_err(|e| rt_err(e.to_string()))?;
        let b: &[u8] = &view;
        // SAFETY: `out` holds `bytes` bytes and the mapped view is exactly that long.
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), out.as_mut_ptr() as *mut u8, b.len()) };
    }
    rd.unmap();
    Ok(out)
}
