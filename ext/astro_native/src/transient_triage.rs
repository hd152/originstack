
//! ZOGY transient triage: real/bogus classification of --transient-detect
//! candidates (src/transient_triage.py, --transient-triage)
//!
//! The crate's ONNX kernel (pure-Rust `tract`): no percentile stretch,
//! no resize -- stamps arrive already extracted at a fixed size and
//! per-channel sigma-normalized in Python (there is no perf-relevant work to
//! move into Rust for a few hundred 31x31x3 stamps), and no multi-head
//! metadata decode, just one scalar sigmoid probability per candidate.
//! Batched: all of a frame's candidates score in a single tract forward pass
//! over (N, 3, size, size), not one call per candidate. Advisory only --
//! never filters `detect_transients`' output, just attaches a
//! `real_probability`.
//!
//! The bundled model (src/data/transient_triage.onnx) is trained entirely on
//! synthetic data (tools/gen_transient_triage_data.py +
//! tools/train_transient_triage.py) -- no labelled real transients exist yet
//! -- so treat it as a first cut, not a production classifier. Deliberately
//! has no numpy/onnxruntime fallback yet, unlike every other native kernel in
//! this crate: a source checkout without astro_native built simply can't use
//! --transient-triage until one is added (self-disables with a warning, see
//! src/transient_triage.py).
//!
//! Behind the `triage` Cargo feature (off by default: tract is ~24 MB of the
//! binary). Build with `maturin build --release --features triage` to get it.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tract_onnx::prelude::*;

type Runnable = Arc<TypedRunnableModel>;

struct Session {
    model: Runnable,
}

fn cache() -> &'static Mutex<HashMap<String, Arc<Session>>> {
    static C: OnceLock<Mutex<HashMap<String, Arc<Session>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn load(path: &str, _size: usize) -> TractResult<Session> {
    // The batch axis must stay symbolic: this kernel scores a whole
    // frame's candidates in one batched call (N varies per frame).
    // The channel/H/W axes are already concrete in the
    // exported graph (torch.onnx.export's dynamic_axes only marks axis
    // 0 as dynamic), so no `with_input_fact` override is needed -- one
    // was tried and made every N != 1 call fail with a tract symbol
    // resolution clash against the fixed batch=1 it forced.
    let proto = tract_onnx::onnx().proto_model_for_path(path)?;
    let model = tract_onnx::onnx()
        .model_for_proto_model(&proto)?
        .into_optimized()?
        .into_runnable()?;
    Ok(Session { model })
}

fn get_session(path: &str, size: usize) -> TractResult<Arc<Session>> {
    // Key on (path, size): a different stamp size is a different
    // compiled graph.
    let key = format!("{path}\u{0}{size}");
    {
        let c = cache().lock().unwrap();
        if let Some(s) = c.get(&key) {
            return Ok(Arc::clone(s));
        }
    }
    let s = Arc::new(load(path, size)?);
    cache().lock().unwrap().insert(key, Arc::clone(&s));
    Ok(s)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Pure compute: batched forward pass over N pre-normalized stamps, no
/// `Python` token -- runs inside `py.detach`.
fn compute(stamps: &[f32], n: usize, size: usize, model_path: &str) -> Result<Vec<f64>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let sess = get_session(model_path, size).map_err(|e| e.to_string())?;
    // `stamps` is already NCHW-ordered per candidate: (n, 3, size, size).
    let input = tract_ndarray::Array4::from_shape_vec((n, 3, size, size), stamps.to_vec())
        .map_err(|e| e.to_string())?
        .into_tensor();
    let outputs = sess.model.run(tvec!(input.into())).map_err(|e| e.to_string())?;
    let raw = outputs
        .first()
        .ok_or_else(|| "model produced no output".to_string())?
        .to_plain_array_view::<f32>()
        .map_err(|e| e.to_string())?;
    if raw.len() != n {
        return Err(format!(
            "model produced {} outputs for {n} candidates -- refusing to guess",
            raw.len()
        ));
    }
    Ok(raw.iter().map(|&v| sigmoid(v as f64)).collect())
}

#[pyfunction]
#[pyo3(signature = (stamps, model_path, size=31))]
pub fn transient_triage_score(
    py: Python<'_>,
    stamps: numpy::PyReadonlyArray4<f32>,
    model_path: &str,
    size: usize,
) -> PyResult<Vec<f64>> {
    let a = stamps.as_array();
    let sh = a.shape();
    if sh.len() != 4 || sh[1] != 3 || sh[2] != size || sh[3] != size {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected stamps shaped (N, 3, {size}, {size}), got {:?}",
            sh
        )));
    }
    let n = sh[0];
    let flat: Vec<f32> = a.iter().cloned().collect();
    let model_path = model_path.to_string();

    // Heavy work off the GIL, panic-guarded: `tract` parses an external
    // .onnx file and a malformed one can panic inside the parser, which would otherwise
    // unwind past callers' `except Exception` as a bare PanicException.
    let outcome = py.detach(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compute(&flat, n, size, &model_path)
        }))
    });

    match outcome {
        Ok(Ok(probs)) => Ok(probs),
        Ok(Err(msg)) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "transient_triage native inference failed: {msg}"
        ))),
        Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "transient_triage native inference panicked (malformed model?)".to_string(),
        )),
    }
}
