//! Moving-object velocity voting (`src/moving_objects.py`).

use crate::*;

/// Vote detections into velocity-space back-projection cells.
///
/// This mirrors `src/moving_objects.py::_vote`: velocities and cells are
/// visited in row-major order, only distinct frame IDs count, and ties retain
/// the first velocity/cell encountered.
#[pyfunction]
pub(crate) fn vote_velocity(
    det: PyReadonlyArray2<'_, f64>,
    t_mid: f64,
    vgrid_x: PyReadonlyArray1<'_, f64>,
    vgrid_y: PyReadonlyArray1<'_, f64>,
    height: usize,
    width: usize,
    cell: f64,
    valid: PyReadonlyArray2<'_, bool>,
) -> PyResult<(usize, f64, f64, f64, f64)> {
    let det = det.as_array();
    let vx = vgrid_x.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("vgrid_x must be C-contiguous")
    })?;
    let vy = vgrid_y.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("vgrid_y must be C-contiguous")
    })?;
    let valid = valid.as_array();
    if valid.shape() != [vy.len(), vx.len()] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "valid shape must be (len(vgrid_y), len(vgrid_x))",
        ));
    }
    if det.ncols() < 5 || cell <= 0.0 || !cell.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "det must have at least five columns and cell must be positive",
        ));
    }
    // Match Python's int(W / cell) rather than integer division. This matters
    // when cell is non-integral.
    let ncx = (width as f64 / cell) as usize + 2;
    let ncy = (height as f64 / cell) as usize + 2;
    let mut best = (0usize, 0.0, 0.0, 0.0, 0.0);
    let tt: Vec<f64> = det.rows().into_iter().map(|r| r[1] - t_mid).collect();
    for (iy, &vy_value) in vy.iter().enumerate() {
        for (ix, &vx_value) in vx.iter().enumerate() {
            if !valid[[iy, ix]] {
                continue;
            }
            let mut counts = vec![0usize; ncx.saturating_mul(ncy)];
            for (row, &time) in det.rows().into_iter().zip(tt.iter()) {
                let cx = ((row[2] - vx_value * time) / cell).floor();
                let cy = ((row[3] - vy_value * time) / cell).floor();
                if cx >= 0.0 && cx < ncx as f64 && cy >= 0.0 && cy < ncy as f64 {
                    counts[cy as usize * ncx + cx as usize] += 1;
                }
            }
            let Some((cell_index, &point_count)) = counts.iter().enumerate()
                .max_by_key(|(index, count)| (*count, std::cmp::Reverse(*index)))
            else {
                continue;
            };
            if point_count < best.0 {
                continue;
            }
            let mut frames = Vec::with_capacity(point_count);
            let mut x_values = Vec::with_capacity(point_count);
            let mut y_values = Vec::with_capacity(point_count);
            for (row, &time) in det.rows().into_iter().zip(tt.iter()) {
                let x0 = row[2] - vx_value * time;
                let y0 = row[3] - vy_value * time;
                let cx = (x0 / cell).floor();
                let cy = (y0 / cell).floor();
                if cx >= 0.0 && cx < ncx as f64 && cy >= 0.0 && cy < ncy as f64
                    && cy as usize * ncx + cx as usize == cell_index
                {
                    frames.push(row[0] as i64);
                    x_values.push(x0);
                    y_values.push(y0);
                }
            }
            frames.sort_unstable();
            frames.dedup();
            if frames.len() <= best.0 {
                continue;
            }
            x_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            y_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = |values: &[f64]| -> f64 {
                let middle = values.len() / 2;
                if values.len() % 2 == 0 {
                    (values[middle - 1] + values[middle]) / 2.0
                } else {
                    values[middle]
                }
            };
            best = (frames.len(), vx_value, vy_value, median(&x_values), median(&y_values));
        }
    }
    Ok(best)
}
