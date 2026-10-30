//! Shared rayon drivers.

use crate::*;

/// Run `f(row_index, row)` over the `width`-element rows of `data` in parallel.
///
/// The one spelling of "each task owns one output row", which most image kernels here use
/// (`data.len()` need not be a multiple of `width`: the last row is then short, as with
/// `par_chunks_mut`). Rows are independent by construction, so the result never depends on
/// the schedule; kernels that need scratch state per worker or a row grain use
/// `for_each_init` / `with_min_len` directly (see `combine::row_parallel_k`).
#[inline]
pub(crate) fn par_rows<T, F>(data: &mut [T], width: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    data.par_chunks_mut(width).enumerate().for_each(|(y, row)| f(y, row));
}
