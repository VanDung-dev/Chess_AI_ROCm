use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod modules;
use modules::mcts::{mcts_loop};

/// Một mô-đun Python được triển khai trong Rust.
#[pymodule]
fn chess_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mcts_loop, m)?)?;
    Ok(())
}