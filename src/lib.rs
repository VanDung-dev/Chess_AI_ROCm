use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use chess::{Board, ChessMove, MoveGen, BoardStatus};
use rand::Rng;
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Clone)]
struct MCTSNode {
    board: Board,
    parent: Option<usize>,
    children: HashMap<ChessMove, usize>,
    visit_count: u32,
    total_value: f32,
    prior: f32,
    move_: Option<ChessMove>,
}

impl MCTSNode {
    fn new(board: Board, parent: Option<usize>, move_: Option<ChessMove>) -> Self {
        MCTSNode {
            board,
            parent,
            children: HashMap::new(),
            visit_count: 0,
            total_value: 0.0,
            prior: 0.0,
            move_,
        }
    }
}

#[pyfunction]
fn mcts_loop(
    _py: Python,
    fen: &str,
    iterations: usize,
) -> PyResult<Vec<(String, f32, u32)>> {
    let board = Board::from_str(fen).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid FEN: {}", e)))?;
    
    let mut nodes: Vec<MCTSNode> = vec![MCTSNode::new(board, None, None)];
    let root_idx = 0;
    
    // Chạy các lần lặp lại MCTS
    for _ in 0..iterations {
        mcts_iteration(root_idx, &mut nodes);
    }
    
    // Trích xuất kết quả
    let mut results = Vec::new();
    if let Some(root) = nodes.get(root_idx) {
        for (chess_move, child_idx) in &root.children {
            if let Some(child) = nodes.get(*child_idx) {
                let visits = child.visit_count;
                let win_rate = if visits > 0 { 
                    child.total_value / (visits as f32) 
                } else { 
                    0.0 
                };
                results.push((format!("{}", chess_move), win_rate, visits));
            }
        }
    }
    
    Ok(results)
}

fn mcts_iteration(root_idx: usize, nodes: &mut Vec<MCTSNode>) {
    // Giai đoạn lựa chọn
    let mut current_idx = root_idx;
    while !nodes[current_idx].children.is_empty() && 
          nodes[current_idx].children.len() == MoveGen::new_legal(&nodes[current_idx].board).count() {
        
        let mut best_move = None;
        let mut best_uct = f32::NEG_INFINITY;
        
        // Chọn child có giá trị UCT tốt nhất
        for (_, &child_idx) in &nodes[current_idx].children {
            let uct = calculate_uct(current_idx, child_idx, &nodes);
            if uct > best_uct {
                best_uct = uct;
                best_move = Some(child_idx);
            }
        }
        
        if let Some(next_idx) = best_move {
            current_idx = next_idx;
        } else {
            break;
        }
    }
    
    // Giai đoạn mở rộng
    if nodes[current_idx].board.status() == BoardStatus::Ongoing {
        let moves: Vec<_> = MoveGen::new_legal(&nodes[current_idx].board).collect();
        
        // Tìm các động thái chưa được truy cập
        let unvisited_moves: Vec<_> = moves.into_iter()
            .filter(|chess_move| !nodes[current_idx].children.contains_key(chess_move))
            .collect();
        
        if !unvisited_moves.is_empty() {
            // Chọn một nước đi ngẫu nhiên chưa được truy cập
            let mut rng = rand::rng();
            let selected_move = unvisited_moves[rng.random_range(0..unvisited_moves.len())];
            
            // Tạo trạng thái bảng mới
            let mut new_board = nodes[current_idx].board;
            new_board = new_board.make_move_new(selected_move);
            
            // Tạo nút mới
            let new_node_idx = nodes.len();
            let new_node = MCTSNode::new(new_board, Some(current_idx), Some(selected_move));
            nodes.push(new_node);
            
            // Thêm child vào parent
            nodes[current_idx].children.insert(selected_move, new_node_idx);
            
            current_idx = new_node_idx;
        }
    }
    
    // Giai đoạn mô phỏng (sử dụng random playout)
    let value = simulate_random_playout(&nodes[current_idx].board);
    
    // Giai đoạn lan truyền ngược
    let mut node_idx = current_idx;
    loop {
        nodes[node_idx].visit_count += 1;
        nodes[node_idx].total_value += value;
        
        if let Some(parent_idx) = nodes[node_idx].parent {
            node_idx = parent_idx;
        } else {
            break;
        }
    }
}

fn calculate_uct(parent_idx: usize, child_idx: usize, nodes: &[MCTSNode]) -> f32 {
    let parent = &nodes[parent_idx];
    let child = &nodes[child_idx];
    
    if child.visit_count == 0 {
        return f32::INFINITY;
    }
    
    let exploration_constant = 1.41;
    let exploitation = child.total_value / (child.visit_count as f32);
    let exploration = exploration_constant * 
        ((parent.visit_count as f32).ln() / (child.visit_count as f32)).sqrt();
    
    exploitation + exploration
}

fn simulate_random_playout(board: &Board) -> f32 {
    let mut current_board = *board;
    let mut rng = rand::rng();
    
    // Giới hạn độ sâu mô phỏng để ngăn chặn các ván game vô hạn
    for _ in 0..200 {
        if current_board.status() != BoardStatus::Ongoing {
            break;
        }
        
        let moves: Vec<_> = MoveGen::new_legal(&current_board).collect();
        if moves.is_empty() {
            break;
        }
        
        let random_move = moves[rng.random_range(0..moves.len())];
        current_board = current_board.make_move_new(random_move);
    }
    
    // Đánh giá vị trí cuối cùng
    // Đánh giá đơn giản cho hiện tại
    0.0
}

/// Một mô-đun Python được triển khai trong Rust.
#[pymodule]
fn chess_mcts_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mcts_loop, m)?)?;
    Ok(())
}