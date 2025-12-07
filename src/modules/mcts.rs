use pyo3::prelude::*;
use chess::{Board, ChessMove, MoveGen, BoardStatus};
use rand::Rng;
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Clone)]
pub struct MCTSNode {
    pub board: Board,
    pub parent: Option<usize>,
    pub children: HashMap<ChessMove, usize>,
    pub visit_count: u32,
    pub total_value: f32,
    pub prior: f32,
    pub move_: Option<ChessMove>,
}

impl MCTSNode {
    pub fn new(board: Board, parent: Option<usize>, move_: Option<ChessMove>) -> Self {
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
pub fn mcts_loop(
    fen: &str,
    iterations: usize,
    evaluator: &Bound<'_, PyAny>,
    root_value: f32,
    root_priors: Vec<(String, f32)>,
) -> PyResult<Vec<(String, f32, u32)>> {
    let board = Board::from_str(fen).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid FEN: {}", e))
    })?;

    let mut nodes: Vec<MCTSNode> = vec![MCTSNode::new(board, None, None)];
    let root_idx = 0;
    
    // Đặt giá trị gốc
    nodes[root_idx].total_value = root_value;

    // Mở rộng gốc bằng priors được cung cấp trước
    let moves: Vec<_> = MoveGen::new_legal(&board).collect();
    for (move_str, prior) in root_priors {
        if let Ok(chess_move) = move_str.parse::<ChessMove>() {
            if moves.contains(&chess_move) {
                let mut new_board = board;
                new_board = new_board.make_move_new(chess_move);
                
                let new_node_idx = nodes.len();
                let mut new_node = MCTSNode::new(new_board, Some(root_idx), Some(chess_move));
                new_node.prior = prior;
                nodes.push(new_node);
                
                nodes[root_idx].children.insert(chess_move, new_node_idx);
            }
        }
    }
    
    // Chạy các lần lặp lại MCTS
    for _ in 0..iterations {
        mcts_iteration(root_idx, &mut nodes, evaluator)?;
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

fn mcts_iteration(
    root_idx: usize,
    nodes: &mut Vec<MCTSNode>,
    evaluator: &Bound<'_, PyAny>,
) -> PyResult<()> {
    // Giai đoạn lựa chọn
    let mut current_idx = root_idx;
    while !nodes[current_idx].children.is_empty()
        && nodes[current_idx].children.len()
            == MoveGen::new_legal(&nodes[current_idx].board).count() {
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

    // Giai đoạn mở rộng và đánh giá
    let mut value = 0.0;

    if nodes[current_idx].board.status() == BoardStatus::Ongoing {
        let moves: Vec<_> = MoveGen::new_legal(&nodes[current_idx].board).collect();
        
        // Tìm các động thái chưa được truy cập
        // Gọi Python evaluator
        let fen = format!("{}", nodes[current_idx].board);
        let result = evaluator.call1((fen,))?;
        let (priors, eval_value): (Vec<(String, f32)>, f32) = result.extract()?;
        value = eval_value;

        // Nếu node này chưa có children (leaf), expand tất cả dựa trên priors từ model
        if nodes[current_idx].children.is_empty() {
            for (move_str, prior) in priors {
                if let Ok(chess_move) = move_str.parse::<ChessMove>() {
                    if moves.contains(&chess_move) {
                        let mut new_board = nodes[current_idx].board;
                        new_board = new_board.make_move_new(chess_move);

                        let new_node_idx = nodes.len();
                        let mut new_node =
                            MCTSNode::new(new_board, Some(current_idx), Some(chess_move));
                        new_node.prior = prior;
                        nodes.push(new_node);

                        nodes[current_idx].children.insert(chess_move, new_node_idx);
                    }
                }
            }

        } else {
        }
    } else {
        let fen = format!("{}", nodes[current_idx].board);
        if let Ok(result) = evaluator.call1((fen,)) {
            if let Ok((_, eval_value)) = result.extract::<(Vec<(String, f32)>, f32)>() {
                value = eval_value;
            }
        }
    }

    // Giai đoạn lan truyền ngược
    let mut node_idx = current_idx;
    loop {
        nodes[node_idx].visit_count += 1;
        nodes[node_idx].total_value += value;

        // Flip value for next level up (parent perspective)
        value = -value;

        if let Some(parent_idx) = nodes[node_idx].parent {
            node_idx = parent_idx;
        } else {
            break;
        }
    }

    Ok(())
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
