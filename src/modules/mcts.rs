use chess::{Board, BoardStatus, ChessMove, MoveGen};
use pyo3::prelude::*;
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

const BATCH_SIZE: usize = 8;
const VIRTUAL_LOSS: f32 = 1.0;

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

    // Chạy các lần lặp lại MCTS theo batch
    let num_batches = (iterations + BATCH_SIZE - 1) / BATCH_SIZE;

    for _ in 0..num_batches {
        // Thu thập hàng loạt
        let mut batch_paths = Vec::with_capacity(BATCH_SIZE);

        // 1. Giai đoạn lựa chọn với Virtual Loss
        for _ in 0..BATCH_SIZE {
            let path = select_leaf_path(root_idx, &mut nodes);
            apply_virtual_loss(&mut nodes, &path);
            batch_paths.push(path);
        }

        // 2. Giai đoạn đánh giá
        let mut fens_to_eval = Vec::new();
        let mut paths_to_expand = Vec::new();

        for path in &batch_paths {
            let leaf_idx = *path.last().unwrap();
            let leaf_node = &nodes[leaf_idx];

            fens_to_eval.push(format!("{}", leaf_node.board));
            paths_to_expand.push(leaf_idx);
        }

        if !fens_to_eval.is_empty() {
            // Gọi Python evaluator.evaluate_batch
            let result = evaluator.call_method1("evaluate_batch", (fens_to_eval,))?;
            // Kết quả là List[(priors, value)]
            let results_list: Vec<(Vec<(String, f32)>, f32)> = result.extract()?;

            // 3. Giai đoạn mở rộng & Backprop
            for (i, (priors, val)) in results_list.iter().enumerate() {
                let leaf_idx = paths_to_expand[i];
                let path = &batch_paths[i];

                let value = *val;

                let is_ongoing = nodes[leaf_idx].board.status() == BoardStatus::Ongoing;

                if is_ongoing && nodes[leaf_idx].children.is_empty() {
                    let moves: Vec<_> = MoveGen::new_legal(&nodes[leaf_idx].board).collect();
                    for (move_str, prior) in priors {
                        if let Ok(chess_move) = move_str.parse::<ChessMove>() {
                            if moves.contains(&chess_move) {
                                let mut new_board = nodes[leaf_idx].board;
                                new_board = new_board.make_move_new(chess_move);

                                let new_node_idx = nodes.len();
                                let mut new_node =
                                    MCTSNode::new(new_board, Some(leaf_idx), Some(chess_move));
                                new_node.prior = *prior;
                                nodes.push(new_node);

                                nodes[leaf_idx].children.insert(chess_move, new_node_idx);
                            }
                        }
                    }
                }

                backpropagate_batch(&mut nodes, path, value);
            }
        } else {
            for path in batch_paths {
                undo_virtual_loss(&mut nodes, &path);
            }
        }
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

fn select_leaf_path(root_idx: usize, nodes: &mut Vec<MCTSNode>) -> Vec<usize> {
    let mut path = Vec::new();
    path.push(root_idx);

    let mut current_idx = root_idx;

    loop {
        if nodes[current_idx].children.is_empty() {
            break;
        }

        let legal_count = MoveGen::new_legal(&nodes[current_idx].board).count();
        if nodes[current_idx].children.len() != legal_count {
            break;
        }

        let mut best_child_idx = None;
        let mut best_uct = f32::NEG_INFINITY;

        for (_, &child_idx) in &nodes[current_idx].children {
            let uct = calculate_uct(current_idx, child_idx, nodes);
            if uct > best_uct {
                best_uct = uct;
                best_child_idx = Some(child_idx);
            }
        }

        if let Some(next_idx) = best_child_idx {
            current_idx = next_idx;
            path.push(current_idx);
        } else {
            break;
        }
    }

    path
}

fn apply_virtual_loss(nodes: &mut Vec<MCTSNode>, path: &[usize]) {
    for &idx in path {
        nodes[idx].visit_count += 1;
        nodes[idx].total_value -= VIRTUAL_LOSS;
    }
}

fn undo_virtual_loss(nodes: &mut Vec<MCTSNode>, path: &[usize]) {
    for &idx in path {
        nodes[idx].visit_count -= 1;
        nodes[idx].total_value += VIRTUAL_LOSS;
    }
}

fn backpropagate_batch(nodes: &mut Vec<MCTSNode>, path: &[usize], real_value: f32) {
    let root_turn = nodes[path[0]].board.side_to_move();
    let leaf_turn = nodes[*path.last().unwrap()].board.side_to_move();

    let score_for_leaf_player = real_value;
    let score_for_root = if leaf_turn == root_turn {
        score_for_leaf_player
    } else {
        -score_for_leaf_player
    };

    for &idx in path {
        nodes[idx].total_value += VIRTUAL_LOSS;

        if nodes[idx].board.side_to_move() == root_turn {
            nodes[idx].total_value += score_for_root;
        } else {
            nodes[idx].total_value -= score_for_root;
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
