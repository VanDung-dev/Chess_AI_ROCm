import chess
from modules.chess_mcts import cached_model_call
from modules.chess_engine import move_to_index
import torch
import chess_rs


def get_best_move(model, board, mcts_iterations=1000, temperature=0.9):
    """
    Chọn nước đi tốt nhất bằng cách sử dụng MCTS và mô hình AI.

    Args:
        model: Mô hình AI (ChessNet).
        board (chess.Board): Trạng thái bàn cờ hiện tại.
        mcts_iterations (int): Số lần lặp MCTS (mặc định 100).
        temperature (float): Hệ số điều chỉnh độ ngẫu nhiên (mặc định 1.0).

    Returns:
        chess.Move: Nước đi tốt nhất hoặc None nếu không tìm thấy.
    """
    try:
        fen = board.fen()
        
        # Get model evaluation for root position
        policy, value = cached_model_call(model, board)
        root_value = value.item()
        
        # Get priors for possible moves
        legal_moves = list(board.legal_moves)
        priors = []
        
        # Convert policy logits to probabilities
        policy_probs = torch.softmax(policy, dim=1)
        
        for move in legal_moves:
            move_idx = move_to_index(move)
            move_prob = policy_probs[0][move_idx].item()
            priors.append((move.uci(), move_prob))
        
        # Gọi triển khai Rust với model evaluation
        results = chess_rs.mcts_loop(fen, mcts_iterations, priors, root_value)
        
        if not results:
            return None
            
        # Tìm động thái có số lượt truy cập cao nhất
        best_move_str = max(results, key=lambda x: x[2])[0]  # x[2] là số lượt truy cập
        return chess.Move.from_uci(best_move_str)
    except Exception as e:
        print(f"Error in Rust MCTS: {e}")
        return None