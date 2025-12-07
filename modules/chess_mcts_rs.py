import chess
from modules.chess_mcts import cached_model_call
from modules.chess_engine import move_to_index
import torch
import chess_rs

class BoardEvaluator:
    def __init__(self, model):
        self.model = model

    def __call__(self, fen):
        try:
            board = chess.Board(fen)
            # Use cached_model_call which returns (policy, value)
            policy, value_tensor = cached_model_call(self.model, board)
            value = value_tensor.item()
            
            # Convert policy to probabilities
            policy_probs = torch.softmax(policy, dim=1)
            
            legal_moves = list(board.legal_moves)
            priors = []
            
            for move in legal_moves:
                move_idx = move_to_index(move)
                move_prob = policy_probs[0][move_idx].item()
                priors.append((move.uci(), move_prob))
                
            return (priors, value)
        except Exception as e:
            print(f"Error in BoardEvaluator: {e}")
            return ([], 0.0)

def get_best_move(model, board, mcts_iterations=100, temperature=0.9):
    """
    Chọn nước đi tốt nhất bằng cách sử dụng MCTS (Rust implementation) và mô hình AI.

    Args:
        model: Mô hình AI (ChessNet).
        board (chess.Board): Trạng thái bàn cờ hiện tại.
        mcts_iterations (int): Số lần lặp MCTS.
        temperature (float): Hệ số điều chỉnh độ ngẫu nhiên.

    Returns:
        chess.Move: Nước đi tốt nhất hoặc None nếu không tìm thấy.
    """
    try:
        fen = board.fen()
        
        # Đánh giá ban đầu cho root
        evaluator = BoardEvaluator(model)
        root_priors, root_value = evaluator(fen)
        
        # Gọi triển khai Rust với evaluator callback
        # Signature: mcts_loop(fen, iterations, evaluator, root_value, root_priors)
        results = chess_rs.mcts_loop(fen, mcts_iterations, evaluator, root_value, root_priors)
        
        if not results:
            return None
            
        # Tìm động thái có số lượt truy cập cao nhất
        best_move_str = max(results, key=lambda x: x[2])[0]  # x[2] là số lượt truy cập
        return chess.Move.from_uci(best_move_str)
    except Exception as e:
        print(f"Error in Rust MCTS: {e}")
        import traceback
        traceback.print_exc()
        return None