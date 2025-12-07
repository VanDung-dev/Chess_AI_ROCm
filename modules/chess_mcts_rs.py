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
            # Sử dụng cached_model_call trả về (policy, value)
            policy, value_tensor = cached_model_call(self.model, board)
            value = value_tensor.item()
            
            # Chuyển đổi policy thành xác suất
            policy_probs = torch.softmax(policy, dim=1)
            
            legal_moves = list(board.legal_moves)
            priors = []
            
            for move in legal_moves:
                move_idx = move_to_index(move)
                move_prob = policy_probs[0][move_idx].item()
                priors.append((move.uci(), move_prob))
                
            return (priors, value)
        except Exception as e:
            print(f"Lỗi trong BoardEvaluator: {e}")
            return ([], 0.0)

import numpy as np

def run_mcts(model, board, mcts_iterations=800, temperature=1.0):
    """
    Chạy MCTS và trả về nước đi tốt nhất cùng với vector policy đã được cải thiện.

    Args:
        model: Mô hình ChessNet
        board: chess.Board
        mcts_iterations: số lần mô phỏng
        temperature: nhiệt độ cho softmax của policy (thường là 1.0 để huấn luyện, 0.1 để chơi)
    Returns:
        tuple: (nước đi tốt nhất, giá trị gốc, mảng xác suất policy)
    """
    try:
        fen = board.fen()
        evaluator = BoardEvaluator(model)
        root_priors, root_value = evaluator(fen)
        
        # Thực thi MCTS bằng Rust
        results = chess_rs.mcts_loop(fen, mcts_iterations, evaluator, root_value, root_priors)
        
        if not results:
            print("MCTS không trả về kết quả.")
            return None, 0.0, None
            
        # Phân tích kết quả: [(move_uci, win_rate, visits), ...]
        total_visits = sum(r[2] for r in results)
        
        # Khởi tạo vector policy (kích thước 4672 cho tất cả các nước đi có thể)
        policy_size = 4672 
        policy_vector = np.zeros(policy_size, dtype=np.float32)
        
        # Điền vào vector policy dựa trên số lần truy cập
        moves_visits = []
        for move_str, win_rate, visits in results:
            move = chess.Move.from_uci(move_str)
            idx = move_to_index(move)
            if 0 <= idx < policy_size:
                policy_vector[idx] = visits
                moves_visits.append((move, visits))
            else:
                print(f"Cảnh báo: Chỉ số {idx} của nước đi {move_str} nằm ngoài giới hạn.")

        # Chuẩn hóa thành xác suất dựa trên nhiệt độ
        if total_visits > 0:
            if temperature == 0:
                # Hành vi Argmax
                best_idx = np.argmax(policy_vector)
                policy_vector[:] = 0
                policy_vector[best_idx] = 1.0
            else:
                # Chuẩn AlphaZero:
                # Đầu ván cờ: pi ~ N^(1/t)
                # Cuối ván cờ: pi = argmax(N)
                
                temp_adj_visits = np.power(policy_vector, 1.0 / temperature)
                policy_sum = np.sum(temp_adj_visits)
                if policy_sum > 0:
                    policy_vector = temp_adj_visits / policy_sum
                else:
                    policy_vector[:] = 0 # Không nên xảy ra nếu total_visits > 0
        
        best_move_str = max(results, key=lambda x: x[2])[0]
        best_move = chess.Move.from_uci(best_move_str)
        
        return best_move, root_value, policy_vector
        
    except Exception as e:
        print(f"Lỗi trong run_mcts: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, None

def get_best_move(model, board, mcts_iterations=1000, temperature=0.1):
    """
    Hàm bao để chơi chỉ trả về nước đi tốt nhất.
    Nhiệt độ thấp hơn theo mặc định để chơi cạnh tranh.
    """
    best_move, _, _ = run_mcts(model, board, mcts_iterations, temperature)
    return best_move