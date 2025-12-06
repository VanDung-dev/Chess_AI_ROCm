import chess
import chess_mcts_rs


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
        # Gọi triển khai Rust
        results = chess_mcts_rs.mcts_loop(fen, mcts_iterations)
        
        if not results:
            return None
            
        # Tìm động thái có số lượt truy cập cao nhất
        best_move_str = max(results, key=lambda x: x[2])[0]  # x[2] là số lượt truy cập
        return chess.Move.from_uci(best_move_str)
    except Exception as e:
        print(f"Error in Rust MCTS: {e}")
        return None