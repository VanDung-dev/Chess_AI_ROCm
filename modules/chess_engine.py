import chess
import numpy as np
import torch


def encode_board(board):
    """
    Mã hóa bàn cờ thành tensor 16x8x8 cho mạng nơ-ron.

    - 12 kênh cho các loại quân cờ (6 loại x 2 màu).
    - 1 kênh cho lượt đi (trắng hoặc đen).
    - 1 kênh cho các ô bị tấn công bởi đối phương.
    - 1 kênh cho các ô được bảo vệ bởi quân cờ của mình.
    - 1 kênh cho giai đoạn trận đấu.

    Args:
        board (chess.Board): Trạng thái bàn cờ hiện tại.

    Returns:
        torch.Tensor: Tensor 16x8x8 mã hóa bàn cờ.
    """
    piece_map = {
        None: 0,
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }
    color_map = {chess.WHITE: 0, chess.BLACK: 6}
    encoded = np.zeros((12, 8, 8), dtype=np.float32)  # 12 kênh cho quân cờ
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            type_idx = piece_map[piece.piece_type]
            color_idx = color_map[piece.color]
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            channel = type_idx + color_idx - 1
            encoded[channel, rank_idx, file_idx] = 1.0

    # Kênh cho lượt đi
    turn = np.zeros((1, 8, 8), dtype=np.float32)
    if board.turn == chess.WHITE:
        turn[0, :, :] = 1.0

    # Kênh cho các ô bị tấn công bởi đối phương (mối đe dọa)
    attacked_by_opponent = np.zeros((1, 8, 8), dtype=np.float32)
    opponent_color = not board.turn
    for square in chess.SQUARES:
        if board.is_attacked_by(opponent_color, square):
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            attacked_by_opponent[0, rank_idx, file_idx] = 1.0

    # Kênh cho các ô được bảo vệ bởi quân cờ của mình (thế cờ)
    protected_by_self = np.zeros((1, 8, 8), dtype=np.float32)
    self_color = board.turn
    for square in chess.SQUARES:
        if board.is_attacked_by(self_color, square):
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            protected_by_self[0, rank_idx, file_idx] = 1.0

    # Kênh cho giai đoạn trận đấu
    game_stage = np.zeros((1, 8, 8), dtype=np.float32)
    piece_count = len(board.piece_map())
    if piece_count > 30:  # Khai cuộc
        stage_value = 0.0
    elif piece_count > 15:  # Trung cuộc
        stage_value = 0.5
    else:  # Tàn cuộc
        stage_value = 1.0
    game_stage[0, :, :] = stage_value

    # Kết hợp tất cả các kênh
    encoded = np.concatenate([encoded, turn, attacked_by_opponent, protected_by_self, game_stage], axis=0)
    return torch.tensor(encoded, dtype=torch.float32)


def move_to_index(move):
    """
    Chuyển nước đi thành chỉ số trong không gian 4672.

    Chỉ số được tính bằng cách kết hợp ô bắt đầu và ô kết thúc của nước đi.

    Args:
        move (chess.Move): Nước đi cần chuyển đổi.

    Returns:
        int: Chỉ số từ 0 đến 4671 tương ứng với nước đi.
    """
    start_idx = move.from_square
    end_idx = move.to_square
    return start_idx * 64 + end_idx