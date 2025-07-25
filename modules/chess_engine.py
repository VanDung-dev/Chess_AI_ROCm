import chess
import numpy as np
import torch

# Pre-computed maps để tránh tạo lại mỗi lần
PIECE_MAP = {
    None: 0,
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

COLOR_MAP = {chess.WHITE: 0, chess.BLACK: 6}


def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Mã hóa bàn cờ thành tensor 16x8x8 cho mạng nơ-ron - phiên bản tối ưu.
    """
    # Sử dụng pre-allocated arrays nếu có thể (tuy nhiên để đơn giản vẫn tạo mới)
    encoded = np.zeros((12, 8, 8), dtype=np.float32)

    # Mã hóa quân cờ - vectorized approach
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            type_idx = PIECE_MAP[piece.piece_type]
            color_idx = COLOR_MAP[piece.color]
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            channel = type_idx + color_idx - 1
            encoded[channel, rank_idx, file_idx] = 1.0

    # Kênh cho lượt đi
    turn = np.zeros((1, 8, 8), dtype=np.float32)
    if board.turn == chess.WHITE:
        turn[0, :, :] = 1.0

    # Kênh cho các ô bị tấn công - tối ưu với numpy
    attacked_by_opponent = np.zeros((1, 8, 8), dtype=np.float32)
    opponent_color = not board.turn

    # Vectorized attack checking
    attacked_squares = []
    for square in chess.SQUARES:
        if board.is_attacked_by(opponent_color, square):
            attacked_squares.append(square)

    # Fill attacked squares
    for square in attacked_squares:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        attacked_by_opponent[0, rank_idx, file_idx] = 1.0

    # Kênh cho các ô được bảo vệ
    protected_by_self = np.zeros((1, 8, 8), dtype=np.float32)
    self_color = board.turn

    # Vectorized protection checking
    protected_squares = []
    for square in chess.SQUARES:
        if board.is_attacked_by(self_color, square):
            protected_squares.append(square)

    # Fill protected squares
    for square in protected_squares:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        protected_by_self[0, rank_idx, file_idx] = 1.0

    # Kênh cho giai đoạn trận đấu
    game_stage = np.zeros((1, 8, 8), dtype=np.float32)
    piece_count = len(board.piece_map())

    # Sử dụng np.where hoặc conditional assignment
    if piece_count > 30:
        stage_value = 0.0
    elif piece_count > 15:
        stage_value = 0.5
    else:
        stage_value = 1.0

    game_stage[0, :, :] = stage_value

    # Kết hợp tất cả các kênh - sử dụng np.concatenate
    result = np.concatenate([encoded, turn, attacked_by_opponent, protected_by_self, game_stage], axis=0)
    return torch.from_numpy(result)


def move_to_index(move: chess.Move) -> int:
    """
    Chuyển nước đi thành chỉ số trong không gian 4672.

    Chỉ số được tính bằng cách kết hợp ô bắt đầu và ô kết thúc của nước đi.

    Args:
        move (chess.Move): Nước đi cần chuyển đổi.

    Returns:
        int: Chỉ số từ 0 đến 4671 tương ứng với nước đi.
    """
    # Tối giản phép tính
    return (move.from_square << 6) + move.to_square  # Dùng bit shift thay vì nhân


def flip_vertical(move: chess.Move) -> chess.Move:
    """
    Lật nước đi theo chiều dọc (đảo tọa độ hàng từ trên xuống dưới và ngược lại).

    Args:
        move (chess.Move): Nước đi ban đầu.

    Returns:
        chess.Move: Nước đi đã được lật theo chiều dọc.
    """
    # Cache các giá trị để tránh gọi hàm nhiều lần
    from_square = move.from_square
    to_square = move.to_square

    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)

    flipped_from = chess.square(from_file, 7 - from_rank)
    flipped_to = chess.square(to_file, 7 - to_rank)

    return chess.Move(flipped_from, flipped_to, move.promotion, move.drop)


def flip_horizontal(move: chess.Move) -> chess.Move:
    """
    Lật nước đi theo chiều ngang (đảo tọa độ cột trái-phải).

    Args:
        move (chess.Move): Nước đi ban đầu.

    Returns:
        chess.Move: Nước đi đã được lật theo chiều ngang.
    """
    # Cache các giá trị
    from_square = move.from_square
    to_square = move.to_square

    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)

    flipped_from = chess.square(7 - from_file, from_rank)
    flipped_to = chess.square(7 - to_file, to_rank)

    return chess.Move(flipped_from, flipped_to, move.promotion, move.drop)
