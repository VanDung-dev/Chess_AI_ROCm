import chess
import numpy as np
import torch

# Ánh xạ để mã hóa nhanh
PIECE_INDICES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Mã hóa bàn cờ đơn giản hóa và nhanh cho ResNet-SE (19 kênh).

    Các kênh:
        0-5: Quân trắng (Tốt, Mã, Tượng, Xe, Hậu, Vua)
        6-11: Quân đen (Tốt, Mã, Tượng, Xe, Hậu, Vua)
        12: Lặp lại/Luật 50 nước (đã chuẩn hóa)
        13-16: Quyền nhập thành (Trắng Vua, Trắng Hậu, Đen Vua, Đen Hậu)
        17: Mục tiêu bắt tốt qua đường
        18: Lượt đi (1=Trắng, 0=Đen)
    """
    state = np.zeros((19, 8, 8), dtype=np.float32)
    
    # 1. Vị trí các quân cờ (Kênh 0-11)
    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        piece_type_idx = PIECE_INDICES[piece.piece_type]
        color_offset = 0 if piece.color == chess.WHITE else 6
        state[piece_type_idx + color_offset, rank, file] = 1.0

    # 2. Luật 50 nước (Kênh 12) - Chuẩn hóa 0-1
    state[12, :, :] = board.halfmove_clock / 100.0

    # 3. Quyền nhập thành (Kênh 13-16)
    if board.has_kingside_castling_rights(chess.WHITE):
        state[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        state[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        state[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        state[16, :, :] = 1.0
        
    # 4. Bắt tốt qua đường (Kênh 17)
    if board.ep_square is not None:
        rank, file = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        state[17, rank, file] = 1.0
        
    # 5. Lượt đi (Kênh 18)
    if board.turn == chess.WHITE:
        state[18, :, :] = 1.0
        
    return torch.from_numpy(state).float()

def move_to_index(move: chess.Move) -> int:
    """
    Mã hóa một nước đi thành một chỉ số (0-4671).
    Nước đi cơ bản: (ô_xuất_phát * 64) + ô_đích (0-4095)
    Phong cấp thấp: Được mã hóa riêng để tránh xung đột hoặc xử lý đơn giản.
    """
    return (move.from_square * 64) + move.to_square

def decode_move(index: int, board: chess.Board) -> chess.Move:
    """
    Giải mã một chỉ số trở lại thành một nước đi chess.Move.

    Lưu ý: Luôn giả định phong Hậu nếu có thể, vì chúng ta đã đơn giản hóa việc mã hóa.
    """
    from_square = index // 64
    to_square = index % 64
    
    # Kiểm tra phong cấp
    promotion = None
    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.PAWN:
        rank = chess.square_rank(to_square)
        if (piece.color == chess.WHITE and rank == 7) or (piece.color == chess.BLACK and rank == 0):
            promotion = chess.QUEEN
            
    return chess.Move(from_square, to_square, promotion=promotion)

# Các hàm trợ giúp để tăng cường dữ liệu, được cung cấp để tương thích
def flip_vertical(move: chess.Move) -> chess.Move:
    from_sq = chess.square(chess.square_file(move.from_square), 7 - chess.square_rank(move.from_square))
    to_sq = chess.square(chess.square_file(move.to_square), 7 - chess.square_rank(move.to_square))
    return chess.Move(from_sq, to_sq, move.promotion)

def flip_horizontal(move: chess.Move) -> chess.Move:
    from_sq = chess.square(7 - chess.square_file(move.from_square), chess.square_rank(move.from_square))
    to_sq = chess.square(7 - chess.square_file(move.to_square), chess.square_rank(move.to_square))
    return chess.Move(from_sq, to_sq, move.promotion)
