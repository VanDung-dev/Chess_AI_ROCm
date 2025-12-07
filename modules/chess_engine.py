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
    Mã hóa bàn cờ thành tensor 24x8x8 cho mạng nơ-ron, bao gồm các kênh bổ sung.

    Returns:
        torch.Tensor: Tensor mã hóa bàn cờ.
    """
    encoded = np.zeros((24, 8, 8), dtype=np.float32)

    # Mã hóa quân cờ
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
    encoded[12, :, :] = turn[0, :, :]

    # Kênh cho các ô bị tấn công
    attacked_by_opponent = np.zeros((1, 8, 8), dtype=np.float32)
    opponent_color = not board.turn
    for square in chess.SQUARES:
        if board.is_attacked_by(opponent_color, square):
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            attacked_by_opponent[0, rank_idx, file_idx] = 1.0
    encoded[13, :, :] = attacked_by_opponent[0, :, :]

    # Kênh cho các ô được bảo vệ
    protected_by_self = np.zeros((1, 8, 8), dtype=np.float32)
    self_color = board.turn
    for square in chess.SQUARES:
        if board.is_attacked_by(self_color, square):
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            protected_by_self[0, rank_idx, file_idx] = 1.0
    encoded[14, :, :] = protected_by_self[0, :, :]

    # Kênh cho giai đoạn trận đấu
    game_stage = np.zeros((1, 8, 8), dtype=np.float32)
    piece_count = len(board.piece_map())
    if piece_count > 30:
        stage_value = 0.0
    elif piece_count > 15:
        stage_value = 0.5
    else:
        stage_value = 1.0
    game_stage[0, :, :] = stage_value
    encoded[15, :, :] = game_stage[0, :, :]

    # Kênh cho kiểm soát trung tâm
    center_control = np.zeros((1, 8, 8), dtype=np.float32)
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    for square in center_squares:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        if board.is_attacked_by(self_color, square):
            center_control[0, rank_idx, file_idx] = 1.0
        if board.is_attacked_by(opponent_color, square):
            center_control[0, rank_idx, file_idx] -= 0.5
    encoded[16, :, :] = center_control[0, :, :]

    # Kênh cho vị trí vua
    king_position = np.zeros((1, 8, 8), dtype=np.float32)
    king_square = board.king(self_color)
    if king_square:
        file_idx = chess.square_file(king_square)
        rank_idx = chess.square_rank(king_square)
        king_position[0, rank_idx, file_idx] = 1.0
    encoded[17, :, :] = king_position[0, :, :]

    # Kênh cho cấu trúc tốt
    pawn_structure = np.zeros((1, 8, 8), dtype=np.float32)
    pawn_squares = [chess.D2, chess.E2, chess.F2, chess.D7, chess.E7, chess.F7]
    for square in pawn_squares:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN and piece.color == self_color:
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            pawn_structure[0, rank_idx, file_idx] = 1.0
    encoded[18, :, :] = pawn_structure[0, :, :]

    # Kênh cho tính di động
    mobility = np.zeros((1, 8, 8), dtype=np.float32)
    mobility[0, :, :] = len(list(board.legal_moves)) / 40.0
    encoded[19, :, :] = mobility[0, :, :]
    
    # Kênh cho các quân đang bị tấn công
    attacked_pieces = np.zeros((1, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and board.is_attacked_by(opponent_color, square):
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            # Tăng giá trị dựa trên giá trị quân cờ (quân quan trọng hơn có giá trị cao hơn)
            piece_value = {chess.PAWN: 0.2, chess.KNIGHT: 0.5, chess.BISHOP: 0.5, 
                          chess.ROOK: 0.8, chess.QUEEN: 1.0, chess.KING: 0.0}[piece.piece_type]
            attacked_pieces[0, rank_idx, file_idx] = piece_value
    encoded[20, :, :] = attacked_pieces[0, :, :]
    
    # Kênh cho các quân đang tấn công quân địch
    attacking_pieces = np.zeros((1, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and board.is_attacked_by(self_color, square):
            # Kiểm tra nếu quân này đang tấn công quân địch
            attackers = []
            for attack_square in chess.SQUARES:
                attacked_piece = board.piece_at(attack_square)
                if attacked_piece and attacked_piece.color != piece.color:
                    # Kiểm tra nếu quân tại square đang tấn công attack_square
                    if board.is_attacked_by(piece.color, attack_square) and \
                       any(attacker == square for attacker in board.attackers(piece.color, attack_square)):
                        attackers.append(attack_square)
            
            if attackers:
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)
                # Tăng giá trị dựa trên số quân đang bị tấn công
                attacking_pieces[0, rank_idx, file_idx] = min(len(attackers) * 0.3, 1.0)
    encoded[21, :, :] = attacking_pieces[0, :, :]
    
    # Kênh cho các quân đang bảo vệ quân khác của mình
    protecting_pieces = np.zeros((1, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Tìm các quân cùng màu đang bảo vệ quân này
            protectors = list(board.attackers(piece.color, square))
            if protectors:
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)
                # Tăng giá trị dựa trên số lượng quân bảo vệ
                protecting_pieces[0, rank_idx, file_idx] = min(len(protectors) * 0.3, 1.0)
    encoded[22, :, :] = protecting_pieces[0, :, :]
    
    # Kênh cho các quân địch đang được bảo vệ
    opponent_protected_pieces = np.zeros((1, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color != self_color:
            # Tìm các quân địch đang bảo vệ quân này
            protectors = list(board.attackers(piece.color, square))
            if protectors:
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)
                # Tăng giá trị dựa trên số lượng quân bảo vệ
                opponent_protected_pieces[0, rank_idx, file_idx] = min(len(protectors) * 0.3, 1.0)
    encoded[23, :, :] = opponent_protected_pieces[0, :, :]

    return torch.from_numpy(encoded)


def move_to_index(move: chess.Move) -> int:
    """
    Chuyển nước đi thành chỉ số trong không gian 4672.
    """
    return (move.from_square << 6) + move.to_square


def flip_vertical(move: chess.Move) -> chess.Move:
    """
    Lật nước đi theo chiều dọc.
    """
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
    Lật nước đi theo chiều ngang.
    """
    from_square = move.from_square
    to_square = move.to_square
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)
    flipped_from = chess.square(7 - from_file, from_rank)
    flipped_to = chess.square(7 - to_file, to_rank)
    return chess.Move(flipped_from, flipped_to, move.promotion, move.drop)