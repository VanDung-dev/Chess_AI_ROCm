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

# Giá trị quân cờ cơ bản (chuẩn hóa về khoảng [0, 1])
PIECE_VALUES = {
    None: 0.0,
    chess.PAWN: 1.0/8.0,      # 1 điểm
    chess.KNIGHT: 3.0/8.0,    # 3 điểm
    chess.BISHOP: 3.0/8.0,    # 3 điểm
    chess.ROOK: 5.0/8.0,      # 5 điểm
    chess.QUEEN: 8.0/8.0,     # 8 điểm (giá trị cao nhất, = 1.0)
    chess.KING: 0.0,          # Vua không có giá trị vật chất
}

# Giá trị phong cấp (cũng chuẩn hóa)
PROMOTION_VALUES = {
    chess.QUEEN: 8.0/8.0,    # 8/8 = 1.0 - mạnh nhất
    chess.KNIGHT: 3.0/8.0,   # 3/8 - có thể hữu ích trong một số tình huống
    chess.ROOK: 5.0/8.0,     # 5/8
    chess.BISHOP: 3.0/8.0,   # 3/8
}


def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Mã hóa bàn cờ thành tensor 32x8x8 cho mạng nơ-ron, bao gồm các kênh bổ sung.

    Returns:
        torch.Tensor: Tensor mã hóa bàn cờ.
    """
    encoded = np.zeros((32, 8, 8), dtype=np.float32)

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
            piece_value = PIECE_VALUES[piece.piece_type]
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
    
    # Kênh cho các ô trống có thể bị phản công
    vulnerable_squares = np.zeros((1, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        # Chỉ xem xét các ô trống
        if not piece:
            # Kiểm tra nếu ô này có thể bị tấn công bởi quân địch
            if board.is_attacked_by(opponent_color, square):
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)
                # Tăng giá trị dựa trên số lượng quân địch có thể tấn công ô này
                attackers = list(board.attackers(opponent_color, square))
                vulnerable_squares[0, rank_idx, file_idx] = min(len(attackers) * 0.3, 1.0)
    encoded[24, :, :] = vulnerable_squares[0, :, :]
    
    # Kênh cho giá trị quân cờ
    piece_values = np.zeros((1, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            # Giá trị dương cho quân mình, âm cho quân địch
            value = PIECE_VALUES[piece.piece_type]
            if piece.color != self_color:
                value = -value
            piece_values[0, rank_idx, file_idx] = value
    encoded[25, :, :] = piece_values[0, :, :]
    
    # Kênh cho tình trạng vua
    king_safety = np.zeros((1, 8, 8), dtype=np.float32)
    king_square = board.king(self_color)
    if king_square:
        # Kiểm tra xem vua có bị chiếu không
        if board.is_check():
            file_idx = chess.square_file(king_square)
            rank_idx = chess.square_rank(king_square)
            king_safety[0, rank_idx, file_idx] = -1.0  # Vua bị chiếu, rất nguy hiểm
            
        # Kiểm tra số lượng attacker đến vua
        attackers = list(board.attackers(opponent_color, king_square))
        if attackers:
            file_idx = chess.square_file(king_square)
            rank_idx = chess.square_rank(king_square)
            # Giá trị âm càng lớn (về phía -1) khi càng nhiều quân tấn công vua
            king_safety[0, rank_idx, file_idx] = -min(len(attackers) * 0.3, 1.0)
    encoded[26, :, :] = king_safety[0, :, :]
    
    # Kênh cho khả năng phong cấp của tốt
    promotion_potential = np.zeros((1, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        # Chỉ xem xét các tốt của mình
        if piece and piece.piece_type == chess.PAWN and piece.color == self_color:
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            
            # Kiểm tra nếu tốt có thể phong cấp (ở hàng 7 cho trắng hoặc hàng 2 cho đen)
            if (self_color == chess.WHITE and rank_idx == 6) or (self_color == chess.BLACK and rank_idx == 1):
                # Kiểm tra nếu ô phong cấp an toàn (không bị tấn công nhiều)
                promotion_square = chess.square(file_idx, 7 if self_color == chess.WHITE else 0)
                attackers = len(list(board.attackers(opponent_color, promotion_square)))
                defenders = len(list(board.attackers(self_color, promotion_square)))
                
                # Tính điểm phong cấp dựa trên quân hậu (mạnh nhất)
                base_promotion_value = PROMOTION_VALUES[chess.QUEEN]
                
                # Điều chỉnh dựa trên tình trạng an toàn của ô phong cấp
                safety_factor = 1.0
                if attackers > defenders:
                    safety_factor = max(0.1, 1.0 - (attackers - defenders) * 0.2)
                    
                promotion_potential[0, rank_idx, file_idx] = base_promotion_value * safety_factor
                
            # Nếu tốt gần hàng phong cấp, cho một giá trị nhỏ
            elif (self_color == chess.WHITE and rank_idx >= 4) or (self_color == chess.BLACK and rank_idx <= 3):
                distance_to_promotion = abs(rank_idx - (7 if self_color == chess.WHITE else 0))
                potential_value = 0.5 * (1.0 - distance_to_promotion / 7.0)  # Tối đa 0.5
                promotion_potential[0, rank_idx, file_idx] = potential_value
    encoded[27, :, :] = promotion_potential[0, :, :]
    
    # Kênh cho quân địch có thể bị phong cấp
    opponent_promotion_threat = np.zeros((1, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        # Chỉ xem xét các tốt của đối phương
        if piece and piece.piece_type == chess.PAWN and piece.color != self_color:
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            
            # Kiểm tra nếu tốt đối phương có thể phong cấp
            if (self_color == chess.BLACK and rank_idx == 1) or (self_color == chess.WHITE and rank_idx == 6):
                # Kiểm tra nếu ô phong cấp an toàn
                promotion_square = chess.square(file_idx, 0 if self_color == chess.BLACK else 7)
                attackers = len(list(board.attackers(self_color, promotion_square)))
                defenders = len(list(board.attackers(not self_color, promotion_square)))
                
                # Tính điểm phong cấp dựa trên quân hậu (mạnh nhất)
                base_threat_value = PROMOTION_VALUES[chess.QUEEN]
                
                # Điều chỉnh dựa trên tình trạng an toàn của ô phong cấp
                safety_factor = 1.0
                if defenders > attackers:
                    safety_factor = max(0.1, 1.0 - (defenders - attackers) * 0.2)
                    
                opponent_promotion_threat[0, rank_idx, file_idx] = base_threat_value * safety_factor
    encoded[28, :, :] = opponent_promotion_threat[0, :, :]
    
    # Kênh cho các trao đổi quân có lợi
    favorable_exchanges = np.zeros((1, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == self_color:
            # Kiểm tra nếu quân này đang tấn công một quân địch
            attacked_squares = []
            for attack_square in chess.SQUARES:
                attacked_piece = board.piece_at(attack_square)
                if attacked_piece and attacked_piece.color != piece.color:
                    # Kiểm tra nếu quân tại square đang tấn công attack_square
                    if board.is_attacked_by(piece.color, attack_square) and \
                        any(attacker == square for attacker in board.attackers(piece.color, attack_square)):
                        attacked_squares.append((attack_square, attacked_piece))
            
            # Nếu có quân địch bị tấn công, kiểm tra xem trao đổi có có lợi không
            for attack_square, attacked_piece in attacked_squares:
                # So sánh giá trị quân đang tấn công và quân bị tấn công
                attacker_value = PIECE_VALUES[piece.piece_type]
                defender_value = PIECE_VALUES[attacked_piece.piece_type]
                
                # Kiểm tra xem quân bị tấn công có được bảo vệ không
                defenders = list(board.attackers(not self_color, attack_square))
                # Kiểm tra xem quân tấn công có được bảo vệ không
                attackers = list(board.attackers(self_color, square))
                
                # Nếu quân địch có giá trị cao hơn và ít hoặc không được bảo vệ
                # hoặc nếu quân địch có giá trị tương đương nhưng được bảo vệ ít hơn
                if defender_value > attacker_value and len(defenders) <= len(attackers):
                    file_idx = chess.square_file(attack_square)
                    rank_idx = chess.square_rank(attack_square)
                    # Giá trị trao đổi có lợi dựa trên chênh lệch giá trị
                    exchange_value = min((defender_value - attacker_value) * 2.0, 1.0)
                    favorable_exchanges[0, rank_idx, file_idx] = exchange_value
                elif defender_value >= attacker_value and len(defenders) < len(attackers):
                    file_idx = chess.square_file(attack_square)
                    rank_idx = chess.square_rank(attack_square)
                    # Giá trị trao đổi có lợi dựa trên sự chênh lệch về số lượng bảo vệ
                    protection_advantage = min((len(attackers) - len(defenders)) * 0.2, 1.0)
                    favorable_exchanges[0, rank_idx, file_idx] = protection_advantage
    encoded[29, :, :] = favorable_exchanges[0, :, :]
    
    # Kênh cho các nước đi chiếu vua đối phương
    giving_check = np.zeros((1, 8, 8), dtype=np.float32)
    # Tạo bản sao của bàn cờ để thử các nước đi
    temp_board = board.copy()
    for move in board.legal_moves:
        # Thử nước đi
        temp_board.set_board_fen(board.fen())  # Reset về trạng thái ban đầu
        temp_board.push(move)
        # Kiểm tra nếu nước đi này tạo ra chiếu
        if temp_board.is_check():
            # Đánh dấu ô đích của nước đi
            to_square = move.to_square
            file_idx = chess.square_file(to_square)
            rank_idx = chess.square_rank(to_square)
            giving_check[0, rank_idx, file_idx] = 1.0
    encoded[30, :, :] = giving_check[0, :, :]
    
    # Kênh cho tình trạng vua đối phương
    opponent_king_safety = np.zeros((1, 8, 8), dtype=np.float32)
    opponent_king_square = board.king(not self_color)
    if opponent_king_square:
        # Kiểm tra xem vua đối phương có bị chiếu không
        temp_board = board.copy()
        if temp_board.is_check():
            file_idx = chess.square_file(opponent_king_square)
            rank_idx = chess.square_rank(opponent_king_square)
            opponent_king_safety[0, rank_idx, file_idx] = 1.0  # Vua đối phương bị chiếu
            
        # Kiểm tra số lượng attacker đến vua đối phương
        attackers = list(board.attackers(self_color, opponent_king_square))
        if attackers:
            file_idx = chess.square_file(opponent_king_square)
            rank_idx = chess.square_rank(opponent_king_square)
            # Giá trị dương càng lớn khi càng nhiều quân tấn công vua đối phương
            opponent_king_safety[0, rank_idx, file_idx] = min(len(attackers) * 0.3, 1.0)
    encoded[31, :, :] = opponent_king_safety[0, :, :]

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