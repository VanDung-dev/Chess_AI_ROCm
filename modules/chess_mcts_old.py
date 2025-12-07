import torch
import chess
import numpy as np
from collections import OrderedDict
from modules.model.engine import encode_board, move_to_index
from modules.config.config import DEVICE

# Bảng Zobrist hashing
ZOBRIST_TABLE = None
ZOBRIST_TURN = np.random.randint(1, 2**63, dtype=np.int64)
ZOBRIST_CASTLING = np.random.randint(1, 2**63, (4,), dtype=np.int64)
ZOBRIST_EN_PASSANT = np.random.randint(1, 2**63, (8,), dtype=np.int64)

def initialize_zobrist_table():
    global ZOBRIST_TABLE
    if ZOBRIST_TABLE is None:
        ZOBRIST_TABLE = np.random.randint(1, 2**63, (12, 64), dtype=np.int64)

# Khởi tạo bảng Zobrist
initialize_zobrist_table()

# Bộ nhớ cache toàn cầu với LRU trục xuất
GLOBAL_CACHE = OrderedDict()
CACHE_SIZE_LIMIT = 10000
class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        """
        Khởi tạo một nút trong cây MCTS.

        Args:
            board (chess.Board): Trạng thái bàn cờ.
            parent (MCTSNode): Nút cha (mặc định None).
            move (chess.Move): Nước đi dẫn đến trạng thái này (mặc định None).
        """
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior = 0
        self.move = move
        self.stage = self.detect_game_stage()
        self.heuristic_value = self.calculate_heuristic()

    def detect_game_stage(self):
        """Xác định giai đoạn trận cờ dựa trên số quân cờ."""
        piece_count = len(self.board.piece_map())
        if piece_count > 30:
            return "opening"
        elif piece_count > 15:
            return "middlegame"
        return "endgame"

    def calculate_heuristic(self):
        """
        Tính giá trị heuristic dựa trên thế cờ, tùy thuộc vào giai đoạn.

        Returns:
            float: Giá trị heuristic.
        """
        heuristic = 0
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        king_safety_squares = [
            chess.F2, chess.G2, chess.H2, chess.F3, chess.G3, chess.H3,
            chess.F6, chess.G6, chess.H6, chess.F7, chess.G7, chess.H7
        ]
        pawn_structure_squares = [chess.D2, chess.E2, chess.F2, chess.D7, chess.E7, chess.F7]

        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                heuristic += value if piece.color == self.board.turn else -value

        # Heuristic cụ thể theo giai đoạn
        if self.stage == "opening":
            for square in center_squares:
                if self.board.is_attacked_by(self.board.turn, square):
                    heuristic += 0.2
                if self.board.is_attacked_by(not self.board.turn, square):
                    heuristic -= 0.1
        elif self.stage == "middlegame":
            legal_moves = len(list(self.board.legal_moves))
            heuristic += legal_moves * 0.05
            for square in king_safety_squares:
                if self.board.is_attacked_by(self.board.turn, square):
                    heuristic += 0.15
        else:  # endgame
            king = self.board.king(self.board.turn)
            if king in king_safety_squares:
                heuristic += 0.3
            for square in pawn_structure_squares:
                piece = self.board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == self.board.turn:
                    heuristic += 0.2

        return heuristic

    def is_fully_expanded(self):
        """
        Kiểm tra xem nút đã được mở rộng hoàn toàn chưa.

        Returns:
            bool: True nếu tất cả nước đi hợp lệ đã được khám phá.
        """
        legal_moves = list(self.board.legal_moves)
        return len(self.children) == len(legal_moves)

    def best_child(self, exploration_weight=1.0):
        """
        Chọn nút con tốt nhất dựa trên UCT và heuristic.

        Args:
            exploration_weight (float): Hệ số khám phá.

        Returns:
            MCTSNode: Nút con tốt nhất hoặc None nếu không có con.
        """
        if not self.children:
            return self

        # Heuristic weight dành riêng cho giai đoạn
        heuristic_weight = {"opening": 0.1, "middlegame": 0.2, "endgame": 0.3}[self.stage]

        weights = {}
        for move, child in self.children.items():
            uct_value = (child.total_value / (child.visit_count + 1e-6))
            exploration_value = exploration_weight * child.prior * (
                (self.visit_count + 1e-6) ** 0.5 / (1 + child.visit_count))
            heuristic_value = heuristic_weight * child.heuristic_value
            weights[move] = uct_value + exploration_value + heuristic_value

        best_move = max(weights, key=weights.get)
        return self.children[best_move]


def mcts_rollout(model, root, cache=None):
    """
    Thực hiện một lần lặp MCTS: selection, expansion, simulation, backpropagation.

    Args:
        model: Mô hình AI (ChessNet).
        root (MCTSNode): Nút gốc của cây MCTS.
        cache (dict): Bộ nhớ đệm để lưu giá trị bàn cờ (mặc định None).
    """
    if cache is None:
        cache = {}

    node = root
    path = []

    # Giai đoạn Selection
    while not node.board.is_game_over() and node.is_fully_expanded():
        node = node.best_child(exploration_weight={"opening": 1.0, "middlegame": 0.8, "endgame": 0.5}[node.stage])
        path.append(node)

    # Giai đoạn Expansion
    if not node.board.is_game_over():
        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            return

        unvisited_moves = [move for move in legal_moves if move not in node.children]
        if not unvisited_moves:
            return

        # Xử lý hàng loạt cho Effective
        batch_threshold = 4 if node.stage == "endgame" else 2
        if len(unvisited_moves) >= batch_threshold:
            new_boards = []
            new_nodes_info = []

            for move in unvisited_moves:
                new_board = node.board.copy()
                new_board.push(move)
                new_node = MCTSNode(new_board, parent=node, move=move)
                new_boards.append(new_board)
                new_nodes_info.append((move, new_node))

            policies, values = batch_model_call(model, new_boards)

            for i, (move, new_node) in enumerate(new_nodes_info):
                policy_probs = torch.softmax(policies[i:i + 1], dim=1)
                move_index = move_to_index(move)
                new_node.prior = policy_probs[0][move_index].item()
                node.children[move] = new_node
        else:
            for move in unvisited_moves:
                new_board = node.board.copy()
                new_board.push(move)
                new_node = MCTSNode(new_board, parent=node, move=move)
                policy, value = cached_model_call(model, new_board)
                move_index = move_to_index(move)
                policy_probs = torch.softmax(policy, dim=1)
                new_node.prior = policy_probs[0][move_index].item()
                node.children[move] = new_node

        # Chọn node con có prior cao nhất
        if node.children:
            best_child = max(node.children.values(), key=lambda x: x.prior)
            node = best_child

    # Giai đoạn Simulation
    _, value = cached_model_call(model, node.board)
    value = value.item()

    # Giai đoạn Backpropagation
    current_node = node
    while current_node is not None:
        if current_node.board.turn == root.board.turn:
            current_node.total_value += value
        else:
            current_node.total_value -= value
        current_node.visit_count += 1
        current_node = current_node.parent


def get_best_move(model, board, mcts_iterations=100, temperature=1.0):
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
    # Clear cache trước mỗi nước đi
    global GLOBAL_CACHE
    GLOBAL_CACHE.clear()

    root = MCTSNode(board)

    # Các lần lặp MCT động dựa trên giai đoạn và độ phức tạp
    legal_moves_count = len(list(board.legal_moves))
    piece_count = len(board.piece_map())
    stage = root.stage
    if stage == "opening":
        mcts_iterations = min(50, mcts_iterations)
    elif stage == "middlegame":
        mcts_iterations = int(mcts_iterations * (1 + legal_moves_count / 20))
    else:  # endgame
        mcts_iterations = min(300, mcts_iterations * 2)

    # Warm-up model
    _ = cached_model_call(model, board)

    for _ in range(mcts_iterations):
        mcts_rollout(model, root)

    if not root.children:
        return None

    # Tính xác suất dựa trên số lần thăm
    total_visits = root.visit_count
    move_probs = {}
    for move, child in root.children.items():
        move_probs[move] = child.visit_count / total_visits

    # Áp dụng nhiệt độ
    if temperature > 0:
        moves_list = list(move_probs.keys())
        probs_list = [move_probs[move] for move in moves_list]
        log_probs = np.log([prob + 1e-10 for prob in probs_list])
        scaled_log_probs = log_probs / temperature
        exp_probs = np.exp(scaled_log_probs - np.max(scaled_log_probs))
        exp_probs /= exp_probs.sum()
        move_probs = dict(zip(moves_list, exp_probs))

    best_move = max(move_probs, key=move_probs.get)
    return best_move


def get_board_cache_key(board):
    """
    Tạo cache key hiệu quả cho board bằng Zobrist hashing.

    Args:
        board (chess.Board): Trạng thái bàn cờ hiện tại.

    Returns:
        int: Mã cache key.
    """
    key = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
            key ^= ZOBRIST_TABLE[piece_idx, square]
    if board.turn == chess.BLACK:
        key ^= ZOBRIST_TURN
    for i, castling in enumerate([
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]):
        if castling:
            key ^= ZOBRIST_CASTLING[i]
    if board.ep_square:
        key ^= ZOBRIST_EN_PASSANT[chess.square_file(board.ep_square)]
    return key


def cached_model_call(model, board):
    """
    Gọi model với caching hiệu quả.

    Args:
        model: The AI model (ChessNet).
        board (chess.Board): The current chess board state.

    Returns:
        tuple: Policy and value from the model.
    """
    cache_key = get_board_cache_key(board)
    if cache_key in GLOBAL_CACHE:
        result = GLOBAL_CACHE.pop(cache_key)
        GLOBAL_CACHE[cache_key] = result
        return result

    encoded = encode_board(board).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    with torch.no_grad():
        policy, value = model(encoded)

    result = (policy, value)
    if len(GLOBAL_CACHE) >= CACHE_SIZE_LIMIT:
        GLOBAL_CACHE.popitem(last=False)
    GLOBAL_CACHE[cache_key] = result
    return result


def batch_model_call(model, boards_list):
    """
    Thực hiện suy luận hàng loạt cho nhiều bảng.

    Args:
        model: Mô hình AI (ChessNet).
        boards_list (list): Danh sách các bàn cờ để đánh giá.

    Returns:
        tuple: Chính sách và giá trị cho lô bảng.
    """
    if not boards_list:
        return [], []

    encoded_list = [encode_board(board) for board in boards_list]
    batch_tensor = torch.stack(encoded_list).to(DEVICE, dtype=torch.float32)
    with torch.no_grad():
        policies, values = model(batch_tensor)
    return policies, values