import torch
import chess
import numpy as np
from collections import OrderedDict
from modules.chess_engine import encode_board, move_to_index
from modules.chess_config import DEVICE

# Global cache với LRU eviction
GLOBAL_CACHE = OrderedDict()
CACHE_SIZE_LIMIT = 5000

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
        self.heuristic_value = self.calculate_heuristic()
        self.stage = self.detect_game_stage()

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
        Tính giá trị heuristic dựa trên thế cờ.

        Returns:
            float: Giá trị heuristic (thưởng bảo vệ, phạt đe dọa).
        """
        heuristic = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                if self.board.is_attacked_by(self.board.turn, square):
                    heuristic += 0.1
            if self.board.is_attacked_by(not self.board.turn, square):
                heuristic -= 0.05
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
            exploration_weight (float): Hệ số khám phá (mặc định 1.0).

        Returns:
            MCTSNode: Nút con tốt nhất hoặc None nếu không có con.
        """
        if not self.children:
            return self

        # Điều chỉnh trọng số heuristic theo giai đoạn
        if self.stage == "opening":
            heuristic_weight = 0.1
        elif self.stage == "middlegame":
            heuristic_weight = 0.2
        else:  # endgame
            heuristic_weight = 0.3

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
        node = node.best_child()
        path.append(node)

    # Giai đoạn Expansion
    if not node.board.is_game_over():
        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            return

        unvisited_moves = [move for move in legal_moves if move not in node.children]
        if not unvisited_moves:
            return

        # Batch processing nếu có nhiều moves
        if len(unvisited_moves) >= 3:  # Threshold cho batch processing
            # Tạo tất cả new boards
            new_boards = []
            new_nodes_info = []

            for move in unvisited_moves:
                new_board = node.board.copy()
                new_board.push(move)
                new_node = MCTSNode(new_board, parent=node, move=move)
                new_boards.append(new_board)
                new_nodes_info.append((move, new_node))

            # Batch inference
            policies, values = batch_model_call(model, new_boards)

            # Process batch results
            for i, (move, new_node) in enumerate(new_nodes_info):
                policy_probs = torch.softmax(policies[i:i + 1], dim=1)
                move_index = move_to_index(move)
                new_node.prior = policy_probs[0][move_index].item()
                node.children[move] = new_node
        else:
            # Individual inference cho ít moves
            for move in unvisited_moves:
                new_board = node.board.copy()
                new_board.push(move)
                new_node = MCTSNode(new_board, parent=node, move=move)

                # Cached model call
                policy, value = cached_model_call(model, new_board)
                move_index = move_to_index(move)
                policy_probs = torch.softmax(policy, dim=1)
                new_node.prior = policy_probs[0][move_index].item()
                node.children[move] = new_node

        # Chọn node con có prior cao nhất
        if node.children:
            best_child = max(node.children.values(), key=lambda x: x.prior)
            node = best_child

    # Giai đoạn Simulation - sử dụng cached model call
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

    # Warm-up model
    _ = cached_model_call(model, board)

    # MCTS iterations với progress tracking tối giản
    for i in range(mcts_iterations):
        mcts_rollout(model, root)

        # Chỉ log mỗi 25% để tránh overhead
        if i > 0 and i % max(1, mcts_iterations // 4) == 0:
            pass  # Có thể thêm log tối giản nếu cần

    if not root.children:
        return None

    # Tính xác suất dựa trên số lần thăm
    total_visits = root.visit_count
    move_probs = {}
    for move, child in root.children.items():
        move_probs[move] = child.visit_count / total_visits

    # Áp dụng nhiệt độ để điều chỉnh độ ngẫu nhiên
    if temperature > 0:
        moves_list = list(move_probs.keys())
        probs_list = [move_probs[move] for move in moves_list]

        # Chuyển sang dạng log để tránh underflow
        log_probs = np.log([prob + 1e-10 for prob in probs_list])
        scaled_log_probs = log_probs / temperature
        exp_probs = np.exp(scaled_log_probs - np.max(scaled_log_probs))
        exp_probs /= exp_probs.sum()

        # Cập nhật lại xác suất
        move_probs = dict(zip(moves_list, exp_probs))

    # Chọn nước đi tốt nhất
    best_move = max(move_probs, key=move_probs.get)
    return best_move


def get_board_cache_key(board):
    """
    Tạo cache key hiệu quả cho board

    Args:
        board (chess.Board): Trạng thái bàn cờ hiện tại.

    Returns:
        int: Mã cache key.
    """
    # Dùng hash nhanh hơn string FEN
    piece_map = tuple(sorted(board.piece_map().items()))
    turn = board.turn
    castling = (
        board.has_kingside_castling_rights(True),
        board.has_queenside_castling_rights(True),
        board.has_kingside_castling_rights(False),
        board.has_queenside_castling_rights(False)
    )
    en_passant = board.ep_square if board.ep_square else -1
    return hash((piece_map, turn, castling, en_passant))


def cached_model_call(model, board):
    """
    Gọi model với caching hiệu quả

    Args:
        model: The AI model (ChessNet).
        board (chess.Board): The current chess board state.

    Returns:
        tuple: Policy and value from the model.
    """
    cache_key = get_board_cache_key(board)

    # Kiểm tra bộ nhớ cache
    if cache_key in GLOBAL_CACHE:
        # Di chuyển đến kết thúc (LRU)
        result = GLOBAL_CACHE.pop(cache_key)
        GLOBAL_CACHE[cache_key] = result
        return result

    # Tính kết quả mới
    encoded = encode_board(board).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    with torch.no_grad():
        policy, value, _ = model(encoded)

    result = (policy, value)

    # Thêm vào bộ đệm với giới hạn kích thước
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
        return []

    # Mã hóa tất cả các bảng
    encoded_list = []
    for board in boards_list:
        encoded = encode_board(board)
        encoded_list.append(encoded)

    # Xếp chồng vào tenor
    batch_tensor = torch.stack(encoded_list).to(DEVICE, dtype=torch.float32)

    # Suy luận hàng loạt
    with torch.no_grad():
        policies, values, _ = model(batch_tensor)

    return policies, values
