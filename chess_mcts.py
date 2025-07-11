import torch
import chess
import numpy as np
from chess_engine import encode_board, move_to_index
from chess_log import setup_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = setup_logger()


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
            if self.board.piece_at(square) and self.board.piece_at(square).color == self.board.turn:
                if self.board.is_attacked_by(self.board.turn, square):
                    heuristic += 0.1  # Thưởng cho quân cờ được bảo vệ
            if self.board.is_attacked_by(not self.board.turn, square):
                heuristic -= 0.05  # Phạt cho quân cờ bị đe dọa
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

        weights = {
            move: (child.total_value / (child.visit_count + 1e-6))
                  + exploration_weight * child.prior * ((self.visit_count + 1e-6) ** 0.5 / (1 + child.visit_count))
                  + heuristic_weight * child.heuristic_value
            for move, child in self.children.items()
        }
        best_move = max(weights, key=weights.get)
        return self.children[best_move]


def mcts_rollout(ai_model, root, cache=None):
    """
    Thực hiện một lần lặp MCTS: selection, expansion, simulation, backpropagation.

    Args:
        ai_model: Mô hình AI (ChessNet).
        root (MCTSNode): Nút gốc của cây MCTS.
        cache (dict): Bộ nhớ đệm để lưu giá trị bàn cờ (mặc định None).
    """
    if cache is None:
        cache = {}
    node = root
    path = []

    # Giai đoạn Selection
    logger.info("Bắt đầu giai đoạn Selection...")
    while not node.board.is_game_over() and node.is_fully_expanded():
        node = node.best_child()
        path.append(node)
    logger.debug(f"Kết thúc giai đoạn Selection. Node hiện tại: {node.board.fen()}")

    # Giai đoạn Expansion
    if not node.board.is_game_over():
        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            return

        unvisited_moves = [move for move in legal_moves if move not in node.children]
        if not unvisited_moves:
            return

        logger.info(f"Bắt đầu giai đoạn Expansion. Thêm {len(unvisited_moves)} nút con.")
        for move in unvisited_moves:
            new_board = node.board.copy()
            new_board.push(move)
            new_node = MCTSNode(new_board, parent=node, move=move)
            encoded = encode_board(new_board).unsqueeze(0).to(device, dtype=torch.float32)
            policy, value, _ = ai_model(encoded)  # Bỏ qua giai đoạn từ mô hình
            move_index = move_to_index(move)
            policy_probs = torch.softmax(policy, dim=1)
            new_node.prior = policy_probs[0][move_index].item()
            node.children[move] = new_node
        # node = list(node.children.values())[0]  # Chọn một node để simulation
        best_child = None
        max_prior = -float('inf')
        for child in node.children.values():
            if child.prior > max_prior:
                max_prior = child.prior
                best_child = child
        node = best_child
        logger.info(f"Kết thúc giai đoạn Expansion. Node mới: {node.board.fen()}")

    # Giai đoạn Simulation
    state_key = str(node.board.fen())
    if state_key not in cache:
        encoded = encode_board(node.board).unsqueeze(0).to(device, dtype=torch.float32)
        _, value, _ = ai_model(encoded)  # Bỏ qua giai đoạn từ mô hình
        value = value.item()
        cache[state_key] = value
    else:
        value = cache[state_key]
    logger.info(f"Giai đoạn Simulation. Giá trị dự đoán: {value}")

    # Giai đoạn Backpropagation
    logger.info("Bắt đầu giai đoạn Backpropagation...")
    current_node = node
    while current_node is not None:
        current_node.total_value += value if current_node.board.turn == root.board.turn else -value
        current_node.visit_count += 1
        current_node = current_node.parent
    logger.info("Kết thúc giai đoạn Backpropagation.")


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
    root = MCTSNode(board)
    cache = {}

    logger.info(f"Bắt đầu MCTS với {mcts_iterations} lần lặp...")
    for i in range(mcts_iterations):
        logger.debug(f"Lần lặp MCTS thứ {i + 1}/{mcts_iterations}")
        mcts_rollout(model, root, cache)

    if not root.children:
        logger.info("Không tìm thấy nước đi hợp lệ.")
        return None

    # Tính xác suất dựa trên số lần thăm
    move_probs = {move: child.visit_count / root.visit_count
                  for move, child in root.children.items()}

    # Áp dụng nhiệt độ để điều chỉnh độ ngẫu nhiên
    if temperature > 0:
        # Chuyển sang dạng log để tránh underflow
        log_probs = np.log([prob + 1e-10 for prob in move_probs.values()])
        scaled_log_probs = log_probs / temperature
        exp_probs = np.exp(scaled_log_probs - np.max(scaled_log_probs))  # Ổn định số học
        exp_probs /= exp_probs.sum()

        # Cập nhật lại xác suất
        move_probs = {
            move: exp_probs[i]
            for i, move in enumerate(move_probs.keys())
        }

    # Chọn nước đi tốt nhất
    best_move = max(move_probs, key=move_probs.get)
    logger.debug(f"Nước đi tốt nhất: {best_move.uci()}, Xác suất: {move_probs[best_move]:.4f}")

    return best_move
