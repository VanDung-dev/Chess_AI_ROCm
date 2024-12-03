import torch
import torch_directml
import numpy as np
import copy
from chess_engine import Move

device = torch_directml.device(0)


class MCTSNode:
    def __init__(self, state, parent=None, move_key=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior = 0
        self.move_key = move_key

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_valid_moves())

    def best_child(self, exploration_weight=1):
        weights = {
            move_key: (child.total_value / (child.visit_count + 1e-6)) +
            exploration_weight * child.prior * (self.visit_count ** 0.5 / (1 + child.visit_count))
            for move_key, child in self.children.items()
        }
        best_move_key = max(weights, key=weights.get)
        return self.children[best_move_key]

def mcts_rollout(ai_model, root):
    node = root
    while not node.state.checkmate and not node.state.stalemate and node.is_fully_expanded():
        node = node.best_child()

    if not node.state.checkmate and not node.state.stalemate:
        valid_moves = node.state.get_valid_moves()
        for move in valid_moves:
            move_key = (move.start_row, move.start_column, move.end_row, move.end_column)

            # Tạo bản sao GameState để không ảnh hưởng trạng thái gốc
            new_state = copy.deepcopy(node.state)
            new_state.make_move(move)

            # Tạo node con
            new_node = MCTSNode(new_state, parent=node, move_key=move_key)
            policy, value = ai_model(encode_board(new_state).unsqueeze(0))
            new_node.prior = policy[0][move_to_index(move)].item()
            node.children[move_key] = new_node

    # Lấy giá trị trạng thái và truyền ngược lại
    _, value = ai_model(encode_board(node.state).unsqueeze(0))
    backpropagate(node, value.item())

def move_to_index(move):
    """Chuyển nước đi thành chỉ số duy nhất."""
    if not isinstance(move, Move):
        raise ValueError("move phải là một đối tượng Move.")
    start_idx = move.start_row * 8 + move.start_column
    end_idx = move.end_row * 8 + move.end_column
    return start_idx * 64 + end_idx

def encode_board(game_state):
    """Mã hóa bàn cờ và thông tin trạng thái thành tensor."""
    piece_map = {
        "--": 0, "wK": 1, "wQ": 2, "wR": 3, "wB": 4, "wN": 5, "wP": 6,
        "bK": 7, "bQ": 8, "bR": 9, "bB": 10, "bN": 11, "bP": 12,
    }
    board = game_state.board
    encoded_board = np.zeros((13, 8, 8), dtype=np.float32)  # [channels, height, width]
    for i, row in enumerate(board):
        for j, piece in enumerate(row):
            encoded_board[piece_map[piece], i, j] = 1  # One-hot encoding cho mỗi quân cờ

    # Chuyển đổi thành tensor PyTorch
    return torch.tensor(encoded_board, dtype=torch.float32, device=device)

def simulate_move(game_state, move):
    """
    Thực hiện một nước đi trên game_state.
    Trả về trạng thái mới sau nước đi và phần thưởng.
    """
    captured_piece = move.piece_captured
    game_state.make_move(move)  # Thực hiện nước đi
    reward = 0
    if game_state.checkmate:
        reward = 1 if game_state.white_to_move else -1
    elif game_state.stalemate or game_state.stalemate_special():
        reward = 0
    elif captured_piece != "--":
        reward += piece_value(captured_piece)

    game_state.undo_move()  # Hoàn tác nước đi sau khi tính toán
    return game_state, reward

def piece_value(piece):
    """Giá trị quân cờ."""
    values = {"wP": 1, "bP": -1, "wN": 3, "bN": -3, "wB": 3, "bB": -3,
              "wR": 5, "bR": -5, "wQ": 9, "bQ": -9, "wK": 0, "bK": 0}
    return values.get(piece, 0)

def backpropagate(node, value):
    """Truyền giá trị ngược lại từ nút hiện tại lên đến nút gốc."""
    while node is not None:
        # Cập nhật tổng giá trị và số lần thăm
        node.total_value += value
        node.visit_count += 1

        # Đảo giá trị nếu lượt là của đối thủ
        value = -value
        node = node.parent