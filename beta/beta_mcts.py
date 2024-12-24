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
            move_key: (child.total_value / (child.visit_count + 1e-6))
            + exploration_weight
            * child.prior
            * (self.visit_count**0.5 / (1 + child.visit_count))
            for move_key, child in self.children.items()
        }
        best_move_key = max(weights, key=weights.get)
        return self.children[best_move_key]


def mcts_rollout(ai_model, root, cache=None):
    """
    Một rollout của MCTS, tối ưu hóa sao chép trạng thái và sử dụng batch inference.
    """
    if cache is None:
        cache = {}

    node = root
    # Duyệt xuống cây đến nút lá
    while (
        not node.state.checkmate
        and not node.state.stalemate
        and node.is_fully_expanded()
    ):
        node = node.best_child()

    # Nếu nút hiện tại chưa phải trạng thái cuối cùng, mở rộng cây
    if not node.state.checkmate and not node.state.stalemate:
        valid_moves = node.state.get_valid_moves()
        new_states = []
        new_nodes = []
        for move in valid_moves:
            move_key = (
                move.start_row,
                move.start_column,
                move.end_row,
                move.end_column,
            )

            if move_key not in node.children:
                # Tạo trạng thái mới từ nước đi
                new_state = copy.deepcopy(node.state)
                new_state.make_move(move)
                new_node = MCTSNode(new_state, parent=node, move_key=move_key)

                # Lưu lại trạng thái và nút để xử lý batch
                new_states.append(new_state)
                new_nodes.append(new_node)
                node.children[move_key] = new_node

        # Xử lý batch inference
        if new_states:
            encoded_boards = torch.stack([encode_board(state) for state in new_states])
            policies, values = ai_model(encoded_boards)

            for i, new_node in enumerate(new_nodes):
                move_index = move_to_index(valid_moves[i])
                new_node.prior = policies[i][move_index].item()
                cache[str(new_states[i].board)] = values[i].item()  # Lưu cache

    # Lấy giá trị từ cache nếu đã đánh giá, nếu không, sử dụng mô hình AI
    state_key = str(node.state.board)
    if state_key in cache:
        value = cache[state_key]
    else:
        _, value = ai_model(encode_board(node.state).unsqueeze(0))
        value = value.item()
        cache[state_key] = value

    # Backpropagate giá trị lên cây
    backpropagate(node, value)


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
        "--": 0,
        "wK": 1,
        "wQ": 2,
        "wR": 3,
        "wB": 4,
        "wN": 5,
        "wP": 6,
        "bK": 7,
        "bQ": 8,
        "bR": 9,
        "bB": 10,
        "bN": 11,
        "bP": 12,
    }
    board = game_state.board
    encoded_board = np.zeros((13, 8, 8), dtype=np.float32)
    for i, row in enumerate(board):
        for j, piece in enumerate(row):
            encoded_board[piece_map[piece], i, j] = 1

    return torch.tensor(encoded_board, dtype=torch.float32, device=device)


def simulate_move(game_state, move):
    """
    Thực hiện một nước đi trên game_state.
    Trả về trạng thái mới sau nước đi và phần thưởng.
    """
    captured_piece = move.piece_captured
    game_state.make_move(move)
    reward = 0
    if game_state.checkmate:
        reward = 1 if game_state.white_to_move else -1
    elif game_state.stalemate or game_state.stalemate_special():
        reward = 0
    elif captured_piece != "--":
        reward += piece_value(captured_piece)

    game_state.undo_move()
    return game_state, reward


def piece_value(piece):
    """Giá trị quân cờ."""
    values = {
        "wP": 1,
        "bP": -1,
        "wN": 3,
        "bN": -3,
        "wB": 3,
        "bB": -3,
        "wR": 5,
        "bR": -5,
        "wQ": 9,
        "bQ": -9,
        "wK": 0,
        "bK": 0,
    }
    return values.get(piece, 0)


def backpropagate(node, value):
    """Truyền giá trị ngược lại từ nút hiện tại lên đến nút gốc."""
    while node is not None:
        node.total_value += value
        node.visit_count += 1

        value = -value
        node = node.parent
