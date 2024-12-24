import torch
import torch_directml
import numpy as np
from chess_engine import GameState
import copy

device = torch_directml.device(0)
game_state = GameState()


class MCTSNode:
    def __init__(self, state, parent=None, move_key=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior = 0
        self.move_key = move_key

    def is_leaf(self):
        """Kiểm tra nếu nút là nút lá."""
        return len(self.children) == 0

    def is_fully_expanded(self):
        """Kiểm tra nếu nút đã mở rộng hoàn toàn."""
        valid_moves = self.state.get_valid_moves()
        return valid_moves and len(self.children) == len(valid_moves)

    def best_child(self, exploration_weight=1):
        if not self.children:
            raise ValueError("No children nodes available to select the best child.")

        weights = {
            move_key: (child.total_value / (child.visit_count + 1e-6))
            + exploration_weight
            * child.prior
            * (self.visit_count**0.5 / (1 + child.visit_count))
            for move_key, child in self.children.items()
        }

        # Ưu tiên hòa nếu không có nước đi thắng
        max_value = max(weights.values())
        if max_value <= 0:
            # Tìm nước đi dẫn đến hòa
            for move_key, child in self.children.items():
                if child.total_value / (child.visit_count + 1e-6) == 0:
                    return child

        # Nếu có nước đi thắng, chọn nước đi tốt nhất
        best_move_key = max(weights, key=weights.get)
        return self.children[best_move_key]


def mcts_rollout(ai_model, root, cache=None):
    if cache is None:
        cache = {}

    node = root

    # Traversing the tree
    while (
        not node.state.checkmate
        and not node.state.stalemate
        and not node.state.stalemate_special()  # Hỗ trợ phát hiện hòa đặc biệt
        and node.is_fully_expanded()
    ):
        if not node.children:
            break
        try:
            node = node.best_child()
        except ValueError as e:
            print(f"Warning: {e}")
            break

    # Expansion phase
    if (
        not node.state.checkmate
        and not node.state.stalemate
        and not node.state.stalemate_special()  # Điều kiện hòa đặc biệt
    ):
        valid_moves = node.state.get_valid_moves()
        new_states, new_nodes = [], []

        for move in valid_moves:
            move_key = (
                move.start_row,
                move.start_column,
                move.end_row,
                move.end_column,
            )

            if move_key not in node.children:
                # Generate new state
                new_state = copy.deepcopy(node.state)
                new_state.make_move(move)
                new_node = MCTSNode(new_state, parent=node, move_key=move_key)

                new_states.append(new_state)
                new_nodes.append(new_node)
                node.children[move_key] = new_node

        # Batch inference for new states
        if new_states:
            board_tensors = torch.stack([encode_board(state) for state in new_states])
            piece_types_batch = torch.stack(
                [get_piece_types(state) for state in new_states]
            )

            # Model inference
            policies, values, *_ = ai_model(board_tensors, piece_types_batch)

            for i, new_node in enumerate(new_nodes):
                move_index = move_to_index(valid_moves[i])
                new_node.prior = policies[i][move_index].item()
                cache[str(new_states[i].board)] = values[i].item()

    # Evaluation phase
    state_key = str(node.state.board)
    if state_key in cache:
        value = cache[state_key]
    else:
        board_tensor = encode_board(node.state).unsqueeze(0)
        piece_types = get_piece_types(node.state).unsqueeze(0)

        # Get the value from the model
        _, value, *_ = ai_model(board_tensor, piece_types)
        value = value.item()
        cache[state_key] = value

    # Special case: Handle draw scenarios
    if node.state.stalemate or node.state.stalemate_special():
        value = 0  # Assign neutral value for draws

    # Backpropagation phase
    backpropagate(node, value)


def get_piece_types(game_state):
    """
    Trả về tensor biểu diễn loại quân cờ trên bàn.
    """
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
    piece_types = np.zeros((8, 8), dtype=np.int64)
    for i, row in enumerate(board):
        for j, piece in enumerate(row):
            piece_types[i, j] = piece_map[piece]
    return torch.tensor(piece_types, dtype=torch.long, device=device)


def adjust_priority_based_on_phase(priority, phase, game_state):
    opening_weight, middlegame_weight, endgame_weight = phase.squeeze().tolist()

    if opening_weight > 0.5:
        return priority * (1.2 + evaluate_center_control(game_state))
    elif middlegame_weight > 0.5:
        return priority * (1.1 + evaluate_piece_synergy(game_state))
    elif endgame_weight > 0.5:
        return priority * (1.3 if has_endgame_advantage(game_state) else 1.0)
    return priority


def evaluate_center_control(game_state):
    """Đánh giá mức độ kiểm soát các ô trung tâm."""
    center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]  # Ô d4, e4, d5, e5
    control_score = 0
    valid_moves = game_state.get_valid_moves()

    for move in valid_moves:
        if (move.end_row, move.end_column) in center_squares:
            control_score += 1  # Tăng điểm nếu kiểm soát ô trung tâm

    return control_score / len(valid_moves) if valid_moves else 0  # Chuẩn hóa


def evaluate_piece_synergy(game_state):
    """Đánh giá sự phối hợp giữa các quân cờ."""
    synergy_score = 0
    valid_moves = game_state.get_valid_moves()
    attacked_squares = calculate_attacked_squares(game_state)

    for move in valid_moves:
        end_row, end_col = move.end_row, move.end_column
        synergy_score += attacked_squares[end_row, end_col]  # Số quân nhắm cùng một ô

    return synergy_score / len(valid_moves) if valid_moves else 0  # Chuẩn hóa


def has_endgame_advantage(game_state):
    """Kiểm tra ưu thế trong Endgame."""
    white_pieces = sum(
        1 for row in game_state.board for piece in row if piece.startswith("w")
    )
    black_pieces = sum(
        1 for row in game_state.board for piece in row if piece.startswith("b")
    )

    # Kiểm tra số lượng tốt
    white_pawns = sum(1 for row in game_state.board for piece in row if piece == "wP")
    black_pawns = sum(1 for row in game_state.board for piece in row if piece == "bP")

    # Ưu thế nếu quân số nhiều hơn hoặc có nhiều Tốt hơn
    return (
        (white_pieces > black_pieces and white_pawns >= black_pawns)
        if game_state.white_to_move
        else (black_pieces > white_pieces and black_pawns >= white_pawns)
    )


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
    encoded_board = np.zeros((25, 8, 8), dtype=np.float32)  # Tăng số kênh lên 25

    # Loại quân cờ
    for i, row in enumerate(board):
        for j, piece in enumerate(row):
            encoded_board[piece_map[piece], i, j] = 1

    # Giá trị quân cờ
    value_map = {
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
        "--": 0,
    }
    for i, row in enumerate(board):
        for j, piece in enumerate(row):
            encoded_board[17, i, j] = value_map.get(piece, 0)

    # Ô được bảo vệ, bị tấn công
    encoded_board[18, :, :] = calculate_protected_squares(game_state)
    encoded_board[19, :, :] = calculate_attacked_squares(game_state)

    # Kiểm soát trung tâm
    encoded_board[20, :, :] = calculate_center_control(game_state)

    # Phối hợp quân cờ
    encoded_board[21, :, :] = calculate_piece_synergy(game_state)

    # Mã hóa di động của quân cờ
    encoded_board[22, :, :] = calculate_piece_mobility(game_state)

    # Vị trí vua an toàn
    encoded_board[23, :, :] = calculate_king_safety(game_state)

    # Đường đi của các quân mạnh
    encoded_board[24, :, :] = calculate_strong_piece_paths(game_state)

    return torch.tensor(encoded_board, dtype=torch.float32, device=device)


def calculate_piece_mobility(game_state):
    """Tính toán tính di động của quân cờ."""
    mobility_map = np.zeros((8, 8), dtype=np.float32)
    valid_moves = game_state.get_valid_moves()

    for move in valid_moves:
        mobility_map[
            move.end_row, move.end_column
        ] += 1  # Cộng điểm cho các ô có thể đi đến

    return mobility_map


def calculate_king_safety(game_state):
    """Tính toán mức độ an toàn của vua."""
    safety_map = np.zeros((8, 8), dtype=np.float32)
    king_position = (
        game_state.white_king_location
        if game_state.white_to_move
        else game_state.black_king_location
    )

    # Tính toán các ô xung quanh vua
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            r, c = king_position[0] + dr, king_position[1] + dc
            if 0 <= r < 8 and 0 <= c < 8:
                safety_map[r, c] += 1  # Cộng điểm bảo vệ

    return safety_map


def calculate_strong_piece_paths(game_state):
    """Tính toán các đường đi của các quân mạnh."""
    strong_pieces = {"wQ", "wR", "wB", "bQ", "bR", "bB"}
    paths_map = np.zeros((8, 8), dtype=np.float32)

    valid_moves = game_state.get_valid_moves()
    for move in valid_moves:
        start_piece = game_state.board[move.start_row][move.start_column]
        if start_piece in strong_pieces:
            paths_map[
                move.end_row, move.end_column
            ] += 1  # Cộng điểm cho đường đi của quân mạnh

    return paths_map


def calculate_threat_score(state, threats, synergy_map):
    """Dự đoán điểm đe dọa dựa trên output mạng và phối hợp quân cờ."""
    if threats.dim() == 4 and threats.shape[2:] == (8, 8):
        threat_map = threats.squeeze(0).squeeze(0).detach().cpu().numpy()
    else:
        raise ValueError(
            f"Unexpected threats shape: {threats.shape}, expected [batch_size, 1, 8, 8]"
        )

    king_position = (
        state.white_king_location if state.white_to_move else state.black_king_location
    )
    threat_score = threat_map[king_position[0], king_position[1]]
    synergy_score = synergy_map[king_position[0], king_position[1]]  # Tính phối hợp
    return threat_score + 0.5 * synergy_score  # Ưu tiên phối hợp


def calculate_protected_squares(game_state):
    """Tính toán các ô được bảo vệ bởi quân cờ của người chơi hiện tại."""
    protected_squares = np.zeros((8, 8), dtype=np.float32)
    valid_moves = game_state.get_valid_moves()
    for move in valid_moves:
        end_row, end_col = move.end_row, move.end_column
        protected_squares[end_row, end_col] += 1
    return protected_squares


def calculate_attacked_squares(game_state):
    """Tính toán các ô bị tấn công bởi quân cờ của đối thủ."""
    attacked_squares = np.zeros((8, 8), dtype=np.float32)
    game_state.white_to_move = not game_state.white_to_move
    valid_moves = game_state.get_valid_moves()
    for move in valid_moves:
        end_row, end_col = move.end_row, move.end_column
        attacked_squares[end_row, end_col] += 1
    game_state.white_to_move = not game_state.white_to_move
    return attacked_squares


def calculate_center_control(game_state):
    """Tính toán kiểm soát trung tâm."""
    center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]  # Ô d4, e4, d5, e5
    center_control = np.zeros((8, 8), dtype=np.float32)

    valid_moves = game_state.get_valid_moves()
    for move in valid_moves:
        if (move.end_row, move.end_column) in center_squares:
            center_control[move.end_row, move.end_column] += 1

    return center_control


def calculate_piece_synergy(game_state):
    """Tính toán phối hợp giữa các quân cờ."""
    synergy_map = np.zeros((8, 8), dtype=np.float32)
    valid_moves = game_state.get_valid_moves()
    for move in valid_moves:
        synergy_map[
            move.end_row, move.end_column
        ] += 1  # Tăng nếu nhiều quân nhắm cùng một ô
    return synergy_map


def move_to_index(move):
    """Chuyển nước đi thành chỉ số duy nhất."""
    start_idx = move.start_row * 8 + move.start_column
    end_idx = move.end_row * 8 + move.end_column
    return start_idx * 64 + end_idx


def backpropagate(node, value):
    """Truyền giá trị ngược lại từ nút hiện tại lên đến nút gốc."""
    while node is not None:
        node.total_value += value
        node.visit_count += 1
        value = -value
        node = node.parent


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
