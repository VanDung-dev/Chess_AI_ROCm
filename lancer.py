import torch
import random
import time
import h5py
import torch_directml
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from chess_engine import GameState

# Cấu hình GPU qua DirectML
device = torch_directml.device(0)
model_path = "ai_model.pth"
data_path = "ai_data.h5"

# === 1. Định nghĩa lớp mạng thần kinh ===
class ChessNet(nn.Module):
    """Mạng nơ-ron với Attention cho cờ vua."""
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.bn1 = nn.LayerNorm(128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)
        self.gelu = nn.GELU()

    def forward(self, x):
        # Layer đầu tiên
        x = self.gelu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        # Attention
        attn_output, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = attn_output.squeeze(1) + x  # Residual connection

        # Layer tiếp theo
        x = self.gelu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

# === 2. Hàm mã hóa bàn cờ và quân cờ ===
def encode_board(game_state):
    """Mã hóa bàn cờ thành tensor với trọng số cải tiến."""
    board = game_state.board
    position_weight = [
        [0.5, 1, 1.5, 2, 2, 1.5, 1, 0.5],
        [1, 2, 2.5, 3, 3, 2.5, 2, 1],
        [1.5, 2.5, 3.5, 4, 4, 3.5, 2.5, 1.5],
        [2, 3, 4, 5, 5, 4, 3, 2],
        [2, 3, 4, 5, 5, 4, 3, 2],
        [1.5, 2.5, 3.5, 4, 4, 3.5, 2.5, 1.5],
        [1, 2, 2.5, 3, 3, 2.5, 2, 1],
        [0.5, 1, 1.5, 2, 2, 1.5, 1, 0.5],
    ]
    encoded = [
        position_weight[i][j] * piece_encoding(piece)
        for i, row in enumerate(board) for j, piece in enumerate(row)
    ]
    return torch.tensor(encoded, dtype=torch.float32, device=device)

def piece_encoding(piece):
    """Mã hóa quân cờ."""
    mapping = {
        "--": 0, "wK": 10, "wQ": 8, "wN": 6, "wR": 5, "wB": 3, "wP": 1,
        "bK": -10, "bQ": -8, "bN": -6, "bR": -5, "bB": -3, "bP": -1
    }
    return mapping.get(piece, 0)

# === 3. Chọn nước đi ===
def select_move(ai_model, game_state, valid_moves, epsilon=0.1):
    """Chọn nước đi sử dụng epsilon-greedy."""
    if random.random() < epsilon:
        return random.choice(valid_moves)
    scores = [
        ai_model(encode_board(simulate_move(game_state, move)).unsqueeze(0)).item()
        for move in valid_moves
    ]
    return valid_moves[scores.index(max(scores))]

def simulate_move(game_state, move):
    """Mô phỏng nước đi."""
    new_state = GameState()
    new_state.board = [row[:] for row in game_state.board]
    new_state.make_move(move)
    return new_state

# === 4. Quá trình tự chơi và huấn luyện ===
def self_play(current_ai, num_games=100, save_path=data_path):
    """AI tự chơi và lưu dữ liệu mới vào file HDF5, thay thế dữ liệu cũ."""

    # Kiểm tra và tải mô hình từ file .pth
    load_model(current_ai)

    # Ghi đè dữ liệu cũ bằng dữ liệu mới
    with h5py.File(save_path, "w") as h5file:
        for game_num in range(num_games):
            print(f"Đang chơi ván thứ {game_num + 1}/{num_games}...")
            game_state = GameState()
            states, actions, rewards = [], [], []

            while not game_state.checkmate and not game_state.stalemate and not game_state.stalemate_special():
                valid_moves = game_state.get_valid_moves()
                if not valid_moves:
                    break

                move = select_move(current_ai, game_state, valid_moves)
                if move is None:
                    break

                # Mã hóa trạng thái bàn cờ
                state_tensor = encode_board(game_state)
                if state_tensor.shape != (64,):
                    raise ValueError(f"self_play: Invalid state tensor shape: {state_tensor.shape}")
                states.append(state_tensor.tolist())  # Lưu trạng thái

                # Lưu nước đi
                actions.append(str(move))
                game_state.make_move(move)
                rewards.append(0 if not game_state.checkmate else 1 if game_state.white_to_move else -1)

            if len(states) == 0 or len(rewards) == 0:
                raise ValueError(f"self_play: No valid data for game {game_num + 1}")

            # Lưu dữ liệu vào HDF5
            group = h5file.create_group(f"game_{game_num}")
            group.create_dataset("states", data=np.array(states, dtype=np.float32))
            group.create_dataset("actions", data=np.array(actions, dtype=h5py.string_dtype()))
            group.create_dataset("rewards", data=np.array(rewards, dtype=np.float32))

    print(f"Đã lưu dữ liệu tự chơi vào {save_path}.")

def train_with(ai_model, optimizer, loss_fn, h5_file=data_path, epochs=10, batch_size=16, gamma=0.99):
    """Huấn luyện mô hình bằng dữ liệu trong file HDF5."""
    print(f"Đang đọc dữ liệu từ {h5_file} để huấn luyện...")
    states, targets = [], []

    # Đọc dữ liệu từ file HDF5
    with h5py.File(h5_file, "r") as h5file:
        for game_key in h5file.keys():
            print(f"  -> Đang xử lý {game_key}...")
            game_states = torch.tensor(h5file[game_key]["states"][:], dtype=torch.float32, device=device)
            game_rewards = torch.tensor(h5file[game_key]["rewards"][:], dtype=torch.float32, device=device)

            for t in range(len(game_states)):
                state = game_states[t]
                reward = game_rewards[t]
                if t < len(game_states) - 1:
                    next_state = game_states[t + 1]
                    target = reward + gamma * ai_model(next_state.unsqueeze(0)).item()
                else:
                    target = reward
                states.append(state)
                targets.append(target)

    # Chuyển đổi dữ liệu thành tensor
    if not states or not targets:
        raise ValueError("Không có dữ liệu để huấn luyện!")
    states_tensor = torch.stack(states)
    targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(1)

    dataset = TensorDataset(states_tensor, targets_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình...")
    ai_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_targets in data_loader:
            optimizer.zero_grad()
            predictions = ai_model(batch_states)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")
    print("Huấn luyện hoàn tất!")

# === 5. Lưu và tải dữ liệu/mô hình ===
def save_model(ai_model, path=model_path):
    torch.save(ai_model.state_dict(), path)
    print(f"Đã lưu mô hình tại {path}.")

def load_model(ai_model, path=model_path):
    try:
        ai_model.load_state_dict(torch.load(path, map_location=device))
        print(f"Đã tải mô hình từ {path}.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy {path}.")

# === 6. Chương trình chính ===
if __name__ == "__main__":
    ai_model = ChessNet().to(device)
    optimizer = optim.AdamW(ai_model.parameters(), lr=1e-4)
    loss_fn = nn.SmoothL1Loss()

    start_time = time.time()
    print(f"DirectML Detected: {torch_directml.device_name(0)}")
    try:
        choice = input("1. Tự chơi để lấy dữ liệu\n"
                       "2. Huấn luyện từ dữ liệu đã lấy được\n"
                       "Chọn: ").strip()
        if choice == "1":
            num = int(input("Nhập số game: "))
            self_play(ai_model, num_games=num, save_path=data_path)
        elif choice == "2":
            train_with(ai_model, optimizer, loss_fn, h5_file=data_path, epochs=10)
            save_model(ai_model)

    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")
    end_time = time.time()
    print(f"Thời gian hoàn tất: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
