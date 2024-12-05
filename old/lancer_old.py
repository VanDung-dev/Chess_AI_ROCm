import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
import random
from chess_engine import GameState

device = torch_directml.device()

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

def encode_board(game_state):
    """Mã hóa bàn cờ thành tensor."""
    board = game_state.board
    encoded = []
    for row in board:
        for piece in row:
            encoded.append(piece_encoding(piece))
    return torch.tensor(encoded, dtype=torch.float32, device=device)

def piece_encoding(piece):
    """Mã hóa quân cờ."""
    mapping = {
        "--": 0, "wP": 1, "wR": 2, "wN": 3, "wB": 4, "wQ": 5, "wK": 6,
        "bP": -1, "bR": -2, "bN": -3, "bB": -4, "bQ": -5, "bK": -6
    }
    return mapping.get(piece, 0)

def select_move(ai_model, game_state, valid_moves, epsilon=0.1):
    """Chọn nước đi sử dụng epsilon-greedy."""
    if random.random() < epsilon:
        return random.choice(valid_moves)
    else:
        scores = []
        for move in valid_moves:
            next_state = simulate_move(game_state, move)
            next_state_tensor = encode_board(next_state).unsqueeze(0)
            scores.append(ai_model(next_state_tensor).item())
        best_move_idx = scores.index(max(scores))
        return valid_moves[best_move_idx]

def simulate_move(game_state, move):
    """Tạo trạng thái mới sau khi thực hiện nước đi."""
    new_state = GameState()
    new_state.board = [row[:] for row in game_state.board]
    new_state.make_move(move)
    return new_state

def calculate_reward(game_state):
    """Tính điểm thưởng/phạt."""
    if game_state.checkmate:
        return 1 if game_state.white_to_move else -1
    if game_state.stalemate:
        return 0
    return 0

def self_play(ai_white, ai_black, num_games=10):
    """Tự chơi giữa AI trắng và AI đen."""
    all_games = []
    for game_num in range(num_games):
        game_state = GameState()
        states, actions, rewards = [], [], []
        while not game_state.checkmate and not game_state.stalemate:
            current_ai = ai_white if game_state.white_to_move else ai_black
            valid_moves = game_state.get_valid_moves()
            move = select_move(current_ai, game_state, valid_moves, epsilon=0.1)
            states.append(encode_board(game_state))
            actions.append(move)
            game_state.make_move(move)
            rewards.append(calculate_reward(game_state))
        all_games.append((states, actions, rewards))
        print(f"Game {game_num + 1}/{num_games} completed.")
    return all_games

def train_model(ai_model, optimizer, loss_fn, all_games, gamma=0.99):
    """Huấn luyện mạng thần kinh từ các trận đấu."""
    for states, actions, rewards in all_games:
        for t in range(len(states)):
            state = states[t].unsqueeze(0)
            reward = rewards[t]
            if t < len(states) - 1:
                next_state = states[t + 1].unsqueeze(0)
                target = reward + gamma * ai_model(next_state).max().item()
            else:
                target = reward
            prediction = ai_model(state)
            loss = loss_fn(prediction, torch.tensor([target], device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def save_model(ai_model, path):
    """Lưu mô hình AI."""
    torch.save(ai_model.state_dict(), path)

def load_model(ai_model, path):
    """Tải mô hình AI."""
    ai_model.load_state_dict(torch.load(path, map_location=device))

if __name__ == "__main__":
    ai_white = ChessNet().to(device)
    ai_black = ChessNet().to(device)
    optimizer_white = optim.Adam(ai_white.parameters(), lr=1e-4)
    optimizer_black = optim.Adam(ai_black.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    all_games = self_play(ai_white, ai_black, num_games=10)
    train_model(ai_white, optimizer_white, loss_fn, all_games)
    save_model(ai_white, f"ai_chess.pth")
