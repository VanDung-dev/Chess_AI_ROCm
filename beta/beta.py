import time
import h5py
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from chess_neuron import *
from chess_mcts import *
from chess_engine import *

model_path = "ai_model_beta.pth"
data_path = "ai_data_beta.h5"


def calculate_reward(move, game_state):
    """Tính điểm thưởng dựa trên nước đi và trạng thái bàn cờ."""
    reward = 0
    reward += piece_value(move.piece_captured)

    if move.is_castle_move:
        reward += 2 if game_state.white_to_move else -2
    if move.is_check:
        reward += 2 if game_state.white_to_move else -2
    if move.is_pawn_promotion:
        promoted_piece_value = piece_value(
            game_state.board[move.end_row][move.end_column]
        )
        reward += (
            promoted_piece_value if game_state.white_to_move else -promoted_piece_value
        )

    return reward


def select_move(ai_model, game_state, mcts_iterations=10, epsilon=0.1):
    """Chọn nước đi tốt nhất hoặc ngẫu nhiên với xác suất epsilon."""
    if np.random.rand() < epsilon:
        valid_moves = game_state.get_valid_moves()
        return np.random.choice(valid_moves)

    root = MCTSNode(game_state)
    for _ in range(mcts_iterations):
        mcts_rollout(ai_model, root)
    best_child = root.best_child(exploration_weight=0)
    for move in game_state.get_valid_moves():
        if (move.start_row, move.start_column, move.end_row, move.end_column) == best_child.move_key:
            return move


def self_play(ai_model, num_games=100, save_path=data_path):
    """AI tự chơi và lưu dữ liệu với trạng thái trước cả nước đi của trắng và đen."""
    with h5py.File(save_path, "w") as h5file:
        for game_num in range(num_games):
            print(f"Đang chơi ván thứ {game_num + 1}/{num_games}...")
            game_state = GameState()
            states, policies, rewards, actions = [], [], [], []
            cumulative_reward = 0

            while (
                not game_state.checkmate
                and not game_state.stalemate
                and not game_state.stalemate_special()
            ):
                valid_moves = game_state.get_valid_moves()
                if not valid_moves:
                    break

                state = encode_board(game_state)
                move = select_move(ai_model, game_state)
                if move not in valid_moves:
                    raise ValueError("Nước đi của AI không hợp lệ!")

                game_state.make_move(move)
                reward = calculate_reward(move, game_state)
                cumulative_reward += reward

                states.append(state.tolist())
                policies.append(np.zeros(4672, dtype=np.float32))
                rewards.append(cumulative_reward)
                actions.append(move.get_chess_notation())

                if (
                    game_state.checkmate
                    or game_state.stalemate
                    or game_state.stalemate_special()
                ):
                    break

            if game_state.checkmate:
                actions.append("#")
            elif game_state.stalemate or game_state.stalemate_special():
                actions.append("1/2 - 1/2")

            group = h5file.create_group(f"game_{game_num}")
            group.create_dataset("states", data=np.array(states, dtype=np.float32))
            group.create_dataset("policies", data=np.array(policies, dtype=np.float32))
            group.create_dataset("rewards", data=np.array(rewards, dtype=np.float32))
            group.create_dataset("actions", data=np.array(actions, dtype="S"))

            print(f"Đã lưu dữ liệu ván {game_num + 1}.")


def train_with(ai_model, optimizer, h5_file=data_path, epochs=10, batch_size=2048):
    """Huấn luyện mô hình bằng dữ liệu trong file HDF5."""
    ai_model.train()
    states, policies, values = [], [], []
    with h5py.File(h5_file, "r") as h5file:
        for game_key in h5file.keys():
            print(f"Đang lấy mẫu từ ván {game_key}")
            game_states = torch.tensor(
                h5file[game_key]["states"][:], dtype=torch.float32
            ).to(device)
            game_policies = torch.tensor(
                h5file[game_key]["policies"][:], dtype=torch.float32
            ).to(device)
            game_values = torch.tensor(
                h5file[game_key]["rewards"][:], dtype=torch.float32
            ).to(device)
            states.extend(game_states)
            policies.extend(game_policies)
            values.extend(game_values)

    dataset = TensorDataset(
        torch.stack(states), torch.stack(policies), torch.tensor(values, device=device)
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Đang học")
    for epoch in range(epochs):
        total_loss, total_policy_loss, total_value_loss = 0, 0, 0
        for batch_states, batch_policies, batch_values in data_loader:
            batch_states, batch_policies, batch_values = (
                batch_states.to(device),
                batch_policies.to(device),
                batch_values.to(device),
            )
            optimizer.zero_grad()
            predicted_policies, predicted_values = ai_model(batch_states)
            policy_loss = torch.nn.functional.cross_entropy(
                predicted_policies, batch_policies
            )
            value_loss = torch.nn.functional.mse_loss(
                predicted_values.squeeze(-1), batch_values
            )
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        epoch_log = (
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}, "
            f"Policy Loss: {total_policy_loss / len(data_loader):.4f}, "
            f"Value Loss: {total_value_loss / len(data_loader):.4f}\n"
        )
        with open(f"training_log.txt", "a") as log_file:
            log_file.write(epoch_log)
        print(epoch_log)


def save_model(ai_model, path=model_path):
    torch.save(ai_model.state_dict(), path)
    print(f"Đã lưu mô hình tại {path}.")


def load_model(ai_model, path=model_path):
    try:
        ai_model.load_state_dict(torch.load(path, map_location=device))
        print(f"Đã tải mô hình từ {path}.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy {path}.")


if __name__ == "__main__":
    ai_model = ChessNet().to(device)
    optimizer = optim.AdamW(ai_model.parameters(), lr=1e-4)
    print(f"DirectML Detected: {device}")
    choice = input(
        "1. Tự chơi để lấy dữ liệu\n" "2. Huấn luyện từ dữ liệu đã lấy được\n" "Chọn: "
    ).strip()
    start_time = time.time()
    if choice == "1":
        num = int(input("Nhập số game: "))
        self_play(ai_model, num_games=num, save_path=data_path)
    elif choice == "2":
        train_with(ai_model, optimizer, h5_file=data_path)
        save_model(ai_model)
    end_time = time.time()
    print(
        f"Thời gian hoàn tất: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}"
    )
