import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from chess_engine import GameState, Move
from chess_mcts import MCTSNode, mcts_rollout, encode_board, move_to_index, device
from chess_neuron import ChessNet


class ChessDataset(Dataset):
    def __init__(self, h5_file):
        with h5py.File(h5_file, "r") as f:
            self.boards = []
            self.policies = []
            self.values = []
            self.phases = []
            for game_group in f.values():
                self.boards.extend(game_group["boards"][:])
                self.policies.extend(game_group["policies"][:])
                self.values.extend(game_group["values"][:])
                self.phases.extend(game_group["phases"][:])

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        policy = self.policies[idx]
        value = self.values[idx]
        phase = self.phases[idx]
        return (
            torch.tensor(board, dtype=torch.float32),
            torch.tensor(policy, dtype=torch.float32),
            torch.tensor(value, dtype=torch.float32),
            torch.tensor(phase, dtype=torch.long),
        )


def append_to_h5(output_file, data):
    """Ghi thêm dữ liệu mới vào file .h5"""
    with h5py.File(output_file, "a") as f:
        if f"game_{data['game_number']}" in f:
            print(f"Game {data['game_number']} đã tồn tại. Bỏ qua.")
            return
        game_group = f.create_group(f"game_{data['game_number']}")
        game_group.create_dataset(
            "boards", data=np.array(data["boards"], dtype=np.float32)
        )
        game_group.create_dataset(
            "policies", data=np.array(data["policies"], dtype=np.float32)
        )
        game_group.create_dataset(
            "values", data=np.array(data["values"], dtype=np.float32)
        )
        game_group.create_dataset(
            "phases", data=np.array(data["phases"], dtype=np.int32)
        )


def select_move(ai_model, game_state, mcts_iterations=50, epsilon=0.1):
    if np.random.rand() < epsilon:
        valid_moves = game_state.get_valid_moves()
        return np.random.choice(valid_moves)

    root = MCTSNode(game_state)
    for _ in range(mcts_iterations):
        mcts_rollout(ai_model, root)

    best_child = root.best_child(exploration_weight=0)
    for move in game_state.get_valid_moves():
        if (
            move.start_row,
            move.start_column,
            move.end_row,
            move.end_column,
        ) == best_child.move_key:
            return move

    valid_moves = game_state.get_valid_moves()
    return valid_moves[0] if valid_moves else None


def self_play(ai_model, num_games, output_file, mcts_iterations=50):
    all_game_values = []  # Danh sách lưu kết quả từng ván đấu

    for game_idx in range(num_games):
        print(f"Game {game_idx + 1}/{num_games}")
        game_state = GameState()
        game_boards = []
        game_policies = []
        game_values = []
        phase_labels = []  # Lưu nhãn giai đoạn trận đấu
        start_time = time.time()
        num = 1
        epsilon = 0.1 - ((game_idx + 1) / (num_games * 10))

        while (
            not game_state.checkmate
            and not game_state.stalemate
            and not game_state.stalemate_special()
        ):
            valid_moves = game_state.get_valid_moves()
            if not valid_moves:
                break

            # Lưu trạng thái hiện tại
            root = MCTSNode(game_state)
            for _ in range(mcts_iterations):
                mcts_rollout(ai_model, root)

            policy = np.zeros(4672)
            for move_key, child in root.children.items():
                start_square = (move_key[0], move_key[1])
                end_square = (move_key[2], move_key[3])
                idx = move_to_index(Move(start_square, end_square, game_state.board))
                policy[idx] = child.visit_count
            policy /= policy.sum()

            # Lấy phase từ mô hình AI
            _, _, _, phase = ai_model(encode_board(game_state).unsqueeze(0))
            phase_np = phase.detach().cpu().numpy().squeeze()

            # Mã hóa giai đoạn trận đấu
            if phase_np[0] > phase_np[1] and phase_np[0] > phase_np[2]:
                phase_label = 0  # Opening
            elif phase_np[1] > phase_np[0] and phase_np[1] > phase_np[2]:
                phase_label = 1  # Middlegame
            else:
                phase_label = 2  # Endgame
            phase_labels.append(phase_label)

            game_boards.append(encode_board(game_state).cpu().numpy())
            game_policies.append(policy)
            game_values.append(1 if game_state.white_to_move else -1)

            # Thực hiện nước đi
            move = select_move(
                ai_model, game_state, mcts_iterations=mcts_iterations, epsilon=epsilon
            )
            game_state.make_move(move)
            print(f"{num}: {move}")
            num += 1

        # Xác định giá trị kết thúc
        final_value = 1 if game_state.checkmate else 0
        game_values = [final_value * value for value in game_values]
        all_game_values.append(final_value)  # Lưu kết quả của ván đấu

        # Save each game's data to the H5 file immediately after the game
        append_to_h5(
            output_file,
            {
                "game_number": game_idx,
                "boards": np.array(game_boards),
                "policies": np.array(game_policies),
                "values": np.array(game_values),
                "phases": np.array(phase_labels),  # Lưu phase labels
            },
        )
        end_time = time.time()
        print(
            f"Thời gian ván {game_idx + 1}: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}"
        )

    # Tóm tắt kết quả sau tất cả các ván đấu
    self_play_summary(all_game_values)


def self_play_summary(all_values):
    """Tóm tắt kết quả self-play."""
    wins = sum(1 for v in all_values if v > 0)
    losses = sum(1 for v in all_values if v < 0)
    draws = sum(1 for v in all_values if v == 0)
    print(f"Summary: {wins} Wins, {losses} Losses, {draws} Draws")


def train(model, train_dataset, val_dataset, batch_size=256, lr=1e-5, patience=5):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    phase_loss_fn = nn.CrossEntropyLoss()
    accumulation_steps = 4

    model.train()
    best_loss = float("inf")
    epochs_no_improve = 0
    epoch = 0

    while epochs_no_improve < patience:
        total_train_loss = 0
        total_val_loss = 0

        # Training phase
        for i, (boards, policies, values, phases) in enumerate(train_dataloader):
            boards = boards.to(device)
            policies = policies.to(device)
            values = values.to(device)
            phases = phases.to(device)  # Chuyển phases sang GPU

            try:
                pred_policies, pred_values, _, pred_phases = model(boards)

                # Kiểm tra NaN trong đầu ra mô hình
                if (
                    torch.isnan(pred_policies).any()
                    or torch.isnan(pred_values).any()
                    or torch.isnan(pred_phases).any()
                ):
                    print(f"NaN detected in predictions at batch {i}")
                    continue

                loss_policy = policy_loss_fn(
                    pred_policies, torch.argmax(policies, dim=1)
                )
                loss_value = value_loss_fn(pred_values.squeeze(), values)
                loss_phase = phase_loss_fn(pred_phases, phases)

                loss = (loss_policy + loss_value + loss_phase) / accumulation_steps
                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_train_loss += loss.item()
            except Exception as e:
                print(f"Error during training batch {i}: {e}")

        # Validation phase
        model.eval()
        with torch.no_grad():
            for boards, policies, values, phases in val_dataloader:
                boards = boards.to(device)
                policies = policies.to(device)
                values = values.to(device)
                phases = phases.to(device)

                try:
                    pred_policies, pred_values, _, pred_phases = model(boards)
                    loss_policy = policy_loss_fn(
                        pred_policies, torch.argmax(policies, dim=1)
                    )
                    loss_value = value_loss_fn(pred_values.squeeze(), values)
                    loss_phase = phase_loss_fn(pred_phases, phases)

                    total_val_loss += (
                        loss_policy.item() + loss_value.item() + loss_phase.item()
                    )
                except Exception as e:
                    print(f"Error during validation: {e}")

        avg_val_loss = total_val_loss / len(val_dataloader)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_chess_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_loss:.4f}"
                )
                break

        print(
            f"Epoch {epoch + 1}, Train Loss: {total_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )
        model.train()


if __name__ == "__main__":
    model = ChessNet().to(device)

    while True:
        print("\nChoose an option:")
        print("1. Self-play and generate data")
        print("2. Train model")
        print("3. Exit")

        choice = input("Enter your choice: ")
        if choice == "1":
            num_games = int(input("Enter number of self-play games: "))

            output_file = (
                input("Enter H5 file path (default: self_play_data.h5): ")
                or "self_play_data.h5"
            )
            model.eval()
            start_time = time.time()
            self_play(model, num_games, output_file)
            end_time = time.time()
            print(
                f"Thời gian hoàn tất: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}"
            )
        elif choice == "2":
            h5_file = (
                input("Enter H5 file path for training (default: self_play_data.h5): ")
                or "self_play_data.h5"
            )
            dataset = ChessDataset(h5_file)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            train(model, train_dataset, val_dataset)
            torch.save(model.state_dict(), "trained_chess_model.pth")
            print("Model saved to trained_chess_model.pth")
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")
