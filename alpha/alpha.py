import h5py
import torch.optim as optim
import time
from torch.utils.data import *
from chess_engine import *
from alpha_mcts import *
from alpha_neuron import *


class ChessDataset(Dataset):
    def __init__(self, h5_file):
        self.boards = []
        self.policies = []
        self.values = []
        self.phases = []
        self.strategies = []
        self.evaluations = []
        self.mobility = []
        with h5py.File(h5_file, "r") as f:
            for game_group in f.values():
                self.boards.extend(game_group.get("boards", []))
                self.policies.extend(game_group.get("policies", []))
                self.values.extend(game_group.get("values", []))
                self.phases.extend(game_group.get("phases", []))
                self.strategies.extend(game_group.get("strategies", []))
                self.evaluations.extend(game_group.get("evaluations", []))
                self.mobility.extend(game_group.get("mobility", []))

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.boards[idx], dtype=torch.float32),
            torch.tensor(self.policies[idx], dtype=torch.float32),
            torch.tensor(self.values[idx], dtype=torch.float32),
            torch.tensor(self.phases[idx], dtype=torch.long),
            torch.tensor(self.strategies[idx], dtype=torch.long),
            torch.tensor(self.evaluations[idx], dtype=torch.float32),
            torch.tensor(self.mobility[idx], dtype=torch.float32),
        )


'''def append_to_h5(output_file, data):
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
        game_group.create_dataset(
            "strategies", data=np.array(data["strategies"], dtype=np.int32)
        )
        game_group.create_dataset(
            "evaluations", data=np.array(data["evaluations"], dtype=np.float32)
        )
        game_group.create_dataset(
            "mobility", data=np.array(data["mobility"], dtype=np.float32)
        )'''


def append_to_h5(output_file, data):
    """Ghi thêm dữ liệu mới vào file .h5."""
    with h5py.File(output_file, "a") as f:
        game_key = f"game_{data['game_number']}"
        if game_key in f:
            print(f"Game {data['game_number']} đã tồn tại. Bỏ qua.")
            return
        game_group = f.create_group(game_key)
        for key, value in data.items():
            if key != "game_number":
                game_group.create_dataset(key, data=np.array(value, dtype=np.float32))


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
    all_game_values = []

    for game_idx in range(num_games):
        print(f"Game {game_idx + 1}/{num_games}")
        game_state = GameState()
        game_boards = []
        game_policies = []
        game_values = []
        phase_labels = []
        strategy_labels = []
        evaluation_values = []
        mobility_values = []
        epsilon = max(0.1 - ((game_idx + 1) / (num_games * 10)), 0)

        while (
            not game_state.checkmate
            and not game_state.stalemate
            and not game_state.stalemate_special()
        ):
            valid_moves = game_state.get_valid_moves()
            if not valid_moves:
                break

            root = MCTSNode(game_state)
            for _ in range(mcts_iterations):
                mcts_rollout(ai_model, root)

            policy = np.zeros(4672)
            for move_key, child in root.children.items():
                idx = move_to_index(
                    Move(
                        (move_key[0], move_key[1]),
                        (move_key[2], move_key[3]),
                        game_state.board,
                    )
                )
                policy[idx] = child.visit_count
            policy /= policy.sum()

            # Lấy đầu ra từ mô hình AI
            board_tensor = encode_board(game_state).unsqueeze(0)
            piece_types = get_piece_types(game_state).unsqueeze(0)
            (_, _, _, phase, strategy, evaluation, mobility) = ai_model(
                board_tensor, piece_types
            )

            # Ghi nhận thông tin đầu ra
            phase_label = torch.argmax(phase, dim=1).item()
            strategy_label = torch.argmax(strategy, dim=1).item()
            evaluation_value = evaluation.item()
            mobility_value = mobility.item()

            # Lưu trạng thái
            phase_labels.append(phase_label)
            strategy_labels.append(strategy_label)
            evaluation_values.append(evaluation_value)
            mobility_values.append(mobility_value)
            game_boards.append(encode_board(game_state).cpu().numpy())
            game_policies.append(policy)
            game_values.append(1 if game_state.white_to_move else -1)

            # Tìm nước đi
            move = select_move(ai_model, game_state, mcts_iterations, epsilon)
            if not move:
                print("Không có nước đi hợp lệ. Kết thúc ván cờ.")
                break
            game_state.make_move(move)

        final_value = 1 if game_state.checkmate else 0
        game_values = [final_value * value for value in game_values]
        all_game_values.append(final_value)

        # Ghi dữ liệu vào file
        append_to_h5(
            output_file,
            {
                "game_number": game_idx,
                "boards": game_boards,
                "policies": game_policies,
                "values": game_values,
                "phases": phase_labels,
                "strategies": strategy_labels,
                "evaluations": evaluation_values,
                "mobility": mobility_values,
            },
        )

    self_play_summary(all_game_values)


def self_play_summary(all_values):
    """Tóm tắt kết quả self-play."""
    wins = sum(1 for v in all_values if v > 0)
    losses = sum(1 for v in all_values if v < 0)
    draws = sum(1 for v in all_values if v == 0)
    print(f"Summary: {wins} Wins, {losses} Losses, {draws} Draws")


def train(
    model, train_dataset, val_dataset, epochs=100, batch_size=256, lr=1e-5, patience=5
):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    phase_loss_fn = nn.CrossEntropyLoss()
    strategy_loss_fn = nn.CrossEntropyLoss()
    evaluation_loss_fn = nn.MSELoss()
    mobility_loss_fn = nn.MSELoss()
    accumulation_steps = 4

    model.train()
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        total_train_loss = 0
        total_val_loss = 0

        # Training phase
        for i, (
            boards,
            policies,
            values,
            phases,
            strategies,
            evaluations,
            mobility,
        ) in enumerate(train_dataloader):
            boards = boards.to(device)
            policies = policies.to(device)
            values = values.to(device)
            phases = phases.to(device)
            strategies = strategies.to(device)
            evaluations = evaluations.to(device)
            mobility = mobility.to(device)

            try:
                (
                    pred_policies,
                    pred_values,
                    _,
                    pred_phases,
                    pred_strategies,
                    pred_evaluations,
                    pred_mobility,
                ) = model(boards)

                loss_policy = policy_loss_fn(
                    pred_policies, torch.argmax(policies, dim=1)
                )
                loss_value = value_loss_fn(pred_values.squeeze(), values)
                loss_phase = phase_loss_fn(pred_phases, phases)
                loss_strategy = strategy_loss_fn(pred_strategies, strategies)
                loss_evaluation = evaluation_loss_fn(
                    pred_evaluations.squeeze(), evaluations
                )
                loss_mobility = mobility_loss_fn(pred_mobility.squeeze(), mobility)

                loss = (
                    loss_policy
                    + loss_value
                    + loss_phase
                    + loss_strategy
                    + loss_evaluation
                    + loss_mobility
                ) / accumulation_steps
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
            for (
                boards,
                policies,
                values,
                phases,
                strategies,
                evaluations,
                mobility,
            ) in val_dataloader:
                boards = boards.to(device)
                policies = policies.to(device)
                values = values.to(device)
                phases = phases.to(device)
                strategies = strategies.to(device)
                evaluations = evaluations.to(device)
                mobility = mobility.to(device)

                try:
                    (
                        pred_policies,
                        pred_values,
                        _,
                        pred_phases,
                        pred_strategies,
                        pred_evaluations,
                        pred_mobility,
                    ) = model(boards)
                    loss_policy = policy_loss_fn(
                        pred_policies, torch.argmax(policies, dim=1)
                    )
                    loss_value = value_loss_fn(pred_values.squeeze(), values)
                    loss_phase = phase_loss_fn(pred_phases, phases)
                    loss_strategy = strategy_loss_fn(pred_strategies, strategies)
                    loss_evaluation = evaluation_loss_fn(
                        pred_evaluations.squeeze(), evaluations
                    )
                    loss_mobility = mobility_loss_fn(pred_mobility.squeeze(), mobility)

                    total_val_loss += (
                        loss_policy.item()
                        + loss_value.item()
                        + loss_phase.item()
                        + loss_strategy.item()
                        + loss_evaluation.item()
                        + loss_mobility.item()
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
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
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
