import time
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import chess
import chess.pgn
from modules.chess_neuron import ChessNet
from modules.chess_engine import encode_board, move_to_index
from modules.chess_model import list_and_select_model, load_model, device
from modules.chess_log import setup_logger

logger = setup_logger()


def train_with(ai_model, optimizer, data_dir="data", batch_size=256, tolerance=1e-4, patience=3):
    """
    Huấn luyện mô hình với dữ liệu PGN.

    Args:
        ai_model: Mô hình AI (ChessNet).
        optimizer: Bộ tối ưu hóa (ví dụ: optim.AdamW).
        data_dir (str): Thư mục chứa file PGN (mặc định "data").
        batch_size (int): Kích thước batch (mặc định 256).
        tolerance (float): Ngưỡng dừng sớm (mặc định 1e-4).
        patience (int): Số epoch chờ trước khi dừng sớm (mặc định 3).
    """
    logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    logger.info(f"GPU Memory Cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

    # Kiểm tra backend có sẵn và thử dùng inductor
    try:
        logger.info(f"Các backend có sẵn: {torch._dynamo.list_backends()}")
        ai_model = torch.compile(ai_model, backend="inductor")
        logger.info("Đã áp dụng torch.compile với backend inductor")
    except Exception as e:
        logger.warning(f"Không thể áp dụng torch.compile: {e}")

    ai_model.train()
    dataset = ChessDataset(data_dir)
    if len(dataset) == 0:
        logger.error("Lỗi: Không tìm thấy ván cờ hoặc nước đi hợp lệ trong dữ liệu.")
        return
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    logger.info(f"DataLoader được tạo với batch_size={batch_size}, kích thước dataset={len(dataset)}")

    for param in ai_model.parameters():
        param.data = param.data.float()
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    conv_params = f"{ai_model.conv.out_channels}{ai_model.conv.in_channels}{ai_model.conv.kernel_size[0] * ai_model.conv.kernel_size[1]}"
    timestamp = time.strftime("%H%M_%d%m%Y")
    model_filename = f"{conv_params}_{timestamp}.pth"
    model_path = os.path.join(model_dir, model_filename)

    best_loss = float("inf")
    no_improve_count = 0
    epoch = 0

    while True:
        total_loss, total_policy_loss, total_value_loss, total_stage_loss = 0, 0, 0, 0
        for i, (batch_states, batch_policies, batch_values, batch_stages) in enumerate(data_loader):
            logger.debug(f"Batch {i + 1}: GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            batch_states = batch_states.to(device, dtype=torch.float32)
            batch_policies = batch_policies.to(device, dtype=torch.long)
            batch_values = batch_values.to(device, dtype=torch.float32)
            batch_stages = batch_stages.to(device, dtype=torch.long)
            optimizer.zero_grad()
            try:
                start_time = time.time()
                predicted_policies, predicted_values, predicted_stages = ai_model(batch_states)
                logger.debug(f"Batch {i + 1}: Thời gian forward pass: {time.time() - start_time:.4f}s")
                policy_loss = torch.nn.functional.cross_entropy(predicted_policies, batch_policies)
                value_loss = torch.nn.functional.mse_loss(predicted_values.squeeze(-1), batch_values)
                stage_loss = torch.nn.functional.cross_entropy(predicted_stages, batch_stages)
                loss = policy_loss + value_loss + 0.5 * stage_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_stage_loss += stage_loss.item()
            except Exception as e:
                logger.error(f"Lỗi trong batch {i + 1}: {e}")
                raise
        if len(data_loader) == 0:
            logger.error("DataLoader rỗng, không thể huấn luyện.")
            break
        avg_loss = total_loss / len(data_loader)
        epoch_log = (
            f"Epoch {epoch + 1}, Tổng Loss: {avg_loss:.4f}, "
            f"Policy Loss: {total_policy_loss / len(data_loader):.4f}, "
            f"Value Loss: {total_value_loss / len(data_loader):.4f}, "
            f"Stage Loss: {total_stage_loss / len(data_loader):.4f}"
        )
        logger.info(epoch_log)

        if best_loss - avg_loss < tolerance:
            no_improve_count += 1
            if no_improve_count >= patience:
                logger.info(f"Dừng huấn luyện sớm tại epoch {epoch + 1}.")
                break
        else:
            no_improve_count = 0
            best_loss = avg_loss
        epoch += 1

    torch.save(ai_model.state_dict(), model_path)
    logger.info(f"Đã lưu mô hình tại {model_path}")


class ChessDataset:
    """
    Dataset cho dữ liệu cờ vua từ các file PGN.
    """

    def __init__(self, data_dir):
        self.games = []
        preprocessed_path = os.path.join(data_dir, "preprocessed.pt")
        if os.path.exists(preprocessed_path):
            self.data = torch.load(preprocessed_path)
            print(f"Đã tải dữ liệu tiền xử lý từ {preprocessed_path}")
        else:
            print(f"Kiểm tra thư mục: {data_dir}")
            if not os.path.exists(data_dir):
                print(f"Lỗi: Thư mục '{data_dir}' không tồn tại.")
                return
            pgn_files = [f for f in os.listdir(data_dir) if f.endswith(".pgn")]
            print(f"Tìm thấy {len(pgn_files)} file PGN: {pgn_files}")

            for filename in pgn_files:
                file_path = os.path.join(data_dir, filename)
                print(f"Đọc file PGN: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        game_count = 0
                        while True:
                            game = chess.pgn.read_game(f)
                            if game is None:
                                break
                            self.games.append(game)
                            game_count += 1
                        print(f"Đã tải {game_count} ván cờ từ {filename}")
                except Exception as e:
                    print(f"Lỗi khi đọc {filename}: {e}")
            print(f"Tổng số ván cờ tải được: {len(self.games)}")
            total_moves = sum(len(list(game.mainline_moves())) for game in self.games)
            print(f"Tổng số nước đi trong dataset: {total_moves}")
            self.preprocess_and_save(preprocessed_path)

    def preprocess_and_save(self, save_path="data/preprocessed.pt"):
        processed_data = []
        for game in self.games:
            board = game.board()
            for move in game.mainline_moves():
                tensor = encode_board(board)
                move_idx = move_to_index(move)
                piece_count = len(board.piece_map())
                stage = 0 if piece_count > 30 else 1 if piece_count > 15 else 2
                processed_data.append((tensor, move_idx, 0.0, stage))
                board.push(move)
        torch.save(processed_data, save_path)
        print(f"Đã lưu dữ liệu tiền xử lý vào {save_path}")

    def __len__(self):
        return len(self.data) if hasattr(self, 'data') else sum(len(list(game.mainline_moves())) for game in self.games)

    def __getitem__(self, idx):
        if hasattr(self, 'data'):
            return self.data[idx]
        game_idx = 0
        move_count = 0
        for game in self.games:
            moves = list(game.mainline_moves())
            if idx < move_count + len(moves):
                board = game.board()
                for i, move in enumerate(moves):
                    if move_count + i == idx:
                        tensor = encode_board(board)
                        move_idx = move_to_index(move)
                        piece_count = len(board.piece_map())
                        if piece_count > 30:
                            stage = 0  # opening
                        elif piece_count > 15:
                            stage = 1  # middlegame
                        else:
                            stage = 2  # endgame
                        return (
                            tensor,
                            torch.tensor(move_idx, dtype=torch.long),
                            torch.tensor(0.0,dtype=torch.float32),
                            torch.tensor(stage, dtype=torch.long)
                        )
                    board.push(move)
                move_count += len(moves)
            else:
                move_count += len(moves)
        raise IndexError("Chỉ số ngoài phạm vi")


if __name__ == "__main__":
    ai_model = ChessNet().to(device)
    model_path = list_and_select_model()
    if model_path:
        ai_model = load_model(model_path)
    else:
        print("Không chọn mô hình. Khởi tạo mô hình mới.")

    optimizer = optim.AdamW(ai_model.parameters(), lr=1e-4)
    start_time = time.time()

    # Kiểm tra và tạo thư mục data nếu chưa tồn tại
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Đã tạo thư mục '{data_dir}'")

    train_with(ai_model, optimizer, data_dir=data_dir)

    elapsed_time = time.time() - start_time
    print(f"Thời gian hoàn tất: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")