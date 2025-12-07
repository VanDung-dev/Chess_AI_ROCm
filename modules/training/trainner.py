import time
import os
import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import chess
import chess.pgn
import chess.engine
from tqdm import tqdm
from typing import List, Tuple, Any
from modules.model.neuron import ChessNet
from modules.model.engine import encode_board, move_to_index, flip_vertical, flip_horizontal
from modules.model.model import load_model
from modules.config.config import STOCKFISH_PATH, DATA_PATH, MODEL_PATH, DEVICE, LOGGER


def get_game_result(pgn_game: chess.pgn.Game) -> float:
    """
    Lấy kết quả ván cờ từ PGN headers.

    Args:
        pgn_game (chess.pgn.Game): Đối tượng ván cờ PGN.

    Returns:
        float: Giá trị kết quả (1.0: trắng thắng, -1.0: đen thắng, 0.0: hòa hoặc không xác định).
    """
    result = pgn_game.headers.get("Result", "*")
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    elif result == "1/2-1/2":
        return 0.0
    return 0.0


def evaluate_position(engine: chess.engine.SimpleEngine, board: chess.Board) -> float:
    """
    Đánh giá trạng thái bàn cờ bằng Stockfish.

    Args:
        engine (chess.engine.SimpleEngine): Engine Stockfish.
        board (chess.Board): Trạng thái bàn cờ hiện tại.

    Returns:
        float: Giá trị bàn cờ trong khoảng [-1, 1].
    """
    try:
        info = engine.analyse(board, chess.engine.Limit(time=0.05))
        score = info["score"].relative.score(mate_score=10000) / 100.0
        return max(min(score / 10.0, 1.0), -1.0)
    except Exception as e:
        LOGGER.warning(f"Lỗi khi đánh giá vị trí: {e}")
        return 0.0


def propagate_values(values: List[float], gamma: float = 0.99) -> List[float]:
    """
    Lan truyền ngược giá trị với hệ số giảm.

    Args:
        values (List[float]): Danh sách giá trị của các trạng thái.
        gamma (float): Hệ số giảm (mặc định 0.99).

    Returns:
        List[float]: Danh sách giá trị đã lan truyền ngược.
    """
    propagated = values.copy()
    for i in range(len(values) - 2, -1, -1):
        propagated[i] = gamma * propagated[i + 1]
    return propagated


def augment_board(board: chess.Board) -> List[Tuple[chess.Board, callable]]:
    """
    Tăng cường dữ liệu bằng cách lật dọc và xoay 180 độ bàn cờ.

    Args:
        board (chess.Board): Trạng thái bàn cờ gốc.

    Returns:
        List[Tuple[chess.Board, callable]]: Danh sách các bàn cờ tăng cường và hàm biến đổi nước đi.
    """
    augmented = [(board, lambda m: m)]
    flipped = board.copy()
    flipped.apply_transform(chess.flip_vertical)
    augmented.append((flipped, lambda move: flip_vertical(move)))
    rotated = board.copy()
    rotated.apply_transform(chess.flip_vertical)
    rotated.apply_transform(chess.flip_horizontal)
    augmented.append((rotated, lambda move: flip_horizontal(flip_vertical(move))))
    return augmented


def get_top_moves(engine: chess.engine.SimpleEngine, board: chess.Board, top_k: int = 1) -> List[chess.Move]:
    """
    Lấy top k nước đi tốt nhất từ Stockfish.

    Args:
        engine (chess.engine.SimpleEngine): Engine Stockfish.
        board (chess.Board): Trạng thái bàn cờ hiện tại.
        top_k (int): Số lượng nước đi tốt nhất cần lấy.

    Returns:
        List[chess.Move]: Danh sách các nước đi tốt nhất.
    """
    try:
        info = engine.analyse(board, chess.engine.Limit(time=0.05), multipv=top_k)
        return [pv["pv"][0] for pv in info]
    except Exception as e:
        LOGGER.warning(f"Lỗi khi lấy top moves: {e}")
        return list(board.legal_moves)[:top_k]


class ChessDataset:
    """
    Dataset cho dữ liệu cờ vua từ các file PGN hoặc danh sách dữ liệu thô.
    """
    def __init__(self, preprocess=False, data=None):
        """
        Khởi tạo dataset.

        Args:
            preprocess (bool): Nếu True, thực hiện tiền xử lý từ PGN.
            data (list): Danh sách dữ liệu thô (nếu có) để load trực tiếp.
        """
        self.games = []
        self.data = []
        self.stage_info = {"opening": 0, "middlegame": 0, "endgame": 0}
        self.preprocessed_path = os.path.join(DATA_PATH, "preprocessed.pt")

        if data is not None:
            self.data = data
            return

        if not preprocess and os.path.exists(self.preprocessed_path):
            self.data = torch.load(self.preprocessed_path, weights_only=False)
            LOGGER.info(f"Đã tải dữ liệu tiền xử lý từ {self.preprocessed_path}")
            return

        if not os.path.exists(DATA_PATH):
            LOGGER.error(f"Thư mục '{DATA_PATH}' không tồn tại.")
            return

        self.load_pgn_files()
        if preprocess:
            self.preprocess_data()

    def load_pgn_files(self):
        """
        Đọc các file PGN từ thư mục DATA_PATH.
        """
        pgn_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pgn")]
        LOGGER.info(f"Tìm thấy {len(pgn_files)} file PGN: {pgn_files}")

        for filename in tqdm(pgn_files, desc="Đang đọc PGN files", total=len(pgn_files)):
            file_path = os.path.join(DATA_PATH, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    while True:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                        self.games.append(game)
            except Exception as e:
                LOGGER.error(f"Lỗi khi đọc {filename}: {e}")

    def preprocess_data(self):
        """
        Tiền xử lý dữ liệu PGN, bao gồm đánh giá Stockfish, tăng cường dữ liệu, và lưu vào file.
        """
        if os.path.exists(self.preprocessed_path):
            overwrite = input(f"Tệp {self.preprocessed_path} đã tồn tại. Bạn có muốn xóa và tạo mới? (y/n): ").strip().lower()
            if overwrite != 'y':
                LOGGER.info("Hủy quá trình tiền xử lý do người dùng chọn giữ file cũ.")
                return
            else:
                os.remove(self.preprocessed_path)
                LOGGER.info(f"Đã xóa tệp {self.preprocessed_path}")

        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH, debug=False)
        processed_data = []
        total_moves = 0

        for game in tqdm(self.games, desc="Tiền xử lý dữ liệu", total=len(self.games)):
            board = game.board()
            final_value = get_game_result(game)
            values = []
            moves = []

            for move in game.mainline_moves():
                top_moves = get_top_moves(engine, board, top_k=1)
                if move not in top_moves:
                    board.push(move)
                    continue

                values.append(evaluate_position(engine, board))
                moves.append(move)
                total_moves += 1
                board.push(move)

            values.append(final_value)
            values = propagate_values(values)

            board = game.board()
            for i, move in enumerate(moves):
                piece_count = len(board.piece_map())
                stage = 0 if piece_count > 30 else 1 if piece_count > 15 else 2
                self.stage_info["opening" if stage == 0 else "middlegame" if stage == 1 else "endgame"] += 1

                augmented_boards = augment_board(board)
                for aug_board, transform in augmented_boards:
                    aug_move = transform(move) if transform else move
                    if not aug_board.is_legal(aug_move):
                        continue
                    tensor = encode_board(aug_board)
                    move_idx = move_to_index(aug_move)
                    stage_onehot = torch.zeros(3, dtype=torch.float32)
                    stage_onehot[stage] = 1.0
                    processed_data.append((tensor, move_idx, values[i], stage_onehot))
                if board.is_legal(move):
                    board.push(move)
                else:
                    continue

        torch.save(processed_data, self.preprocessed_path)
        LOGGER.info(f"Đã lưu dữ liệu tiền xử lý vào {self.preprocessed_path}")
        self.data = processed_data
        engine.quit()

    def __len__(self) -> int:
        """
        Trả về số lượng mẫu trong dataset.

        Returns:
            int: Số lượng mẫu.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor, Any]:
        """
        Lấy mẫu dữ liệu tại chỉ số idx.

        Args:
            idx (int): Chỉ số mẫu.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]: Tensor bàn cờ, chỉ số nước đi,
            giá trị bàn cờ, và one-hot vector cho giai đoạn.
        """
        # Định dạng dữ liệu: (fen/tensor, policy/move_idx, value, stage_onehot)
        # Kiểm tra xem mục có phải là bộ 4 (định dạng pgn cũ) hay 3 (định dạng tự phát) hay không
        item = self.data[idx]
        
        if len(item) == 4:
            tensor, move_idx, value, stage_onehot = item
            # move_idx là chỉ mục int
            return tensor, torch.tensor(move_idx, dtype=torch.long), torch.tensor(value, dtype=torch.float32), stage_onehot
        elif len(item) == 3:
            fen, policy, value = item
            if isinstance(fen, str):
                board = chess.Board(fen)
                tensor = encode_board(board)
            else:
                tensor = fen
            return tensor, torch.tensor(policy, dtype=torch.float32), torch.tensor(value, dtype=torch.float32), torch.zeros(3) # dummy stage
        else:
             raise ValueError("Unknown data format")


def analyze_data(data_path: str = DATA_PATH) -> Tuple[dict, int]:
    """
    Phân tích dữ liệu PGN để thống kê số lượng nước đi theo từng giai đoạn.

    Args:
        data_path (str): Thư mục chứa file PGN.

    Returns:
        Tuple[dict, int]: Từ điển chứa số lượng nước đi theo giai đoạn và tổng số nước đi.
    """
    LOGGER.info("\nPhân tích dữ liệu huấn luyện...")
    pgn_files = [f for f in os.listdir(data_path) if f.endswith(".pgn")]
    stage_counts = {"Khai cuộc": 0, "Trung cuộc": 0, "Tàn cuộc": 0}
    total_moves = 0

    for filename in tqdm(pgn_files, desc="Phân tích dữ liệu", total=len(pgn_files)):
        file_path = os.path.join(data_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                for move in game.mainline_moves():
                    piece_count = len(board.piece_map())
                    if piece_count > 30:
                        stage_counts["Khai cuộc"] += 1
                    elif piece_count > 15:
                        stage_counts["Trung cuộc"] += 1
                    else:
                        stage_counts["Tàn cuộc"] += 1
                    total_moves += 1
                    board.push(move)

    return stage_counts, total_moves


def train_with(ai_model: ChessNet, optimizer: optim.Optimizer,
               batch_size: int = 256, tolerance: float = 1e-4, patience: int = 3, dataset=None) -> None:
    """
    Huấn luyện mô hình AI.

    Args:
        ai_model (ChessNet): Mô hình AI cần huấn luyện.
        optimizer (optim.Optimizer): Bộ tối ưu hóa.
        batch_size (int): Kích thước batch.
        tolerance (float): Ngưỡng dừng sớm.
        patience (int): Số epoch chờ trước khi dừng sớm.
        dataset (ChessDataset): Dataset tùy chỉnh (nếu có).
    """
    try:
        LOGGER.info(f"Các backend có sẵn: {torch._dynamo.list_backends()}")
        ai_model = torch.compile(ai_model, backend="inductor")
        LOGGER.info("Đã áp dụng torch.compile với backend inductor")
    except Exception as e:
        LOGGER.warning(f"Không thể áp dụng torch.compile: {e}")

    ai_model.train()
    if dataset is None:
        dataset = ChessDataset()
    
    if len(dataset) == 0:
        LOGGER.error("Lỗi: Không tìm thấy ván cờ hoặc nước đi hợp lệ trong dữ liệu.")
        return
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) # num_workers=0 for safety in some envs
    LOGGER.info(f"DataLoader được tạo với batch_size={batch_size}, kích thước dataset={len(dataset)}")

    for param in ai_model.parameters():
        param.data = param.data.float()

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    conv_params = f"{ai_model.conv.out_channels}{ai_model.conv.in_channels}{ai_model.conv.kernel_size[0] * ai_model.conv.kernel_size[1]}"
    timestamp = time.strftime("%Y%m%d_%H%M")
    model_filename = f"{conv_params}_{timestamp}.pth"
    model_path = os.path.join(MODEL_PATH, model_filename)

    best_loss = float("inf")
    no_improve_count = 0
    epoch = 0

    while True:
        total_loss, total_policy_loss, total_value_loss = 0, 0, 0
        for i, (batch_states, batch_policies, batch_values, batch_stages) in enumerate(data_loader):
            batch_states = batch_states.to(DEVICE, dtype=torch.float32)
            # batch_policies có thể là Long (index) hoặc Float (probs) tùy thuộc vào tập dữ liệu
            if batch_policies.dtype == torch.float32:
                 batch_policies = batch_policies.to(DEVICE, dtype=torch.float32)
            else:
                 batch_policies = batch_policies.to(DEVICE, dtype=torch.long)
                 
            batch_values = batch_values.to(DEVICE, dtype=torch.float32)
            # batch_stages is ignored
            
            optimizer.zero_grad()
            try:
                predicted_policies, predicted_values = ai_model(batch_states)
                
                # Check target type for loss
                if batch_policies.dim() > 1 and batch_policies.dtype == torch.float32:
                     policy_loss = torch.nn.functional.cross_entropy(predicted_policies, batch_policies)
                else:
                     policy_loss = torch.nn.functional.cross_entropy(predicted_policies, batch_policies)
                     
                value_loss = torch.nn.functional.mse_loss(predicted_values.squeeze(-1), batch_values)
                
                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
            except Exception as e:
                LOGGER.error(f"Lỗi trong batch {i + 1}: {e}")
                raise
        if len(data_loader) == 0:
            LOGGER.error("DataLoader rỗng, không thể huấn luyện.")
            break
        avg_loss = total_loss / len(data_loader)
        epoch_log = (
            f"Epoch {epoch + 1}, Tổng Loss: {avg_loss:.4f}, "
            f"Policy Loss: {total_policy_loss / len(data_loader):.4f}, "
            f"Value Loss: {total_value_loss / len(data_loader):.4f}"
        )
        LOGGER.info(epoch_log)

        if best_loss - avg_loss < tolerance:
            no_improve_count += 1
            if no_improve_count >= patience:
                LOGGER.info(f"Dừng huấn luyện sớm tại epoch {epoch + 1}.")
                break
        else:
            no_improve_count = 0
            best_loss = avg_loss
        epoch += 1

    torch.save(ai_model.state_dict(), model_path)
    LOGGER.info(f"Đã lưu mô hình tại {model_path}")

def train_on_data(ai_model, data_files, epochs=1):
    """
    Người trợ giúp đào tạo về danh sách tệp dữ liệu (self-play data).
    """
    all_data = []
    for f in data_files:
        if os.path.exists(f):
            all_data.extend(torch.load(f))
    
    LOGGER.info(f"Training on {len(all_data)} samples from {len(data_files)} files.")
    dataset = ChessDataset(data=all_data)
    optimizer = optim.AdamW(ai_model.parameters(), lr=1e-4)
    
    train_with(ai_model, optimizer, patience=1)


def run_train(selected_model_path=None):
    """
    Chương trình học AI.

    Args:
        selected_model_path (str): Đường dẫn file .pth của mô hình chọn.
    """
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        LOGGER.info(f"Đã tạo thư mục '{DATA_PATH}'")

    # Khởi tạo mô hình
    if selected_model_path and os.path.exists(selected_model_path):
        ai_model = load_model(selected_model_path)
        LOGGER.info(f"Tải mô hình từ {selected_model_path}")
    else:
        ai_model = ChessNet().to(DEVICE)
        LOGGER.info("Không chọn mô hình. Khởi tạo mô hình mới.")

    while True:
        print("\nChọn chế độ:")
        print("1. Tiền xử lý dữ liệu với Stockfish")
        print("2. Huấn luyện mô hình")
        print("3. Cả hai")
        print("0. Thoát")
        choice = input("Nhập lựa chọn của bạn: ").strip()

        if choice == "1":
            mode = "preprocess"
            break
        elif choice == "2":
            mode = "train"
            break
        elif choice == "3":
            mode = "both"
            break
        elif choice == "0":
            print("Thoát chương trình.")
            return
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")

    optimizer = optim.AdamW(ai_model.parameters(), lr=1e-4)
    start_time = time.time()

    if mode in ["preprocess", "both"]:
        dataset = ChessDataset(preprocess=True)
        if len(dataset) == 0:
            LOGGER.error("Không thể tạo dữ liệu tiền xử lý. Thoát chương trình.")
            return
        if mode == "preprocess":
            LOGGER.info("Đã hoàn tất tiền xử lý dữ liệu.")
            return

    if mode in ["train", "both"]:
        train_with(ai_model, optimizer)
        elapsed_time = time.time() - start_time
        LOGGER.info(f"Thời gian hoàn tất: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")