import torch
import chess
import os
import time
from tqdm import tqdm
from modules.model.model import ChessNet
from modules.engine.mcts import run_mcts


def play_self_play_game(model, mcts_iterations=800, temperature=1.0):
    """
    Chơi một ván cờ tự chơi.
    
    Returns:
        list of tuples: (fen, policy_vector, player_turn)
        float: kết quả ván cờ (+1 cho Trắng thắng, -1 cho Đen thắng, 0 cho Hòa)
    """
    board = chess.Board()
    game_history = []
    
    # 0 = Trắng, 1 = Đen (trong logic, nhưng python-chess: True=Trắng)
    
    while not board.is_game_over():
        # Lấy policy từ MCTS
        temp = temperature if board.fullmove_number <= 30 else 0.5 # Giảm nhiệt độ sau đó
        
        best_move, _, policy_vector = run_mcts(model, board, mcts_iterations, temp)
        
        if best_move is None:
            break
            
        # Lưu trữ trạng thái (FEN là đủ để tái tạo, nhưng chúng ta có thể muốn mã hóa sau này)
        # Chúng ta lưu trữ FEN để cho nhẹ
        game_history.append((board.fen(), policy_vector, board.turn))
        
        board.push(best_move)
        
    # Xác định kết quả
    outcome = board.outcome()
    if outcome is None:
        result = 0.0 # Hòa hoặc dừng
    else:
        if outcome.winner == chess.WHITE:
            result = 1.0
        elif outcome.winner == chess.BLACK:
            result = -1.0
        else:
            result = 0.0
            
    return game_history, result

def process_game_data(game_history, result):
    """
    Gán các mục tiêu giá trị cho mỗi bước.
    Nếu kết quả là +1 (Trắng thắng):
        Các bước của lượt Trắng -> mục tiêu = +1
        Các bước của lượt Đen -> mục tiêu = -1
    """
    processed_data = []
    for fen, policy, turn in game_history:
        # turn là boolean (True=Trắng)
        if turn == chess.WHITE:
            value_target = result
        else:
            value_target = -result
            
        processed_data.append((fen, policy, value_target))
        
    return processed_data

def self_play_worker(model_path, num_games=10, output_file="data/self_play_data.pt", device='cpu', mcts_iterations=800):
    """
    Hàm worker để tạo ra các ván cờ.
    """
    # Tải mô hình
    device = torch.device(device)
    model = ChessNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_data = []
    
    print(f"Bắt đầu tự chơi cho {num_games} ván cờ...")
    for i in tqdm(range(num_games)):
        history, result = play_self_play_game(model, mcts_iterations=mcts_iterations)
        game_data = process_game_data(history, result)
        all_data.extend(game_data)
        
    # Lưu dữ liệu đã xử lý
    # Định dạng: Danh sách các (FEN, Policy, Value)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Nếu tệp tồn tại, có nối thêm không? Không, torch.save ghi đè.
    # Thường thì chúng ta lưu các đoạn riêng biệt: "data_timestamp_workerid.pt"
    torch.save(all_data, output_file)
    print(f"Đã lưu {len(all_data)} mẫu vào {output_file}")


if __name__ == "__main__":
    # Đảm bảo thư mục này tồn tại
    os.makedirs("modules/data", exist_ok=True)
    
    # Kiểm tra mô hình
    model_path = "modules/chess_model.pth"
    if not os.path.exists(model_path):
        print("Không tìm thấy mô hình, khởi tạo mô hình ngẫu nhiên...")
        model = ChessNet()
        torch.save(model.state_dict(), model_path)
        
    timestamp = int(time.time())
    self_play_worker(model_path, num_games=1, output_file=f"modules/data/self_play_{timestamp}.pt", device='cuda' if torch.cuda.is_available() else 'cpu')
