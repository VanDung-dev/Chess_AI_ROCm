import os
import time
import torch
import glob
from modules.game.self_play import self_play_worker
from modules.training.chess_train import train_on_data, ChessNet
from modules.model.chess_model import load_model
from modules.config.chess_config import MODEL_PATH, DATA_PATH


def train_loop(
    games_per_iteration=10, 
    iterations=100, 
    retention_window=50, 
    device='cuda' if torch.cuda.is_available() else 'cpu',
    mcts_iterations=800
):
    """
    Vòng lặp huấn luyện liên tục:
    1. Tạo ván cờ (Tự chơi)
    2. Huấn luyện trên dữ liệu gần đây
    3. Lặp lại
    
    Args:
        games_per_iteration: Số ván cờ để chơi trước khi huấn luyện lại.
        iterations: Tổng số chu kỳ để chạy.
        retention_window: Số lượng tệp dữ liệu gần đây để giữ lại trong tập huấn luyện.
        mcts_iterations: Số lần mô phỏng MCTS cho mỗi nước đi.
    """
    print(f"Bắt đầu Vòng lặp Huấn luyện trên {device}...")
    
    # Khởi tạo đường dẫn
    model_path = os.path.join(MODEL_PATH, "latest_model.pth")
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Khởi tạo Mô hình
    if os.path.exists(model_path):
        print(f"Đang tải mô hình hiện có từ {model_path}")
        model = load_model(model_path).to(device)
    else:
        print("Đang khởi tạo mô hình mới...")
        model = ChessNet().to(device)
        torch.save(model.state_dict(), model_path)
    
    for it in range(iterations):
        print(f"\n=== Vòng lặp {it+1}/{iterations} ===")
        
        # 1. Tự chơi
        # Tệp có dấu thời gian cho vòng lặp này
        timestamp = int(time.time())
        data_file = os.path.join(DATA_PATH, f"self_play_{timestamp}_{it}.pt")
        
        print(f"Đang tạo {games_per_iteration} ván cờ...")
        self_play_worker(model_path, num_games=games_per_iteration, output_file=data_file, device=device)
        
        # 2. Thu thập dữ liệu
        # Lấy tất cả các tệp .pt trong thư mục dữ liệu, được sắp xếp theo thời gian
        data_files = sorted(glob.glob(os.path.join(DATA_PATH, "self_play_*.pt")), key=os.path.getmtime)
        
        # Chỉ giữ lại các tệp gần đây (Triển khai cửa sổ Replay Buffer)
        train_files = data_files[-retention_window:]
        print(f"Đang huấn luyện trên {len(train_files)} tệp gần đây (Kích thước cửa sổ: {retention_window})")
        
        # 3. Huấn luyện
        train_on_data(model, train_files, epochs=1)
        
        # 4. Lưu
        torch.save(model.state_dict(), model_path)
        
        # Tùy chọn: Lưu điểm kiểm tra
        if (it + 1) % 5 == 0:
            ckpt_path = os.path.join(MODEL_PATH, f"model_iter_{it+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Đã lưu điểm kiểm tra: {ckpt_path}")
            
    print("Vòng lặp Huấn luyện đã Hoàn thành.")

if __name__ == "__main__":
    train_loop(games_per_iteration=2, iterations=5, retention_window=10)
