import os
import torch
from modules.chess_neuron import ChessNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_and_select_model(model_dir="model"):
    """
    Hiển thị danh sách các mô hình .pth trong thư mục và cho phép người dùng chọn.

    Args:
        model_dir (str): Thư mục chứa các file mô hình (mặc định "model").

    Returns:
        str: Đường dẫn đến file mô hình được chọn hoặc None nếu không chọn.
    """
    if not os.path.exists(model_dir):
        print(f"Lỗi: Thư mục '{model_dir}' không tồn tại.")
        return None

    pth_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not pth_files:
        print(f"Lỗi: Không tìm thấy file .pth nào trong '{model_dir}'.")
        return None

    print("\nDanh sách mô hình có sẵn:")
    for i, file in enumerate(pth_files, 1):
        print(f"{i}. {file}")

    while True:
        try:
            choice = input("Chọn mô hình (nhập số hoặc 'quit' để thoát): ").strip()
            if choice.lower() == "quit":
                return None
            choice = int(choice)
            if 1 <= choice <= len(pth_files):
                return os.path.join(model_dir, pth_files[choice - 1])
            else:
                print(f"Vui lòng nhập số từ 1 đến {len(pth_files)}.")
        except ValueError:
            print("Đầu vào không hợp lệ. Vui lòng nhập số hoặc 'quit'.")


def load_model(model_path):
    """
    Tải mô hình ChessNet đã huấn luyện từ file .pth.

    Args:
        model_path (str): Đường dẫn đến file mô hình.

    Returns:
        ChessNet: Mô hình đã tải.
    """
    model = ChessNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        model.eval()
        print(f"Đã tải mô hình thành công từ {model_path}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{model_path}'.")
        raise
    return model