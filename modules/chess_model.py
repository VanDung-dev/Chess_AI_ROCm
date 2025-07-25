import os
import torch
from modules.chess_neuron import ChessNet
from modules.chess_config import MODEL_PATH, DEVICE

def list_and_select_model():
    """
    Hiển thị danh sách các mô hình .pth trong thư mục và cho phép người dùng chọn.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Thư mục '{MODEL_PATH}' không tồn tại.")
        return None

    pth_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".pth")]
    if not pth_files:
        print(f"Lỗi: Không tìm thấy file .pth nào trong '{MODEL_PATH}'.")
        return None

    print("\nDanh sách mô hình có sẵn:")
    for i, file in enumerate(pth_files, 1):
        print(f"{i}. {file}")

    while True:
        try:
            choice = input("Chọn mô hình (nhập số hoặc '0' để không sử dụng mô hình): ").strip()
            if choice.lower() == "0":
                return None
            choice = int(choice)
            if 1 <= choice <= len(pth_files):
                return os.path.join(MODEL_PATH, pth_files[choice - 1])
            else:
                print(f"Vui lòng nhập số từ 1 đến {len(pth_files)}.")
        except ValueError:
            print("Đầu vào không hợp lệ. Vui lòng nhập số hoặc 'quit'.")


def load_model(model_path):
    """
    Tải mô hình ChessNet đã huấn luyện từ file .pth.
    """
    model = ChessNet().to(DEVICE)
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=True),
            strict=False
        )
        model.eval()
        print(f"Đã tải mô hình thành công từ {model_path}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{model_path}'.")
        raise
    return model
