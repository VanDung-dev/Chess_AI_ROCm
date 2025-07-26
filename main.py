import os
import torch
from modules.chess_neuron import ChessNet
from modules.chess_model import list_and_select_model
from modules.chess_train import run_train
from modules.chess_play import run_play
from modules.chess_config import DEVICE, MODEL_PATH
from stockfish.download_stockfish import download_stockfish


if __name__ == "__main__":
    download_stockfish()
    ai_model = ChessNet().to(DEVICE)
    selected_model_path = list_and_select_model(MODEL_PATH)

    # Biến để theo dõi đã thử fix GPU chưa
    tried_fix_gpu = False

    # Hàm kiểm tra GPU
    def check_gpu():
        if torch.cuda.is_available():
            return torch.device("cuda"), True
        else:
            return torch.device("cpu"), False

    # Lần đầu: thử kiểm tra GPU mà không set HSA_OVERRIDE_GFX_VERSION
    device, success = check_gpu()

    if not success:
        print("Không phát hiện GPU. Đang cố gắng sửa bằng cách đặt HSA_OVERRIDE_GFX_VERSION...")
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
        device, success = check_gpu()
        tried_fix_gpu = True

    # Kiểm tra thành công hay không
    if success:
        if tried_fix_gpu:
            print("Đã sửa thành công: Phát hiện GPU sau khi áp dụng HSA_OVERRIDE_GFX_VERSION.")
        print(f"Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
        print(f"Phiên bản ROCm: {torch.version.hip}")
        print(f"Phiên bản PyTorch: {torch.__version__}")
    else:
        print("Không tìm thấy GPU, đang sử dụng CPU.")
        device = torch.device("cpu")

    # Hiển thị menu chọn chức năng
    while True:
        print(
            "\nChọn chức năng:\n"
            "1. Huấn luyện AI\n"
            "2. Chơi cờ với AI\n"
            "0. Thoát chương trình"
        )
        choose = input("Nhập lựa chọn của bạn (0/1/2): ").strip()

        if choose == "1":
            run_train(selected_model_path)
        elif choose == "2":
            run_play(selected_model_path)
        elif choose == "0":
            print("Thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")

