import os
import subprocess
import sys
import torch

if __name__ == "__main__":
    # Đặt biến môi trường cho ROCm 6.3
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    # Kiểm tra GPU và ROCm
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
        print(f"Phiên bản ROCm: {torch.version.hip}")
        print(f"Phiên bản PyTorch: {torch.__version__}")
    else:
        device = torch.device("cpu")
        print("Không tìm thấy GPU, đang sử dụng CPU.")
        sys.exit(1)

    python_executable = ".venv/bin/python" if not os.path.exists("/.dockerenv") else "python3"
    print(
        "1. chess_train.py\n"
        "2. chess_play.py\n"
    )

    choose = input("Choose: ")
    if choose == "1":
        script_path = "modules/chess_train.py"
    elif choose == "2":
        script_path = "modules/chess_play.py"
    else:
        print("Invalid choice")
        sys.exit(1)

    subprocess.run([python_executable, script_path])