import torch
import platform
from modules.config.logger import setup_logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = setup_logger()

# Phát hiện nền tảng và đặt đường dẫn Stockfish chính xác
if platform.system().lower() == "windows":
    STOCKFISH_PATH = "./stockfish/stockfish-windows-x86-64-avx2.exe"
else:
    STOCKFISH_PATH = "./stockfish/stockfish-ubuntu-x86-64-avx2"

DATA_PATH = "./data"
MODEL_PATH = "./models"