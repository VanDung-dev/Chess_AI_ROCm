import torch
from modules.chess_log import setup_logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = setup_logger()
STOCKFISH_PATH = "./stockfish/stockfish-ubuntu-x86-64-avx2"
DATA_PATH = "data"
MODEL_PATH = "models"