import chess
import os
import time
import chess.pgn
from modules.chess_mcts import get_best_move
from modules.chess_model import load_model
from modules.chess_config import DATA_PATH, MODEL_PATH


def detect_game_stage(board):
    """Xác định giai đoạn trận cờ"""
    piece_count = len(board.piece_map())
    if piece_count > 30:
        return "Khai cuộc"
    elif piece_count > 15:
        return "Trung cuộc"
    return "Tàn cuộc"


def play_game(ai_model, human_color=chess.WHITE):
    """
    Chơi một ván cờ vua giữa người và AI, sử dụng MCTS để điều khiển trò chơi.

    Args:
        ai_model: Mô hình AI (ChessNet).
        human_color (bool): Màu của người chơi (mặc định chess.WHITE).
    """
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    timestamp = time.strftime("%H%M_%d%m%Y")
    pgn_filename = f"HA_{timestamp}.pgn"
    pgn_file_path = os.path.join(DATA_PATH, pgn_filename)

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Human vs AI game with MCTS"
    game.headers["Site"] = "Local"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["White"] = "Human" if human_color == chess.WHITE else "AI_MCTS"
    game.headers["Black"] = "AI_MCTS" if human_color == chess.WHITE else "Human"
    node = game

    print("Bắt đầu ván cờ mới. Bạn là", "Trắng" if human_color == chess.WHITE else "Đen")
    print(board)

    while not board.is_game_over():
        # Hiển thị giai đoạn hiện tại
        stage = detect_game_stage(board)
        print(f"Giai đoạn hiện tại: {stage}")

        if board.turn == human_color:
            # Lượt của người chơi
            print("\nNước đi của bạn (ví dụ: 'e2e4' hoặc 'Nf3'):")
            while True:
                try:
                    move_input = input("> ")
                    if move_input.lower() == "quit":
                        print("Ván cờ kết thúc do người dùng thoát.")
                        game.headers["Result"] = "*"
                        with open(pgn_file_path, "a", encoding="utf-8") as pgn_file:
                            exporter = chess.pgn.FileExporter(pgn_file)
                            game.accept(exporter)
                            pgn_file.write("\n\n")
                        print(f"Ván cờ đã được lưu vào {pgn_file_path}")
                        return
                    move = board.parse_san(move_input)
                    if move in board.legal_moves:
                        board.push(move)
                        node = node.add_variation(move)
                        break
                    else:
                        print("Nước đi không hợp lệ. Thử lại.")
                except ValueError:
                    print("Định dạng nước đi không hợp lệ. Sử dụng ký hiệu SAN (ví dụ: 'e2e4' hoặc 'Nf3').")
        else:
            # Lượt của AI
            print("\nAI đang suy nghĩ...")
            print(f"Trạng thái bàn cờ hiện tại: {board.fen()}")
            # Điều chỉnh số lần lặp MCTS theo giai đoạn
            stage = detect_game_stage(board)
            if stage == "Tàn cuộc":
                move = get_best_move(ai_model, board, mcts_iterations=200, temperature=0.5)
            else:
                move = get_best_move(ai_model, board, mcts_iterations=100, temperature=1.0)
            if move is None:
                print("AI không tìm thấy nước đi hợp lệ. Kết thúc ván.")
                game.headers["Result"] = "*"
                with open(pgn_file_path, "a", encoding="utf-8") as pgn_file:
                    exporter = chess.pgn.FileExporter(pgn_file)
                    game.accept(exporter)
                    pgn_file.write("\n\n")
                print(f"Ván cờ đã được lưu vào {pgn_file_path}")
                return
            move_san = board.san(move)
            print(f"Nước đi của AI: {move_san} (UCI: {move.uci()})")
            board.push(move)
            node = node.add_variation(move)  # Thêm nước đi của AI vào game node

        print("\nVị trí hiện tại:")
        print(board)

    game.headers["Result"] = board.result()
    with open(pgn_file_path, "a", encoding="utf-8") as pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        game.accept(exporter)
        pgn_file.write("\n\n")

    print(f"\nVán cờ kết thúc. Kết quả: {game.headers['Result']}")
    print(f"Ván cờ đã được lưu vào {pgn_file_path}")
    if game.headers["Result"] == "1-0":
        print("Trắng thắng!")
    elif game.headers["Result"] == "0-1":
        print("Đen thắng!")
    else:
        print("Hòa!")


def self_play(ai_model, num_games=100):
    """
    AI tự chơi và lưu các ván cờ dưới dạng PGN, sử dụng MCTS để chọn nước đi.

    Args:
        ai_model: Mô hình AI (ChessNet).
        num_games (int): Số ván cờ cần chơi (mặc định 100).
    """
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    timestamp = time.strftime("%H%M_%d%m%Y")
    pgn_filename = f"SP_{num_games}_{timestamp}.pgn"
    pgn_file_path = os.path.join(DATA_PATH, pgn_filename)

    with open(pgn_file_path, "w", encoding="utf-8") as pgn_file:
        for game_num in range(num_games):
            print(f"Đang chơi ván thứ {game_num + 1}/{num_games}...")
            start_time = time.time()
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = "Self-play game with MCTS"
            game.headers["Site"] = "Local"
            game.headers["Date"] = time.strftime("%Y.%m.%d")
            game.headers["Round"] = str(game_num + 1)
            game.headers["White"] = "AI_MCTS"
            game.headers["Black"] = "AI_MCTS"
            node = game

            while not board.is_game_over():
                # Điều chỉnh số lần lặp MCTS theo giai đoạn
                stage = detect_game_stage(board)
                if stage == "Tàn cuộc":
                    move = get_best_move(ai_model, board, mcts_iterations=200, temperature=0.5)
                else:
                    move = get_best_move(ai_model, board, mcts_iterations=100, temperature=1.0)
                if move is None:
                    print("Không tìm thấy nước đi hợp lệ. Kết thúc ván.")
                    break
                move_san = board.san(move)
                print(f"Nước đi của AI: {move_san} (UCI: {move.uci()})")
                board.push(move)
                node = node.add_variation(move)

            game.headers["Result"] = board.result()
            exporter = chess.pgn.FileExporter(pgn_file)
            game.accept(exporter)
            pgn_file.write("\n\n")
            print(
                f"Đã lưu ván cờ {game_num + 1} vào {pgn_file_path}, thời gian: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print(f"Đã lưu tất cả {num_games} ván cờ vào {pgn_file_path}")

def run_play(ai_model):
    """
    Chương trình chơi cờ với AI.

    Args:
        ai_model (ChessNet): Mô hình AI.
    """
    if MODEL_PATH is None:
        print("Không chọn mô hình. Thoát chương trình.")
        exit(1)
    else:
        load_model()

    while True:
        print("\nChọn chế độ:")
        print("1. Chơi với AI")
        print("2. AI tự chơi")
        print("0. Thoát")
        choice = input("Nhập lựa chọn của bạn: ").strip()

        if choice == "1":
            print("Chọn màu của bạn:")
            print("1. Trắng")
            print("2. Đen")
            color_choice = input("Nhập 1 hoặc 2: ").strip()
            human_color = chess.WHITE if color_choice == "1" else chess.BLACK
            play_game(ai_model, human_color=human_color)
        elif choice == "2":
            try:
                num = int(input("Nhập số ván cờ AI tự chơi: "))
                if num <= 0:
                    print("Số ván phải lớn hơn 0. Sử dụng giá trị mặc định là 10.")
                    num = 10
            except ValueError:
                print("Giá trị không hợp lệ. Sử dụng giá trị mặc định là 10.")
                num = 10

            start_time = time.time()
            self_play(ai_model, num_games=num)
            print(f"Thời gian hoàn tất: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        elif choice == "0":
            print("Đã thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")