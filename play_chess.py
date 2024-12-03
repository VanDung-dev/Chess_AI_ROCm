from chess_engine import *
from negamaxAI import *
import pygame
import os

pygame.init()
game_state = GameState()
valid_moves = game_state.get_valid_moves()
screen = pygame.display.set_mode((1000, 800))
clock = pygame.time.Clock()
images = {}
captures_images = {}
SQ_SIZE = 100
color_board = ['#EBECD0', '#739552']
color_str = color_board[::-1]

def draw_game_state(screen, game_state, square_selected, SQ_SIZE):
    """Vẽ bộ hiện của trò chơi"""
    draw_board(screen, SQ_SIZE)
    highlight_squares(screen, game_state, square_selected, SQ_SIZE)
    if game_state.in_check:
        king_position = game_state.find_king(game_state.white_to_move)
        highlight_king_in_check(screen, king_position, SQ_SIZE)
    draw_pieces(screen, game_state.board, SQ_SIZE)
    draw_captured_pieces(screen, game_state.white_captured_pieces, game_state.black_captured_pieces, SQ_SIZE)
    if square_selected != ():
        row, col = square_selected
        piece = game_state.board[row][col]
        if (piece[0] == 'w' and game_state.white_to_move) or (
        piece[0] == 'b' and not game_state.white_to_move, SQ_SIZE):
            valid_moves = game_state.get_valid_moves()
            highlight_valid_moves(screen, valid_moves, square_selected, SQ_SIZE)

def load_and_scale_image(image_path, size):
    """Tải ảnh PNG và thay đổi kích thước"""
    image = pygame.image.load(image_path)
    image = image.convert_alpha()
    image = pygame.transform.scale(image, size)
    return image

def load_images(SQ_SIZE):
    """Tải ảnh các quân cờ trên bàn cờ"""
    pieces = ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR', 'wP',
              'bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR', 'bP']

    for piece in pieces:
        image_path = os.path.join('images', f'{piece}.png')
        images[piece] = load_and_scale_image(image_path, (SQ_SIZE, SQ_SIZE))

def load_captured_images(SQ_SIZE):
    """Tải ảnh các quân cờ bị ăn"""
    pieces = ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR', 'bP',
              'wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR', 'wP']
    for piece in pieces:
        image_path = os.path.join('images', f'{piece}.png')
        captures_images[piece] = load_and_scale_image(image_path, (SQ_SIZE // 2, SQ_SIZE // 2))

def draw_board(screen, SQ_SIZE):
    """Vẽ giao diện bàn cờ"""
    font_board = pygame.font.Font(None, SQ_SIZE // 5)
    for row in range(8):
        for column in range(8):
            colour = color_board[((row + column) % 2)]
            pygame.draw.rect(screen, colour, pygame.Rect(column * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

    for i in range(8):
        color = color_board[(i % 2)]
        text = font_board.render(chr(ord('a') + i), True, color)
        screen.blit(text, ((SQ_SIZE - SQ_SIZE // 6) + i * SQ_SIZE, SQ_SIZE * 8 - SQ_SIZE // 4))

    for i in range(8):
        color = color_str[(i % 2)]
        text = font_board.render(str(8 - i), True, color)
        screen.blit(text, (SQ_SIZE // 30, SQ_SIZE // 100 + i * SQ_SIZE))

def draw_pieces(screen, board, SQ_SIZE):
    """Vẽ quân cờ trên bàn cờ"""
    for row in range(8):
        for column in range(8):
            piece = board[row][column]
            if piece != '--' and piece in images:
                screen.blit(images[piece], pygame.Rect(column * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_captured_pieces(screen, white_captured_pieces, black_captured_pieces, SQ_SIZE):
    """Vẽ các quân cờ đã bị ăn"""
    pygame.draw.rect(screen, 'grey', pygame.Rect(
        SQ_SIZE * 8 + SQ_SIZE // 8, SQ_SIZE, SQ_SIZE // 2, SQ_SIZE * 7 - SQ_SIZE // 8))
    pygame.draw.rect(screen, 'grey', pygame.Rect(
        SQ_SIZE * 9 - SQ_SIZE // 3, SQ_SIZE, SQ_SIZE // 2, SQ_SIZE * 7 - SQ_SIZE // 8))

    for i, piece in enumerate(white_captured_pieces):
        piece_image = captures_images.get(piece)
        if piece_image:
            screen.blit(piece_image, (SQ_SIZE * 8 + SQ_SIZE // 8, SQ_SIZE + i * (SQ_SIZE // 2 - SQ_SIZE // 15)))

    for i, piece in enumerate(black_captured_pieces):
        piece_image = captures_images.get(piece)
        if piece_image:
            screen.blit(piece_image, (SQ_SIZE * 9 - SQ_SIZE // 3, SQ_SIZE + i * (SQ_SIZE // 2 - SQ_SIZE // 15)))

def highlight_king_in_check(screen, king_position, SQ_SIZE):
    """Làm nổi bật vua khi bị chiếu"""
    row, col = king_position
    highlight_king = pygame.Surface((SQ_SIZE, SQ_SIZE))
    highlight_king.set_alpha(125)
    highlight_king.fill(pygame.Color('red'))
    screen.blit(highlight_king, (col * SQ_SIZE, row * SQ_SIZE))

def highlight_squares(screen, game_state, square_selected, SQ_SIZE):
    """Làm nổi bật quân cờ được chọn"""
    if square_selected != ():
        row, column = square_selected
        if game_state.board[row][column][0] == ('w' if game_state.white_to_move else 'b'):
            highlight_selected = pygame.Surface((SQ_SIZE, SQ_SIZE))
            highlight_selected.set_alpha(125)
            highlight_selected.fill(pygame.Color('black'))
            screen.blit(highlight_selected, (column * SQ_SIZE, row * SQ_SIZE))

    if len(game_state.move_log) != 0:
        last_move = game_state.move_log[-1]
        start_row, start_column = last_move.start_row, last_move.start_column
        end_row, end_column = last_move.end_row, last_move.end_column
        highlight_last = pygame.Surface((SQ_SIZE, SQ_SIZE))
        highlight_last.set_alpha(125)
        highlight_last.fill(pygame.Color('orange'))
        screen.blit(highlight_last, (start_column * SQ_SIZE, start_row * SQ_SIZE))
        screen.blit(highlight_last, (end_column * SQ_SIZE, end_row * SQ_SIZE))

def highlight_valid_moves(screen, valid_moves, square_selected, SQ_SIZE):
    """Vẽ các nước đi hợp lệ của quân cờ được chọn, và nước đi ăn quân sẽ là hình tròn rỗng"""
    for move in valid_moves:
        if move.start_row == square_selected[0] and move.start_column == square_selected[1]:
            highlight_moves = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            highlight_moves.set_alpha(125)
            if move.is_capture:
                pygame.draw.circle(highlight_moves, 'black', (SQ_SIZE // 2, SQ_SIZE // 2),
                                   SQ_SIZE // 3, SQ_SIZE // 10)
            else:
                pygame.draw.circle(highlight_moves, 'black', (SQ_SIZE // 2, SQ_SIZE // 2), SQ_SIZE // 6)
            screen.blit(highlight_moves, (move.end_column * SQ_SIZE, move.end_row * SQ_SIZE))

def animate_move(move, screen, board, clock, game_state, SQ_SIZE):
    """Hoạt ảnh di chuyển quân cờ"""
    delta_row = move.end_row - move.start_row
    delta_column = move.end_column - move.start_column
    total_duration = 0.3
    total_frames = int(clock.get_fps() * total_duration)

    if total_frames == 0:
        total_frames = 1
    for frame in range(total_frames + 1):
        row, column = (
            move.start_row + delta_row * frame / total_frames,
            move.start_column + delta_column * frame / total_frames)
        draw_board(screen, SQ_SIZE)
        draw_pieces(screen, board, SQ_SIZE)
        draw_captured_pieces(screen, game_state.white_captured_pieces, game_state.black_captured_pieces, SQ_SIZE)

        colour = color_board[(move.end_row + move.end_column) % 2]
        end_square = pygame.Rect(move.end_column * SQ_SIZE, move.end_row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        pygame.draw.rect(screen, colour, end_square)

        if move.piece_captured != '--':
            if move.is_en_passant_move:
                en_passant_row = move.end_row + 1 if move.piece_captured[0] == 'b' else move.end_row - 1
                end_square = pygame.Rect(move.end_column * SQ_SIZE, en_passant_row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            screen.blit(images[move.piece_captured], end_square)

        screen.blit(images[move.piece_moved], pygame.Rect(column * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))
        pygame.display.flip()

def play_game():
    """Giao diện chơi cờ giữa người chơi và AI hoặc giữa hai AI"""
    global text, valid_moves
    screen.fill('wheat4')
    load_images(SQ_SIZE)
    load_captured_images(SQ_SIZE)
    square_selected = ()
    player_clicks = []
    animate = False
    game_over = False
    move_made = False
    in_game = True
    valid_moves = game_state.get_valid_moves()
    num = 0

    while in_game:
        draw_game_state(screen, game_state, square_selected, SQ_SIZE)
        pygame.draw.rect(screen, 'black', pygame.Rect(SQ_SIZE * 8, 0, SQ_SIZE * 2, SQ_SIZE))
        human_turn = game_state.white_to_move

        if not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not game_over:
                        location = pygame.mouse.get_pos()
                        column = location[0] // SQ_SIZE
                        row = location[1] // SQ_SIZE
                        if square_selected == (row, column) or column >= 8:
                            square_selected = ()
                            player_clicks = []
                        else:
                            square_selected = (row, column)
                            player_clicks.append(square_selected)
                        if len(player_clicks) == 2:
                            move = Move(player_clicks[0], player_clicks[1], game_state.board)
                            for i in range(len(valid_moves)):
                                if move == valid_moves[i]:
                                    game_state.make_move(valid_moves[i])
                                    num += 1
                                    print(f"{num}: {game_state.move_log[-1]}")
                                    move_made = True
                                    animate = True
                                    square_selected = ()
                                    player_clicks = []
                                    break
                            if not move_made:
                                player_clicks = [square_selected]

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_z:
                        if len(game_state.move_log) > 0:
                            for _ in range(2):
                                game_state.undo_move()
                            valid_moves = game_state.get_valid_moves()
                            square_selected = ()
                            player_clicks = []
                            move_made = False
                            animate = False
                    elif event.key == pygame.K_r:
                        game_state.__init__()
                        valid_moves = game_state.get_valid_moves()
                        square_selected = ()
                        player_clicks = []
                        move_made = False
                        animate = False

        if move_made:
            if animate:
                animate_move(game_state.move_log[-1], screen, game_state.board, clock, game_state, SQ_SIZE)
                animate = False
            valid_moves = game_state.get_valid_moves()
            move_made = False

        if game_state.checkmate or game_state.stalemate or game_state.stalemate_special():
            game_over = True
            if game_state.stalemate:
                text = "Stalemate"
            elif game_state.checkmate:
                text = "Black win" if game_state.white_to_move else "White win"
            promote_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            pygame.draw.rect(promote_surface, pygame.Color('grey'), pygame.Rect(0, 0, SQ_SIZE * 8, SQ_SIZE * 8))
            promote_surface.set_alpha(100)
            screen.blit(promote_surface, (0, 0))
            font = pygame.font.Font(None, SQ_SIZE // 2)
            text_surface = font.render(text, True, "white")
            text_rect = text_surface.get_rect(center=(SQ_SIZE * 9, SQ_SIZE // 2))
            screen.blit(text_surface, text_rect)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        game_state.__init__()
                        valid_moves = game_state.get_valid_moves()
                        square_selected = ()
                        player_clicks = []
                        move_made = False
                        animate = False
                        game_over = False

        clock.tick(60)
        pygame.display.flip()

# Main: Chạy chương trình
if __name__ == "__main__":
    play_game()
