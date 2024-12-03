class GameState:
    """
    Lớp chịu trách nhiệm lưu trữ các thông tin cơ bản của trò chơi.
    Các hàm trong lớp chịu trách nhiệm về:
        - Di chuyển quân cờ
        - Hoàn tác nước đi
        - Xác định các nước đi hợp lệ
        - Nhập kí nước đi
    """

    def __init__(self):
        self.board = [
            ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR'],
            ['bP', 'bP', 'bP', 'bP', 'bP', 'bP', 'bP', 'bP'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['wP', 'wP', 'wP', 'wP', 'wP', 'wP', 'wP', 'wP'],
            ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR']]
        self.move_functions = {'P': self.get_pawn_moves, 'R': self.get_rook_moves, 'N': self.get_knight_moves,
                               'B': self.get_bishop_moves, 'Q': self.get_queen_moves, 'K': self.get_king_moves}
        self.white_to_move = True
        self.move_log = []
        self.white_king_location = (7, 4)
        self.black_king_location = (0, 4)
        self.checkmate = False
        self.stalemate = False
        self.in_check = False
        self.pins = []
        self.checks = []
        self.white_captured_pieces = []
        self.black_captured_pieces = []
        self.castle_move = []
        self.state_count = {}
        self.fifty_move_counter = 0

        # Bắt tốt qua đường
        self.en_passant_possible = ()
        self.en_passant_possible_log = [self.en_passant_possible]

        # Nhập thành
        self.white_castle_king_side = True
        self.white_castle_queen_side = True
        self.black_castle_king_side = True
        self.black_castle_queen_side = True
        self.castle_rights_log = [CastleRights(self.white_castle_king_side, self.black_castle_king_side,
                                               self.white_castle_queen_side, self.black_castle_queen_side)]

    def make_move(self, move):
        """Thực hiện một di chuyển làm tham số, thực thi nó và cập nhật nhật ký di chuyển"""
        global promoted_piece

        self.board[move.start_row][move.start_column] = '--'  # Làm trống ô bắt đầu
        self.board[move.end_row][move.end_column] = move.piece_moved  # Di chuyển mảnh đến vị trí mới
        self.move_log.append(move)  # Nhật ký di chuyển


        if move.piece_moved == 'wK':
            self.white_king_location = (move.end_row, move.end_column)
        elif move.piece_moved == 'bK':
            self.black_king_location = (move.end_row, move.end_column)

        # Phong cấp cho quân tốt
        if move.is_pawn_promotion:
            # AI luôn chọn phong cấp thành Hậu
            promoted_piece = 'wQ' if self.white_to_move else 'bQ'
            self.board[move.end_row][move.end_column] = promoted_piece

        # Bắt tốt qua đường
        if move.is_en_passant_move:
            self.board[move.start_row][move.end_column] = '--'

        # Cập nhật biến en_passant_posible
        if move.piece_moved[1] == 'P' and abs(
                move.start_row - move.end_row) == 2:  # Chỉ có giá trị cho quân tốt di chuyển 2 ô
            self.en_passant_possible = ((move.start_row + move.end_row) // 2, move.start_column)
        else:
            self.en_passant_possible = ()

        self.en_passant_possible_log.append(self.en_passant_possible)

        # Nhập thành
        if move.is_castle_move:
            if move.end_column - move.start_column == 2:  # Nhập thành phía vua
                self.board[move.end_row][move.end_column - 1] = self.board[move.end_row][
                    move.end_column + 1]  # di chuyển xe
                self.board[move.end_row][move.end_column + 1] = '--'  # xóa xe cũ
            else:  # Nhập thành phía hậu
                self.board[move.end_row][move.end_column + 1] = self.board[move.end_row][
                    move.end_column - 2]
                self.board[move.end_row][move.end_column - 2] = '--'
            self.castle_move = True
        else:
            self.castle_move = False

        # Cập nhật quyền đúc
        self.update_castle_rights(move)
        self.castle_rights_log.append(CastleRights(self.white_castle_king_side, self.black_castle_king_side,
                                                   self.white_castle_queen_side, self.black_castle_queen_side))

        self.white_to_move = not self.white_to_move  # Biến các công tắc

        # Ăn quân cờ
        if move.piece_captured != '--':
            if move.piece_captured[0] == 'w':
                self.white_captured_pieces.append(move.piece_captured)
            elif move.piece_captured[0] == 'b':
                self.black_captured_pieces.append(move.piece_captured)

        # Kiểm tra với luận 50 di chuyển
        if move.piece_moved[1] == 'P' or move.piece_captured != '--':
            self.fifty_move_counter = 0
        else:
            self.fifty_move_counter += 1

        # Ghi nhận trạng thái sau nước đi
        state_key = self._current_state_key()
        self.state_count[state_key] = self.state_count.get(state_key, 0) + 1

        valid_moves = self.get_valid_moves()
        if not valid_moves:
            # Kiểm tra chiếu hết hoặc hòa cờ
            self.checkmate = True if self.in_check else False
            move.is_checkmate = self.checkmate
        else:
            # Kiểm tra trạng thái chiếu
            self.in_check, _, _ = self.check_for_pins_and_checks()
            move.is_check = self.in_check

        #print(f"Move: {move.start_row}, {move.start_column}, {move.end_row}, {move.end_column}")

    def undo_move(self):
        """Hoàn tác nước đi"""
        if not self.move_log:
            return  # Không có nước đi để hoàn tác

        move = self.move_log.pop()
        self.board[move.start_row][move.start_column] = move.piece_moved
        self.board[move.end_row][move.end_column] = move.piece_captured
        self.white_to_move = not self.white_to_move  # Đổi lượt

        # Cập nhật vị trí vua
        if move.piece_moved == 'wK':
            self.white_king_location = (move.start_row, move.start_column)
        elif move.piece_moved == 'bK':
            self.black_king_location = (move.start_row, move.start_column)

        # Bắt tốt qua đường
        if move.is_en_passant_move:
            self.board[move.end_row][move.end_column] = '--'
            self.board[move.start_row][move.end_column] = move.piece_captured

        if self.en_passant_possible_log:
            self.en_passant_possible_log.pop()
            self.en_passant_possible = self.en_passant_possible_log[-1] if self.en_passant_possible_log else None

        # Quyền đúc
        if self.castle_rights_log:
            self.castle_rights_log.pop()
            castle_rights = self.castle_rights_log[-1] if self.castle_rights_log else None
            if castle_rights:
                self.white_castle_king_side = castle_rights.white_king_side
                self.black_castle_king_side = castle_rights.black_king_side
                self.white_castle_queen_side = castle_rights.white_queen_side
                self.black_castle_queen_side = castle_rights.black_queen_side

        # Nhập thành
        if move.is_castle_move:
            if move.end_column - move.start_column == 2:  # Phía vua
                self.board[move.end_row][move.end_column + 1] = self.board[move.end_row][move.end_column - 1]
                self.board[move.end_row][move.end_column - 1] = '--'
            else:  # Phía hậu
                self.board[move.end_row][move.end_column - 2] = self.board[move.end_row][move.end_column + 1]
                self.board[move.end_row][move.end_column + 1] = '--'

        # Khôi phục quân cờ bị ăn
        if move.piece_captured:
            if move.piece_captured[0] == 'w':
                self.white_captured_pieces.remove(move.piece_captured)
            elif move.piece_captured[0] == 'b':
                self.black_captured_pieces.remove(move.piece_captured)

        self.checkmate = False
        self.stalemate = False

    def get_valid_moves(self):
        """Nhận tất cả các động tác xem xét séc"""
        valid_moves = []
        self.in_check, self.pins, self.checks = self.check_for_pins_and_checks()

        # Cập nhật vị trí vua
        if self.white_to_move:
            king_row, king_column = self.white_king_location[0], self.white_king_location[1]
        else:
            king_row, king_column = self.black_king_location[0], self.black_king_location[1]

        if self.in_check:
            if len(self.checks) == 1:  # Chỉ có 1 Kiểm tra: Kiểm tra khối hoặc Di chuyển King
                valid_moves = self.get_all_possible_moves()
                check = self.checks[0]
                check_row, check_column = check[0], check[1]
                piece_checking = self.board[check_row][check_column]  # Mảnh kẻ thù gây ra séc
                valid_squares = []
                if piece_checking == 'N':
                    valid_squares = [(check_row, check_column)]
                else:
                    for i in range(1, len(self.board)):
                        valid_square = (
                        king_row + check[2] * i, king_column + check[3] * i)  # 2 và 3 = Hướng dẫn kiểm tra
                        valid_squares.append(valid_square)
                        if valid_square[0] == check_row and valid_square[1] == check_column:
                            break
                for i in range(len(valid_moves) - 1, -1,
                               -1):  # Loại bỏ việc di chuyển không chặn, kiểm tra hoặc di chuyển vua
                    if valid_moves[i].piece_moved[1] != 'K':
                        if not (valid_moves[i].end_row, valid_moves[i].end_column) in valid_squares:
                            valid_moves.remove(valid_moves[i])
            else:  # Kiểm tra gấp đôi, King phải di chuyển
                self.get_king_moves(king_row, king_column, valid_moves)
        else:  # Không kiểm tra
            valid_moves = self.get_all_possible_moves()

        if len(valid_moves) == 0:  # Kiểm tra thắng hay hòa
            if self.in_check:
                self.checkmate = True
            else:
                self.stalemate = True
        else:
            self.checkmate = False
            self.stalemate = False

        return valid_moves

    def get_all_possible_moves(self):
        """Có tất cả các động tác mà không cần xem xét séc"""
        moves = []
        for row in range(len(self.board)):  # Số lượng hàng
            for column in range(len(self.board[row])):  # Số lượng cột trong mỗi hàng
                turn = self.board[row][column][0]
                if (turn == 'w' and self.white_to_move) or (turn == 'b' and not self.white_to_move):
                    piece = self.board[row][column][1]
                    self.move_functions[piece](row, column, moves)  # Cuộc gọi chức năng di chuyển dựa trên mảnh loại
        return moves

    def get_pawn_moves(self, row, column, moves):
        """Nhận tất cả các động tác cầm đồ cho quân tốt nằm ở (hàng, cột) và thêm các động tác để di chuyển nhật ký"""
        piece_pinned = False
        pin_direction = ()
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == row and self.pins[i][1] == column:
                piece_pinned = True
                pin_direction = (self.pins[i][2], self.pins[i][3])
                self.pins.remove(self.pins[i])
                break

        if self.white_to_move:
            move_amount = -1
            start_row = 6
            back_row = 0
            opponent = 'b'
            king_row, king_column = self.white_king_location
        else:
            move_amount = 1
            start_row = 1
            back_row = 7
            opponent = 'w'
            king_row, king_column = self.black_king_location
        pawn_promotion = False

        if self.board[row + move_amount][column] == '--':  # Di chuyển 1 ô
            if not piece_pinned or pin_direction == (move_amount, 0):
                if row + move_amount == back_row:  # Nếu mảnh được xếp hạng trở lại, đó là một chương trình khuyến mãi cầm đồ
                    pawn_promotion = True
                moves.append(
                    Move((row, column), (row + move_amount, column), self.board, pawn_promotion=pawn_promotion))
                if row == start_row and self.board[row + 2 * move_amount][column] == '--':  # Di chuyển 2 ô
                    moves.append(Move((row, column), (row + 2 * move_amount, column), self.board))
        if column - 1 >= 0:  # Ăn chéo
            if not piece_pinned or pin_direction == (move_amount, -1):
                if self.board[row + move_amount][column - 1][0] == opponent:
                    if row + move_amount == back_row:  # Nếu mảnh được xếp hạng trở lại, đó là một chương trình khuyến mãi cầm đồ
                        pawn_promotion = True
                    moves.append(Move((row, column), (row + move_amount, column - 1),
                                      self.board, pawn_promotion=pawn_promotion))
                if (row + move_amount, column - 1) == self.en_passant_possible:
                    attacking_piece = blocking_piece = False
                    if king_row == row:
                        if king_column < column:  # Vua còn lại của quân tốt
                            # Inside_range giữa vua và cầm đồ; Bên ngoài_range giữa cầm đồ và biên giới
                            inside_range = range(king_column + 1, column - 1)
                            outside_range = range(column + 1, len(self.board))
                        else:  # Vua là đúng của quân tốt
                            inside_range = range(king_column - 1, column, -1)
                            outside_range = range(column - 2, -1, -1)
                        for i in inside_range:
                            if self.board[row][i] != '--':
                                blocking_piece = True
                        for i in outside_range:
                            square = self.board[row][i]
                            if square[0] == opponent and (square[1] == 'R' or square[1] == 'Q'):
                                attacking_piece = True
                            elif square != '--':
                                blocking_piece = True
                    if not attacking_piece or blocking_piece:
                        moves.append(Move((row, column), (row + move_amount, column - 1), self.board, en_passant=True))
        if column + 1 <= len(self.board) - 1:  # Bắt đúng
            if not piece_pinned or pin_direction == (move_amount, 1):
                if self.board[row + move_amount][column + 1][0] == opponent:
                    if row + move_amount == back_row:  # Nếu mảnh được xếp hạng trở lại, đó là một chương trình khuyến mãi cầm đồ
                        pawn_promotion = True
                    moves.append(Move((row, column), (row + move_amount, column + 1),
                                      self.board, pawn_promotion=pawn_promotion))
                if (row + move_amount, column + 1) == self.en_passant_possible:
                    attacking_piece = blocking_piece = False
                    if king_row == row:
                        if king_column < column:  # Vua còn lại của quân tốt
                            # Inside_range giữa vua và cầm đồ; Bên ngoài_range giữa cầm đồ và biên giới
                            inside_range = range(king_column + 1, column)
                            outside_range = range(column + 2, len(self.board))
                        else:  # Vua là đúng của quân tốt
                            inside_range = range(king_column - 1, column + 1, -1)
                            outside_range = range(column - 1, -1, -1)
                        for i in inside_range:
                            if self.board[row][i] != '--':
                                blocking_piece = True
                        for i in outside_range:
                            square = self.board[row][i]
                            if square[0] == opponent and (square[1] == 'R' or square[1] == 'Q'):
                                attacking_piece = True
                            elif square != '--':
                                blocking_piece = True
                    if not attacking_piece or blocking_piece:
                        moves.append(Move((row, column), (row + move_amount, column + 1), self.board, en_passant=True))

    def get_rook_moves(self, row, column, moves):
        """Nhận tất cả các di chuyển quân xe cho quân xe nằm ở (hàng, cột) và thêm các di chuyển để di chuyển nhật ký"""
        opponent = 'b' if self.white_to_move else 'w'

        piece_pinned = False
        pin_direction = ()
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == row and self.pins[i][1] == column:
                piece_pinned = True
                pin_direction = (self.pins[i][2], self.pins[i][3])
                if self.board[row][column][
                    1] != 'Q':  # Không thể loại bỏ nữ hoàng khỏi pin trên các động tác quân xe (chỉ di chuyển của giám mục)
                    self.pins.remove(self.pins[i])
                break

        directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # Tuples chỉ ra các chuyển động (hàng, cột) có thể
        for d in directions:
            for i in range(1, len(self.board)):
                end_row = row + d[0] * i  # Di chuyển mạnh lên/xuống 7 hàng
                end_column = column + d[1] * i  # Di chuyển mạnh lên/xuống 7 cột
                if 0 <= end_row < len(self.board) and 0 <= end_column < len(self.board):  # Đảm bảo trên bảng
                    if not piece_pinned or pin_direction == d or pin_direction == (-d[0], -d[1]):
                        end_piece = self.board[end_row][end_column]
                        if end_piece == '--':  # Di chuyển hợp lệ đến không gian trống
                            moves.append(Move((row, column), (end_row, end_column), self.board))
                        elif end_piece[0] == opponent:  # Di chuyển hợp lệ để bắt giữ
                            moves.append(Move((row, column), (end_row, end_column), self.board))
                            break
                        else:  # Không thể lấy mảnh thân thiện
                            break
                else:  # Không thể di chuyển khỏi bảng
                    break

    def get_knight_moves(self, row, column, moves):
        """Nhận tất cả các di chuyển quân tươợng cho quân tượng nằm ở (hàng, cột) và thêm các động tác để di chuyển nhật ký"""
        opponent = 'b' if self.white_to_move else 'w'

        piece_pinned = False
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == row and self.pins[i][1] == column:
                piece_pinned = True
                self.pins.remove(self.pins[i])
                break

        # Tuples chỉ ra các chuyển động (hàng, cột) có thể
        directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

        for d in directions:
            end_row = row + d[0]
            end_column = column + d[1]
            if 0 <= end_row < len(self.board) and 0 <= end_column < len(self.board):  # Đảm bảo trên bảng
                if not piece_pinned:
                    end_piece = self.board[end_row][end_column]
                    if end_piece[0] == opponent:  # Di chuyển hợp lệ để bắt giữ
                        moves.append(Move((row, column), (end_row, end_column), self.board))
                    elif end_piece == '--':  # Di chuyển hợp lệ đến không gian trống
                        moves.append(Move((row, column), (end_row, end_column), self.board))

    def get_bishop_moves(self, row, column, moves):
        """Nhận tất cả các di chuyển của quân mã cho quân mã nằm ở (hàng, cột) và thêm các động tác để di chuyển nhật ký"""
        opponent = 'b' if self.white_to_move else 'w'

        piece_pinned = False
        pin_direction = ()
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == row and self.pins[i][1] == column:
                piece_pinned = True
                pin_direction = (self.pins[i][2], self.pins[i][3])
                self.pins.remove(self.pins[i])
                break

        directions = [(-1, -1), (-1, 1), (1, 1), (1, -1)]  # Tuples chỉ ra các chuyển động (hàng, cột) có thể
        for d in directions:
            for i in range(1, len(self.board)):

                # Xem get_rook_moves để giải thích; tương tự như vậy nhưng cho các đường chéo
                end_row = row + d[0] * i
                end_column = column + d[1] * i

                if 0 <= end_row < len(self.board) and 0 <= end_column < len(self.board):  # Đảm bảo trên bảng
                    if not piece_pinned or pin_direction == d or pin_direction == (-d[0], -d[1]):
                        end_piece = self.board[end_row][end_column]
                        if end_piece == '--':  # Di chuyển hợp lệ đến không gian trống
                            moves.append(Move((row, column), (end_row, end_column), self.board))
                        elif end_piece[0] == opponent:  # Di chuyển hợp lệ để bắt giữ
                            moves.append(Move((row, column), (end_row, end_column), self.board))
                            break
                        else:  # Không thể lấy mảnh thân thiện
                            break
                else:  # Không thể di chuyển khỏi bảng
                    break

    def get_queen_moves(self, row, column, moves):
        """Nhận tất cả các động thái của quân hậu cho quân hậu nằm ở (hàng, cột) và thêm các động tác để di chuyển nhật ký"""
        self.get_bishop_moves(row, column, moves)
        self.get_rook_moves(row, column, moves)

    def get_king_moves(self, row, column, moves):
        """Nhận tất cả các động thái của quân vua cho quân vua nằm ở (hàng, cột) và thêm các động tác để di chuyển nhật ký"""
        ally = 'w' if self.white_to_move else 'b'
        row_moves = (-1, -1, -1, 0, 0, 1, 1, 1)
        column_moves = (-1, 0, 1, -1, 1, -1, 0, 1)
        for i in range(len(self.board)):
            end_row = row + row_moves[i]
            end_column = column + column_moves[i]
            if 0 <= end_row < len(self.board) and 0 <= end_column < len(self.board):  # Đảm bảo trên bảng
                end_piece = self.board[end_row][end_column]
                if end_piece[0] != ally:  # Mảnh trống hoặc kẻ thù

                    # Vua trên quảng trường và kiểm tra séc
                    if ally == 'w':
                        self.white_king_location = (end_row, end_column)
                    else:
                        self.black_king_location = (end_row, end_column)
                    in_check, pins, checks = self.check_for_pins_and_checks()
                    if not in_check:
                        moves.append(Move((row, column), (end_row, end_column), self.board))

                    # Vua trở lại cho thuê ban đầu
                    if ally == 'w':
                        self.white_king_location = (row, column)
                    else:
                        self.black_king_location = (row, column)
        self.get_castle_moves(row, column, moves, ally)

    def get_castle_moves(self, row, column, moves, ally):
        """
        Tạo tất cả các di chuyển nhập thành hợp lệ cho quân vua tại (hàng, cột).
        Thêm di chuyển nhập thành hợp lệ vào danh sách các di chuyển.
        """
        if self.square_under_attack(row, column, ally):
            return  # Không thể nhập thành khi đang bị chiếu
        if (self.white_to_move and self.white_castle_king_side) or \
                (not self.white_to_move and self.black_castle_king_side):
            self.get_king_side_castle_moves(row, column, moves, ally)
        if (self.white_to_move and self.white_castle_queen_side) or \
                (not self.white_to_move and self.black_castle_queen_side):
            self.get_queen_side_castle_moves(row, column, moves, ally)

    def get_king_side_castle_moves(self, row, column, moves, ally):
        """Nhập thành phía vua"""
        if self.board[row][column + 1] == '--' and self.board[row][column + 2] == '--' and \
                not self.square_under_attack(row, column + 1, ally) and not self.square_under_attack(row, column + 2,
                                                                                                     ally):
            moves.append(Move((row, column), (row, column + 2), self.board, castle=True))

    def get_queen_side_castle_moves(self, row, column, moves, ally):
        """Nhập thành phía hậu"""
        if self.board[row][column - 1] == '--' and self.board[row][column - 2] == '--' and \
                self.board[row][column - 3] == '--' and not self.square_under_attack(row, column - 1, ally) and \
                not self.square_under_attack(row, column - 2, ally):
            moves.append(Move((row, column), (row, column - 2), self.board, castle=True))

    def update_castle_rights(self, move):
        """Cập nhật quyền nhập thành được đưa ra để di chuyển"""
        # Nếu quân vua hay quân xe di chuyển
        if move.piece_moved == 'wK':
            self.white_castle_queen_side = False
            self.white_castle_king_side = False
        elif move.piece_moved == 'bK':
            self.black_castle_queen_side = False
            self.black_castle_king_side = False
        elif move.piece_moved == 'wR':
            if move.start_row == 7:
                if move.start_column == 7:
                    self.white_castle_king_side = False
                elif move.start_column == 0:
                    self.white_castle_queen_side = False
        elif move.piece_moved == 'bR':
            if move.start_row == 0:
                if move.start_column == 7:
                    self.black_castle_king_side = False
                elif move.start_column == 0:
                    self.black_castle_queen_side = False

        # Nếu xe bị bắt
        if move.piece_captured == 'wR':
            if move.end_row == 7:
                if move.end_column == 0:
                    self.white_castle_queen_side = False
                elif move.end_column == 7:
                    self.white_castle_king_side = False
        elif move.piece_captured == 'bR':
            if move.end_row == 0:
                if move.end_column == 0:
                    self.black_castle_queen_side = False
                elif move.end_column == 7:
                    self.black_castle_king_side = False

    def square_under_attack(self, row, column, ally):
        """Kiểm tra bên ngoài từ một hình vuông để xem nó có bị tấn công hay không, do đó làm mất hiệu lực Castling"""
        opponent = 'b' if self.white_to_move else 'w'
        directions = ((-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
        for j in range(len(directions)):
            d = directions[j]
            for i in range(1, len(self.board)):
                end_row = row + d[0] * i
                end_column = column + d[1] * i
                if 0 <= end_row < len(self.board) and 0 <= end_column < len(self.board):
                    end_piece = self.board[end_row][end_column]
                    if end_piece[0] == ally:  # Không có cuộc tấn công từ hướng đó
                        break
                    elif end_piece[0] == opponent:
                        piece_type = end_piece[1]
                        if (0 <= j <= 3 and piece_type == 'R') or (4 <= j <= 7 and piece_type == 'B') or \
                                (i == 1 and piece_type == 'P' and ((opponent == 'w' and 6 <= j <= 7)
                                                                   or (opponent == 'b' and 4 <= j <= 5))) or \
                                (piece_type == 'Q') or (i == 1 and piece_type == 'K'):
                            return True
                        else:  # Mảnh kẻ thù nhưng không áp dụng kiểm tra
                            break
                else:  # TẮT
                    break
        knight_moves = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
        for move in knight_moves:
            end_row = row + move[0]
            end_column = column + move[1]
            if 0 <= end_row < len(self.board) and 0 <= end_column < len(self.board):
                end_piece = self.board[end_row][end_column]
                if end_piece[0] == opponent and end_piece[1] == 'N':
                    return True
        return False

    def check_for_pins_and_checks(self):
        """
        Trả về trạng thái vua có đang bị chiếu (check)
        Danh sách các quân bị ghim (pin)
        Danh sách các nước chiếu (check)"""
        pins = []
        checks = []
        in_check = False

        if self.white_to_move:
            opponent = 'b'  # Quân đối thủ là đen
            ally = 'w'  # Quân đồng minh là trắng
            start_row, start_column = self.white_king_location[0], self.white_king_location[1]  # Vị trí của vua trắng
        else:
            opponent = 'w'  # Quân đối thủ là trắng
            ally = 'b'  # Quân đồng minh là đen
            start_row, start_column = self.black_king_location[0], self.black_king_location[1]  # Vị trí của vua đen

        # Các hướng đi để kiểm tra nước chiếu và ghim
        directions = ((-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
        for j in range(len(directions)):
            d = directions[j]
            possible_pin = ()  # Khởi tạo ghim có thể xảy ra
            for i in range(1, len(self.board)):
                end_row = start_row + d[0] * i
                end_column = start_column + d[1] * i
                if 0 <= end_row < len(self.board) and 0 <= end_column < len(
                        self.board):  # Kiểm tra vị trí trong bảng cờ
                    end_piece = self.board[end_row][end_column]
                    if end_piece[0] == ally and end_piece[1] != 'K':
                        if possible_pin == ():  # Quân đồng minh đầu tiên có thể bị ghim
                            possible_pin = (end_row, end_column, d[0], d[1])
                        else:  # Gặp quân đồng minh thứ hai, không thể ghim hoặc chiếu
                            break
                    elif end_piece[0] == opponent:
                        piece_type = end_piece[1]
                        # Kiểm tra nếu là quân xe, tượng, hậu, tốt hoặc vua đối phương đang chiếu
                        if (0 <= j <= 3 and piece_type == 'R') or (4 <= j <= 7 and piece_type == 'B') or \
                                (i == 1 and piece_type == 'P' and ((opponent == 'w' and 6 <= j <= 7) or
                                                                   (opponent == 'b' and 4 <= j <= 5))) or (
                                piece_type == 'Q') or (i == 1 and piece_type == 'K'):
                            if possible_pin == ():  # Không có quân chặn, vua đang bị chiếu
                                in_check = True
                                checks.append((end_row, end_column, d[0], d[1]))
                                break
                            else:  # Có quân chặn, ghim quân đồng minh
                                pins.append(possible_pin)
                                break
                        else:  # Quân đối phương nhưng không có chiếu
                            break
                else:  # Ra ngoài bảng cờ
                    break

        # Kiểm tra nước đi của quân mã
        knight_moves = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
        for move in knight_moves:
            end_row = start_row + move[0]
            end_column = start_column + move[1]
            if 0 <= end_row < len(self.board) and 0 <= end_column < len(self.board):  # Kiểm tra trong bảng cờ
                end_piece = self.board[end_row][end_column]
                if end_piece[0] == opponent and end_piece[1] == 'N':  # Nếu quân mã của đối phương đang chiếu vua
                    in_check = True
                    checks.append((end_row, end_column, move[0], move[1]))
        return in_check, pins, checks  # Trả về kết quả: vua có bị chiếu, danh sách ghim, danh sách chiếu

    def find_king(self, white_to_move):
        """Tìm vị trí của quân vua dựa trên màu sắc"""
        king = 'wK' if white_to_move else 'bK'
        for row in range(len(self.board)):
            for column in range(len(self.board[row])):
                if self.board[row][column] == king:
                    return row, column
        return None

    def insufficient_material(self):
        """Kiểm tra cờ hòa do thế cờ chết"""
        white_pieces = []
        black_pieces = []

        for row in self.board:
            for piece in row:
                if piece != '--':
                    if piece[0] == 'w':
                        white_pieces.append(piece[1])
                    else:
                        black_pieces.append(piece[1])

        if (len(white_pieces) == 1 and white_pieces[0] == 'K' and
            len(black_pieces) == 1 and black_pieces[0] == 'K'):
            return True
        if white_pieces == (['K', 'N'] or ['K', 'B']) and black_pieces == (['K', 'N'] or ['K', 'B']):
            return True

        return False

    def pat(self):
        """Kiểm tra hòa do hết nước đi"""
        if not self.in_check:
            valid_moves = self.get_valid_moves()
            if len(valid_moves) == 0:
                return True
        return False

    def _current_state_key(self):
        """Trả về khóa trạng thái đầy đủ của bàn cờ."""
        board_hash = ''.join(''.join(row) for row in self.board)
        castle_rights = f"{self.white_castle_king_side}_{self.white_castle_queen_side}_{self.black_castle_king_side}_{self.black_castle_queen_side}"
        en_passant = f"{self.en_passant_possible}"  # Trạng thái en passant
        turn = "w" if self.white_to_move else "b"  # Trắng hoặc đen
        state_key = f"{board_hash}_{castle_rights}_{en_passant}_{turn}"
        return state_key

    def threefold_repetition(self):
        """Kiểm tra lặp lại ba lần."""
        if len(self.move_log) < 3:
            return False

        state_key = self._current_state_key()
        return self.state_count.get(state_key, 0) >= 3

    def stalemate_special(self):
        """Kiểm tra các điều kiện hòa đặc biệt."""
        if self.insufficient_material():
            print("Hòa do thế cờ chết.")
            self.stalemate = True
            return True

        if self.pat():
            print("Hòa do hết nước đi.")
            self.stalemate = True
            return True

        if self.threefold_repetition():
            print("Hòa do lặp lại ba lần.")
            self.stalemate = True
            return True

        if self.fifty_move_counter >= 100:
            print("Hòa do quy tắc 50 nước.")
            self.stalemate = True
            return True

        return False

class CastleRights:
    """Lưu trữ dữ liệu về các trạng thái hiện tại của nhập thành"""

    def __init__(self, white_king_side, black_king_side, white_queen_side, black_queen_side):
        self.white_king_side = white_king_side
        self.black_king_side = black_king_side
        self.white_queen_side = white_queen_side
        self.black_queen_side = black_queen_side

class Move:
    """
    Lớp chịu trách nhiệm lưu trữ thông tin về các động thái cụ thể,
    Bao gồm các vị trí bắt đầu và kết thúc, mà các mảnh được và bị bắt,
    Và các động tác đặc biệt như nhân tiện, quảng cáo cầm đồ và đúc.
    """
    ranks_to_rows = {'1': 7, '2': 6, '3': 5, '4': 4,
                     '5': 3, '6': 2, '7': 1, '8': 0}
    rows_to_ranks = {v: k for k, v in ranks_to_rows.items()}
    files_to_columns = {'a': 0, 'b': 1, 'c': 2, 'd': 3,
                        'e': 4, 'f': 5, 'g': 6, 'h': 7}
    columns_to_files = {v: k for k, v in files_to_columns.items()}

    def __init__(self, start_square, end_square, board, en_passant=False, pawn_promotion=False, castle=False):
        self.start_row, self.start_column = start_square[0], start_square[1]
        self.end_row, self.end_column = end_square[0], end_square[1]
        self.piece_moved = board[self.start_row][self.start_column]
        self.piece_captured = board[self.end_row][self.end_column]
        self.is_pawn_promotion = pawn_promotion

        self.start_row = start_square[0]
        self.start_col = start_square[1]
        self.end_row = end_square[0]
        self.end_col = end_square[1]

        # En passant
        self.is_en_passant_move = en_passant
        if self.is_en_passant_move:
            self.piece_captured = 'wP' if self.piece_moved == 'bP' else 'bP'

        self.is_castle_move = castle
        self.is_capture = self.piece_captured != '--'
        self.move_id = self.start_row * 1000 + self.start_column * 100 + self.end_row * 10 + self.end_column

        self.is_check = False
        self.is_checkmate = False

    def __eq__(self, other):
        """Ghi đè phương thức bằng vì"""
        if isinstance(other, Move):
            return self.move_id == other.move_id
        return False

    def get_chess_notation(self):
        """Tạo ra một ký hiệu cờ vua và tập tin"""
        return self.get_rank_file(self.start_row, self.start_column) + self.get_rank_file(self.end_row, self.end_column)

    def get_rank_file(self, rank, column):
        return self.columns_to_files[column] + self.rows_to_ranks[rank]

    def __str__(self):
        """
        Chức năng Chuỗi ghi đè để cải thiện ký hiệu cờ vua.
        Không 1) séc cụ thể, 2) Checkmate, Gold 3) Khi
        Nhiều mảnh cùng loại có thể chụp cùng một hình vuông.
        """
        # Nhập thành
        if self.is_castle_move:
            return 'O-O' if self.end_column == 6 else 'O-O-O'

        end_square = self.get_rank_file(self.end_row, self.end_column)
        if self.is_check:
            end_square += '+'
        elif self.is_checkmate:
            end_square += '#'

        # Quân tốt di chuyển
        if self.piece_moved[1] == 'P':
            capture_indicator = 'x' if self.is_capture else ''
            promotion_indicator = '=' if self.is_pawn_promotion else ''
            promoted_piece_new = promoted_piece[-1] if self.is_pawn_promotion else ''
            return f'{capture_indicator}{end_square}{promotion_indicator}{promoted_piece_new}'

        # Các quân còn lại
        move_string = self.piece_moved[1]
        if self.is_capture:
            move_string += 'x'

        return f'{move_string}{end_square}'

