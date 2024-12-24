if move.is_pawn_promotion:
    # Phong cấp dựa trên lựa chọn
    if hasattr(move, "promoted_to") and move.promoted_to in ["Q", "R", "B", "N"]:
        promoted_piece = (
            f"w{move.promoted_to}" if self.white_to_move else f"b{move.promoted_to}"
        )
    else:
        promoted_piece = "wQ" if self.white_to_move else "bQ"  # Mặc định là hậu
    self.board[move.end_row][move.end_column] = promoted_piece
