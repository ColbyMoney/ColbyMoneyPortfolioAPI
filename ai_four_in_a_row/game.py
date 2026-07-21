import numpy as np

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2


class FourInARow:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = PLAYER1
        self.game_over = False
        self.winner = None

    def copy(self):
        clone = FourInARow()
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.game_over = self.game_over
        clone.winner = self.winner
        return clone

    def get_valid_moves(self):
        return [col for col in range(COLS) if self.board[0][col] == EMPTY]

    def is_valid_move(self, col: int) -> bool:
        return 0 <= col < COLS and self.board[0][col] == EMPTY

    def drop_piece(self, col: int) -> int:
        """Drop a piece in the given column. Returns the row it landed in."""
        for row in range(ROWS - 1, -1, -1):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = self.current_player
                return row
        raise ValueError(f"Column {col} is full.")

    def make_move(self, col: int) -> bool:
        """Make a move for the current player. Returns True if the game ended."""
        if self.game_over or not self.is_valid_move(col):
            raise ValueError("Invalid move.")

        row = self.drop_piece(col)

        if self._check_win(row, col):
            self.winner = self.current_player
            self.game_over = True
        elif not self.get_valid_moves():
            self.game_over = True  # draw

        self.current_player = PLAYER2 if self.current_player == PLAYER1 else PLAYER1
        return self.game_over

    def _check_win(self, row: int, col: int) -> bool:
        player = self.board[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False

    def get_board_tensor(self):
        """
        Returns the board as a (3, ROWS, COLS) float32 tensor:
          channel 0 = current player's pieces
          channel 1 = opponent's pieces
          channel 2 = empty squares
        """
        opponent = PLAYER2 if self.current_player == PLAYER1 else PLAYER1
        planes = np.stack([
            (self.board == self.current_player).astype(np.float32),
            (self.board == opponent).astype(np.float32),
            (self.board == EMPTY).astype(np.float32),
        ])
        return planes

    def is_terminal(self) -> bool:
        return self.game_over

    def get_outcome(self, player: int) -> float:
        """Return +1 win, -1 loss, 0 draw from the perspective of `player`."""
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def __repr__(self):
        symbols = {EMPTY: ".", PLAYER1: "X", PLAYER2: "O"}
        rows = ["  " + " ".join(str(c) for c in range(COLS))]
        for r in range(ROWS):
            rows.append(f"{r} " + " ".join(symbols[self.board[r][c]] for c in range(COLS)))
        return "\n".join(rows)
