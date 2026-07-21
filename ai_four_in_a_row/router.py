"""Connect 4 AI router — all endpoints for the Connect 4 game."""

import os
import threading
import time as _time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from .game import FourInARow, ROWS, COLS, PLAYER1, PLAYER2
from .inference import get_ai, reload_ai, get_ai_for_difficulty, DEFAULT_CHECKPOINT

# ---------------------------------------------------------------------------
# Lifespan: load model once on startup, watch for updates
# ---------------------------------------------------------------------------
_stop_event = threading.Event()


def _start_checkpoint_watcher() -> threading.Thread:
    def _watcher():
        last_mtime = 0.0
        while not _stop_event.is_set():
            try:
                mtime = os.path.getmtime(DEFAULT_CHECKPOINT)
                if mtime > last_mtime and last_mtime > 0:
                    print("[model watcher] New checkpoint detected — reloading...")
                    reload_ai()
                    print("[model watcher] Model reloaded.")
                last_mtime = mtime
            except FileNotFoundError:
                pass
            _stop_event.wait(timeout=10)

    t = threading.Thread(target=_watcher, daemon=True)
    t.start()
    return t


def startup() -> None:
    try:
        get_ai()
    except FileNotFoundError as exc:
        print(f"[WARNING] {exc}")

    _stop_event.clear()
    _start_checkpoint_watcher()


def shutdown() -> None:
    _stop_event.set()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(on_startup=[startup], on_shutdown=[shutdown])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class BoardState(BaseModel):
    """
    board: 6×7 grid, row 0 = top.
           0 = empty, 1 = Player 1 (human), 2 = Player 2 (AI).
    current_player: whose turn it is (1 or 2).
    use_mcts: set False for a fast greedy response (weaker).
    """

    board: List[List[int]] = Field(
        ...,
        examples=[[[0] * 7 for _ in range(6)]],
    )
    current_player: int = Field(1, ge=1, le=2)
    use_mcts: bool = True
    difficulty: str = "medium"  # 'easy' | 'medium' | 'hard'

    @field_validator("board")
    @classmethod
    def validate_board(cls, v):
        if len(v) != ROWS:
            raise ValueError(f"Board must have {ROWS} rows, got {len(v)}.")
        for i, row in enumerate(v):
            if len(row) != COLS:
                raise ValueError(f"Row {i} must have {COLS} columns, got {len(row)}.")
        return v


class MoveResponse(BaseModel):
    move: int
    probabilities: dict
    value: float
    game_over: bool
    winner: Optional[int]


class AiMoveResponse(BaseModel):
    column: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _board_from_request(data: BoardState) -> FourInARow:
    import numpy as np

    game = FourInARow()
    game.board = np.array(data.board, dtype=np.int8)
    game.current_player = data.current_player
    return game


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    try:
        ai = get_ai()
        loaded = ai is not None
    except Exception:
        loaded = False
    return {"status": "ok", "model_loaded": loaded}


@router.post("/get-move", response_model=MoveResponse, tags=["game"])
def get_move(data: BoardState):
    """
    Given a board state, return the AI's recommended move plus analysis.
    """
    try:
        ai = get_ai_for_difficulty(data.difficulty)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    game = _board_from_request(data)

    if game.is_terminal():
        raise HTTPException(status_code=400, detail="The game is already over.")

    valid_moves = game.get_valid_moves()
    if not valid_moves:
        raise HTTPException(status_code=400, detail="No valid moves available.")

    move = ai.best_move(game, use_mcts=True)
    probs = ai.move_probabilities(game) if data.use_mcts else {m: 0.0 for m in valid_moves}
    value = ai.evaluate_position(game)

    # Apply move to report post-move game state
    game_copy = game.copy()
    game_copy.make_move(move)

    return MoveResponse(
        move=move,
        probabilities=probs,
        value=round(value, 4),
        game_over=game_copy.is_terminal(),
        winner=game_copy.winner,
    )


@router.post("/valid-moves", tags=["game"])
def valid_moves(data: BoardState):
    """Return the list of columns that are legal to play."""
    game = _board_from_request(data)
    return {"valid_moves": game.get_valid_moves()}


@router.post("/evaluate", tags=["game"])
def evaluate(data: BoardState):
    """
    Return only the neural network's raw value estimate for the position,
    without running MCTS (very fast).
    """
    try:
        ai = get_ai()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    game = _board_from_request(data)
    value = ai.evaluate_position(game)
    return {"value": round(value, 4), "current_player": game.current_player}
