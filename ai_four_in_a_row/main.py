"""FastAPI application — Connect 4 AI endpoints.

Start the server:
    uvicorn main:app --reload
"""

import os
import threading
import time as _time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .game import Connect4, ROWS, COLS, PLAYER1, PLAYER2
from .inference import get_ai, reload_ai, get_ai_for_difficulty, load_all_models, DEFAULT_CHECKPOINT

# ---------------------------------------------------------------------------
# Lifespan: load model once on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all three difficulty models at startup so every request is served
    # from pre-warmed weights with no first-request latency.
    load_all_models()

    # Background thread: reload the model whenever the checkpoint file changes.
    # Training saves a new checkpoint after every iteration, so the API will
    # automatically serve the latest weights within ~10 seconds.
    stop_event = threading.Event()

    def _checkpoint_watcher():
        last_mtime = 0.0
        while not stop_event.is_set():
            try:
                mtime = os.path.getmtime(DEFAULT_CHECKPOINT)
                if mtime > last_mtime and last_mtime > 0:
                    print("[model watcher] New checkpoint detected — reloading...")
                    reload_ai()
                    print("[model watcher] Model reloaded.")
                last_mtime = mtime
            except FileNotFoundError:
                pass
            stop_event.wait(timeout=10)

    watcher = threading.Thread(target=_checkpoint_watcher, daemon=True)
    watcher.start()

    yield

    stop_event.set()


app = FastAPI(
    title="AI Four-in-a-Row API",
    description="Connect 4 powered by a self-play trained neural network + MCTS.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "https://colbymoney.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
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
            for cell in row:
                if cell not in (0, 1, 2):
                    raise ValueError("Cell values must be 0, 1, or 2.")
        return v


class MoveResponse(BaseModel):
    move: int = Field(..., description="Recommended column (0-indexed).")
    probabilities: dict[int, float] = Field(
        ..., description="MCTS visit-count probabilities for each valid column."
    )
    value: float = Field(
        ..., description="Position evaluation (+1 AI winning, -1 AI losing)."
    )
    game_over: bool
    winner: Optional[int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# Schema matched to the Angular AiService contract
class AiMoveRequest(BaseModel):
    board: List[List[int]]
    player: int = Field(..., ge=1, le=2)

    @field_validator("board")
    @classmethod
    def validate_board(cls, v):
        if len(v) != ROWS:
            raise ValueError(f"Board must have {ROWS} rows, got {len(v)}.")
        for i, row in enumerate(v):
            if len(row) != COLS:
                raise ValueError(f"Row {i} must have {COLS} columns.")
        return v


class AiMoveResponse(BaseModel):
    column: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _board_from_request(data: BoardState) -> Connect4:
    import numpy as np

    game = Connect4()
    game.board = np.array(data.board, dtype=np.int8)
    game.current_player = data.current_player
    return game


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    try:
        ai = get_ai()
        loaded = ai is not None
    except Exception:
        loaded = False
    return {"status": "ok", "model_loaded": loaded}


@app.post("/api/ai-four-in-a-row/get-move", response_model=MoveResponse, tags=["game"])
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


@app.post("/valid-moves", tags=["game"])
def valid_moves(data: BoardState):
    """Return the list of columns that are legal to play."""
    game = _board_from_request(data)
    return {"valid_moves": game.get_valid_moves()}


@app.post("/evaluate", tags=["game"])
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
