"""
Inference module: load a trained model checkpoint and select the best move.

Usage example:
    from four_in_a_row.inference import FourInARowAI
    ai = FourInARowAI()
    move = ai.best_move(game)
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .game import FourInARow, COLS, ROWS, EMPTY
from .model import build_model, FourInARowNet
from .train import batched_mcts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

DEFAULT_CHECKPOINT = MODELS_DIR / "ai_four_in_a_row_model_iteration_current.pt"
DEFAULT_MCTS_SIMS = 500  # more sims at inference time for stronger play

DIFFICULTY_CHECKPOINTS: dict[str, Path] = {
    "medium":    MODELS_DIR / "ai_four_in_a_row_model_iteration_v1_45000.pt",
    "hard":      MODELS_DIR / "ai_four_in_a_row_model_iteration_v2_13000.pt",
    "legendary": MODELS_DIR / "ai_four_in_a_row_model_iteration_v3_50000.pt",
}

# Cache one FourInARowAI instance per difficulty so repeated API calls don't reload weights.
_ai_cache: dict[str, "FourInARowAI"] = {}


class FourInARowAI:
    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        mcts_sims: int = DEFAULT_MCTS_SIMS,
    ):
        self.checkpoint_path = checkpoint_path
        self.mcts_sims = mcts_sims
        self.net = build_model().to(DEVICE)
        self._load(checkpoint_path)
        self.net.eval()

    def _load(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"No model checkpoint found at '{path}'. "
                "Run train.py first to generate a checkpoint."
            )
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        state = checkpoint.get("model_state", checkpoint)
        self.net.load_state_dict(state)
        total_games = checkpoint.get("total_games", checkpoint.get("iteration", "?"))
        print(f"Loaded model from '{path}' (total_games={total_games})")

    def best_move(self, game: FourInARow, use_mcts: bool = True) -> int:
        """
        Return the column index of the best move.

        Args:
            game:      Current game state.
            use_mcts:  If True (default), run MCTS for stronger play.
                       If False, use the raw policy head (faster but weaker).
        """
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available.")

        if use_mcts:
            pi = batched_mcts(self.net, [game], self.mcts_sims)[0]
            return int(np.argmax(pi))

        # Fast greedy from raw policy
        board_tensor = torch.tensor(
            game.get_board_tensor(), dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            policy_logits, _ = self.net(board_tensor)

        mask = torch.full((COLS,), float("-inf"), device=DEVICE)
        for m in valid_moves:
            mask[m] = 0.0
        probs = F.softmax(policy_logits.squeeze(0) + mask, dim=-1).cpu().numpy()
        return int(np.argmax(probs))

    def move_probabilities(self, game: FourInARow) -> dict[int, float]:
        """Return a dict mapping each valid column to its MCTS visit-count probability."""
        pi = batched_mcts(self.net, [game], self.mcts_sims)[0]
        return {col: float(pi[col]) for col in game.get_valid_moves()}

    def evaluate_position(self, game: FourInARow) -> float:
        """
        Return the model's value estimate for the current position.
        +1.0 = current player is winning, -1.0 = current player is losing.
        """
        board_tensor = torch.tensor(
            game.get_board_tensor(), dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, value = self.net(board_tensor)
        return float(value.item())


# Singleton used by router.py so the model is only loaded once at startup.
_ai_instance: Optional[FourInARowAI] = None


def get_ai(checkpoint_path: Path = DEFAULT_CHECKPOINT, mcts_sims: int = DEFAULT_MCTS_SIMS) -> FourInARowAI:
    global _ai_instance
    if _ai_instance is None:
        _ai_instance = FourInARowAI(checkpoint_path, mcts_sims)
    return _ai_instance


def reload_ai() -> None:
    """Replace the singleton with a freshly loaded copy of the checkpoint."""
    global _ai_instance
    if _ai_instance is not None:
        _ai_instance = FourInARowAI(_ai_instance.checkpoint_path, _ai_instance.mcts_sims)


def get_ai_for_difficulty(difficulty: str = "medium") -> FourInARowAI:
    """Return a cached FourInARowAI for the requested difficulty level.

    Falls back to the default current-model singleton when a versioned
    checkpoint file doesn't exist yet (e.g. hard/v3 still in training).
    """
    ckpt = DIFFICULTY_CHECKPOINTS.get(difficulty)
    if ckpt is None or not ckpt.exists():
        return get_ai()
    if difficulty not in _ai_cache:
        _ai_cache[difficulty] = FourInARowAI(ckpt, DEFAULT_MCTS_SIMS)
    return _ai_cache[difficulty]
