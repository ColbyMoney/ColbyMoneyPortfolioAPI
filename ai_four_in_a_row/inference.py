"""
Inference module: load a trained model checkpoint and select the best move.

Usage example:
    from inference import FourInARowAI
    ai = FourInARowAI("models/connect4_v1.pt")
    move = ai.best_move(game)
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .game import Connect4, COLS, ROWS, EMPTY
from .model import build_model, FourInARowNet
from .train import batched_mcts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MCTS_SIMS = 50  # more sims at inference time for stronger play

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

DEFAULT_CHECKPOINT = MODELS_DIR / "ai_four_in_a_row_model_iteration_v3_41000.pt"

DIFFICULTY_CHECKPOINTS: dict[str, Path] = {
    "medium":    MODELS_DIR / "ai_four_in_a_row_model_iteration_v1_45000.pt",
    "hard":      MODELS_DIR / "ai_four_in_a_row_model_iteration_v2_13000.pt",
    "legendary": MODELS_DIR / "ai_four_in_a_row_model_iteration_v3_41000.pt",
}

# All difficulty models are lazily loaded on first request and kept for the lifetime of the
# process.  No eviction — Python's GC handles cleanup if a reference is ever dropped.
_ai_cache: dict[str, "FourInARowAI"] = {}

class FourInARowAI:
    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        mcts_sims: int = DEFAULT_MCTS_SIMS,
    ):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No model checkpoint found at '{checkpoint_path}'. "
                "Run train.py first to generate a checkpoint."
            )
        self.checkpoint_path = checkpoint_path
        self.mcts_sims = mcts_sims
        self.net = build_model().to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        state = checkpoint.get("model_state", checkpoint)
        self.net.load_state_dict(state)
        total_games = checkpoint.get("total_games", checkpoint.get("iteration", "?"))
        print(f"Loaded model from '{checkpoint_path}' (total_games={total_games})")
        self.net.eval()

    def best_move(self, game: Connect4, use_mcts: bool) -> int:
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

    def move_probabilities(self, game: Connect4) -> dict[int, float]:
        """Return a dict mapping each valid column to its MCTS visit-count probability."""
        pi = batched_mcts(self.net, [game], self.mcts_sims)[0]
        return {col: float(pi[col]) for col in game.get_valid_moves()}

    def evaluate_position(self, game: Connect4) -> float:
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


# Singleton used by main.py so the model is only loaded once at startup.
_ai_instance: Optional[FourInARowAI] = None


def get_ai(checkpoint_path: str = DEFAULT_CHECKPOINT, mcts_sims: int = DEFAULT_MCTS_SIMS) -> FourInARowAI:
    global _ai_instance
    if _ai_instance is None:
        _ai_instance = FourInARowAI(checkpoint_path, mcts_sims)
    return _ai_instance


def reload_ai() -> None:
    """Replace the singleton with a freshly loaded copy of the checkpoint."""
    global _ai_instance
    if _ai_instance is not None:
        # Construct a new instance so in-flight requests finish on the old one
        _ai_instance = FourInARowAI(_ai_instance.checkpoint_path, _ai_instance.mcts_sims)


def get_ai_for_difficulty(difficulty: str = "medium") -> FourInARowAI:
    """Return the pre-loaded FourInARowAI for the requested difficulty level.

    All models are loaded at startup via load_all_models().  Falls back to the
    default singleton if the requested difficulty is not in the cache.
    """
    return _ai_cache.get(difficulty) or get_ai()


def load_all_models() -> None:
    """Load every difficulty model into the cache at startup.

    Called once from the FastAPI lifespan so all models are warm before the
    first request arrives.  Missing checkpoint files are logged as warnings
    rather than hard failures so the server still starts during training.
    """
    for difficulty, ckpt in DIFFICULTY_CHECKPOINTS.items():
        try:
            _ai_cache[difficulty] = FourInARowAI(ckpt, DEFAULT_MCTS_SIMS)
        except FileNotFoundError as exc:
            print(f"[WARNING] Could not load '{difficulty}' model: {exc}")
