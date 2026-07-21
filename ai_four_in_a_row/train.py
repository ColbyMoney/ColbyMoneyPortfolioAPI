"""
Monte Carlo Tree Search (MCTS) + self-play training loop.
Optimised for GPU parallelism: all MCTS leaf evaluations within a simulation
step are batched into a single GPU forward pass across PARALLEL_GAMES games.

Run offline to produce a trained model:
    python train.py

A snapshot is saved after every iteration:
    models/ai_four_in_a_row_model_iteration_{total_games}.pt
"""

import logging
import math
import os
import re
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .game import FourInARow, COLS
from .model import build_model, FourInARowNet

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ITERATIONS     = 50
GAMES_PER_ITER     = 1000
MCTS_SIMS          = 250
PARALLEL_GAMES     = 200 # games evaluated simultaneously; sets GPU batch size
REPLAY_BUFFER_SIZE = 50_000
BATCH_SIZE         = 512
EPOCHS_PER_ITER    = 5
LR                 = 1e-3
L2_REG             = 1e-4
C_PUCT             = 1.5
TEMPERATURE_MOVES  = 20
CHECKPOINT_DIR     = Path(__file__).parent / "models"
CHECKPOINT_PATH    = CHECKPOINT_DIR / "ai_four_in_a_row_model_iteration_current.pt"

# GPU performance flags (no-op on CPU)
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True


def ts() -> str:
    """Timestamp prefix for log lines."""
    return datetime.now().strftime("[%H:%M:%S]")


# ---------------------------------------------------------------------------
# Persistent file logging
# ---------------------------------------------------------------------------
class _FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after every record so partial logs are readable."""
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


_logger: logging.Logger | None = None


def lprint(msg: str, level: int = logging.INFO) -> None:
    """
    Print to the terminal (unchanged) and mirror to the training log file.
    Falls back to print-only before logging is configured.
    """
    print(msg)
    if _logger is not None:
        _logger.log(level, msg)


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------
class MCTSNode:
    __slots__ = ("game", "parent", "move", "children", "visit_count",
                 "value_sum", "prior", "is_expanded")

    def __init__(self, game: FourInARow, parent=None, move=None, prior: float = 0.0):
        self.game        = game
        self.parent      = parent
        self.move        = move
        self.children: dict[int, "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum   = 0.0
        self.prior       = prior
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int) -> float:
        u = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u


def _backpropagate(node: MCTSNode, value: float) -> None:
    while node is not None:
        node.visit_count += 1
        node.value_sum   += value
        value             = -value
        node              = node.parent


def _select_leaf(root: MCTSNode) -> MCTSNode:
    node = root
    while node.is_expanded and not node.game.is_terminal():
        best_move = max(
            node.children,
            key=lambda m: node.children[m].ucb_score(node.visit_count),
        )
        node = node.children[best_move]
    return node


def _expand_node(node: MCTSNode, policy_logits: torch.Tensor, value: float) -> None:
    """Expand a leaf using pre-computed network outputs."""
    valid_moves = node.game.get_valid_moves()
    mask        = torch.full((COLS,), float("-inf"), device=policy_logits.device)
    for m in valid_moves:
        mask[m] = 0.0
    priors = F.softmax(policy_logits + mask, dim=-1).cpu().numpy()
    for move in valid_moves:
        child_game = node.game.copy()
        child_game.make_move(move)
        node.children[move] = MCTSNode(
            child_game, parent=node, move=move, prior=float(priors[move])
        )
    node.is_expanded = True


def batched_mcts(
    net: FourInARowNet, games: List[FourInARow], num_sims: int
) -> List[np.ndarray]:
    """
    Run MCTS for multiple games simultaneously.

    All leaf nodes that need neural-network evaluation within a single
    simulation step are stacked into one GPU forward pass (batch size up to
    PARALLEL_GAMES), replacing the original batch-size-1 approach.
    """
    roots = [MCTSNode(g.copy()) for g in games]

    for _ in range(num_sims):
        to_expand: List[MCTSNode] = []

        for root in roots:
            leaf = _select_leaf(root)
            if leaf.game.is_terminal():
                value = leaf.game.get_outcome(leaf.game.current_player)
                _backpropagate(leaf, -value)
            else:
                to_expand.append(leaf)

        if not to_expand:
            continue

        # Single batched forward pass for all leaves this simulation step
        boards   = np.stack([n.game.get_board_tensor() for n in to_expand])
        boards_t = torch.tensor(boards, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            policy_logits_batch, values_batch = net(boards_t)

        for idx, node in enumerate(to_expand):
            val = values_batch[idx].item()
            _expand_node(node, policy_logits_batch[idx], val)
            _backpropagate(node, -val)

    policies = []
    for root in roots:
        policy = np.zeros(COLS, dtype=np.float32)
        for move, child in root.children.items():
            policy[move] = child.visit_count
        if policy.sum() > 0:
            policy /= policy.sum()
        policies.append(policy)
    return policies


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------
Experience = Tuple[np.ndarray, np.ndarray, float]


def self_play_batch(net: FourInARowNet, num_games: int) -> List[Experience]:
    """
    Run num_games self-play games using batched MCTS.
    Up to PARALLEL_GAMES games share each GPU evaluation round.
    """
    all_experiences: List[Experience] = []
    games_completed = 0

    while games_completed < num_games:
        batch_size  = min(PARALLEL_GAMES, num_games - games_completed)
        games       = [FourInARow() for _ in range(batch_size)]
        histories: List[List[Tuple]] = [[] for _ in range(batch_size)]
        move_counts = [0] * batch_size
        active      = list(range(batch_size))

        while active:
            policies = batched_mcts(net, [games[i] for i in active], MCTS_SIMS)

            still_active = []
            for local_idx, game_idx in enumerate(active):
                game = games[game_idx]
                pi   = policies[local_idx]
                histories[game_idx].append(
                    (game.get_board_tensor(), pi, game.current_player)
                )

                if move_counts[game_idx] < TEMPERATURE_MOVES:
                    valid  = game.get_valid_moves()
                    probs  = np.array([pi[m] for m in valid], dtype=np.float64)
                    probs /= probs.sum()
                    move   = random.choices(valid, weights=probs)[0]
                else:
                    move = int(np.argmax(pi))

                # Force the first two moves to be uniformly random regardless
                # of what MCTS recommends.  This prevents the model from
                # converging on a single opening line early in training and
                # ensures all 7 opening columns (and 49 response pairs) are
                # explored.  The network will naturally learn that the center
                # column is optimal by seeing diverse outcomes, not by being
                # steered there.
                if move_counts[game_idx] < 2:
                    move = random.choice(game.get_valid_moves())

                game.make_move(move)
                move_counts[game_idx] += 1

                if game.is_terminal():
                    for board_t, pol, player in histories[game_idx]:
                        all_experiences.append(
                            (board_t, pol, game.get_outcome(player))
                        )
                    games_completed += 1
                else:
                    still_active.append(game_idx)

            active = still_active

    return all_experiences


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_step(
    net: FourInARowNet,
    optimizer: Adam,
    scaler: GradScaler,
    replay_buffer: deque,
) -> Optional[float]:
    if len(replay_buffer) < BATCH_SIZE:
        return None

    batch                    = random.sample(replay_buffer, BATCH_SIZE)
    boards, policies, values = zip(*batch)

    boards_t   = torch.tensor(np.array(boards),   dtype=torch.float32, device=DEVICE)
    policies_t = torch.tensor(np.array(policies), dtype=torch.float32, device=DEVICE)
    values_t   = torch.tensor(np.array(values),   dtype=torch.float32, device=DEVICE)

    # bfloat16 has the same exponent range as float32 — safe with custom BatchNorm
    with autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
        policy_logits, value_preds = net(boards_t)
        policy_loss = F.cross_entropy(policy_logits, policies_t)
        value_loss  = F.mse_loss(value_preds, values_t)
        loss        = policy_loss + value_loss

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


def save_iteration_checkpoint(net: FourInARowNet, total_games: int) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    state     = {"total_games": total_games, "model_state": net.state_dict()}
    iter_path = CHECKPOINT_DIR / f"ai_four_in_a_row_model_iteration_v3_{total_games}.pt"
    torch.save(state, iter_path)
    torch.save(state, CHECKPOINT_PATH)
    lprint(f"{ts()} Saved → {iter_path}")


# ---------------------------------------------------------------------------
# Checkpoint resume helpers
# ---------------------------------------------------------------------------
_CKPT_RE = re.compile(r"ai_four_in_a_row_model_iteration_v3_(\d+)\.pt$")


def find_latest_checkpoint() -> tuple[str, int] | None:
    """
    Scan CHECKPOINT_DIR for versioned checkpoint files.
    Returns (path, total_games) for the highest total_games found,
    or None if the directory doesn't exist or contains no valid files.
    Ignores the iteration-0 seed checkpoint (no training done yet).
    """
    if not CHECKPOINT_DIR.is_dir():
        return None

    best_games = 0
    best_path: Path | None = None

    for fname in os.listdir(CHECKPOINT_DIR):
        m = _CKPT_RE.match(fname)
        if m:
            games = int(m.group(1))
            if games > best_games:
                best_games = games
                best_path  = CHECKPOINT_DIR / fname

    return (best_path, best_games) if best_path else None

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    global _logger

    # ── Logging setup (file only — terminal handled by print) ────────────────
    log_dir  = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    _logger = logging.getLogger("train")
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False          # don't duplicate to root logger
    _fh = _FlushingFileHandler(log_file, encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _logger.addHandler(_fh)

    lprint(f"{ts()} Log file: {log_file}")
    lprint(f"{ts()} Training on: {DEVICE}")
    if DEVICE == "cuda":
        lprint(f"{ts()} GPU: {torch.cuda.get_device_name(0)}")

    net       = build_model().to(DEVICE)
    optimizer = Adam(net.parameters(), lr=LR, weight_decay=L2_REG)
    scaler    = GradScaler("cuda", enabled=(DEVICE == "cuda"))
    replay_buffer: deque = deque(maxlen=REPLAY_BUFFER_SIZE)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    total_games = 0
    start_iter  = 1
    completed   = 0
    latest      = find_latest_checkpoint()

    if latest:
        ckpt_path, ckpt_games = latest
        completed_found = ckpt_games // GAMES_PER_ITER
        ans = input(
            f"{ts()} Resume from {os.path.basename(ckpt_path)} "
            f"({ckpt_games} games, {completed_found} iterations done)? (y/n): "
        ).strip().lower()
        if ans == "y":
            try:
                state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                if "model_state" not in state:
                    raise KeyError("checkpoint is missing 'model_state'")
                net.load_state_dict(state["model_state"])
                total_games = ckpt_games
                start_iter  = completed_found + 1
                completed   = completed_found
                # Keep CHECKPOINT_PATH current so workers always find latest weights
                torch.save(state, CHECKPOINT_PATH)
                lprint(f"{ts()} Resumed  | total_games={total_games} | "
                      f"starting iteration {start_iter}/{NUM_ITERATIONS}")
            except Exception as exc:
                lprint(
                    f"{ts()} WARNING: could not load checkpoint ({exc}) — starting fresh",
                    logging.ERROR,
                )
                if _logger:
                    _logger.exception("Full traceback for checkpoint load failure")
                total_games = 0
                start_iter  = 1
                completed   = 0
        else:
            lprint(f"{ts()} Starting fresh.")
    else:
        lprint(f"{ts()} No checkpoint found — starting fresh.")

    # Pass last_epoch=completed-1 so __init__'s internal step() lands exactly
    # at position `completed` with no manual loop and no PyTorch UserWarning.
    # When resuming (completed > 0, last_epoch >= 0) PyTorch requires
    # 'initial_lr' to already exist in the param groups — seed it from the
    # current LR before the scheduler reads it.
    if completed > 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_ITERATIONS,
                                  last_epoch=completed - 1)

    iter_times: List[float] = []

    for iteration in range(start_iter, NUM_ITERATIONS + 1):
        t_iter = time.time()
        lprint(f"\n{ts()} === Iteration {iteration}/{NUM_ITERATIONS} | "
              f"total games so far: {total_games} ===")

        # ── Self-play ────────────────────────────────────────────────────────
        net.eval()
        t_sp = time.time()
        lprint(f"{ts()} Self-play  {GAMES_PER_ITER} games | "
              f"{MCTS_SIMS} sims/move | {PARALLEL_GAMES} parallel")

        experiences = self_play_batch(net, GAMES_PER_ITER)
        replay_buffer.extend(experiences)
        total_games += GAMES_PER_ITER
        sp_elapsed   = time.time() - t_sp

        lprint(f"{ts()} Self-play done | {len(experiences)} experiences | "
              f"buffer={len(replay_buffer)} | "
              f"{sp_elapsed:.1f}s  ({sp_elapsed / GAMES_PER_ITER:.2f}s/game)")

        # ── Train ────────────────────────────────────────────────────────────
        net.train()
        t_tr       = time.time()
        total_loss = 0.0
        steps      = 0
        for _ in range(EPOCHS_PER_ITER):
            for _ in range(max(1, len(experiences) // BATCH_SIZE)):
                loss = train_step(net, optimizer, scaler, replay_buffer)
                if loss is not None:
                    total_loss += loss
                    steps      += 1
        scheduler.step()

        avg_loss   = total_loss / steps if steps else 0.0
        tr_elapsed = time.time() - t_tr
        lprint(f"{ts()} Training done | avg_loss={avg_loss:.4f} | "
              f"steps={steps} | {tr_elapsed:.1f}s")

        # ── Checkpoint ───────────────────────────────────────────────────────
        save_iteration_checkpoint(net, total_games)

        # ── ETA ──────────────────────────────────────────────────────────────
        iter_elapsed = time.time() - t_iter
        iter_times.append(iter_elapsed)
        rolling_avg  = sum(iter_times[-5:]) / len(iter_times[-5:])
        remaining    = rolling_avg * (NUM_ITERATIONS - iteration)
        lprint(f"{ts()} Iteration {iteration} done | "
              f"{iter_elapsed:.1f}s | ETA {remaining / 60:.1f} min")

    lprint(f"\n{ts()} Training complete. Total self-play games: {total_games}")


if __name__ == "__main__":
    main()
