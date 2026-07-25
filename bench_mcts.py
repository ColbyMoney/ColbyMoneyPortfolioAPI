"""
Benchmark batched_mcts to validate PARALLEL_MCTS_TREES and MCTS_SIMS settings
before committing to a full training run.

Run from the repo root:
    python -m bench_mcts

Runs a sweep over PARALLEL_MCTS_TREES_TO_TEST for each MCTS_SIMS_TO_TEST value,
always benchmarking the same TOTAL_GAMES_TO_BENCH for a fair comparison.

Set PROFILE = True to run a single representative config under cProfile
instead of the full sweep, to see where time is actually being spent.
"""

import time
import pstats
import cProfile
import io
import torch
from ai_four_in_a_row.game import Connect4
from ai_four_in_a_row.model import build_model
from ai_four_in_a_row.train import batched_mcts

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_GAMES_TO_BENCH = 10

MCTS_SIMS_TO_TEST           = [50, 250]
PARALLEL_MCTS_TREES_TO_TEST = [500, 1000, 2000, 4000]

# Profiling mode — when True, skips the full sweep and instead profiles
# one config in detail to find the actual bottleneck functions.
PROFILE                 = True
PROFILE_MCTS_SIMS       = 250
PROFILE_PARALLEL_TREES  = 1000
PROFILE_TOTAL_GAMES     = 20   # keep small — profiling adds overhead per call
PROFILE_TOP_N           = 25   # how many rows of the profile report to print

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True


def bench_full_games(net, parallel_mcts_trees, total_games, mcts_sims):
    games_completed = 0
    total_elapsed = 0.0
    total_moves = 0
    peak_mb = 0.0

    while games_completed < total_games:
        batch_size = min(parallel_mcts_trees, total_games - games_completed)
        games = [Connect4() for _ in range(batch_size)]
        active = list(range(batch_size))
        move_counts = [0] * batch_size

        if DEVICE == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()

        while active:
            policies = batched_mcts(net, [games[i] for i in active], mcts_sims)
            still_active = []
            for local_idx, game_idx in enumerate(active):
                move = int(policies[local_idx].argmax())
                games[game_idx].make_move(move)
                move_counts[game_idx] += 1
                if not games[game_idx].is_terminal():
                    still_active.append(game_idx)
            active = still_active

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        total_elapsed += time.perf_counter() - start
        if DEVICE == "cuda":
            peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() / 1024**2)
        total_moves += sum(move_counts)
        games_completed += batch_size

    return {
        "parallel_mcts_trees": parallel_mcts_trees,
        "mcts_sims": mcts_sims,
        "total_games": total_games,
        "elapsed": total_elapsed,
        "s_per_game": total_elapsed / total_games,
        "avg_moves": total_moves / total_games,
        "peak_mb": peak_mb,
    }


def warm_up(net, total_games):
    warmup_games = [Connect4() for _ in range(min(8, total_games))]
    with torch.no_grad():
        _ = batched_mcts(net, warmup_games, 1)
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def run_profile(net):
    """
    Profiles a single representative bench_full_games() call and prints
    the top time-consuming functions, sorted by cumulative time.
    """
    print(f"Profiling: sims={PROFILE_MCTS_SIMS}, trees={PROFILE_PARALLEL_TREES}, "
          f"games={PROFILE_TOTAL_GAMES}")
    print("(Note: profiling itself adds overhead — absolute times will be")
    print(" slower than an unprofiled run; relative breakdown is what matters.)")
    print()

    profiler = cProfile.Profile()
    profiler.enable()

    bench_full_games(
        net=net,
        parallel_mcts_trees=PROFILE_PARALLEL_TREES,
        total_games=PROFILE_TOTAL_GAMES,
        mcts_sims=PROFILE_MCTS_SIMS,
    )

    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(PROFILE_TOP_N)
    print(stream.getvalue())

    # Also show a version sorted by total time in the function itself
    # (excludes time spent in sub-calls) — useful for spotting hot leaf functions
    # like _check_win or .copy() that don't call much else.
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2).sort_stats("tottime")
    stats2.print_stats(PROFILE_TOP_N)
    print("--- Sorted by tottime (self time, excludes sub-calls) ---")
    print(stream2.getvalue())


def run_sweep(net):
    print(f"Device               : {DEVICE}")
    print(f"Total games / point  : {TOTAL_GAMES_TO_BENCH}")
    print(f"MCTS sims sweep      : {MCTS_SIMS_TO_TEST}")
    print(f"Parallel trees sweep : {PARALLEL_MCTS_TREES_TO_TEST}")
    print()

    results = []
    for mcts_sims in MCTS_SIMS_TO_TEST:
        for parallel_mcts_trees in PARALLEL_MCTS_TREES_TO_TEST:
            r = bench_full_games(
                net=net,
                parallel_mcts_trees=parallel_mcts_trees,
                total_games=TOTAL_GAMES_TO_BENCH,
                mcts_sims=mcts_sims,
            )
            results.append(r)
            print(
                f"sims={r['mcts_sims']:4d} | trees={r['parallel_mcts_trees']:5d} | "
                f"{r['elapsed']:8.3f}s | {r['s_per_game']:6.3f}s/game | "
                f"{r['avg_moves']:5.1f} avg moves | {r['peak_mb']:8.1f}MB peak"
            )

    print()
    print("Summary (sorted by s/game, fastest first):")
    for r in sorted(results, key=lambda x: x["s_per_game"]):
        print(
            f"  sims={r['mcts_sims']:4d} | trees={r['parallel_mcts_trees']:5d} | "
            f"{r['s_per_game']:6.3f}s/game | {r['peak_mb']:8.1f}MB peak"
        )


def main():
    net = build_model().to(DEVICE)
    net.eval()
    warm_up(net, PROFILE_TOTAL_GAMES if PROFILE else TOTAL_GAMES_TO_BENCH)

    if PROFILE:
        run_profile(net)
    else:
        run_sweep(net)


if __name__ == "__main__":
    main()