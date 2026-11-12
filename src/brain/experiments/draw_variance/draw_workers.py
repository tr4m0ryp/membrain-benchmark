"""One draw-runner per statistic: replay-or-compute every seed, in order.

`run_f23` is called out as the priority worker (coordinator instruction: "F23
is the result that would hurt most if it moved"). `run_f21` is the one worker
that can go genuinely BLOCKED (no usable LLM transport) rather than merely
slow, so it is the only one with a persisted blocked-reason sentinel: a
concurrent process not running f21 itself still needs to know whether f21 is
blocked, in progress, or simply not its job.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from brain.config import ensure_dir
from brain.experiments.draw_variance.consolidation_draws import draw_consolidation_delta
from brain.experiments.draw_variance.precision_draws import draw_precision
from brain.experiments.draw_variance.preregistration import MERGE_TAU
from brain.experiments.draw_variance.relations_harm import draw_relations_harm
from brain.experiments.draw_variance.scratch_store import ScratchStore
from brain.experiments.draw_variance.synthetic_harm import draw_merge_harm

__all__ = ["run_statistic", "run_f17a", "run_f17b", "run_f23", "run_f21",
           "F21_BLOCKED_SENTINEL", "read_f21_blocked", "write_f21_blocked"]

OnSeed = Callable[[], None]


def run_statistic(
    statistic_id: str,
    seeds: tuple[int, ...],
    draw_one: Callable[[int], dict[str, Any]],
    scratch: ScratchStore,
    *,
    say: Callable[[str], None],
    on_seed: OnSeed | None = None,
) -> list[dict[str, Any]]:
    """Run (or replay from scratch) every seed for one statistic, in order.

    `on_seed` fires after EVERY seed, cached replay or freshly computed, so a
    caller can checkpoint the combined result file per seed rather than only
    once the whole (possibly hours-long) statistic finishes.
    """
    draws: list[dict[str, Any]] = []
    for seed in seeds:
        cached = scratch.get(seed)
        if cached is not None:
            say(f"{statistic_id} seed={seed}: replayed from scratch")
            draws.append(cached)
        else:
            started = time.monotonic()
            draw = draw_one(seed)
            draw["wall_seconds"] = round(time.monotonic() - started, 2)
            scratch.put(seed, draw)
            say(f"{statistic_id} seed={seed}: done in {draw['wall_seconds']:.1f}s -> {draw}")
            draws.append(draw)
        if on_seed is not None:
            on_seed()
    return draws


def run_f17a(
    seeds: tuple[int, ...], scratch: ScratchStore, say: Callable[[str], None],
    *, on_seed: OnSeed | None = None,
) -> list[dict[str, Any]]:
    from brain.embed.embedder import get_embedder

    embedder = get_embedder()
    return run_statistic(
        "F17a_merge_harm_synthetic", seeds,
        lambda seed: draw_merge_harm(seed, embedder, tau=MERGE_TAU), scratch, say=say, on_seed=on_seed,
    )


def run_f17b(
    seeds: tuple[int, ...], scratch: ScratchStore, say: Callable[[str], None],
    *, on_seed: OnSeed | None = None,
) -> list[dict[str, Any]]:
    from brain.embed.embedder import get_embedder

    embedder = get_embedder()
    return run_statistic(
        "F17b_merge_harm_relations_v2", seeds,
        lambda seed: draw_relations_harm(seed, embedder, tau=MERGE_TAU), scratch, say=say, on_seed=on_seed,
    )


def run_f23(
    seeds: tuple[int, ...], scratch: ScratchStore, say: Callable[[str], None],
    *, on_seed: OnSeed | None = None,
) -> list[dict[str, Any]]:
    """The priority statistic: consolidation delta at tau=0.75."""
    return run_statistic(
        "F23_consolidation_delta", seeds,
        lambda seed: draw_consolidation_delta(seed), scratch, say=say, on_seed=on_seed,
    )


#: A sentinel a concurrent process (not running f21 itself) can read, so it does
#: not misreport "not run in this process" as BLOCKED when another process
#: already produced real draws, or hide a genuine block behind that same text.
F21_BLOCKED_SENTINEL = "F21_memory_precision.blocked.json"


def write_f21_blocked(scratch_dir: str, reason: str) -> None:
    Path(ensure_dir(scratch_dir) / F21_BLOCKED_SENTINEL).write_text(
        json.dumps({"reason": reason}) + "\n"
    )


def read_f21_blocked(scratch_dir: str) -> str | None:
    path = Path(scratch_dir) / F21_BLOCKED_SENTINEL
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("reason")


def run_f21(
    seeds: tuple[int, ...], scratch: ScratchStore, say: Callable[[str], None],
    *, scratch_dir: str, on_seed: OnSeed | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Returns `(draws, blocked_reason)`. `blocked_reason` is None on success."""
    from brain.eval.precision.entailment import EntailmentJudge

    judge = EntailmentJudge()
    if judge.blocked_reason:
        reason = f"entailment judge unavailable: {judge.blocked_reason}"
        write_f21_blocked(scratch_dir, reason)
        return [], reason
    try:
        draws = run_statistic(
            "F21_memory_precision", seeds, lambda seed: draw_precision(seed, judge), scratch,
            say=say, on_seed=on_seed,
        )
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
        reason = f"live judging failed after some seeds: {type(exc).__name__}: {exc}"
        write_f21_blocked(scratch_dir, reason)
        return [], reason
    return draws, None
