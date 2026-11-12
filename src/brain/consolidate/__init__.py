"""Baseline consolidation: the summarise-and-merge pass every system ships.

This package is explicitly a BASELINE, not our mechanism. It exists so task 005
can measure **consolidation delta**, accuracy on a fixed store before versus
after a consolidation pass, which no vendor has ever published.

It is built to be the strongest fair version of the industry default, because a
weak comparator would make that measurement worthless in our own favour, which
is exactly the vendor failure mode this project exists to avoid:

* **Complete linkage by default.** Every pair inside a merged group is itself
  above tau, so the baseline only makes merges its own policy would defend. The
  naive transitive closure (`linkage="single"`) chains distant episodes together
  and would inflate the measured harm; it stays available and swept rather than
  being the silent default.
* **A merge prompt told to preserve distinguishing detail**, to record several
  occasions separately instead of blending them, and never to trade a specific
  value for a vague summary.
* **The threshold is a swept parameter**, never a constant, and `plan_groups`
  prices a whole sweep without calling a model.
* **Both modes.** `destructive=True` replaces the group; `destructive=False`
  keeps the originals and adds the summary. Task 005 needs both to attribute any
  measured loss to merging rather than to summarising.

Deliberately absent, because they are the mechanism being measured against and
not the baseline: supersession classification, evidence marking, nogoods,
belief-to-episode provenance, and the consolidation interlock.

**Recorded property: the pass is not idempotent.** Measured on the frozen corpus
(120 episodes, complete linkage), running it a second time over its own output:

| mode | tau | pass 1 groups | pass 2 groups | pass 2 episodes |
|---|---|---|---|---|
| destructive | 0.90 | 9 | 0 | 0 |
| destructive | 0.85 | 25 | 2 | 4 |
| non-destructive | 0.90 | 9 | 9 | 19 |
| non-destructive | 0.85 | 25 | 26 | 70 |

Destructive consolidation is close to a fixed point but not at one: two merged
summaries can themselves land above tau. Non-destructive consolidation is not
even close, because every summary sits next to the originals it was written
from. Both are properties of the shipped default, not defects introduced here,
and neither is repaired: a baseline that quietly stopped re-merging would no
longer be the thing being measured. Task 005 should run ONE pass per condition.

Entry point: `python -m brain.consolidate.smoke` verifies the corpus hash,
prices the whole sweep for free, smokes N=10 and extrapolates.
"""

from brain.consolidate.baseline import (
    DEFAULT_CACHE_DIR,
    BaselineConsolidator,
    ConsolidationBudgetExceeded,
    consolidate_baseline,
    run_consolidation,
    sweep_thresholds,
)
from brain.consolidate.cache import MergeCache
from brain.consolidate.grouping import group_traces, groups_for, plan_groups
from brain.consolidate.prefetch import merge_request, warm_cache
from brain.consolidate.prompt import (
    MAX_OUTPUT_TOKENS,
    PROMPT_VERSION,
    merge_cache_key,
    merge_schema,
    merge_system_prompt,
    merge_user_prompt,
    parse_merged,
)
from brain.consolidate.types import (
    DEFAULT_LINKAGE,
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLDS,
    LINKAGES,
    MAX_SPEND_USD,
    ConsolidationResult,
    ConsolidationSpend,
    MergeGroup,
    price_usd,
)

__all__ = [
    "consolidate_baseline",
    "run_consolidation",
    "sweep_thresholds",
    "BaselineConsolidator",
    "ConsolidationBudgetExceeded",
    "ConsolidationResult",
    "ConsolidationSpend",
    "MergeGroup",
    "MergeCache",
    "group_traces",
    "groups_for",
    "plan_groups",
    "warm_cache",
    "merge_request",
    "price_usd",
    "merge_cache_key",
    "merge_schema",
    "merge_system_prompt",
    "merge_user_prompt",
    "parse_merged",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_LINKAGE",
    "DEFAULT_THRESHOLD",
    "DEFAULT_THRESHOLDS",
    "LINKAGES",
    "MAX_OUTPUT_TOKENS",
    "MAX_SPEND_USD",
    "PROMPT_VERSION",
]
