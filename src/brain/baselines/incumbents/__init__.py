"""Incumbent memory systems, executed rather than read.

The paper this project targets compares against Mem0, Graphiti (the engine
behind Zep), TOKI and Nous by reading their source and recording commit hashes.
That is the weakest point in its argument, and this package is the answer to it:
every incumbent that can be executed without a paid credential is executed here,
against the frozen corpora, through the same statistics as every other arm.

What is measured is the write path, because that is what the paper's
architectural claim is about: every shipped system has a two-branch write, merge
or invalidate, with a silent fall-through, and none has an affirmative "these
are distinct, keep both, mark as siblings" operation that leaves a trace. Three
rates, defined in the pre-registration and computed in `outcomes` and
`aggregate`:

* **D**, the distinct-event destruction rate;
* **E**, third-branch expressiveness under a judge that is trying to record a
  relation; and
* **I**, residual indistinguishability against a matched unrelated write.

`published` holds the numbers these systems report for themselves. Nothing in it
was measured here and every row says so.
"""

from __future__ import annotations

from brain.baselines.incumbents import aggregate, bridge, cases, outcomes, published, specs

__all__ = ["aggregate", "bridge", "cases", "outcomes", "published", "specs"]
