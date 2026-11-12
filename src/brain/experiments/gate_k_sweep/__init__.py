"""The k sweep: is recall@5 = 0.958 near a plateau or on a slope?

The 0.958 ceiling is `recall@5`, and k was never swept. This experiment sweeps k over {1, 2, 3, 5, 10, 20, 50} for the two
free arms (plain_vector, bm25) on the frozen synthetic corpus, reporting
recall@k, disambiguation accuracy, precision@k, MRR and accuracy by cluster
size, all with cluster-level BCa intervals, plus the F15/I4 recollection-veto
precondition (bounded, enumerable competitor set) and the headroom formula's
behaviour across k.

Entry point: `python -m brain.experiments.gate_k_sweep`, or
`brain.experiments.gate_k_sweep.sweep.run()` programmatically.
"""

from __future__ import annotations

from brain.experiments.gate_k_sweep.sweep import SweepResult, run

__all__ = ["SweepResult", "run"]
