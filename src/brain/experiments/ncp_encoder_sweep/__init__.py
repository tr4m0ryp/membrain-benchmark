"""Sweep the winner: NCP's encoder and neighbourhood size (F26 -> "lost a comparison").

F26 measured NCP's local-df rule against its two named incumbents (global-df,
SIF) with the ENCODER frozen at all-MiniLM-L6-v2 and the neighbourhood size
frozen at k = 8. Neither was searched. Per this project's improvement
discipline, a mechanism that loses a comparison gets its frozen knobs swept
before it is shelved, because the measured ceiling is usually a parameter
rather than a law.

This package answers one question: does the LOCAL-vs-GLOBAL separation NCP
needs (F26's paired delta over SIF / global-IDF) become conclusive under any
combination of (encoder, k_neighbours), or does it stay unresolved everywhere?

Everything here IMPORTS from `brain.experiments.gate_ncp` (the pre-registered
grid values, the metric/bootstrap plumbing) and from `brain.embed` (the
embedder and the token-level pooling machinery). Neither is modified.

    from brain.experiments.ncp_encoder_sweep.report import main
    main(["--seeds", "5"])
"""

from __future__ import annotations

__all__: list[str] = []
