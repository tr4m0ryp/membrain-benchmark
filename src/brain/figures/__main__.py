"""Render the paper figures.

    python -m brain.figures                 # all of them, into figures/
    python -m brain.figures --only ceiling  # one, matched on a key substring
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import build_all, build_one, io, manifest
from .manifest import REGISTRY


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brain.figures", description=__doc__)
    parser.add_argument("--out", type=Path, default=io.FIGURES_DIR, help="output directory")
    parser.add_argument("--only", default=None, help="render only figures whose key contains this")
    args = parser.parse_args(argv)

    if args.only:
        selected = [f for f in REGISTRY if args.only in f.key]
        if not selected:
            parser.error(f"no figure key contains {args.only!r}; have {[f.key for f in REGISTRY]}")
        entries = [manifest.entry(f, build_one(f, args.out)) for f in selected]
        # Rewrite the manifest for the whole registry, not just the rebuilt
        # subset. A partial build that left the manifest holding a stale hash
        # would defeat the one thing the manifest is for, and the drift would
        # only surface later as a failing test.
        rebuilt = {e["id"]: e for e in entries}
        merged = []
        for figure in REGISTRY:
            if figure.key in rebuilt:
                merged.append(rebuilt[figure.key])
            else:
                path = args.out / f"{figure.key}.pdf"
                merged.append(manifest.entry(figure, path) if path.is_file() else None)
        manifest.write([e for e in merged if e is not None], args.out)
    else:
        entries = build_all(args.out)

    for e in entries:
        print(f"{e['id']:22s} -> {e['file']}  from {', '.join(e['sources'])}")
    print(f"{len(entries)} figure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
