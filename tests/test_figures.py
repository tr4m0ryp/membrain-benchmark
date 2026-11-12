"""The figures must stay tied to the runs they were drawn from.

A figure that drifts away from its result file is worse than no figure, because
it looks like evidence. Three things are checked: every registered figure names
a pinned source that exists, the manifest on disk agrees with the registry, and
one figure actually renders. Rendering all five is left to the builder, since it
is slow and the failure modes are shared.

Skipped entirely when matplotlib is absent: it is an optional extra, and the
core manifest that reproduces every reported number does not depend on it.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("matplotlib", reason="figures are an optional extra: pip install -e '.[figures]'")

from brain.figures import REGISTRY, build_one, io  # noqa: E402
from brain.figures import manifest as manifest_mod  # noqa: E402


def test_every_registered_source_is_pinned_and_present():
    for figure in REGISTRY:
        assert figure.sources, f"{figure.key} names no source"
        for key in figure.sources:
            path = io.source_path(key)
            assert path.is_file(), f"{figure.key} points at a missing run: {path}"


def test_registry_keys_and_labels_are_unique():
    keys = [f.key for f in REGISTRY]
    labels = [f.label for f in REGISTRY]
    assert len(set(keys)) == len(keys), "duplicate figure key"
    assert len(set(labels)) == len(labels), "duplicate LaTeX label"


def test_manifest_on_disk_matches_the_registry():
    path = io.FIGURES_DIR / "manifest.json"
    if not path.is_file():
        pytest.skip("figures have not been built in this checkout")
    entries = json.loads(path.read_text(encoding="utf-8"))
    assert [e["id"] for e in entries] == [f.key for f in REGISTRY]
    for entry, figure in zip(entries, REGISTRY):
        assert entry["sources"] == [io.rel_source(k) for k in figure.sources]
        rendered = io.REPO_ROOT / entry["file"]
        assert rendered.is_file(), f"manifest names a missing file: {entry['file']}"
        assert manifest_mod._sha256(rendered) == entry["sha256"], (
            f"{entry['file']} changed without the manifest being rewritten; "
            f"run `python -m brain.figures`"
        )


def test_one_figure_renders(tmp_path):
    figure = next(f for f in REGISTRY if f.key == "separation-sweep")
    out = build_one(figure, tmp_path)
    assert out.is_file()
    assert out.stat().st_size > 4096, "a PDF that small did not draw anything"
    assert out.read_bytes()[:4] == b"%PDF"


def test_unknown_source_is_refused():
    with pytest.raises(KeyError):
        io.source_path("no-such-run")
