"""Tests for the transcript-derived candidate corpus pipeline.

The safety properties get the most coverage, because they are the ones whose
failure is unrecoverable: a committed secret cannot be un-committed. Three of
them are tested as invariants rather than as behaviours, so a future edit that
reintroduces the risk fails here rather than in a repository:

- no detected value reaches any record, report or aggregate (`test_scan_*`);
- writing refuses outright when the output directory is not gitignored
  (`test_assert_ignored_*`);
- a mined pair carries no class (`test_mined_pairs_are_unlabelled`).
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from brain.corpus.transcripts.emit.artifacts import NotIgnored, assert_ignored, write_jsonl
from brain.corpus.transcripts.extract.episodes import episodes_of, parse_diff
from brain.corpus.transcripts.extract.reader import read_session, session_id_of
from brain.corpus.transcripts.inventory import take_inventory
from brain.corpus.transcripts.mine import MiningConfig, mine_pairs
from brain.corpus.transcripts.protocol import PROTOCOL, effort_estimate
from brain.corpus.transcripts.scrub.pseudonymise import Pseudonymiser, load_or_create_salt
from brain.corpus.transcripts.scrub.scan import MASK, aggregate, mask_benign, scan_text, verdict
from brain.corpus.transcripts.stats import compare, corpus_profile, implied_beta
from brain.corpus.transcripts.types import STRATA, CandidateEpisode, ScanFinding, SessionRecord

# A synthetic credential built at import time from harmless parts, so this file
# never contains a value that looks like a live key to a scanner of its own.
FAKE_AWS = "AKIA" + "Q" * 16
FAKE_GH = "ghp_" + "b" * 36


def _session_line(**kwargs) -> str:
    return json.dumps(kwargs)


def _write_session(tmp_path: Path, session_id: str, prompt: str, answer: str) -> Path:
    path = tmp_path / f"{session_id}.jsonl"
    lines = [
        _session_line(
            type="user",
            timestamp="2026-07-24T10:00:00.000Z",
            cwd="/home/somebody/projects/thing",
            gitBranch="main",
            message={"role": "user", "content": prompt},
        ),
        _session_line(
            type="assistant",
            timestamp="2026-07-24T10:00:05.000Z",
            message={"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ),
        _session_line(
            type="user",
            timestamp="2026-07-24T10:00:06.000Z",
            message={"role": "user", "content": [{"type": "tool_result", "content": "SECRETS"}]},
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


DIFF_PROMPT = """\
Write ONE git commit subject line for the staged change below.
Format: type(scope): imperative summary. Max 72 chars.

 notes/plan.md | 4 ++++
 1 file changed, 4 insertions(+)

diff --git a/notes/plan.md b/notes/plan.md
index a5b1a90..896e838 100644
--- a/notes/plan.md
+++ b/notes/plan.md
@@ -10,4 +10,8 @@ ## Threshold policy
+The retrieval threshold moves from 0.75 to 0.85 because the lower value merged
+two distinct incidents in the July replay and the merge was not recoverable.
+We keep both records and mark them siblings instead of collapsing them.
-The retrieval threshold is 0.75.
"""


# --------------------------------------------------------------------------- scan


def test_scan_finds_credential_shapes():
    _, counts = scan_text(f"key = {FAKE_AWS} and token {FAKE_GH}")
    assert counts["aws_access_key_id"] == 1
    assert counts["github_token"] == 1
    assert verdict(counts) == "block"


def test_scan_never_returns_the_matched_value():
    masked, counts = scan_text(f"AWS={FAKE_AWS}")
    assert FAKE_AWS not in masked
    assert MASK in masked
    for key in counts:
        assert FAKE_AWS not in key
    assert all(isinstance(value, int) for value in counts.values())


def test_scan_finding_has_no_field_that_could_hold_a_value():
    """Structural: the record shape itself makes leaking a value impossible."""
    names = {f.name for f in fields(ScanFinding)}
    assert names == {"category", "severity", "count"}


def test_aggregate_reports_counts_only():
    report = aggregate([{"aws_access_key_id": 2}, {"email": 1}, {}])
    payload = report.to_dict()
    assert report.n_scanned == 3
    assert report.n_blocked == 1
    assert report.n_redacted == 1
    assert report.n_clean == 1
    assert set(payload["findings"][0]) == {"category", "severity", "count"}


def test_benign_git_shapes_do_not_become_high_entropy_hits():
    text = "index a5b1a90..896e838 100644"
    masked, counts = mask_benign(text)
    assert counts["git_index_line"] == 1
    _, after = scan_text(text)
    assert after.get("high_entropy_hex", 0) == 0


def test_content_addressed_cache_paths_are_not_treated_as_secrets():
    """The exclusion that saved 59 percent of the corpus. It must stay verifiable.

    Every high-entropy hit in the first pass over the real store was a committed
    cache filename of this shape. The directory prefix has to be consumed too,
    because `/` is inside the base64 character class.
    """
    text = "data/cache/full_context/" + "a1" * 32 + ".json updated"
    masked, counts = mask_benign(text)
    assert counts["content_addressed_path"] == 1
    _, after = scan_text(text)
    assert after.get("high_entropy_hex", 0) == 0
    assert after.get("high_entropy_base64", 0) == 0
    assert verdict(after) == "clean"  # counted under its own category, quarantines nothing


def test_a_bare_digest_is_still_flagged():
    """The recall tier stays armed: only the FILENAME shape is excused."""
    _, counts = scan_text("value " + "9f" * 32)
    assert counts["high_entropy_hex"] == 1
    assert verdict(counts) == "flag"


def test_identifiers_are_redact_tier_not_block_tier():
    """Quarantining every home path would quarantine the whole corpus."""
    _, counts = scan_text("see /home/someone/projects/x and mail a@b.com")
    assert counts["home_path"] == 1
    assert counts["email"] == 1
    assert verdict(counts) == "redact"


# ------------------------------------------------------------------ pseudonymise


def test_pseudonyms_are_stable_and_do_not_leak_the_original(tmp_path):
    salt = load_or_create_salt(tmp_path)
    scrubber = Pseudonymiser(salt=salt)
    first = scrubber.apply("/home/alice/projects/brain and alice@example.com")
    second = scrubber.apply("/home/alice/other")
    assert "alice" not in first
    assert "alice" not in second
    home = first.split("/projects")[0]
    assert home.startswith("/home/user_")
    assert second.startswith(home)  # the same real path maps to the same pseudonym
    assert "@example.invalid" in first


def test_salt_is_created_once_and_reused(tmp_path):
    first = load_or_create_salt(tmp_path)
    assert load_or_create_salt(tmp_path) == first
    assert (tmp_path / ".salt").stat().st_mode & 0o777 == 0o600


def test_map_is_written_private_and_is_the_only_inverse(tmp_path):
    scrubber = Pseudonymiser(salt=load_or_create_salt(tmp_path))
    scrubber.apply("mail bob@example.com")
    path = scrubber.write_map(tmp_path)
    assert path.stat().st_mode & 0o777 == 0o600
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "bob@example.com" in payload["map"].values()


def test_literal_names_are_replaced():
    scrubber = Pseudonymiser(salt=b"s" * 32, literals={"Ada Lovelace": "Ada Lovelace"})
    out = scrubber.apply("committed by Ada Lovelace today")
    assert "Ada Lovelace" not in out
    assert "person_name_" in out


# ----------------------------------------------------------------- read/extract


def test_read_session_keeps_prompt_and_answer_but_drops_tool_results(tmp_path):
    path = _write_session(tmp_path, "s1", DIFF_PROMPT, "feat(notes): raise merge threshold")
    session = read_session(path)
    assert session.genre == "commit_subject"
    assert session.n_user_turns == 1
    assert "SECRETS" not in session.prompt
    assert session.answer.startswith("feat(notes)")
    assert session_id_of(path) == "s1"


def test_parse_diff_splits_files_and_recovers_prose():
    changes = parse_diff(DIFF_PROMPT)
    assert len(changes) == 1
    assert changes[0].path == "notes/plan.md"
    assert changes[0].section
    assert len(changes[0].added) == 3
    assert len(changes[0].removed) == 1


def test_episodes_are_emitted_in_both_regimes(tmp_path):
    path = _write_session(tmp_path, "s1", DIFF_PROMPT, "feat(notes): raise merge threshold")
    episodes = list(episodes_of(read_session(path)))
    assert {e.regime for e in episodes} == {"long", "short"}
    long_ep = next(e for e in episodes if e.regime == "long")
    short_ep = next(e for e in episodes if e.regime == "short")
    assert long_ep.n_words > short_ep.n_words
    assert long_ep.referent["entity"] == "notes/plan.md"


def test_non_commit_sessions_yield_nothing(tmp_path):
    path = _write_session(tmp_path, "s2", "Just chatting about the weather", "sure")
    assert list(episodes_of(read_session(path))) == []


def test_inventory_counts_genres(tmp_path):
    _write_session(tmp_path, "s1", DIFF_PROMPT, "feat(notes): raise merge threshold")
    _write_session(tmp_path, "s2", "You are the autonomous GLOBAL-MEMORY curator", "ok")
    inventory = take_inventory(tmp_path)
    assert len(inventory) == 2
    assert inventory.genres == {"commit_subject": 1, "memory_curator": 1}
    assert len(inventory.episode_bearing) == 1


# ------------------------------------------------------------------------- mine


def _episodes(n: int) -> list[CandidateEpisode]:
    return [
        CandidateEpisode(
            episode_id=f"e{i}",
            session_id=f"s{i}",
            regime="long",
            timestamp=f"2026-07-{10 + i:02d}T00:00:00Z",
            text=f"episode {i}",
            referent={"entity": f"file{i % 3}.py", "attribute": ""},
        )
        for i in range(n)
    ]


def _vectors(n: int):
    import numpy as np

    rng = np.random.default_rng(0)
    return rng.normal(size=(n, 8))


def test_mined_pairs_are_unlabelled():
    episodes = _episodes(24)
    pairs, audit = mine_pairs(episodes, _vectors(24), config=MiningConfig(per_stratum=3))
    assert pairs
    assert all(pair.label is None for pair in pairs)
    assert all(pair.stratum in STRATA for pair in pairs)
    assert set(audit["selected"]) == set(STRATA)


def test_pair_a_precedes_b_in_time():
    episodes = _episodes(20)
    pairs, _ = mine_pairs(episodes, _vectors(20), config=MiningConfig(per_stratum=5))
    by_id = {e.episode_id: e for e in episodes}
    for pair in pairs:
        assert by_id[pair.a].timestamp <= by_id[pair.b].timestamp


def test_mining_is_deterministic():
    episodes, vectors = _episodes(30), _vectors(30)
    first, _ = mine_pairs(episodes, vectors, config=MiningConfig(per_stratum=4))
    second, _ = mine_pairs(episodes, vectors, config=MiningConfig(per_stratum=4))
    assert [p.pair_id for p in first] == [p.pair_id for p in second]
    assert [p.a for p in first] == [p.a for p in second]


def test_same_entity_stratum_really_shares_the_entity():
    episodes = _episodes(24)
    pairs, _ = mine_pairs(episodes, _vectors(24), config=MiningConfig(per_stratum=8))
    for pair in pairs:
        if pair.stratum == "same_entity":
            assert pair.same_entity == 1


# ------------------------------------------------------------------------ stats


def test_implied_beta_matches_the_papers_worked_value():
    """cos = 0.824 implies beta ~ 0.684 in the paper. Reproduce it."""
    assert implied_beta(0.824) == pytest.approx(0.684, abs=0.002)
    assert implied_beta(0.5) == pytest.approx(0.5, abs=1e-6)


def test_corpus_profile_reports_length_and_overlap():
    texts = ["alpha beta gamma delta", "alpha beta gamma epsilon", "zeta eta theta"]
    clusters = ["c1", "c1", "c2"]
    profile = corpus_profile(texts, clusters, None)
    assert profile["n_episodes"] == 3
    assert profile["n_clusters"] == 2
    assert profile["words"]["mean"] == pytest.approx(3.667, abs=0.01)
    assert profile["lexical"]["token_overlap"]["mean"] == pytest.approx(0.6, abs=1e-6)


def test_compare_reports_a_delta_per_quantity():
    real = corpus_profile(["a b c", "a b d"], ["c1", "c1"], None)
    synthetic = corpus_profile(["a b c d e", "a b c d f"], ["c1", "c1"], None)
    rows = compare(real, synthetic)
    assert rows["episode_words_mean"]["delta"] == pytest.approx(-2.0)
    assert rows["token_overlap_mean"]["real"] < rows["token_overlap_mean"]["synthetic"]


# ------------------------------------------------------------- write safety


def test_assert_ignored_accepts_the_configured_data_path():
    from brain.config import REPO_ROOT

    assert_ignored(REPO_ROOT / "data" / "transcripts")


def test_assert_ignored_refuses_a_tracked_path(tmp_path):
    from brain.config import REPO_ROOT

    tracked = REPO_ROOT / "src" / "brain" / "corpus" / "transcripts"
    with pytest.raises(NotIgnored):
        assert_ignored(tracked)
    assert not (tracked / ".ignore-probe").exists()


def test_written_jsonl_is_private(tmp_path):
    path = write_jsonl(tmp_path / "rows.jsonl", [{"a": 1}])
    assert path.stat().st_mode & 0o777 == 0o600


# -------------------------------------------------------------------- protocol


def test_protocol_states_the_two_bits_and_forbids_the_cosine():
    assert "rho" in PROTOCOL and "nu" in PROTOCOL
    assert "Do not look at the cosine" in PROTOCOL
    for cls in ("DUPLICATE", "SUPERSESSION", "COINCIDENCE", "INTERFERENCE"):
        assert cls in PROTOCOL
    assert "UNCLEAR" in PROTOCOL


def test_effort_estimate_includes_double_labelling():
    pairs, _ = mine_pairs(_episodes(24), _vectors(24), config=MiningConfig(per_stratum=3))
    estimate = effort_estimate(pairs)
    assert estimate["n_judgements_total"] > estimate["n_pairs_total"]
    assert estimate["hours_total"] > 0
    assert "not measured" in estimate["basis"]


def test_pipeline_end_to_end_writes_only_into_the_ignored_path(tmp_path, monkeypatch):
    """Full pass on a two-session store, with the encoder stubbed out.

    Asserts the properties that matter: a credential-bearing episode is
    quarantined rather than written, the surviving episode is pseudonymised, and
    the result payload announces itself as unvalidated with no ground truth.
    """
    import numpy as np

    from brain.config import REPO_ROOT
    from brain.corpus.transcripts import pipeline as pipeline_module

    store = tmp_path / "store"
    store.mkdir()
    _write_session(store, "s1", DIFF_PROMPT, "feat(notes): raise merge threshold")
    # The credential goes on an ADDED line, which is the only text an episode
    # carries; putting it on a removed line would never reach the scanner.
    poisoned = DIFF_PROMPT.replace(
        "+We keep both records and mark them siblings instead of collapsing them.",
        f"+We keep both records and rotate the deploy key aws_key={FAKE_AWS} today.",
    )
    _write_session(store, "s2", poisoned, "chore(notes): rotate the key")

    monkeypatch.setattr(pipeline_module, "embed", lambda texts, **_: np.eye(len(texts), 8))
    monkeypatch.setattr(pipeline_module, "synthetic_profiles", lambda *_a, **_k: {})

    out_dir = REPO_ROOT / "data" / "transcripts" / "_pytest"
    results = tmp_path / "results"
    payload = pipeline_module.run_pipeline(
        store, out_dir=out_dir, results_dir=results, mining=MiningConfig(per_stratum=2)
    )

    assert payload["unvalidated"] is True
    assert payload["ground_truth"] is None
    assert payload["extraction"]["n_quarantined"] >= 1
    assert payload["scan"]["post_pseudonymisation"]["n_blocked"] == 0
    assert payload["scan"]["post_pseudonymisation"]["n_flagged"] == 0

    written = (out_dir / "episodes.jsonl").read_text(encoding="utf-8")
    assert FAKE_AWS not in written
    assert "somebody" not in written  # the cwd owner never reaches the artifact
    quarantine = (out_dir / "quarantine" / "episodes.jsonl").read_text(encoding="utf-8")
    assert FAKE_AWS not in quarantine

    report = next(results.glob("transcripts-*.md")).read_text(encoding="utf-8")
    assert "UNVALIDATED" in report
    assert FAKE_AWS not in report
    assert "Do not look at the cosine" in report

    # results/ is TRACKED, so the two files that land there must survive their
    # own scanner and must carry no absolute home path.
    for artifact in sorted(results.glob("transcripts-*")):
        text = artifact.read_text(encoding="utf-8")
        _, counts = scan_text(text, redact=False)
        assert verdict(counts) in ("clean", "redact"), f"{artifact.name} would be quarantined"
        assert counts.get("home_path", 0) == 0, f"{artifact.name} leaks an absolute home path"


def test_session_record_dict_drops_the_raw_text():
    record = SessionRecord(
        session_id="s", genre="commit_subject", started_at="", ended_at="",
        n_lines=1, n_bytes=1, n_user_turns=1, n_tool_uses=0, prompt="SECRET", answer="SECRET",
    )
    assert "prompt" not in record.to_dict()
    assert "answer" not in record.to_dict()
