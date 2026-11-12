"""Tests for the four-class relational corpus (post-write correctness, task 001).

Three things are load bearing here and each has a test that would fail loudly:

1. The labels implement the spec's action map on `(rho, nu)` EXACTLY, and every
   stored label is re-derivable from the episodes rather than trusted.
2. Generation is reproducible to the byte, and the freeze detects a single byte
   of drift.
3. `data/corpus/synthetic/` is byte-identical. Every result published so far,
   including the kill and its theorem, is stated against that hash.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from brain.corpus.loader import CorpusHashMismatch, CorpusNotFrozen
from brain.corpus.relations import (
    ACTION_MAP, CLASSES, COINCIDENCE, DUPLICATE, INTERFERENCE, REGIMES, SUPERSESSION,
    SYNTHETIC_CONTENT_SHA256, RelationCorpus, RelationsAlreadyFrozen, RelationsConfig,
    assert_synthetic_corpus_unchanged, classify, content_hashes, default_relations_config,
    freeze_relations, generate_relations, load_relations, pair_of, spotcheck_rows,
    validate_relations,
)
from brain.corpus.relations.freeze import (
    RELATIONS_V1_CONTENT_SHA256,
    RELATIONS_V2_DIR,
    assert_relations_v1_unchanged,
)
from brain.corpus.relations.generate import SCALED_COUNTS, V2_SCALE, scaled_relations_config
from brain.corpus.relations.loader import RELATIONS_DIR
from brain.corpus.relations.types import serialize_records
from brain.corpus.validate import CorpusValidationError

FULL_TARGETS = {SUPERSESSION: 60, INTERFERENCE: 60, DUPLICATE: 30, COINCIDENCE: 30}


@pytest.fixture(scope="module")
def small_config() -> RelationsConfig:
    return default_relations_config(smoke=True, seed=7)


@pytest.fixture(scope="module")
def small_corpus(small_config: RelationsConfig) -> RelationCorpus:
    return generate_relations(small_config)


@pytest.fixture(scope="module")
def frozen() -> RelationCorpus:
    try:
        return load_relations()
    except CorpusNotFrozen:  # pragma: no cover - only before the first freeze
        pytest.skip("no frozen relational corpus at data/corpus/relations")


def _freeze(tmp_path, corpus, config, **kwargs):
    return freeze_relations(corpus, config, tmp_path, **kwargs)


# --- the immutability gate ------------------------------------------------


def test_existing_frozen_corpus_is_byte_identical() -> None:
    """The single hard constraint on this task. Printed so a run records it."""
    sha = assert_synthetic_corpus_unchanged()
    print(f"\ndata/corpus/synthetic content_sha256={sha}")
    assert sha == SYNTHETIC_CONTENT_SHA256
    assert sha == "25c2953a0b38dc4066e8cff99f01aa3cbd39aca6296018966b57fbde34d24ebb"


# --- the action map -------------------------------------------------------


def test_action_map_is_the_spec_table() -> None:
    assert classify(1, 1) == DUPLICATE
    assert classify(1, 0) == SUPERSESSION
    assert classify(0, 1) == COINCIDENCE
    assert classify(0, 0) == INTERFERENCE
    assert set(ACTION_MAP) == {(1, 1), (1, 0), (0, 1), (0, 0)}
    with pytest.raises(ValueError):
        classify(2, 0)


def test_every_label_re_derives_from_the_episodes(small_corpus: RelationCorpus) -> None:
    """rho and nu come from the slot and the value, never from a score."""
    by_id = small_corpus.episode_by_id()
    for pair in small_corpus.pairs:
        a, b = by_id[pair.a], by_id[pair.b]
        assert pair.rho == int(a.referent == b.referent)
        assert pair.nu == int(a.value_id == b.value_id)
        assert pair.cls == classify(pair.rho, pair.nu)
        assert pair == pair_of(pair.pair_id, a, b)


def test_supersession_pairs_are_time_ordered(small_corpus: RelationCorpus) -> None:
    by_id = small_corpus.episode_by_id()
    supersessions = [p for p in small_corpus.pairs if p.cls == SUPERSESSION]
    assert supersessions
    for pair in supersessions:
        assert by_id[pair.a].date < by_id[pair.b].date
        assert by_id[pair.a].referent == by_id[pair.b].referent
        assert by_id[pair.a].value_id != by_id[pair.b].value_id


def test_pair_ordering_is_independent_of_argument_order(small_corpus: RelationCorpus) -> None:
    by_id = small_corpus.episode_by_id()
    for pair in small_corpus.pairs:
        a, b = by_id[pair.a], by_id[pair.b]
        assert pair_of(pair.pair_id, b, a) == pair_of(pair.pair_id, a, b)


def test_validation_rejects_a_mislabelled_pair(small_corpus: RelationCorpus) -> None:
    pairs = list(small_corpus.pairs)
    index = next(i for i, p in enumerate(pairs) if p.cls == SUPERSESSION)
    pairs[index] = dataclasses.replace(pairs[index], cls=DUPLICATE, nu=1)
    with pytest.raises(CorpusValidationError):
        validate_relations(RelationCorpus(episodes=list(small_corpus.episodes), pairs=pairs))


def test_validation_rejects_a_cross_regime_pair(small_corpus: RelationCorpus) -> None:
    long_episode = next(e for e in small_corpus.episodes if e.regime == "long")
    short_episode = next(e for e in small_corpus.episodes if e.regime == "short")
    with pytest.raises(CorpusValidationError):
        pair_of("x", long_episode, short_episode)


def test_validation_rejects_duplicate_episode_text(small_corpus: RelationCorpus) -> None:
    episodes = list(small_corpus.episodes)
    episodes[1] = dataclasses.replace(episodes[1], text=episodes[0].text)
    with pytest.raises(CorpusValidationError):
        validate_relations(RelationCorpus(episodes=episodes, pairs=list(small_corpus.pairs)))


# --- generation -----------------------------------------------------------


def test_generation_is_reproducible(small_config: RelationsConfig) -> None:
    first, second = generate_relations(small_config), generate_relations(small_config)
    assert content_hashes(first.episodes, first.pairs) == content_hashes(
        second.episodes, second.pairs
    )


def test_a_different_seed_gives_different_content(small_config: RelationsConfig) -> None:
    other = generate_relations(dataclasses.replace(small_config, seed=small_config.seed + 1))
    base = generate_relations(small_config)
    assert content_hashes(base.episodes, base.pairs) != content_hashes(other.episodes, other.pairs)


def test_every_episode_carries_a_referent_a_value_and_a_regime(small_corpus) -> None:
    for episode in small_corpus.episodes:
        assert set(episode.slot) == {"entity", "attribute"}
        assert all(episode.slot.values())
        assert episode.value and episode.value_id
        assert episode.regime in REGIMES
        assert episode.cluster_id


def test_both_regimes_render_the_same_relational_plan(small_corpus: RelationCorpus) -> None:
    """The paired design: only the surface differs, so the regime comparison is clean."""
    by_regime = small_corpus.episodes_by_regime()
    assert len(by_regime["long"]) == len(by_regime["short"]) > 0
    for long_episode, short_episode in zip(by_regime["long"], by_regime["short"]):
        assert long_episode.trace_id.split("-", 1)[1] == short_episode.trace_id.split("-", 1)[1]
        assert long_episode.referent == short_episode.referent
        assert long_episode.value_id == short_episode.value_id
        assert long_episode.date == short_episode.date
        assert long_episode.cluster_id == short_episode.cluster_id
        assert long_episode.text != short_episode.text


def test_the_short_regime_is_shorter_than_the_long_one(small_corpus: RelationCorpus) -> None:
    by_regime = small_corpus.episodes_by_regime()
    longest_short = max(len(e.text.split()) for e in by_regime["short"])
    shortest_long = min(len(e.text.split()) for e in by_regime["long"])
    assert longest_short < shortest_long


def test_generation_refuses_to_invent_pairs_it_cannot_find() -> None:
    """Under-supply is an error, never a quietly smaller class."""
    with pytest.raises(ValueError):
        generate_relations(default_relations_config(smoke=True, coincidence_pairs=10_000))


def test_config_rejects_a_store_larger_than_the_slot_bank() -> None:
    with pytest.raises(ValueError):
        RelationsConfig(n_slots=10_000)


# --- the released artifact ------------------------------------------------


def test_frozen_split_hits_the_evaluation_plan_composition(frozen: RelationCorpus) -> None:
    counts = {cls: len(pairs) for cls, pairs in frozen.pairs_by_class().items()}
    assert counts == FULL_TARGETS
    per_regime = frozen.class_regime_counts()
    for cls in CLASSES:
        assert per_regime[cls]["long"] > 0 and per_regime[cls]["short"] > 0
        assert per_regime[cls]["long"] == per_regime[cls]["short"] == FULL_TARGETS[cls] // 2


def test_frozen_split_verifies_and_validates(frozen: RelationCorpus) -> None:
    validate_relations(frozen)
    assert frozen.content_hash == content_hashes(frozen.episodes, frozen.pairs)["content_sha256"]
    assert frozen.manifest["counts"]["episodes"] == len(frozen.episodes)
    assert frozen.manifest["hash_scope"].startswith("episodes.jsonl and pairs.jsonl only")
    assert frozen.manifest["immutability"]["synthetic_content_sha256"] == SYNTHETIC_CONTENT_SHA256


def test_regeneration_from_the_recorded_seed_reproduces_the_hash(frozen: RelationCorpus) -> None:
    fields = {f.name for f in dataclasses.fields(RelationsConfig)}
    recorded = {k: v for k, v in frozen.manifest["generator"].items() if k in fields}
    regenerated = generate_relations(RelationsConfig(**recorded))
    assert (
        content_hashes(regenerated.episodes, regenerated.pairs)["content_sha256"]
        == frozen.content_hash
    )


def test_spotcheck_ships_forty_pairs_with_full_text(frozen: RelationCorpus) -> None:
    rows = [
        json.loads(line)
        for line in (RELATIONS_DIR / "spotcheck.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == frozen.manifest["spotcheck"]["n"] == 40
    assert frozen.manifest["spotcheck"]["seed"] is not None
    assert {row["cls"] for row in rows} == set(CLASSES)
    assert {row["regime"] for row in rows} == set(REGIMES)
    for row in rows:
        assert row["text_a"] and row["text_b"] and row["text_a"] != row["text_b"]
        assert row["human_verdict"] == ""


def test_spotcheck_sampling_is_seeded(small_corpus: RelationCorpus) -> None:
    first = spotcheck_rows(small_corpus, n=6, seed=3)
    assert first == spotcheck_rows(small_corpus, n=6, seed=3)
    assert first != spotcheck_rows(small_corpus, n=6, seed=4)


# --- freeze, verify, tamper ------------------------------------------------


def test_freeze_writes_every_file_and_a_complete_manifest(tmp_path, small_corpus, small_config):
    manifest = _freeze(tmp_path, small_corpus, small_config)
    assert manifest["seed"] == small_config.seed
    assert manifest["counts"]["pairs"] == len(small_corpus.pairs)
    assert len(manifest["content_sha256"]) == 64
    assert "similarity" not in manifest  # measuring the geometry here invites tuning
    for name in ("episodes.jsonl", "pairs.jsonl", "manifest.json", "spotcheck.jsonl"):
        assert (tmp_path / name).is_file()
    assert list(json.loads((tmp_path / "manifest.json").read_text())) == sorted(manifest)


def test_round_trip_through_the_loader(tmp_path, small_corpus, small_config) -> None:
    _freeze(tmp_path, small_corpus, small_config)
    loaded = load_relations(tmp_path)
    assert [e.to_dict() for e in loaded.episodes] == [e.to_dict() for e in small_corpus.episodes]
    assert [p.to_dict() for p in loaded.pairs] == [p.to_dict() for p in small_corpus.pairs]


def test_loader_detects_a_single_byte_edit(tmp_path, small_corpus, small_config) -> None:
    _freeze(tmp_path, small_corpus, small_config)
    path = tmp_path / "episodes.jsonl"
    raw = path.read_bytes()
    index = raw.index(b"Set the ") if b"Set the " in raw else 0
    path.write_bytes(raw[:index] + bytes([raw[index] ^ 0x20]) + raw[index + 1 :])
    with pytest.raises(CorpusHashMismatch):
        load_relations(tmp_path)
    assert load_relations(tmp_path, verify=False) is not None


def test_loader_detects_a_tampered_pair(tmp_path, small_corpus, small_config) -> None:
    _freeze(tmp_path, small_corpus, small_config)
    pairs = list(small_corpus.pairs)
    pairs[0] = dataclasses.replace(pairs[0], cls=DUPLICATE)
    (tmp_path / "pairs.jsonl").write_bytes(serialize_records(pairs, key="pair_id"))
    with pytest.raises(CorpusHashMismatch):
        load_relations(tmp_path)


def test_loader_requires_a_manifest(tmp_path) -> None:
    with pytest.raises(CorpusNotFrozen):
        load_relations(tmp_path)


def test_refreezing_identical_content_is_a_no_op(tmp_path, small_corpus, small_config) -> None:
    first = _freeze(tmp_path, small_corpus, small_config)
    second = _freeze(tmp_path, small_corpus, small_config)
    assert (second["created_at"], second["content_sha256"]) == (
        first["created_at"],
        first["content_sha256"],
    )


def test_refreezing_different_content_is_refused(tmp_path, small_corpus, small_config) -> None:
    _freeze(tmp_path, small_corpus, small_config)
    other_config = dataclasses.replace(small_config, seed=small_config.seed + 1)
    other = generate_relations(other_config)
    with pytest.raises(RelationsAlreadyFrozen):
        _freeze(tmp_path, other, other_config)
    forced = _freeze(tmp_path, other, other_config, force=True)
    assert load_relations(tmp_path).content_hash == forced["content_sha256"]


# --- task 007: the v2 power split -----------------------------------------
#
# v2 exists for one reason: the K1 interval straddled 0.90 in the long regime, so
# the ceiling had to be recomputed at larger n. That only means anything if v2 is
# the SAME distribution as v1 with more draws, so the tests below check exactly
# that: counts scale by an integer, every shape parameter is identical, and the
# conventions block, the action map and the record schema are byte-for-byte the
# ones F22 was measured under.

V2_SCALE_UNDER_TEST = V2_SCALE


@pytest.fixture(scope="module")
def v2_config() -> RelationsConfig:
    return scaled_relations_config(V2_SCALE_UNDER_TEST)


@pytest.fixture(scope="module")
def v2_corpus(v2_config: RelationsConfig) -> RelationCorpus:
    return generate_relations(v2_config)


@pytest.fixture(scope="module")
def v2_frozen() -> RelationCorpus:
    try:
        return load_relations(RELATIONS_V2_DIR)
    except CorpusNotFrozen:  # pragma: no cover - only before the first v2 freeze
        pytest.skip("no frozen v2 split at data/corpus/relations-v2")


def _content_hash(corpus: RelationCorpus) -> str:
    return content_hashes(corpus.episodes, corpus.pairs)["content_sha256"]


def _same_topic_supersessions(corpus: RelationCorpus, regime: str) -> list:
    """Derivable same-topic SUPERSESSION pairs, from the episodes alone.

    Only same-referent episodes can be (rho = 1), so grouping by `slot_id` gives
    the same answer as the gate's exhaustive enumeration at a fraction of the work.
    """
    by_slot: dict[str, list] = {}
    for episode in sorted(corpus.episodes_by_regime()[regime], key=lambda e: e.trace_id):
        by_slot.setdefault(episode.slot_id, []).append(episode)
    pairs = []
    for slot in sorted(by_slot):
        group = by_slot[slot]
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                pair = pair_of(f"{a.trace_id}|{b.trace_id}", a, b)
                if pair.cls == SUPERSESSION and pair.same_topic:
                    pairs.append(pair)
    return pairs


def test_the_v1_relational_split_is_byte_identical() -> None:
    """F22 is stated against this hash; v2 is additive over it, never a rewrite."""
    sha = assert_relations_v1_unchanged()
    print(f"\ndata/corpus/relations content_sha256={sha}")
    assert sha == RELATIONS_V1_CONTENT_SHA256


def test_the_frozen_gate_still_points_at_v1() -> None:
    """The measurement is frozen: v2 is read through the loader, not by editing it."""
    from brain.experiments.gate_threshold.report import RELATIONS_SHA

    assert RELATIONS_SHA == RELATIONS_V1_CONTENT_SHA256


def test_the_scale_moves_counts_and_nothing_else(v2_config: RelationsConfig) -> None:
    base = RelationsConfig()
    for field in dataclasses.fields(RelationsConfig):
        got, want = getattr(v2_config, field.name), getattr(base, field.name)
        if field.name in SCALED_COUNTS:
            assert got == want * V2_SCALE_UNDER_TEST, field.name
        elif field.name != "label":
            assert got == want, f"{field.name} is a shape parameter and must not move"
    assert v2_config.states_per_slot == base.states_per_slot == 3
    assert (v2_config.home_topic_prob, v2_config.restatement_prob) == (0.6, 0.4)
    assert (v2_config.seed, v2_config.generator) == (base.seed, base.generator)


def test_the_scale_must_be_a_positive_integer() -> None:
    with pytest.raises(ValueError):
        scaled_relations_config(0)
    assert scaled_relations_config(1) == RelationsConfig()


def test_v2_class_composition_is_the_v1_composition_times_the_scale(v2_corpus) -> None:
    """Growing one class and not the others would move the marginal composition."""
    v1 = generate_relations(default_relations_config())
    v1_counts = {cls: len(p) for cls, p in v1.pairs_by_class().items()}
    v2_counts = {cls: len(p) for cls, p in v2_corpus.pairs_by_class().items()}
    assert v2_counts == {cls: n * V2_SCALE_UNDER_TEST for cls, n in v1_counts.items()}
    for cls in CLASSES:
        per_regime = v2_corpus.class_regime_counts()[cls]
        assert per_regime["long"] == per_regime["short"] > 0


def test_v2_clears_the_power_target(v2_corpus: RelationCorpus) -> None:
    """The task 007 target, stated in pairs and in CLUSTERS, which is the unit."""
    pairs = _same_topic_supersessions(v2_corpus, "long")
    clusters = {p.cluster_id for p in pairs}
    print(f"\nv2 long same-topic supersession pairs={len(pairs)} clusters={len(clusters)}")
    assert len(pairs) >= 180
    assert len(clusters) >= 30
    v1_pairs = _same_topic_supersessions(generate_relations(default_relations_config()), "long")
    assert len(v1_pairs) == 69 and len({p.cluster_id for p in v1_pairs}) == 19


def test_v2_keeps_the_v1_surface_statistics(v2_corpus: RelationCorpus) -> None:
    """Same renderer, same regimes: the text length distribution must not move.

    This is the cheapest available check that v2 is the same distribution rather
    than a differently shaped corpus, and it is the one that would catch a
    template change hiding inside a count change.
    """
    v1 = generate_relations(default_relations_config())
    for regime in REGIMES:
        one = [len(e.text.split()) for e in v1.episodes_by_regime()[regime]]
        two = [len(e.text.split()) for e in v2_corpus.episodes_by_regime()[regime]]
        assert abs(sum(one) / len(one) - sum(two) / len(two)) < 1.0, regime
        assert min(two) >= min(one) - 3 and max(two) <= max(one) + 3, regime


def test_v2_frozen_split_verifies_and_is_a_separate_artifact(v2_frozen, frozen) -> None:
    validate_relations(v2_frozen)
    assert v2_frozen.content_hash == content_hashes(v2_frozen.episodes, v2_frozen.pairs)["content_sha256"]
    assert v2_frozen.content_hash != frozen.content_hash
    assert v2_frozen.manifest["counts"]["episodes"] == len(v2_frozen.episodes)
    immutability = v2_frozen.manifest["immutability"]
    assert immutability["synthetic_content_sha256"] == SYNTHETIC_CONTENT_SHA256
    assert immutability["relations_v1_content_sha256"] == RELATIONS_V1_CONTENT_SHA256
    print(f"\ndata/corpus/relations-v2 content_sha256={v2_frozen.content_hash}")


def test_v2_preserves_the_v1_conventions_and_action_map(v2_frozen, frozen) -> None:
    """The one thing that would break comparability with F22, tested directly."""
    assert v2_frozen.manifest["conventions"] == frozen.manifest["conventions"]
    assert v2_frozen.manifest["action_map"] == frozen.manifest["action_map"]
    assert v2_frozen.manifest["schema_version"] == frozen.manifest["schema_version"]
    assert v2_frozen.manifest["hash_scope"] == frozen.manifest["hash_scope"]
    generator = v2_frozen.manifest["generator"]
    assert generator["generator"] == frozen.manifest["generator"]["generator"]
    assert generator["regimes"] == frozen.manifest["generator"]["regimes"]
    for shape in ("states_per_slot", "restatement_prob", "home_topic_prob", "date_span", "seed"):
        assert generator[shape] == frozen.manifest["generator"][shape], shape


def test_v2_regeneration_from_the_recorded_seed_reproduces_the_hash(v2_frozen) -> None:
    fields = {f.name for f in dataclasses.fields(RelationsConfig)}
    recorded = {k: v for k, v in v2_frozen.manifest["generator"].items() if k in fields}
    regenerated = generate_relations(RelationsConfig(**recorded))
    assert (
        content_hashes(regenerated.episodes, regenerated.pairs)["content_sha256"]
        == v2_frozen.content_hash
    )


def test_v2_freeze_detects_tampering(tmp_path, v2_corpus, v2_config) -> None:
    _freeze(tmp_path, v2_corpus, v2_config)
    assert load_relations(tmp_path).content_hash == _content_hash(v2_corpus)
    path = tmp_path / "episodes.jsonl"
    raw = path.read_bytes()
    index = raw.index(b"Set the ") if b"Set the " in raw else 0
    path.write_bytes(raw[:index] + bytes([raw[index] ^ 0x20]) + raw[index + 1:])
    with pytest.raises(CorpusHashMismatch):
        load_relations(tmp_path)
