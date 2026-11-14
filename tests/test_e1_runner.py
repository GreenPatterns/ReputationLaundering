"""Task 4.4: keeper.scem.e1_runner (E1 of the protocol, Section 2.3) and its task list, keeper.cluster.e1_tasks.

Small synthetic streams with known answers: batching (one record per warner, the 2% cap, first fit), R34's skips as
the injector applies them, the comparator (no bad-mouthing row, the attacked stream's levels), the revoked share,
the flagged, future-only and honest-mass definitions, latency, the registered grid and closure ages, and determinism.
No test loads a network. CPU only.
"""
import json

import numpy as np
import pytest

from keeper.cluster import e1_tasks
from keeper.laundering import core as C
from keeper.scem import e1_runner as R
from keeper.scem import stage_b
from keeper.scem import witness_attack as WA
from keeper.scem import witness_cred as WC

EPS = {"capped": 0.5, "uncapped": 0.5}


def honest_background(rng, n: int, E: int) -> tuple:
    """E random ratings (85% positive) among n accounts over snapshots 0..20: (edges, sign, snap)."""
    u = rng.integers(0, n, E)
    v = (u + rng.integers(1, n, E)) % n
    snap = rng.integers(0, 21, E)
    sign = np.where(rng.random(E) < 0.15, -1, 1)
    return np.column_stack([u, v]), sign, snap


def toy_stream(n=300, E=3000, n_targets=5, seed=3) -> R.Stream:
    """A random honest stream (85% positive) on n accounts, plus n_targets cohort targets that each receive two
    warnings, at snapshots 7..16, from honest accounts; n_ref is large enough for a batch at k = 5."""
    rng = np.random.default_rng(seed)
    edges, sign, snap = honest_background(rng, n, E)
    targets = np.arange(n, n + n_targets)
    warn = np.column_stack([rng.integers(0, 20, 2 * n_targets), np.repeat(targets, 2)])
    edges = np.concatenate([edges, warn])
    sign = np.concatenate([sign, -np.ones(2 * n_targets, dtype=np.int64)])
    snap = np.concatenate([snap, rng.integers(7, 17, 2 * n_targets)])
    order = np.argsort(snap, kind="stable")
    return R.Stream("toy", 0, edges[order], sign[order], snap[order], n + n_targets, 21, 4, EPS)


def interacting_stream() -> R.Stream:
    """Two victim records on an honest background: A = (1 -> 300) at snapshot 8 and B = (2 -> 301) at 6, plus the
    warning 2 -> 1 at snapshot 10. At lag 3, B's attack on account 2 (snapshot 9) changes the evidence that 2's warning
    lays on account 1 while A is in escrow (quarantine commits A at age 4), whereas A's own attack (snapshot 11) is
    eligible under quarantine (lag + L_f = 4 >= A)."""
    edges, sign, snap = honest_background(np.random.default_rng(1), 300, 3000)
    extra = np.array([(1, 300, -1, 8), (2, 301, -1, 6), (2, 1, -1, 10)])
    edges, sign, snap = (np.concatenate([x, y]) for x, y in ((edges, extra[:, :2]), (sign, extra[:, 2]),
                                                             (snap, extra[:, 3])))
    order = np.argsort(snap, kind="stable")
    return R.Stream("toy", 0, edges[order], sign[order], snap[order], 302, 21, 4, EPS)


# ------------------------------------------------------------------ batches and skips


def test_batch_cap_is_the_largest_count_within_two_percent_of_n_ref():
    for k, n_ref in [(1, 3629), (5, 3629), (10, 5684), (50, 3629), (50, 5684), (50, 10152), (5, 250)]:
        cap = R.batch_cap(k, n_ref)
        assert k * cap <= 0.02 * n_ref < k * (cap + 1)
    assert [R.batch_cap(50, n) for n in (3629, 5684, 10152)] == [1, 2, 4]      # clarification 1
    with pytest.raises(ValueError, match="2% of n_ref"):
        R.batch_cap(5, 249)


def test_batches_hold_one_record_per_warner_and_at_most_cap_records_first_fit():
    records = np.array([10, 11, 12, 13, 14, 15, 16])
    warners = np.array([7, 7, 8, 7, 9, 8, 9])
    batches = [b.tolist() for b in R.make_batches(records, warners, cap=2)]
    assert batches == [[10, 12], [11, 14], [13, 15], [16]]
    flat = [r for b in batches for r in b]
    assert sorted(flat) == records.tolist()
    for b in R.make_batches(records, warners, cap=3):
        ws = warners[np.searchsorted(records, b)]
        assert len(b) <= 3 and len(set(ws.tolist())) == len(b)


def test_attackable_skips_exactly_what_the_injector_skips():
    s = toy_stream()
    records = R.victim_records(s)
    base = R.witness_levels(s, *s.clean, s.n)
    for lag in (0, 1, 7, 14):
        attacked, early, late = R.attackable(s, records, lag)
        att = WA.inject_adversarial_witnesses(*s.clean, s.n, records, 5, lag, 0, np.random.default_rng(0), base, s.K)
        assert attacked.tolist() == att.victims.tolist()
        assert early.tolist() == att.skipped_seed_window.tolist()
        assert late.tolist() == att.skipped_beyond_stream.tolist()


def test_registered_grid_and_closure_ages():
    assert len(R.E1B_CELLS) == 4 * 3 * 5 + 3 + 2 and len(set(R.E1B_CELLS)) == len(R.E1B_CELLS)
    for k in (5, 50):                                    # the H4 cells, ring off and on (clarification 3)
        for ring in (False, True):
            assert R.Cell(k, 0, 1, ring) in R.E1B_CELLS
    assert {c for c in R.E1B_CELLS if c.farming} == {R.Cell(5, 0, 1, True, True), R.Cell(50, 0, 1, True, True)}
    A = {p: round(R.horizon_A(p), 2) for p in R.POLICIES}
    assert A == dict(arrival=1.0, exponential=18.42, erlang2=13.28, erlang3=11.21, erlang5=9.28, quarantine=4.0,
                     no_commit=float("inf"))
    assert R.engine_defaults()["L_f"] == 1
    assert tuple(R.H5_LAG_SCALE * lag for lag in R.E1B_LAGS) == (0, 4, 12, 28, 56)


# ------------------------------------------------------------------ H5's single-record studies


def test_single_plan_runs_exactly_the_pairs_eligible_under_some_committing_policy():
    s = toy_stream()
    records = R.victim_records(s)
    n_run = n_not = 0
    for lag in (0, 1, 3, 7, 14):
        attacked = R.attackable(s, records, lag)[0]
        plan = R.single_plan(s, records, lag)
        # quarantine has the smallest A (4); eligible: lag + L_f >= 4 and t_k + lag + L_f <= K - 1
        want = [r for r in attacked.tolist() if lag + 1 >= 4 and s.snap[r] + lag + 1 <= 20]
        assert [b.tolist() for b in plan["batches"]] == [[r] for r in want]
        assert sorted(want + plan["not_eligible"].tolist()) == sorted(attacked.tolist())
        n_run, n_not = n_run + len(want), n_not + len(plan["not_eligible"])
    assert n_run > 0 and n_not > 0


def test_single_record_units_isolate_a_record_that_a_batch_mate_reaches_in_escrow():
    s = interacting_stream()
    a, b = (int(np.flatnonzero((s.edges[:, 0] == w) & (s.edges[:, 1] == t))[0]) for w, t in ((1, 300), (2, 301)))
    assert {a, b} <= set(R.victim_records(s).tolist())
    clean = R.revoked_share(R.run_engine(s, *s.clean, s.n, None, "quarantine"))[a]
    att = R.inject(s, R.witness_levels(s, *s.clean, s.n), R.Cell(5, 0, 3, False), 0, 0, np.array([a, b]))
    together = R.revoked_share(R.run_engine(s, att.edges, att.sign, att.snap, att.n_nodes, None, "quarantine"))[a]
    assert abs(together - clean) > 1e-3                       # attacked with B, record A changes after it commits
    body = R.study_e1c(s, (3,), single=True)
    assert body["n_units"] == len(body["units"]) and all(len(u["victims"]) == 1 for u in body["units"])
    unit = {int(u["victims"][0]): u for u in body["units"]}[a]
    assert unit["attack"]["quarantine"][0] == unit["clean"]["quarantine"][0] == clean     # alone: exactly unchanged
    assert b in [int(u["victims"][0]) for u in body["units"]]


# ------------------------------------------------------------------ the comparator and the revoked share


def test_comparator_drops_the_badmouthing_rows_and_keeps_the_record_ids():
    s = toy_stream()
    records = R.attackable(s, R.victim_records(s), 1)[0][:3]
    base = R.witness_levels(s, *s.clean, s.n)
    att = R.inject(s, base, R.Cell(5, 1, 1, True), 0, 0, records)
    edges, sign, snap = R.comparator_stream(att)
    bad = att.badmouth_edge_ids.reshape(-1)
    assert len(sign) == len(att.sign) - len(bad) == len(s.sign) + len(records) * (1 + 5)
    E = len(s.sign)
    assert np.array_equal(edges[:E], s.edges) and np.array_equal(sign[:E], s.sign) and np.array_equal(snap[:E], s.snap)
    keep = np.setdiff1d(np.arange(len(att.sign)), bad)
    assert np.array_equal(edges, att.edges[keep]) and (sign[E:] > 0).all()      # attack edges and ring stay


def test_attack_unit_runs_the_comparator_with_the_attacked_levels(monkeypatch):
    s = toy_stream()
    records = R.attackable(s, R.victim_records(s), 1)[0][:2]
    base = R.witness_levels(s, *s.clean, s.n)
    clean = {mb: R.revoked_share(R.run_engine(s, *s.clean, s.n, lv, "erlang2")) for mb, lv in (("off", None),
                                                                                                ("on", base))}
    cell = R.Cell(5, 1, 1, True)
    calls, engine = [], R.run_engine

    def spy(stream, edges, sign, snap, n_nodes, levels, policy):
        calls.append((len(sign), levels))
        return engine(stream, edges, sign, snap, n_nodes, levels, policy)

    monkeypatch.setattr(R, "run_engine", spy)
    unit = R.attack_unit(s, base, clean, cell, 3, 0, records)
    monkeypatch.undo()
    (n_att, off_a), (n_comp, off_c), (_, on_a), (_, on_c) = calls       # attacked, comparator; M-B off, then on
    assert off_a is None and off_c is None and on_c is on_a and n_comp == n_att - 5 * len(records)
    att = R.inject(s, base, cell, 3, 0, records)
    levels = WC.witness_levels(att.edges, att.sign, att.snap, att.n_nodes, 0.5, clean=s.clean, T=21)
    comp = R.comparator_stream(att)
    for mb, lv in (("off", None), ("on", levels)):
        attacked = R.run_engine(s, att.edges, att.sign, att.snap, att.n_nodes, lv, "erlang2")
        want = R.revoked_share(R.run_engine(s, *comp, att.n_nodes, lv, "erlang2"))[records]
        assert np.array_equal(unit[mb]["comparator"], want)
        assert np.array_equal(unit[mb]["attack"], R.revoked_share(attacked)[records])
        assert np.array_equal(unit[mb]["clean"], clean[mb][records])
        rec = attacked["records"]
        assert np.allclose(R.revoked_share(attacked), rec["revoked"], atol=1e-12, rtol=0)   # 1 - omega - pi
    assert np.array_equal(on_a, levels)


# ------------------------------------------------------------------ E1a metrics on a hand-made stream


HAND = [  # (u, v, sign, t)
    (0, 1, +1, 2),    # 0: sender 0 gets 2 negatives after t = 2 and none before: flagged, future-only
    (5, 1, -1, 1),    # 1: 1 receives a negative at t = 1
    (1, 3, +1, 3),    # 2: sender 1 gets 2 negatives after t = 3, one before: flagged, not future-only
    (2, 0, -1, 5),    # 3
    (0, 2, +1, 5),    # 4: after t = 5 sender 0 gets one negative only: not flagged, yet 0 is a flagged sender
    (3, 0, -1, 6),    # 5
    (2, 1, -1, 8),    # 6
    (4, 1, -1, 9),    # 7
]


def hand_stream():
    rows = np.asarray(HAND, dtype=np.int64)
    return R.Stream("toy", 0, rows[:, :2], rows[:, 2], rows[:, 3], 6, 21, 4, EPS)


def test_flagged_future_only_and_honest_mass_follow_their_definitions():
    s = hand_stream()
    out = R.run_engine(s, *s.clean, s.n, None, "erlang2")
    lost = 1.0 - out["effective_weight"]
    m = R.escrow_metrics(s, out)
    assert (m["n_flagged"], m["n_future_only"]) == (2, 1)
    assert m["silent_sender_revocation"] == pytest.approx(lost[[0, 2]].mean(), abs=1e-15)
    assert m["silent_sender_revocation_future_only"] == pytest.approx(lost[0], abs=1e-15)
    honest = [1, 3, 5, 6, 7]                         # senders 2, 3, 4, 5; never senders 0 or 1
    assert m["n_honest_ratings"] == 5 and m["n_honest_senders"] == 4
    assert m["honest_mass_revoked"] == pytest.approx(lost[honest].mean(), abs=1e-15)
    per_sender = [lost[[1]].mean(), lost[[3, 6]].mean(), lost[[5]].mean(), lost[[7]].mean()]
    assert m["honest_mass_revoked_sender_mean"] == pytest.approx(np.mean(per_sender), abs=1e-15)
    assert m["all_retained"] == pytest.approx(out["effective_weight"].mean(), abs=1e-15)


def test_default_rule_is_stage_b_engine_metrics_attenuation():
    s = toy_stream()
    out = R.run_engine(s, *s.clean, s.n, None, "erlang2")
    graph = WC.stream_graph(*s.clean)
    want = stage_b.engine_metrics(graph, s.sign, s.n, dict(N=2, phi=0.5))
    m = R.escrow_metrics(s, out)
    assert m["n_flagged"] == want["n_flagged"] and m["all_retained"] == want["all_retained"]
    assert m["silent_sender_revocation"] == want["attenuation"]


def test_latency_is_the_released_share_of_the_surviving_mass():
    s = toy_stream()
    shares = {p: [R.released_share(R.run_engine(s, *s.clean, s.n, None, p), s.snap, a, s.K) for a in (1, 4)]
              for p in ("arrival", "quarantine", "no_commit", "erlang2")}
    assert shares["arrival"] == [1.0, 1.0] and shares["quarantine"] == [0.0, 1.0] and shares["no_commit"] == [0, 0]
    assert 0 < shares["erlang2"][0] < shares["erlang2"][1] < 1


# ------------------------------------------------------------------ determinism, shards, task list


def test_e1b_is_deterministic_and_its_shards_partition_the_units():
    s = toy_stream()
    cells = (R.Cell(1, 0, 1, True), R.Cell(5, 1, 3, True))
    whole = R.study_e1b(s, cells=cells)
    again = R.study_e1b(s, cells=cells)
    dump = lambda body: json.dumps(C._finite(body), default=C._json_default, sort_keys=True)  # noqa: E731
    assert dump(whole) == dump(again)
    parts = [R.study_e1b(s, shard=(i, 2), cells=cells) for i in range(2)]
    merged = sorted((u for p in parts for u in p["units"]), key=lambda u: (u["cell"], u["batch"]))
    assert dump(merged) == dump(whole["units"]) and whole["n_units"] == len(whole["units"]) >= 3
    for u in whole["units"]:
        assert len(set(u["warners"].tolist())) == len(u["victims"])
        assert len(u["victims"]) <= R.batch_cap(cells[u["cell"]].k, s.n_ref)


def test_e1_task_list_is_cpu_only_and_names_every_shard(tmp_path):
    rows = e1_tasks.e1_tasks(str(tmp_path))
    assert len(rows) == sum(sum(v.values()) for v in e1_tasks.SHARDS.values()) == len({r.id for r in rows})
    for row in rows:
        assert row.command.startswith("CUDA_VISIBLE_DEVICES= ") and f"--out {row.done_path}" in row.command
        assert row.done_path == str(tmp_path / "results" / "scem" / "e1" / f"{row.id}.json")
    assert rows[0].id == "e1b_wiki-rfa_part0of24" and "--shard 0/24" in rows[0].command


def test_single_task_list_holds_only_the_single_record_studies(tmp_path):
    out = tmp_path / "tasks.tsv"
    assert e1_tasks.main(["--results-root", str(tmp_path), "--out", str(out), "--single"]) == 0
    ids = [line.split("\t")[0] for line in out.read_text().splitlines()]
    assert len(ids) == len(set(ids)) == sum(sum(v.values()) for v in e1_tasks.SINGLE_SHARDS.values())
    assert set(e1_tasks.SINGLE_SHARDS) == {"e1c_single", "e1d_h5_single"}
    assert all(i.startswith(("e1c_single_", "e1d_h5_single_")) for i in ids)
    assert ids[0] == "e1d_h5_single_wiki-rfa_part0of66" and "e1c_single_bitcoinalpha_part1of2" in ids
