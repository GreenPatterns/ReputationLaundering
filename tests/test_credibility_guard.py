"""Ruling R56, the credible-negatives guard of BTP eligibility (Task 3.1b).

An account is eligible for Tier >= 1 when it is trusted and its positive in-edges from trusted raters outnumber
its negative in-edges. With neg_from="trusted", KEEPER's default, only trusted raters' negative in-edges count;
neg_from="all" counts every rater's, the rule of Task 3.1, kept as the ablation. Tested here:
1. the default and the validation of neg_from in every function that applies the guard;
2. neg_from="all" reproduces the rule before R56 bit for bit, and each rule equals an explicit loop oracle;
3. by hand: an untrusted rater's negative rating demotes only under "all", a trusted rater's under both;
4. the exact property: under a frozen calibration, adding any number of accounts that only send ratings (either
   sign, to anyone, at any snapshot) changes no other account's trust, eligibility, tier or tercile cut point
   under "trusted", and does change tiers under "all". On a toy, per snapshot (causal tiers), and on
   Bitcoin-Alpha's training graph with 50 Sybils bad-mouthing 20 honest Tier >= 2 accounts.
The recalibrated eps on the three networks is pinned in tests/test_credibility_btp.py, next to Task 3.1's pins.
CPU only.
"""
import inspect

import numpy as np
import pytest

from keeper.credibility import tiers
from keeper.laundering import signed as S
from tests.credibility_common import (HONEST_N, SYBIL_SNAP, TOY_EPS, graph, honest_rows, in_time_order,
                                      with_rows)

GUARDED = (tiers.btp_tiers, tiers.choose_eps, tiers.causal_tiers, tiers.causal_weights)


def pre_r56_tiers(Tr, n_nodes, calibration):
    """btp_tiers' levels and trust before ruling R56 (commit 18322d0) under a frozen calibration, rebuilt from
    public functions with the old guard copied verbatim: neg_in counts every negative edge. The oracle for "all"."""
    edges = np.asarray(Tr.edges, dtype=np.int64).reshape(-1, 2)
    is_neg = S.neg_mask(Tr.labels).reshape(-1)
    trust = tiers.btp_trust(n_nodes, tiers.endorsement_edges(Tr), calibration.seeds, calibration.K,
                            edge_cap=calibration.edge_cap)
    trusted = trust >= calibration.theta0
    pos_in = np.bincount(edges[~is_neg & trusted[edges[:, 0]], 1], minlength=n_nodes)
    neg_in = np.bincount(edges[is_neg, 1], minlength=n_nodes)
    eligible = trusted & (pos_in > neg_in)
    return tiers.tercile_levels(trust, eligible), trust


def loop_oracle(Tr, n_nodes, calibration, neg_from):
    """Levels by an explicit loop over the rating edges: a positive rating counts when its rater is trusted, a
    negative one when its rater is trusted ("trusted") or always ("all")."""
    trust = tiers.btp_trust(n_nodes, tiers.endorsement_edges(Tr), calibration.seeds, calibration.K,
                            edge_cap=calibration.edge_cap)
    trusted = trust >= calibration.theta0
    pos_in = np.zeros(n_nodes, dtype=np.int64)
    neg_in = np.zeros(n_nodes, dtype=np.int64)
    for (u, v), is_neg in zip(np.asarray(Tr.edges).tolist(), S.neg_mask(Tr.labels).reshape(-1).tolist()):
        if is_neg and (neg_from == "all" or trusted[u]):
            neg_in[v] += 1
        elif not is_neg and trusted[u]:
            pos_in[v] += 1
    return tiers.tercile_levels(trust, trusted & (pos_in > neg_in))


def cut_points(levels, trust):
    """The tercile cut points of trust over the eligible (Tier >= 1) accounts."""
    return np.quantile(trust[levels > 0], [1.0 / 3.0, 2.0 / 3.0]).tolist()


def sybil_rows(n_sybils, targets, seed):
    """(u, v, sign, snapshot) rows sent by n_sybils fresh accounts HONEST_N, HONEST_N + 1, ... that nobody else rates:
    a positive ring and chords among them (endorsements that carry no trust without an attack edge); random ratings
    of either sign to any account, themselves included, at any snapshot, the seed window included; a rating that
    turns from positive to negative and a repeated negative one; and one negative rating of each account in
    ``targets``, so that the rule "all" has something to count."""
    rng = np.random.default_rng(seed)
    fakes = list(range(HONEST_N, HONEST_N + n_sybils))
    rows = []
    for j, f in enumerate(fakes):
        if n_sybils > 3:
            rows += [(f, fakes[(j + 1) % n_sybils], +1, SYBIL_SNAP), (f, fakes[(j + 3) % n_sybils], +1, SYBIL_SNAP)]
        for _ in range(6):
            rows.append((f, int(rng.integers(0, HONEST_N + n_sybils)), int(rng.choice([-1, 1])), int(rng.integers(0, 20))))
        a, b = (int(x) for x in rng.integers(0, HONEST_N, 2))
        rows += [(f, a, +1, 6), (f, a, -1, 7), (f, b, -1, 8), (f, b, -1, 9)]
        rows += [(f, int(v), -1, SYBIL_SNAP) for v in targets]
    return rows


# ------------------------------------------------------------------ 1. the default and the validation
def test_neg_from_defaults_to_trusted_and_is_validated():
    assert tiers.NEG_FROM == ("trusted", "all") and tiers.KEEPER_NEG_FROM == "trusted"
    for fn in GUARDED:
        assert inspect.signature(fn).parameters["neg_from"].default == "trusted", fn.__name__
    G = graph(honest_rows(0))
    for bad in ("none", "ALL", None, 1):
        with pytest.raises(ValueError):
            tiers.btp_tiers(G, HONEST_N, TOY_EPS, neg_from=bad)
        with pytest.raises(ValueError):
            tiers.choose_eps(G, HONEST_N, neg_from=bad)
        with pytest.raises(ValueError):
            tiers.causal_tiers(G, HONEST_N, TOY_EPS, neg_from=bad)
        with pytest.raises(ValueError):
            tiers.causal_weights(G, HONEST_N, TOY_EPS, neg_from=bad)
    # The info dict, which records K and theta0, records the guard as well.
    assert tiers.btp_tiers(G, HONEST_N, TOY_EPS)[2]["neg_from"] == "trusted"
    assert tiers.btp_tiers(G, HONEST_N, TOY_EPS, neg_from="all")[2]["neg_from"] == "all"


def test_choose_eps_applies_and_records_the_guard():
    # Each row of the eps rule comes from btp_tiers under the same guard, and the table records it. On this toy
    # the two rules retain different shares at eps = 0.5 (untrusted raters' negatives stop counting) and pick the
    # same eps.
    G = graph(honest_rows(0))
    legacy = tiers.legacy_bfs(G, HONEST_N) > 0
    tables = {}
    for neg_from in tiers.NEG_FROM:
        eps, tables[neg_from] = tiers.choose_eps(G, HONEST_N, neg_from=neg_from)
        assert eps == 0.2 and tables[neg_from]["neg_from"] == neg_from
        for row in tables[neg_from]["rows"]:
            btp = tiers.btp_tiers(G, HONEST_N, row["eps"], neg_from=neg_from)[0] > 0
            assert (row["n_btp_tier1"], row["n_overlap"]) == (int(btp.sum()), int((btp & legacy).sum()))
    retention = {neg_from: [row["retention"] for row in tables[neg_from]["rows"]] for neg_from in tiers.NEG_FROM}
    assert retention["trusted"][1] > retention["all"][1]
    assert all(t >= a for t, a in zip(retention["trusted"], retention["all"]))   # the eligible set only grows


# ------------------------------------------------------------------ 2. "all" is the old rule; both equal the oracle
@pytest.mark.parametrize("units", [None, 1.0, tiers.KEEPER_EDGE_CAP_UNITS])
@pytest.mark.parametrize("seed", [0, 2, 4])
def test_each_rule_equals_its_oracle_and_all_is_the_rule_before_r56(seed, units):
    G = graph(honest_rows(seed))
    cal = tiers.calibrate_btp(G, HONEST_N, TOY_EPS, edge_cap_units=units)
    old_levels, old_trust = pre_r56_tiers(G, HONEST_N, cal)
    levels, trust, info = tiers.btp_tiers(G, HONEST_N, calibration=cal, neg_from="all")
    assert np.array_equal(levels, old_levels) and np.array_equal(trust, old_trust)
    assert info["n_eligible"] == int((old_levels > 0).sum())
    for neg_from in tiers.NEG_FROM:
        got = tiers.btp_tiers(G, HONEST_N, calibration=cal, neg_from=neg_from)[0]
        assert np.array_equal(got, loop_oracle(G, HONEST_N, cal, neg_from)), neg_from
    # Without a calibration the graph calibrates itself; the guard is applied the same way.
    self_cal = tiers.btp_tiers(G, HONEST_N, TOY_EPS, edge_cap_units=units, neg_from="all")
    assert np.array_equal(self_cal[0], old_levels) and self_cal[2] == info


@pytest.mark.parametrize("units, eps", [(None, 0.05), (tiers.KEEPER_EDGE_CAP_UNITS, 0.02)])
def test_both_rules_equal_their_oracles_on_bitcoinalpha(units, eps, training_graphs):
    Tr, n = training_graphs["bitcoinalpha"]
    cal = tiers.calibrate_btp(Tr, n, eps, edge_cap_units=units)
    old_levels, old_trust = pre_r56_tiers(Tr, n, cal)
    levels, trust, _ = tiers.btp_tiers(Tr, n, calibration=cal, neg_from="all")
    assert np.array_equal(levels, old_levels) and np.array_equal(trust, old_trust)
    assert np.array_equal(tiers.btp_tiers(Tr, n, calibration=cal)[0], loop_oracle(Tr, n, cal, "trusted"))


# ------------------------------------------------------------------ 3. by hand
# Seed 0 (rated by 9 in snapshot 0) endorses 1 and 2; 1 endorses 2; 2 endorses 3. With K = 2 and eps = 0.5
# (theta0 = 0.1) accounts 0-3 are trusted, with trust 0.25, 0.25, 0.375, 0.125; account 9 holds no trust.
CHAIN_ROWS = [(9, 0, +1, 0), (0, 1, +1, 5), (0, 2, +1, 5), (1, 2, +1, 6), (2, 3, +1, 7)]


@pytest.mark.parametrize("neg_from", ["trusted", "all"])
def test_an_untrusted_raters_negative_rating_demotes_only_under_all(neg_from):
    # Account 3's one positive rating comes from 2 (trusted). A negative rating of 3 by untrusted 9 ties it under
    # "all", so 3 drops to Tier 0; under "trusted" it does not count. A negative rating of 3 by trusted 1 counts
    # under both rules. Neither rating is an endorsement, so the trust is the same in every case.
    plain = tiers.btp_tiers(graph(CHAIN_ROWS), 10, 0.5, K=2, neg_from=neg_from)
    assert plain[0].tolist()[:4] == [0, 2, 3, 1]
    by_untrusted = tiers.btp_tiers(graph(CHAIN_ROWS + [(9, 3, -1, 8)]), 10, 0.5, K=2, neg_from=neg_from)
    by_trusted = tiers.btp_tiers(graph(CHAIN_ROWS + [(1, 3, -1, 8)]), 10, 0.5, K=2, neg_from=neg_from)
    assert by_untrusted[0].tolist()[:4] == ([0, 2, 3, 1] if neg_from == "trusted" else [0, 1, 3, 0])
    assert by_trusted[0].tolist()[:4] == [0, 1, 3, 0]
    assert np.array_equal(by_untrusted[1], plain[1]) and np.array_equal(by_trusted[1], plain[1])
    assert by_untrusted[2]["n_eligible"] == (3 if neg_from == "trusted" else 2)


# ------------------------------------------------------------------ 4. the exact property
@pytest.mark.parametrize("units", [None, tiers.KEEPER_EDGE_CAP_UNITS])
@pytest.mark.parametrize("n_sybils", [1, 7, 60])
def test_untrusted_raters_move_no_tier_under_the_credible_negatives_guard(n_sybils, units):
    rows = honest_rows(0)
    H = graph(rows)
    cal = tiers.calibrate_btp(H, HONEST_N, TOY_EPS, edge_cap_units=units)
    honest = {neg_from: tiers.btp_tiers(H, HONEST_N, calibration=cal, neg_from=neg_from) for neg_from in tiers.NEG_FROM}
    targets = np.flatnonzero(honest["all"][0] > 0)
    A, n = graph(in_time_order(rows + sybil_rows(n_sybils, targets, seed=n_sybils))), HONEST_N + n_sybils
    for neg_from in tiers.NEG_FROM:
        levels_h, trust_h, info_h = honest[neg_from]
        levels, trust, info = tiers.btp_tiers(A, n, calibration=cal, neg_from=neg_from)
        # The Sybils hold exactly zero trust, so they are untrusted and the honest trust is unchanged bit for bit.
        assert np.array_equal(trust[:HONEST_N], trust_h) and not trust[HONEST_N:].any()
        assert not levels[HONEST_N:].any() and info["n_trusted"] == info_h["n_trusted"]
        if neg_from == "trusted":
            assert np.array_equal(levels[:HONEST_N], levels_h) and info["n_eligible"] == info_h["n_eligible"]
            assert cut_points(levels, trust) == cut_points(levels_h, trust_h)
        else:
            assert not np.array_equal(levels[:HONEST_N], levels_h)
            assert info["n_eligible"] < info_h["n_eligible"]


def test_untrusted_raters_move_no_causal_row_under_the_credible_negatives_guard():
    # Per snapshot, under the calibration frozen on each honest prefix: the Sybils' ratings, spread over every
    # snapshot (the seed window included), leave every row's honest tiers unchanged under "trusted".
    rows = honest_rows(4)
    H = graph(rows)
    ccal = tiers.calibrate_btp_causal(H, HONEST_N, TOY_EPS, edge_cap_units=tiers.KEEPER_EDGE_CAP_UNITS)
    targets = np.flatnonzero(tiers.btp_tiers(H, HONEST_N, TOY_EPS, neg_from="all")[0] > 0)
    A, n = graph(in_time_order(rows + sybil_rows(20, targets, seed=11))), HONEST_N + 20
    for neg_from in tiers.NEG_FROM:
        L_h = tiers.causal_tiers(H, HONEST_N, calibration=ccal, neg_from=neg_from)
        L_a = tiers.causal_tiers(A, n, calibration=ccal, neg_from=neg_from)
        assert not L_a[:, HONEST_N:].any()
        assert np.array_equal(L_a[:, :HONEST_N], L_h) is (neg_from == "trusted")
        # causal_weights applies the same rule: its rows after t_seed are the weights of these levels.
        W_a = tiers.causal_weights(A, n, calibration=ccal, neg_from=neg_from)
        assert np.array_equal(W_a[5:], np.asarray(tiers.TIER_WEIGHTS)[L_a[5:]]) and (W_a[:5] == 1.0).all()


@pytest.mark.parametrize("units, eps", [(None, 0.05), (tiers.KEEPER_EDGE_CAP_UNITS, 0.02)])
def test_fifty_sybils_bad_mouthing_twenty_alpha_accounts_move_no_tier_only_under_the_guard(units, eps, training_graphs):
    # Bitcoin-Alpha's training graph, frozen calibration (the rule's eps for each variant). 50 fresh Sybils form a
    # positive ring (endorsements among themselves, without an attack edge) and each rates 20 honest Tier >= 2
    # accounts negatively, in the last training snapshot. Under "trusted" no honest account's trust, eligibility,
    # tier or tercile cut point moves; under "all" victims fall to Tier 0. Both rules equal the loop oracle.
    Tr, n = training_graphs["bitcoinalpha"]
    cal = tiers.calibrate_btp(Tr, n, eps, edge_cap_units=units)
    honest = {neg_from: tiers.btp_tiers(Tr, n, calibration=cal, neg_from=neg_from) for neg_from in tiers.NEG_FROM}
    pool = np.flatnonzero((honest["trusted"][0] >= 2) & (honest["all"][0] >= 2))
    victims = np.sort(np.random.default_rng(56).choice(pool, size=20, replace=False))
    fakes = np.arange(n, n + 50)
    ring = [(int(f), int(fakes[(j + 1) % 50]), +1) for j, f in enumerate(fakes)]
    A = with_rows(Tr, ring + [(int(f), int(v), -1) for f in fakes for v in victims], int(np.max(Tr.snap)))
    for neg_from in tiers.NEG_FROM:
        levels_h, trust_h, info_h = honest[neg_from]
        levels, trust, info = tiers.btp_tiers(A, n + 50, calibration=cal, neg_from=neg_from)
        assert np.array_equal(trust[:n], trust_h) and not trust[n:].any() and not levels[n:].any()
        assert np.array_equal(levels, loop_oracle(A, n + 50, cal, neg_from))
        if neg_from == "trusted":
            assert np.array_equal(levels[:n], levels_h) and info["n_eligible"] == info_h["n_eligible"]
            assert cut_points(levels, trust) == cut_points(levels_h, trust_h)
        else:
            dropped = levels[victims] < levels_h[victims]
            assert dropped.sum() >= 10 and (levels[victims][dropped] == 0).all()
