"""BTP credibility tiers of keeper.credibility.tiers (Task 3.1, upgrade M-A; rulings R32, R33b, R37, R38).

The legacy tiers are tested in tests/test_credibility_legacy.py. Here, against the design note and rulings:
5. seed_set equals the legacy seeds; endorsement_edges keeps each pair's last rating in scope; propagation
   steps by hand, uncapped and capped; total trust stays 1; edge_cap=None is the approved rule bit for bit.
6. The Sybil property: a Sybil region without attack edges holds exactly 0 trust; with g attack edges its
   trust is at most bound_sybil_trust over the iteration history (attained without outflow), and at most
   floor(bound / theta0) fakes reach Tier >= 1 while legacy_bfs admits the whole region. With the calibration
   frozen on the honest graph (R32) that cap does not grow with the region; with the per-endorsement cap (R37)
   it is at most floor(K g edge_cap_units). Admission is also pinned from below, so a regression to zero fails.
7. btp_tiers (eps required, info, terciles, determinism, calibration), causal_tiers (strict prefix, frozen
   per-snapshot calibration), causal_weights, choose_eps (toys with known retentions, the fallback grid, the
   three networks, KEEPER's cap of R38), endorser_inflow_scores (R33b), and the --report CLI and blocks.
Ruling R56 (Task 3.1b) makes the credible-negatives guard (neg_from="trusted") the default; its tests are in
tests/test_credibility_guard.py. Task 3.1's pins were measured under the rule that counts every rater's negative
ratings, so the tests that pin them are parameterized on neg_from="all", next to the new pins for "trusted".
"""
import dataclasses
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from keeper.credibility import tiers
from keeper.credibility.tiers import TIER_WEIGHTS, legacy_bfs
from keeper.laundering import signed as S
from tests.credibility_common import (HONEST_N, NETWORKS, RING, SEED_CORE, SYBIL_SNAP, TOY_EPS,
                                      attacked_region, graph, honest_rows, hub_rows, in_time_order,
                                      max_inflow_endorsers, seed_attacker, seed_endorsers)

ROOT = Path(__file__).resolve().parents[1]


# ================================================================== Task 3.1: BTP tiers (upgrade M-A)
# The BTP functions are reached through the module (tiers.seed_set, ...), so each new test fails on its own
# until its function exists, and the Task 2.2 tests above keep running.
EPS_GRID = (1.0, 0.5, 0.2, 0.1, 0.05)


def legacy_seeds(Tr, n_nodes):
    """The seed initialisation of legacy_levels (seeded branch), copied verbatim: the oracle for seed_set."""
    y_neg = S.class_ids(Tr.labels)
    early_pos = np.zeros(n_nodes, dtype=np.int64)
    early_neg = np.zeros(n_nodes, dtype=np.int64)
    for (u, v), is_neg, sn in zip(Tr.edges, y_neg, Tr.snap):
        if int(sn) <= 4:
            if is_neg:
                early_neg[int(v)] += 1
            else:
                early_pos[int(v)] += 1
    return np.where(early_pos > early_neg)[0]


def btp_history(G, n_nodes, K):
    """The iterates t^(0..K) on G's endorsement graph and seeds, and that graph's out-degrees."""
    endorse = tiers.endorsement_edges(G)
    history = tiers.btp_trust(n_nodes, endorse, tiers.seed_set(G, n_nodes), K, return_history=True)
    return history, np.bincount(endorse[:, 0], minlength=n_nodes)


# ------------------------------------------------------------------ 5. seeds, endorsements, propagation
# Node 0: two positive and one negative rating in snapshots 0-4 (seed). Node 1: one of each (no seed).
# Node 2: positive ratings only in snapshots 5 and 6 (no seed). Node 3: two positive, one negative (seed).
SEED_ROWS = [(1, 0, +1, 0), (2, 0, -1, 1), (3, 0, +1, 2), (0, 1, +1, 3), (2, 1, -1, 4),
             (0, 2, +1, 5), (1, 2, +1, 6), (2, 3, -1, 0), (0, 3, +1, 4), (1, 3, +1, 4)]


def test_seed_set_equals_the_legacy_seeds_on_toy_graphs():
    for rows, n in ((SEED_ROWS, 4), (RING, 8), (honest_rows(0), HONEST_N)):
        G = graph(rows)
        got = tiers.seed_set(G, n)
        assert got.dtype == np.int64 and np.array_equal(got, legacy_seeds(G, n))
    assert tiers.seed_set(graph(SEED_ROWS), 4).tolist() == [0, 3]
    assert tiers.seed_set(graph(SEED_ROWS), 4, t_seed=6).tolist() == [0, 2, 3]   # snapshots 5, 6 count


def test_seed_set_equals_the_legacy_seeds_on_bitcoinalpha(training_graphs):
    Tr, n = training_graphs["bitcoinalpha"]
    got = tiers.seed_set(Tr, n)
    assert np.array_equal(got, legacy_seeds(Tr, n))
    assert len(got) == 1188          # |S0| of the Task 2.2 census (results/analysis/defense_extra.json)


ENDORSE_ROWS = [(0, 1, +1, 0), (0, 1, -1, 2),       # + then -: not an endorsement
                (0, 2, -1, 0), (0, 2, +1, 1),       # - then +: an endorsement
                (0, 3, +1, 1), (0, 3, +1, 3),       # repeated +: one row
                (2, 2, +1, 1),                      # a self-rating is never an endorsement
                (3, 0, +1, 3), (3, 0, -1, 3),       # same snapshot: the later row is the last rating
                (1, 0, +1, 4), (2, 3, -1, 5), (4, 1, +1, 6), (1, 4, -1, 7), (1, 4, +1, 7)]


def test_endorsement_edges_keep_the_last_rating_of_each_pair():
    G = graph(ENDORSE_ROWS)
    got = tiers.endorsement_edges(G)
    assert got.dtype == np.int64 and got.tolist() == [[0, 2], [0, 3], [1, 0], [1, 4], [4, 1]]
    assert tiers.endorsement_edges(G, t_max=6).tolist() == [[0, 2], [0, 3], [1, 0], [4, 1]]
    assert tiers.endorsement_edges(G, t_max=6, strict=True).tolist() == [[0, 2], [0, 3], [1, 0]]
    assert tiers.endorsement_edges(G, t_max=1).tolist() == [[0, 1], [0, 2], [0, 3]]   # 0 -> 1 still + at 1
    assert tiers.endorsement_edges(G, t_max=0, strict=True).shape == (0, 2)


def test_n_active_counts_the_distinct_endpoints_in_scope():
    G = graph(ENDORSE_ROWS)
    assert tiers.n_active(G) == 5                         # accounts 0..4
    assert tiers.n_active(G, t_max=5) == 4                # account 4 first appears in snapshot 6
    assert tiers.n_active(G, t_max=6, strict=True) == 4
    assert tiers.n_active(G, t_max=0) == 3                # snapshot 0 holds 0 -> 1 and 0 -> 2
    assert tiers.n_active(G, t_max=0, strict=True) == 0


def test_propagation_steps_by_hand():
    # Seed 0 keeps half its trust and splits the other half over its endorsees 1 and 2. Node 1 passes half of
    # what it holds to node 2; node 2, without endorsees, keeps everything.
    endorse = np.array([[0, 1], [0, 2], [1, 2]])
    history = tiers.btp_trust(3, endorse, np.array([0]), 2, return_history=True)
    assert [t.tolist() for t in history] == [[1.0, 0.0, 0.0], [0.5, 0.25, 0.25], [0.25, 0.25, 0.5]]
    assert tiers.btp_trust(3, endorse, np.array([0]), 2).tolist() == [0.25, 0.25, 0.5]
    assert tiers.btp_trust(3, endorse, np.array([0]), 1, laziness=1.0).tolist() == [0.0, 0.5, 0.5]
    assert tiers.btp_trust(3, endorse, np.array([0, 2]), 0).tolist() == [0.5, 0.0, 0.5]    # 1/|S0| each


@pytest.mark.parametrize("laziness", [0.5, 0.2, 1.0])
def test_trust_is_conserved_on_a_random_graph(laziness):
    rng = np.random.default_rng(3)
    n = 500
    pairs = np.unique(rng.integers(0, n, size=(4000, 2)), axis=0)
    endorse = pairs[(pairs[:, 0] != pairs[:, 1]) & (pairs[:, 0] < 450)]   # accounts 450..499 are sinks
    seeds = rng.choice(n, size=40, replace=False)
    K = 9                                                                   # ceil(log2 500)
    history = tiers.btp_trust(n, endorse, seeds, K, laziness=laziness, return_history=True)
    assert len(history) == K + 1
    for t in history:
        assert abs(t.sum() - 1.0) <= 1e-12 and (t >= 0).all()
    assert (history[-1][450:] > 0).any()                                    # trust reached the sinks


def approved_btp_rule(n_nodes, endorse, seeds, K, laziness=0.5):
    """The uncapped propagation of commit 4a3b8b2 (the approved M-A rule), copied verbatim: the oracle."""
    trust = np.zeros(n_nodes, dtype=np.float64)
    if len(seeds):
        trust[seeds] = 1.0 / len(seeds)
    src, dst = endorse[:, 0], endorse[:, 1]
    out_deg = np.bincount(src, minlength=n_nodes)
    kept = np.where(out_deg > 0, 1.0 - laziness, 1.0)
    edge_share = laziness / out_deg[src]
    history = [trust]
    for _ in range(K):
        trust = kept * trust + np.bincount(dst, weights=trust[src] * edge_share, minlength=n_nodes)
        history.append(trust)
    return history


def random_endorsements(seed, n=400, m=3000):
    rng = np.random.default_rng(seed)
    pairs = np.unique(rng.integers(0, n, size=(m, 2)), axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]], np.sort(rng.choice(n, size=30, replace=False))


def test_uncapped_rule_is_byte_identical_to_the_approved_rule():
    # Ruling R37 keeps edge_cap=None exactly the approved rule. An infinite cap never binds, so it takes the
    # capped code path and must still reproduce every iterate bit for bit.
    endorse, seeds = random_endorsements(5)
    want = approved_btp_rule(400, endorse, seeds, 9)
    for edge_cap in (None, math.inf):
        got = tiers.btp_trust(400, endorse, seeds, 9, return_history=True, edge_cap=edge_cap)
        assert len(got) == len(want) and all(np.array_equal(a, b) for a, b in zip(got, want))


def test_capped_propagation_by_hand():
    # c_e = 0.1. Step 1: seed 0 would send 0.25 per endorsement, so each carries 0.1 and 0 keeps 0.8. Step 2:
    # 0 again sends 0.1 to each of 1 and 2 and keeps 0.6; node 1 would send 0.05 < 0.1, so it sends 0.05 and
    # keeps 0.05; node 2, a sink, keeps its 0.1 and gains 0.15.
    endorse = np.array([[0, 1], [0, 2], [1, 2]])
    history = tiers.btp_trust(3, endorse, np.array([0]), 2, return_history=True, edge_cap=0.1)
    for got, want in zip(history, ([1.0, 0.0, 0.0], [0.8, 0.1, 0.1], [0.6, 0.15, 0.25])):
        assert got.tolist() == pytest.approx(want, rel=0, abs=1e-15)
    for bad in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError):
            tiers.btp_trust(3, endorse, np.array([0]), 2, edge_cap=bad)


@pytest.mark.parametrize("edge_cap", [1e-4, 1e-3, 5e-3])          # all below the largest share, 1/120
def test_capped_trust_is_conserved_on_a_random_graph(edge_cap):
    endorse, seeds = random_endorsements(6)
    history = tiers.btp_trust(400, endorse, seeds, 9, return_history=True, edge_cap=edge_cap)
    for t in history:
        assert abs(t.sum() - 1.0) <= 1e-12 and (t >= 0).all()
    # Each endorsement carries at most c_e per step and inflows are non-negative, so an account loses at most
    # dout * c_e per step. The uncapped rule breaks this, so the cap binds.
    out_deg = np.bincount(endorse[:, 0], minlength=400)

    def loses_at_most_its_cap(h):
        return all((after >= before - out_deg * edge_cap - 1e-15).all() for before, after in zip(h[:-1], h[1:]))

    assert loses_at_most_its_cap(history)
    assert not loses_at_most_its_cap(tiers.btp_trust(400, endorse, seeds, 9, return_history=True))


def test_btp_trust_rejects_malformed_input():
    one = np.array([[0, 1]])
    for bad in ([[0, 0]], [[0, 1], [0, 1]], [[0, 3]], [[-1, 0]]):          # self-loop, repeat, out of range
        with pytest.raises(ValueError):
            tiers.btp_trust(3, np.array(bad), np.array([0]), 1)
    for seeds in ([0, 0], [3]):
        with pytest.raises(ValueError):
            tiers.btp_trust(3, one, np.array(seeds), 1)
    with pytest.raises(ValueError):
        tiers.btp_trust(3, one, np.array([0]), -1)
    with pytest.raises(ValueError):
        tiers.btp_trust(3, one, np.array([0]), 1, laziness=1.5)
    no_seed = tiers.btp_trust(3, one, np.zeros(0, dtype=np.int64), 2)      # no seed: no trust anywhere
    assert no_seed.tolist() == [0.0, 0.0, 0.0]


# ------------------------------------------------------------------ 6. the Sybil property
@pytest.mark.parametrize("non_endorsing", [False, True])
def test_sybil_region_without_attack_edges_holds_exactly_zero_trust(non_endorsing):
    # Ten Sybils rate each other (a ring and chords) and endorse honest accounts, after the seed window. No
    # honest account endorses a Sybil. With non_endorsing, honest accounts send Sybils a negative rating and
    # a rating that turns from positive to negative; neither is an endorsement.
    s = 10
    fakes = list(range(HONEST_N, HONEST_N + s))
    syb = [(fakes[j], fakes[(j + 1) % s], +1, SYBIL_SNAP) for j in range(s)]
    syb += [(fakes[j], fakes[(j + 3) % s], +1, SYBIL_SNAP) for j in range(s)]
    syb += [(fakes[j], j, +1, SYBIL_SNAP) for j in range(s)]
    if non_endorsing:
        syb += [(0, fakes[0], -1, SYBIL_SNAP), (1, fakes[1], +1, 15), (1, fakes[1], -1, SYBIL_SNAP)]
    G, n = graph(in_time_order(honest_rows(1) + syb)), HONEST_N + s
    levels, trust, info = tiers.btp_tiers(G, n, TOY_EPS)
    assert not set(tiers.seed_set(G, n).tolist()) & set(fakes)
    assert trust[fakes].tolist() == [0.0] * s and not levels[fakes].any()
    history, _ = btp_history(G, n, info["K"])
    assert all(t[fakes].max() == 0.0 for t in history)
    assert np.array_equal(history[-1], trust) and abs(trust.sum() - 1.0) <= 1e-12


@pytest.mark.parametrize("outflow", [True, False])
@pytest.mark.parametrize("g", [1, 2, 5])
def test_sybil_trust_is_bounded_by_the_attack_edge_inflow(g, outflow):
    # g honest accounts that hold trust endorse Sybils (the attack edges). The Sybils rate each other and,
    # with outflow, endorse and bad-mouth honest accounts. Their total trust after K steps is at most
    # 0.5 * sum_{i<K} sum_a t_i(u_a) / dout(u_a). Without outflow everything that enters stays: equality.
    s = 8
    fakes = list(range(HONEST_N, HONEST_N + s))
    rows = honest_rows(2)
    pre_trust = tiers.btp_tiers(graph(rows), HONEST_N, TOY_EPS)[1]
    attackers = np.random.default_rng(10 + g).choice(np.flatnonzero(pre_trust > 0), size=g, replace=False)
    attack = [(int(u), fakes[a % s]) for a, u in enumerate(attackers)]
    syb = [(u, f, +1, SYBIL_SNAP - 1) for u, f in attack]
    syb += [(fakes[j], fakes[(j + 1) % s], +1, SYBIL_SNAP) for j in range(s)]
    syb += [(fakes[j], fakes[(j + 3) % s], +1, SYBIL_SNAP) for j in range(0, s, 2)]
    if outflow:
        syb += [(fakes[j], 3 * j, +1, SYBIL_SNAP) for j in range(s)] + [(fakes[0], 1, -1, SYBIL_SNAP)]
    G, n = graph(in_time_order(rows + syb)), HONEST_N + s
    history, out_deg = btp_history(G, n, tiers.btp_tiers(G, n, TOY_EPS)[2]["K"])
    bound = tiers.bound_sybil_trust(history, np.array(attack), out_deg)
    sybil_trust = history[-1][fakes].sum()
    assert not set(tiers.seed_set(G, n).tolist()) & set(fakes)
    assert 0.0 < sybil_trust <= bound + 1e-15
    if not outflow:
        assert abs(sybil_trust - bound) <= 1e-15


def test_bound_sybil_trust_by_hand():
    history = [np.array([0.6, 0.4, 0.0]), np.array([0.3, 0.5, 0.2]), np.array([0.2, 0.4, 0.4])]
    out_deg = np.array([2, 1, 0])
    # K = 2 steps: 0.5 * [(0.6/2 + 0.4/1) + (0.3/2 + 0.5/1)]; the last iterate t^(K) carries nothing in.
    got = tiers.bound_sybil_trust(history, np.array([[0, 2], [1, 2]]), out_deg)
    assert got == pytest.approx(0.5 * ((0.3 + 0.4) + (0.15 + 0.5)), rel=0, abs=1e-15)
    assert tiers.bound_sybil_trust(history[:1], np.array([[0, 2]]), out_deg) == 0.0     # K = 0
    with pytest.raises(ValueError):
        tiers.bound_sybil_trust(history, np.array([[2, 0]]), out_deg)   # a sender without endorsees
    # With c_e = 0.2 each term is min(0.5 t_i(u) / dout(u), 0.2): 0.15 + 0.2 (capped from 0.2) at i = 0, and
    # 0.075 + 0.2 (capped from 0.25) at i = 1.
    capped = tiers.bound_sybil_trust(history, np.array([[0, 2], [1, 2]]), out_deg, edge_cap=0.2)
    assert capped == pytest.approx(0.15 + 0.2 + 0.075 + 0.2, rel=0, abs=1e-15)
    with pytest.raises(ValueError):
        tiers.bound_sybil_trust(history, np.array([[0, 2]]), out_deg, edge_cap=0.0)


@pytest.mark.parametrize("s", [3, 10, 50])
def test_one_attack_edge_admits_at_most_the_cap_while_legacy_admits_the_whole_ring(s):
    # The attacker is a seed, so it is in the legacy endorsed set D from the start. Its attack edge is listed
    # after the honest edges and before the ring's edges, so legacy_bfs admits every fake.
    rows = honest_rows(0)
    attacker = seed_attacker(rows)
    G, n, fakes, attack = attacked_region(rows, attacker, s)
    levels, trust, info = tiers.btp_tiers(G, n, TOY_EPS)
    history, out_deg = btp_history(G, n, info["K"])
    assert np.array_equal(history[-1], trust)
    bound = tiers.bound_sybil_trust(history, attack, out_deg)
    cap = math.floor(bound / info["theta0"])
    assert trust[fakes].sum() <= bound + 1e-15
    assert int((levels[fakes] > 0).sum()) <= cap
    assert (legacy_bfs(G, n)[fakes] > 0).all()
    if s > 3:
        assert cap < s, "the cap must be informative: BTP provably admits fewer fakes than legacy"


REGION_SIZES = (3, 10, 50, 350)


@pytest.mark.parametrize("topology", ["ring", "star"])
def test_frozen_calibration_caps_admission_independently_of_region_size(topology):
    # Ruling R32: seeds, K and theta0 = eps / n_ref are calibrated once on the honest graph and applied
    # unchanged to the attacked one. One attack edge from a seed endorses a region of s fakes (a ring, or a
    # star whose hub endorses every other fake) that endorses no honest account. For s = 3, 10, 50 and 350:
    # (i) the bound B on the region's trust and the cap floor(B / theta0) are identical; (ii) the region holds
    # exactly B and at most cap fakes reach Tier >= 1, while legacy_bfs admits all s. The fakes in Tier >= 1
    # therefore never exceed one cap that does not depend on s. For the ring, (iii) the admitted count is the
    # same at s = 50 and 350: trust moves one hop per step, so fakes more than K hops away hold none.
    rows = honest_rows(0)
    attacker = seed_attacker(rows)
    cal = tiers.calibrate_btp(graph(rows), HONEST_N, TOY_EPS)
    bounds, caps, admitted = [], [], []
    for s in REGION_SIZES:
        G, n, fakes, attack = attacked_region(rows, attacker, s, topology)
        levels, trust, info = tiers.btp_tiers(G, n, calibration=cal)
        assert (info["K"], info["theta0"], info["n_ref"]) == (cal.K, cal.theta0, cal.n_ref)
        assert info["n_active"] == cal.n_ref + s
        endorse = tiers.endorsement_edges(G)
        history = tiers.btp_trust(n, endorse, cal.seeds, cal.K, return_history=True)
        assert np.array_equal(history[-1], trust)
        bounds.append(tiers.bound_sybil_trust(history, attack, np.bincount(endorse[:, 0], minlength=n)))
        caps.append(math.floor(bounds[-1] / cal.theta0))
        admitted.append(int((levels[fakes] > 0).sum()))
        assert abs(trust[fakes].sum() - bounds[-1]) <= 1e-15 and admitted[-1] <= caps[-1]
        assert (legacy_bfs(G, n)[fakes] > 0).all()
    assert bounds[0] == bounds[1] == bounds[2] == bounds[3] > 0
    assert caps[0] == caps[1] == caps[2] == caps[3] < 10
    assert max(admitted) > 0                                          # pinned from below: some fakes do get in
    if topology == "ring":
        assert admitted[2] == admitted[3]


def test_uncalibrated_tiers_let_the_count_cap_grow_with_the_region():
    # calibration=None recalibrates on the attacked graph, so theta0 = eps / n_active and K = ceil(log2
    # n_active) count the fakes. For one attack edge into a ring that endorses no honest account, the cap
    # floor(B / theta0) then grows strictly over s = 3, 10, 50, 350: theta0 falls as 1 / (n_honest + s), and
    # K rises from 6 to 9, which raises B too. Holding K at 7 isolates theta0: B is then the same for every s
    # and each ring holds exactly B, yet the cap still grows. Hence ruling R32 freezes the calibration.
    rows = honest_rows(0)
    attacker = seed_attacker(rows)
    caps, Ks, fixed_bounds, held, fixed_ratios = [], [], [], [], []
    for s in REGION_SIZES:
        G, n, fakes, attack = attacked_region(rows, attacker, s)
        _, _, info = tiers.btp_tiers(G, n, TOY_EPS)
        history, out_deg = btp_history(G, n, info["K"])
        caps.append(math.floor(tiers.bound_sybil_trust(history, attack, out_deg) / info["theta0"]))
        Ks.append(info["K"])
        _, trust, info7 = tiers.btp_tiers(G, n, TOY_EPS, K=7)
        history7, _ = btp_history(G, n, 7)
        fixed_bounds.append(tiers.bound_sybil_trust(history7, attack, out_deg))
        held.append(trust[fakes].sum())
        fixed_ratios.append(fixed_bounds[-1] / info7["theta0"])
    assert caps[0] < caps[1] < caps[2] < caps[3] and Ks == [6, 7, 7, 9]
    assert fixed_bounds[0] == fixed_bounds[1] == fixed_bounds[2] == fixed_bounds[3] > 0
    assert np.allclose(held, fixed_bounds, rtol=0, atol=1e-15)
    assert fixed_ratios[0] < fixed_ratios[1] < fixed_ratios[2] < fixed_ratios[3]


@pytest.mark.parametrize("topology", ["ring", "star"])
@pytest.mark.parametrize("selection", ["seeds", "max_trust"])
def test_edge_cap_bounds_admission_by_a_constant_per_attack_edge(selection, topology):
    # Ruling R37, frozen calibration with edge_cap_units = 1: each endorsement carries at most c_e = theta0
    # per step. A region reached by g attack edges therefore holds at most B_cap = sum_{i<K} sum_a
    # min(0.5 t_i(u_a) / dout(u_a), c_e) <= K g c_e, and at most floor(K g edge_cap_units) = K g of its fakes
    # reach Tier >= 1. This holds for g = 1, 2, 5, s = 3, 10, 50, 350, a ring or a star, and endorsers that
    # are seeds or the accounts with the most trust per out-edge: the constant ignores the endorsers' trust.
    rows = honest_rows(0)
    cal = tiers.calibrate_btp(graph(rows), HONEST_N, TOY_EPS, edge_cap_units=1.0)
    assert cal.edge_cap == cal.theta0 and cal.edge_cap_units == 1.0
    endorsers = seed_endorsers(rows, 5) if selection == "seeds" else max_inflow_endorsers(rows, 5, 1.0)
    assert len(endorsers) == 5
    admitted = []
    for g in (1, 2, 5):
        for s in REGION_SIZES:
            G, n, fakes, attack = attacked_region(rows, endorsers[:g], s, topology)
            levels, trust, info = tiers.btp_tiers(G, n, calibration=cal)
            assert (info["edge_cap"], info["K"], info["theta0"]) == (cal.edge_cap, cal.K, cal.theta0)
            endorse = tiers.endorsement_edges(G)
            history = tiers.btp_trust(n, endorse, cal.seeds, cal.K, edge_cap=cal.edge_cap, return_history=True)
            assert np.array_equal(history[-1], trust)
            out_deg = np.bincount(endorse[:, 0], minlength=n)
            bound = tiers.bound_sybil_trust(history, attack, out_deg, edge_cap=cal.edge_cap)
            assert trust[fakes].sum() <= bound + 1e-15
            assert bound <= cal.K * g * cal.edge_cap * (1 + 1e-12)
            admitted.append(int((levels[fakes] > 0).sum()))
            assert admitted[-1] <= math.floor(bound / cal.theta0 + 1e-9) <= cal.K * g
    assert max(admitted) > 0                                          # pinned from below: some fakes do get in


def test_a_sybil_rated_in_the_seed_window_becomes_a_seed_only_without_frozen_calibration():
    # The attack edge lands in snapshot 2, inside the seed window, and gives fake 0 a positive majority there.
    # Recalibrating on the attacked graph makes fake 0 a seed with trust from the start, so the region holds
    # more than the attack-edge inflow bound. The frozen calibration keeps the honest seeds: the bound holds.
    rows = honest_rows(0)
    attacker = seed_attacker(rows)
    cal = tiers.calibrate_btp(graph(rows), HONEST_N, TOY_EPS)
    G, n, fakes, attack = attacked_region(rows, attacker, 10, attack_snap=2)
    assert fakes[0] in tiers.seed_set(G, n).tolist() and fakes[0] not in cal.seeds
    endorse = tiers.endorsement_edges(G)
    out_deg = np.bincount(endorse[:, 0], minlength=n)
    recalibrated, _ = btp_history(G, n, tiers.btp_tiers(G, n, TOY_EPS)[2]["K"])
    assert recalibrated[-1][fakes].sum() > tiers.bound_sybil_trust(recalibrated, attack, out_deg)
    frozen = tiers.btp_trust(n, endorse, cal.seeds, cal.K, return_history=True)
    assert frozen[-1][fakes].sum() <= tiers.bound_sybil_trust(frozen, attack, out_deg) + 1e-15
    assert np.array_equal(tiers.btp_tiers(G, n, calibration=cal)[1], frozen[-1])


# ------------------------------------------------------------------ 7. btp_tiers, causal tiers, eps, CLI
# Seed 0 (rated by 9 in snapshot 0) endorses 1 and 2; 1 endorses 2; 2 endorses 3. With K = 2 the trust is
# 0.25, 0.25, 0.375, 0.125 on accounts 0-3. n_active = 5, so eps = 0.5 gives theta0 = 0.1 and accounts 0-3
# are trusted. Account 0's only positive rating comes from 9, which holds no trust, so 0 is not eligible;
# 1, 2 and 3 are, with distinct trusts 0.25, 0.375 and 0.125: Tiers 2, 3 and 1.
CHAIN_ROWS = [(9, 0, +1, 0), (0, 1, +1, 5), (0, 2, +1, 5), (1, 2, +1, 6), (2, 3, +1, 7)]


@pytest.mark.parametrize("neg_from", ["all", "trusted"])
def test_btp_tiers_info_and_terciles_by_hand(neg_from):
    levels, trust, info = tiers.btp_tiers(graph(CHAIN_ROWS), 10, 0.5, K=2, neg_from=neg_from)
    assert trust.tolist() == [0.25, 0.25, 0.375, 0.125, 0, 0, 0, 0, 0, 0]
    assert info == dict(K=2, theta0=0.1, n_active=5, n_ref=5, n_seeds=1, n_trusted=4, n_eligible=3, edge_cap=None,
                        neg_from=neg_from)
    assert levels.dtype == np.int64 and levels.tolist() == [0, 2, 3, 1, 0, 0, 0, 0, 0, 0]
    # eps = 0.625 puts theta0 exactly on account 3's trust, 0.125: trust >= theta0 still counts as trusted.
    assert tiers.btp_tiers(graph(CHAIN_ROWS), 10, 0.625, K=2, neg_from=neg_from)[2]["n_trusted"] == 4
    # A negative rating of 3 by account 9, which holds no trust, ties 3's trusted positive one. The guard of
    # Task 3.1 ("all") drops 3 to Tier 0; the credible-negatives guard (ruling R56) does not count it.
    levels, _, info = tiers.btp_tiers(graph(CHAIN_ROWS + [(9, 3, -1, 8)]), 10, 0.5, K=2, neg_from=neg_from)
    want = ([0, 1, 3, 0], 2) if neg_from == "all" else ([0, 2, 3, 1], 3)
    assert (levels.tolist()[:4], info["n_eligible"]) == want
    # pos_in counts rating edges, not pairs: a repeated positive rating of 3 by 2 outweighs one negative rating,
    # from untrusted 9 or from trusted 1 (which counts under both rules).
    for rater in (9, 1):
        levels, trust_rep, _ = tiers.btp_tiers(graph(CHAIN_ROWS + [(2, 3, +1, 8), (rater, 3, -1, 8)]), 10, 0.5, K=2,
                                               neg_from=neg_from)
        assert np.array_equal(trust_rep, trust) and levels.tolist()[:4] == [0, 2, 3, 1]
    assert tiers.btp_tiers(graph(CHAIN_ROWS), 10, 0.5)[2]["K"] == 3                 # ceil(log2 5)


def test_calibrate_btp_freezes_the_honest_seeds_K_and_theta0():
    G = graph(CHAIN_ROWS)
    cal = tiers.calibrate_btp(G, 10, 0.5)
    assert (cal.seeds, cal.K, cal.eps, cal.n_ref, cal.t_seed, cal.theta0, cal.n_seeds) == ((0,), 3, 0.5, 5, 4, 0.1, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cal.K = 4
    # Calibrating on the graph itself is the uncalibrated path.
    plain, frozen = tiers.btp_tiers(G, 10, 0.5), tiers.btp_tiers(G, 10, calibration=cal)
    assert np.array_equal(plain[0], frozen[0]) and np.array_equal(plain[1], frozen[1]) and plain[2] == frozen[2]
    assert tiers.btp_tiers(G, 10, 0.5, calibration=cal)[2] == frozen[2]        # a matching eps is accepted
    # On a graph with three more accounts the calibration still sets K, theta0 and the seeds. With K = 3 the
    # trust on accounts 0, 1, 2, 3, 5 is 1/8, 3/16, 3/8, 1/4, 1/16: theta0 = 0.1 trusts four of them, where
    # eps / n_active = 0.5 / 8 would also trust account 5.
    _, trust, info = tiers.btp_tiers(graph(CHAIN_ROWS + [(3, 5, +1, 8), (5, 6, +1, 8), (6, 7, +1, 8)]), 10,
                                     calibration=cal)
    assert trust.tolist()[:6] == [0.125, 0.1875, 0.375, 0.25, 0.0, 0.0625]
    assert (info["K"], info["theta0"], info["n_ref"], info["n_seeds"], info["n_active"], info["n_trusted"]) == (
        3, 0.1, 5, 1, 8, 4)
    assert tiers.calibrate_btp(G, 10, 0.5, K=2).K == 2
    for mismatch in (dict(eps=0.4), dict(K=2), dict(t_seed=3)):
        with pytest.raises(ValueError):
            tiers.btp_tiers(G, 10, calibration=cal, **mismatch)
    with pytest.raises(TypeError):
        tiers.btp_tiers(G, 10, calibration=(0.5,))
    for bad in (dict(seeds=(1, 1)), dict(seeds=(-1,)), dict(n_ref=-1), dict(eps=0.0), dict(K=-1)):
        with pytest.raises(ValueError):
            tiers.BTPCalibration(**{**dict(seeds=(0,), K=3, eps=0.5, n_ref=5, t_seed=4), **bad})
    assert tiers.BTPCalibration(seeds=(), K=0, eps=0.5, n_ref=0).theta0 == math.inf   # empty graph


def test_endorser_inflow_scores_by_hand():
    # R33b: B(u) = sum_{i<K} min(0.5 t_i(u) / (dout(u) + 1), c_e) from the honest iterates of the calibration.
    # CHAIN_ROWS plus account 5, rated by 9 in snapshot 0, a second seed that endorses nobody: |S0| = 2,
    # n_active = 6, K = 3, and eps = 0.5 gives theta0 = 1/12. Seed 5 keeps its 1/2 at every step, so a new
    # endorsement from it would carry min(0.25, c_e) per step: exactly c_e = theta0 under the cap (B(5) = K c_e),
    # 0.25 uncapped (B(5) = 0.75). Account 9 holds no trust, so B(9) = 0.
    G = graph(CHAIN_ROWS + [(9, 5, +1, 0)])
    capped = tiers.calibrate_btp(G, 10, 0.5, edge_cap_units=1.0)
    scores = tiers.endorser_inflow_scores(G, 10, capped)
    assert (capped.seeds, capped.K) == ((0, 5), 3) and scores.shape == (10,) and (scores >= 0).all()
    assert scores[5] == pytest.approx(capped.K * capped.edge_cap, rel=1e-14) and scores[9] == 0.0
    assert scores.max() <= capped.K * capped.edge_cap * (1 + 1e-12)     # no account exceeds K c_e
    uncapped = tiers.endorser_inflow_scores(G, 10, tiers.calibrate_btp(G, 10, 0.5))
    assert uncapped[5] == pytest.approx(0.75, rel=0, abs=1e-15) and uncapped[9] == 0.0
    with pytest.raises(TypeError):
        tiers.endorser_inflow_scores(G, 10, 0.5)


# Six accounts. Seeds 0 and 1 (rated in snapshot 0). The last rating of 4 -> 3 is negative and 5 -> 5 rates itself,
# so the endorsements are 2->0, 3->1, 0->2, 0->3, 1->3, 3->4, 4->5, 5->2, with out-degrees 2, 1, 1, 2, 1, 1.
INFLOW_ROWS = [(2, 0, +1, 0), (3, 1, +1, 0), (0, 2, +1, 5), (0, 3, +1, 5), (1, 3, +1, 6), (3, 4, +1, 7),
               (4, 5, +1, 8), (5, 2, +1, 8), (4, 3, +1, 9), (4, 3, -1, 10), (5, 5, +1, 11)]


def test_endorser_inflow_scores_on_a_six_account_toy_by_hand():
    # Review of Task 3.1: a hand-verified case beyond a sink seed. n_ref = 6, K = 3, theta0 = 0.6 / 6 = 0.1 and
    # c_e = 0.3 theta0 = 0.03. Capped iterates: t0 = (.5, .5, 0, 0, 0, 0), t1 = (.44, .47, .03, .06, 0, 0) (seeds 0
    # and 1 send c_e per endorsement), t2 = (.395, .455, .045, .09, .015, 0). B(u) = sum_{i<3} min(0.5 t_i(u) /
    # (dout(u) + 1), c_e): 3 c_e for both seeds; .0075 + .01125 for 2; .01 + .015 for 3; .00375 for 4; 0 for 5.
    # Uncapped: t1 = (.25, .25, .125, .375, 0, 0), t2 = (.1875, .21875, .125, .375, .09375, 0), so B(0) =
    # 0.5 (.5 + .25 + .1875) / 3 and B(1) = 0.5 (.5 + .25 + .21875) / 2.
    G = graph(INFLOW_ROWS)
    capped = tiers.calibrate_btp(G, 6, 0.6, K=3, edge_cap_units=0.3)
    assert (capped.seeds, capped.K, capped.n_ref) == ((0, 1), 3, 6)
    assert capped.theta0 == pytest.approx(0.1, rel=0, abs=1e-15) and capped.edge_cap == pytest.approx(0.03, abs=1e-15)
    assert tiers.endorser_inflow_scores(G, 6, capped).tolist() == pytest.approx(
        [0.09, 0.09, 0.01875, 0.025, 0.00375, 0.0], rel=0, abs=1e-12)
    uncapped = tiers.calibrate_btp(G, 6, 0.6, K=3)
    assert tiers.endorser_inflow_scores(G, 6, uncapped).tolist() == pytest.approx(
        [0.15625, 0.2421875, 0.0625, 0.125, 0.0234375, 0.0], rel=0, abs=1e-12)


def test_calibration_carries_the_edge_cap():
    G = graph(CHAIN_ROWS)
    cal = tiers.calibrate_btp(G, 10, 0.5, edge_cap_units=2.0)
    assert (cal.edge_cap_units, cal.edge_cap) == (2.0, 0.2) and tiers.calibrate_btp(G, 10, 0.5).edge_cap is None
    plain, frozen = tiers.btp_tiers(G, 10, 0.5, edge_cap_units=2.0), tiers.btp_tiers(G, 10, calibration=cal)
    assert np.array_equal(plain[0], frozen[0]) and np.array_equal(plain[1], frozen[1]) and plain[2] == frozen[2]
    # K = 3 and c_e = 0.2. Seed 0 would send 0.25 per endorsement in step 1, so it sends 0.2 to each of 1 and
    # 2 and keeps 0.6; no later share exceeds 0.2. Accounts 0-3 end at 0.15, 0.2, 0.375 and 0.275.
    assert frozen[1].tolist()[:4] == pytest.approx([0.15, 0.2, 0.375, 0.275], rel=0, abs=1e-15)
    assert frozen[2]["edge_cap"] == 0.2 and tiers.btp_tiers(G, 10, 0.5)[2]["edge_cap"] is None
    assert tiers.btp_tiers(G, 10, calibration=cal, edge_cap_units=2.0)[2] == frozen[2]   # matching units
    with pytest.raises(ValueError):
        tiers.btp_tiers(G, 10, calibration=cal, edge_cap_units=1.0)
    with pytest.raises(ValueError):
        tiers.btp_tiers(G, 10, calibration=tiers.calibrate_btp(G, 10, 0.5), edge_cap_units=2.0)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            tiers.BTPCalibration(seeds=(0,), K=3, eps=0.5, n_ref=5, edge_cap_units=bad)
    # Per-snapshot calibrations carry the same units; each row's cap is units * theta0(t).
    causal_cal = tiers.calibrate_btp_causal(G, 10, 0.5, edge_cap_units=1.0)
    assert causal_cal.edge_cap_units == 1.0 and all(r.edge_cap == r.theta0 for r in causal_cal.rows)
    assert np.array_equal(tiers.causal_tiers(G, 10, calibration=causal_cal),
                          tiers.causal_tiers(G, 10, 0.5, edge_cap_units=1.0))
    with pytest.raises(ValueError):
        tiers.BTPCausalCalibration((tiers.calibrate_btp(G, 10, 0.5), cal))


def test_causal_calibration_freezes_each_strict_prefix():
    rows = honest_rows(4)
    G = graph(rows)
    ccal = tiers.calibrate_btp_causal(G, HONEST_N, TOY_EPS)
    assert (ccal.T, ccal.eps, ccal.t_seed) == (21, TOY_EPS, 4) and ccal.seeds == ccal.rows[20].seeds
    assert ccal.rows[0] == tiers.BTPCalibration(seeds=(), K=0, eps=TOY_EPS, n_ref=0)
    for t in (1, 3, 5, 12, 20):
        prefix = graph([r for r in rows if r[3] < t])
        assert ccal.rows[t] == tiers.calibrate_btp(prefix, HONEST_N, TOY_EPS)
        assert ccal.rows[t].n_ref == tiers.n_active(G, t_max=t, strict=True)
    assert ccal.K == tuple(c.K for c in ccal.rows) and ccal.theta0 == tuple(c.theta0 for c in ccal.rows)
    assert ccal.n_ref == tuple(c.n_ref for c in ccal.rows)
    # On the honest graph the frozen causal tiers and weights equal the recalibrated ones.
    assert np.array_equal(tiers.causal_tiers(G, HONEST_N, calibration=ccal), tiers.causal_tiers(G, HONEST_N, TOY_EPS))
    assert np.array_equal(tiers.causal_weights(G, HONEST_N, calibration=ccal),
                          tiers.causal_weights(G, HONEST_N, TOY_EPS))
    # A region injected in snapshot 12 leaves rows <= 12 unchanged; row t > 12 keeps the honest K(t),
    # theta0(t) and seeds, so the fakes count in n_active only.
    attacker = seed_attacker(rows)
    fakes = list(range(HONEST_N, HONEST_N + 50))
    region = [(attacker, fakes[0], +1, 12)] + [(fakes[j], fakes[(j + 1) % 50], +1, 12) for j in range(50)]
    G2, n2 = graph(in_time_order(rows + region)), HONEST_N + 50
    L_honest = tiers.causal_tiers(G, HONEST_N, calibration=ccal)
    L_attacked = tiers.causal_tiers(G2, n2, calibration=ccal)
    assert np.array_equal(L_attacked[:13, :HONEST_N], L_honest[:13]) and not L_attacked[:13, HONEST_N:].any()
    _, _, info = tiers.btp_tiers(G2, n2, t_max=20, strict=True, calibration=ccal.rows[20])
    assert (info["K"], info["theta0"], info["n_active"]) == (ccal.K[20], ccal.theta0[20], ccal.n_ref[20] + 50)
    with pytest.raises(ValueError):
        tiers.causal_tiers(G, HONEST_N, T=20, calibration=ccal)                  # T differs from the calibration
    with pytest.raises(ValueError):
        tiers.causal_tiers(G, HONEST_N, 0.3, calibration=ccal)                   # eps differs
    with pytest.raises(TypeError):
        tiers.causal_tiers(G, HONEST_N, calibration=ccal.rows[20])               # not a causal calibration


def legacy_terciles(score, eligible):
    """The tercile loop of legacy_levels, copied verbatim over the eligible accounts: the oracle."""
    levels = np.zeros(len(score), dtype=np.int64)
    endorsed = np.where(eligible)[0]
    if len(endorsed) > 0:
        sc_all = score[endorsed]
        q1, q2 = np.quantile(sc_all, [1.0 / 3.0, 2.0 / 3.0])
        for idx, u in enumerate(endorsed):
            sc = sc_all[idx]
            if sc <= q1:
                levels[int(u)] = 1
            elif sc <= q2:
                levels[int(u)] = 2
            else:
                levels[int(u)] = 3
    return levels


def test_tercile_levels_follow_the_legacy_tie_rule():
    # Scores 1, 1, 1, 2, 2, 2 put both terciles on a tie: q1 = 1 and q2 = 2, so the ones take Tier 1, the
    # twos Tier 2, and Tier 3 stays empty. A strict comparison would move both groups up one tier.
    score, eligible = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 5.0]), np.array([1, 1, 1, 1, 1, 1, 0], dtype=bool)
    assert tiers.tercile_levels(score, eligible).tolist() == [1, 1, 1, 2, 2, 2, 0]
    rng = np.random.default_rng(8)
    for _ in range(20):
        score = rng.integers(0, 6, 200).astype(np.float64) / 4.0        # many ties
        eligible = rng.random(200) < 0.7
        assert np.array_equal(tiers.tercile_levels(score, eligible), legacy_terciles(score, eligible))
    assert not tiers.tercile_levels(score, np.zeros(200, dtype=bool)).any()


def test_btp_tiers_require_a_positive_finite_eps():
    for eps in (None, 0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            tiers.btp_tiers(graph(CHAIN_ROWS), 10, eps)


def test_btp_tiers_are_deterministic(training_graphs):
    Tr, n = training_graphs["bitcoinalpha"]
    a, b = tiers.btp_tiers(Tr, n, 0.05), tiers.btp_tiers(Tr, n, 0.05)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    G = graph(honest_rows(3))
    assert np.array_equal(tiers.causal_tiers(G, HONEST_N, TOY_EPS), tiers.causal_tiers(G, HONEST_N, TOY_EPS))


@pytest.mark.parametrize("t_star", [3, 12])
def test_causal_tiers_use_only_the_strict_prefix(t_star):
    # Snapshot t_star gains endorsements from the core accounts and a rating of a new account (60). At
    # t_star = 3 the new ratings also create seeds (snapshots 0-4). Rows t <= t_star must not change.
    rows, n = honest_rows(4), HONEST_N + 1
    extra = [(u, 40 + u, +1, t_star) for u in range(SEED_CORE)] + [(0, HONEST_N, +1, t_star)]
    G1, G2 = graph(rows), graph(in_time_order(rows + extra))
    L1, L2 = tiers.causal_tiers(G1, n, TOY_EPS), tiers.causal_tiers(G2, n, TOY_EPS)
    assert L1.shape == L2.shape == (21, n) and L1.dtype == np.int64
    assert np.array_equal(L1[:t_star + 1], L2[:t_star + 1])
    assert any(not np.array_equal(L1[t], L2[t]) for t in range(t_star + 1, 21))
    for t in (0, 3, t_star, 20):
        assert np.array_equal(L1[t], tiers.btp_tiers(G1, n, TOY_EPS, t_max=t, strict=True)[0])
    assert not L1[0].any()                                            # nothing precedes snapshot 0


def test_causal_weights_are_one_in_the_seed_period_then_the_tier_weights():
    G = graph(honest_rows(4))
    levels = tiers.causal_tiers(G, HONEST_N, TOY_EPS, T=21, t_seed=4)
    weights = tiers.causal_weights(G, HONEST_N, TOY_EPS, T=21, t_seed=4)
    assert weights.shape == (21, HONEST_N) and weights.dtype == np.float64
    assert (weights[:5] == 1.0).all()
    assert np.array_equal(weights[5:], np.asarray(TIER_WEIGHTS)[levels[5:]])
    assert (levels[5:] == 0).any() and (levels[5:] > 0).any()


def test_choose_eps_on_a_toy_graph_with_known_retentions():
    # n_active = 36, so theta0 = eps / 36. eps = 1: theta0 = 0.028 keeps the hubs with 1 and 2 leaves and
    # those 3 leaves (5 of 36). eps = 0.5: theta0 = 0.014 adds the hub with 4 leaves and its leaves (10 of
    # 36). eps <= 0.2: theta0 <= 0.0056 is below every leaf's trust (at least 1/160): all 36.
    G = graph(hub_rows())
    eps, table = tiers.choose_eps(G, 36)
    assert eps == 0.2 and table["met"] is True and table["n_legacy_tier1"] == 36
    assert [row["eps"] for row in table["rows"]] == list(EPS_GRID)
    assert [row["retention"] for row in table["rows"]] == pytest.approx([5 / 36, 10 / 36, 1.0, 1.0, 1.0])
    assert tiers.choose_eps(G, 36, keep=0.25)[0] == 0.5
    assert tiers.choose_eps(G, 36, keep=0.1)[0] == 1.0
    eps, table = tiers.choose_eps(G, 36, grid=(1.0, 0.5))
    assert eps == 0.5 and table["met"] is False                       # none qualifies: min(grid)


def test_choose_eps_extends_its_grid_only_when_no_value_qualifies():
    # Hub toy, uncapped (retentions 5/36, 10/36, 1, 1, 1 for eps = 1 ... 0.05): the grid (1, 0.5) fails, so the
    # fallback grid adds 0.2, 0.1 and 0.05 and 0.2 is chosen; a grid that already qualifies is not extended.
    G = graph(hub_rows())
    eps, table = tiers.choose_eps(G, 36, grid=(1.0, 0.5), fallback_grid=EPS_GRID)
    assert eps == 0.2 and table["met"] is True and table["extended"] is True
    assert [row["eps"] for row in table["rows"]] == list(EPS_GRID)
    eps, table = tiers.choose_eps(G, 36, grid=(1.0, 0.5, 0.2), fallback_grid=EPS_GRID)
    assert eps == 0.2 and table["extended"] is False and len(table["rows"]) == 3
    assert tiers.choose_eps(G, 36)[1]["extended"] is False and tiers.choose_eps(G, 36)[1]["edge_cap_units"] is None
    # With a cap, each row's retention comes from btp_tiers with the same cap (theta0 and c_e scale together).
    assert tiers.EPS_GRID_EXTENDED == (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)
    eps, table = tiers.choose_eps(G, 36, edge_cap_units=1.0, fallback_grid=tiers.EPS_GRID_EXTENDED)
    legacy = legacy_bfs(G, 36) > 0
    assert table["edge_cap_units"] == 1.0 and eps in [row["eps"] for row in table["rows"]]
    for row in table["rows"]:
        levels = tiers.btp_tiers(G, 36, row["eps"], edge_cap_units=1.0)[0]
        assert row["retention"] == ((levels > 0) & legacy).sum() / legacy.sum()
    # On the random toy the cap costs coverage: uncapped retention reaches 1 at eps = 0.2, while the capped
    # one stays below 0.95 at every value. With keep = 0.95 the uncapped rule picks 0.2, and the capped rule
    # extends its grid, still fails, and returns the smallest value evaluated. Both guards (ruling R56) agree.
    H = graph(honest_rows(0))
    for neg_from in ("all", "trusted"):
        assert tiers.choose_eps(H, HONEST_N, keep=0.95, neg_from=neg_from)[0] == 0.2
        eps, table = tiers.choose_eps(H, HONEST_N, keep=0.95, edge_cap_units=1.0,
                                      fallback_grid=tiers.EPS_GRID_EXTENDED, neg_from=neg_from)
        assert (eps, table["met"], table["extended"]) == (0.005, False, True)
        assert [row["eps"] for row in table["rows"]] == list(tiers.EPS_GRID_EXTENDED)
        assert all(row["retention"] < 0.95 for row in table["rows"])


# The eps rule on the three networks, per guard: (eps, BTP Tier >= 1 accounts, their overlap with the legacy Tier >= 1
# set) at the chosen eps. Task 3.1 measured the "all" rows. Ruling R56's guard ("trusted") keeps every eps and admits
# accounts whose only negative ratings come from untrusted raters: none on Bitcoin-Alpha, 5 on Bitcoin-OTC and about
# 170 on Wiki-RfA, some of them outside the legacy Tier >= 1 set, which counts every rater's negative ratings.
LEGACY_TIER1 = {"bitcoinalpha": 3368, "bitcoinotc": 4950, "wiki-rfa": 1598}
UNCAPPED_PINS = {"all": {"bitcoinalpha": (0.05, 3202, 3202), "bitcoinotc": (0.05, 4656, 4656),
                         "wiki-rfa": (0.5, 1498, 1498)},
                 "trusted": {"bitcoinalpha": (0.05, 3202, 3202), "bitcoinotc": (0.05, 4661, 4657),
                             "wiki-rfa": (0.5, 1668, 1532)}}
CAPPED_PINS = {"all": {"bitcoinalpha": (0.02, 3101, 3101), "bitcoinotc": (0.01, 4571, 4571),
                       "wiki-rfa": (0.5, 1475, 1475)},
               "trusted": {"bitcoinalpha": (0.02, 3101, 3101), "bitcoinotc": (0.01, 4576, 4572),
                           "wiki-rfa": (0.5, 1647, 1510)}}


def rule_at_chosen_eps(Tr, n, neg_from, edge_cap_units=None):
    """(eps, met, extended, BTP Tier >= 1 count, overlap, legacy Tier >= 1 count) of the preregistered eps rule."""
    eps, table = tiers.choose_eps(Tr, n, edge_cap_units=edge_cap_units, fallback_grid=tiers.EPS_GRID_EXTENDED,
                                  neg_from=neg_from)
    row = next(r for r in table["rows"] if r["eps"] == eps)
    return eps, table["met"], table["extended"], row["n_btp_tier1"], row["n_overlap"], table["n_legacy_tier1"]


@pytest.mark.parametrize("neg_from", ["all", "trusted"])
@pytest.mark.parametrize("ds,n_act,K", [("bitcoinalpha", 3629, 12), ("bitcoinotc", 5684, 13), ("wiki-rfa", 10152, 14)])
def test_choose_eps_and_K_on_the_three_networks(ds, n_act, K, neg_from, training_graphs):
    # K = ceil(log2 n_active) = 12, 13, 14 as the design note states; uncapped eps by the honest-data rule.
    Tr, n = training_graphs[ds]
    eps, n_tier1, n_overlap = UNCAPPED_PINS[neg_from][ds]
    assert rule_at_chosen_eps(Tr, n, neg_from) == (eps, True, False, n_tier1, n_overlap, LEGACY_TIER1[ds])
    _, trust, info = tiers.btp_tiers(Tr, n, eps, neg_from=neg_from)
    assert (info["n_active"], info["K"]) == (n_act, K) and abs(trust.sum() - 1.0) <= 1e-12


@pytest.mark.parametrize("neg_from", ["all", "trusted"])
def test_keeper_edge_cap_is_the_smallest_cap_meeting_the_rule_on_the_three_networks(neg_from, training_graphs):
    # Ruling R38, honest data only: KEEPER's cap is the smallest edge_cap_units in (1, 2, 4, 8) for which the
    # preregistered rule (retain >= 90% of the legacy Tier >= 1 set; eps grid extended to 0.005) holds on all
    # three networks. At 1 the rule fails (Bitcoin-OTC levels off at 86.2%); at 2 it picks eps = 0.02, 0.01
    # and 0.5, and the guarantee is floor(2 K g) = 24, 26 and 28 fakes per attack edge. Under ruling R56's guard
    # the rule picks the same cap and the same eps; only the Tier >= 1 counts move (CAPPED_PINS).
    assert tiers.KEEPER_EDGE_CAP_UNITS == 2.0
    chosen = {(ds, units): rule_at_chosen_eps(*training_graphs[ds], neg_from, units)
              for ds in NETWORKS for units in (1.0, tiers.KEEPER_EDGE_CAP_UNITS)}
    assert [chosen[ds, 1.0][:2] for ds in NETWORKS] == [(0.005, False), (0.005, False), (0.5, True)]
    assert [chosen[ds, 2.0][:2] for ds in NETWORKS] == [(0.02, True), (0.01, True), (0.5, True)]
    assert [(chosen[ds, 2.0][0], *chosen[ds, 2.0][3:5]) for ds in NETWORKS] == [CAPPED_PINS[neg_from][ds]
                                                                                for ds in NETWORKS]
    Ks = [tiers.calibrate_btp(*training_graphs[ds], 0.5).K for ds in NETWORKS]
    assert [math.floor(tiers.KEEPER_EDGE_CAP_UNITS * K) for K in Ks] == [24, 26, 28]


def test_tiers_module_loads_no_torch_dataset_or_report_code():
    # SCEM and CPPR import this module; the CLI imports keeper.analysis.tiers_report only when it runs.
    probe = ("import sys, keeper.credibility.tiers; print(sorted(m for m in sys.modules if m == 'torch' or "
             "m.startswith(('keeper.analysis', 'keeper.laundering.core', 'keeper.scem', 'keeper.cppr'))))")
    res = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True)
    assert res.stdout.strip() == "[]"
