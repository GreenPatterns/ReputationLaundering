"""The random and ranked attack-edge endorsers of the CPPR driver (Task 4.1c: rulings R43 and R44, --pool-eps), on
Bitcoin-Alpha and on a toy platform, without training. These tests moved verbatim from tests/test_cp_vpe_inject.py
(Task 4.1c fix round 1) so that every test file stays under 800 lines; they use that file's helpers.

  I26 random endorsers are identical across BFS, capped and uncapped BTP runs of the same seed, arm and attack flags
  I27 each criterion of the common pool removes exactly its accounts (a toy whose tiers are set explicitly)
  I28 criterion (e), reference B(u) >= theta0: an endorser below theta0 admits no fake as the only endorser of a
      region under capped BTP
  I29 ranked max-B(u) endorsers: group i takes ranks i g + 1 .. (i + 1) g, ties to the smaller id, targets
      excluded, and the record holds each group's ranks
  I30 an uncapped run ranks by its own uncapped B(u); BFS and capped runs by the reference B(u)

Run from the repository root (CPU is enough):
    CUDA_VISIBLE_DEVICES="" <ps python> -m pytest -q tests/test_cp_vpe_endorsers.py
"""
import dataclasses

import numpy as np
import pytest

from keeper.analysis import tiers_report
from keeper.cppr import cp_vpe_gpu as G
from keeper.credibility import tiers
from keeper.laundering import core as C
from tests.cppr_common import ALPHA_EPS, ALPHA_MAX_B, alpha_calibration, background, rng_with_first_snapshot
from tests.test_cp_vpe_inject import TOY_N, _alpha_setting, _injected, _toy_inputs, _toy_platform


# ================================================================== Task 4.1c (rulings R43, R44)
POOL_FIELDS = ("pool_size", "pool_size_untimed", "share_removed_by_timing", "timing_applied", "pool_criteria")


def _both_arms(protocol, tier_fn, inputs, k, topology):
    """Both injected arms of Bitcoin-Alpha seed 0, built in the driver's order from the stored stream: per arm, its
    targets and its sybil_admission record."""
    bg = background()
    rng, out = np.random.default_rng(3000), {}
    for arm, sign in (("whitewash", +1), ("badmouth", -1)):
        targets = list(bg.edge_t) if arm == "whitewash" else rng.choice(bg.trusted, size=15, replace=False).tolist()
        picker = G.arm_picker(protocol, arm, 0, inputs, targets)
        _, G_inj, cred_inj, _ = G.build_inject_then_erase_graph(bg.D, bg.G, bg.tt, None, targets, sign, k, bg.honest,
                                                                rng, sybil_ring=sign > 0, endorsers=picker,
                                                                topology=topology, tier_fn=tier_fn)
        out[arm] = targets, G.admission_record(cred_inj, C.train_split(G_inj, bg.tt), bg.D.n, len(targets), k,
                                               protocol, tier_fn, picker, inputs.join)
    return out


@pytest.mark.parametrize("topology", ["ring", "star"])
def test_random_endorsers_are_identical_across_tier_settings(topology):
    # Ruling R43: one pool for every tier method, drawn from the arm's own stream, so BFS, capped and uncapped runs of
    # the same seed, arm and attack flags face the same attack edges (H3b's paired attack). The pool reads the BFS
    # tiers, the reference capped tiers and B(u), and legacy's join snapshots, whatever the run's tiers.
    bg = background()
    n, g = bg.D.n, 2
    runs = {name: _both_arms(*_alpha_setting(name, g), 3, topology) for name in ("bfs", "capped", "uncapped")}
    inputs = _alpha_setting("bfs", g)[2]
    reference = _alpha_setting("bfs", g)[0].reference_calibration(bg.Tr, n)
    assert reference == alpha_calibration() and inputs.reference_theta0 == reference.theta0
    assert np.array_equal(inputs.bfs_levels, G.compute_credibility_levels(bg.Tr, n, seeded=True))
    assert np.array_equal(inputs.reference_levels, tiers.btp_tiers(bg.Tr, n, calibration=reference)[0])
    assert np.array_equal(inputs.join, G.legacy_pass(bg.Tr, n)[0])
    for arm, (targets, rec) in runs["bfs"].items():
        pool, _ = G.common_pool(inputs, targets)
        assert all(len(set(e)) == g and set(e) <= set(pool.tolist()) for e in rec["endorsers"])
        assert all((inputs.join[e] <= s).all() for e, s in zip(rec["endorsers"], rec["snap_f"]))
        assert rec["timing_applied"] and rec["endorsers_in_D_at_attack"] == [[True] * g] * len(targets)
        for other in ("capped", "uncapped"):
            other_targets, other_rec = runs[other][arm]
            assert other_targets == targets
            for key in ("endorsers", "snap_f", "endorsers_in_D_at_attack") + POOL_FIELDS:
                assert other_rec[key] == rec[key], (other, arm, key)


def test_each_criterion_of_the_common_pool_removes_exactly_its_accounts():
    # The toy platform with tiers and B(u) set explicitly and legacy's own join snapshots. One account fails each
    # criterion of ruling R43: 1 (a) BFS tier 1; 2 (b) reference tier 1; 3 (c) the target; 4 (e) B(u) below theta0;
    # 11 (d) joins D in snapshot 6, after snap_f = 5. Accounts 5-9 have BFS tier 0. Account 12 sits on every boundary:
    # tiers 2, B(u) = theta0 and joining D in snapshot 5 itself; it is kept.
    D, Gt, tt = _toy_platform()
    Tr = C.train_split(Gt, tt)
    inputs = _toy_inputs(Tr, bfs_levels=np.array([2, 1, 2, 2, 2, 0, 0, 0, 0, 0, 3, 2, 2]),
                         reference_levels=np.array([3, 2, 1, 2, 2, 0, 0, 0, 0, 0, 2, 3, 2]),
                         reference_inflow=np.array([1, 1, 1, 1, 0.25, 1, 1, 1, 1, 1, 1, 1, 0.5]))
    pool, sizes = G.common_pool(inputs, [3])
    assert pool.tolist() == [0, 10, 11, 12]
    assert sizes == (("honest", 13), ("bfs_tier", 7), ("reference_tier", 6), ("not_target", 5), ("inflow", 4))
    protocol = G.RunProtocol(attack_edges=3, pool_eps=ALPHA_EPS)
    picker = G.arm_picker(protocol, "whitewash", 0, inputs, [3])
    assert picker.candidates(5).tolist() == [0, 10, 12] and picker.candidates(6).tolist() == [0, 10, 11, 12]
    # Passing one more criterion returns exactly the account that failed it, and nobody else.
    for account, change in ((1, dict(bfs_levels=np.where(np.arange(13) == 1, 2, inputs.bfs_levels))),
                            (2, dict(reference_levels=np.where(np.arange(13) == 2, 2, inputs.reference_levels))),
                            (4, dict(reference_inflow=np.where(np.arange(13) == 4, 0.5, inputs.reference_inflow)))):
        assert set(G.common_pool(dataclasses.replace(inputs, **change), [3])[0].tolist()) == {0, 10, 11, 12, account}
    assert set(G.common_pool(inputs, [])[0].tolist()) == {0, 3, 10, 11, 12}
    nudged = np.where(np.arange(13) == 12, np.nextafter(0.5, 0.0), inputs.reference_inflow)
    assert 12 not in G.common_pool(dataclasses.replace(inputs, reference_inflow=nudged), [3])[0]
    # The record of an injection at snap_f = 5 lists each criterion's sizes and share; timing has its R39 fields.
    seed = rng_with_first_snapshot(5, tt)
    _, G_inj, cred_inj, _ = G.build_inject_then_erase_graph(D, Gt, tt, None, [3], +1, 3, [],
                                                            np.random.default_rng(seed), endorsers=picker)
    rec = G.admission_record(cred_inj, C.train_split(G_inj, tt), TOY_N, 1, 3, protocol, G.CredibilityTiers("bfs"),
                             picker, inputs.join)
    assert sorted(rec["endorsers"][0]) == [0, 10, 12] and rec["endorsers_in_D_at_attack"] == [[True] * 3]
    assert rec["pool_criteria"] == [dict(criterion="bfs_tier", size_before=13, size=7, share_removed=6 / 13),
                                    dict(criterion="reference_tier", size_before=7, size=6, share_removed=1 / 7),
                                    dict(criterion="not_target", size_before=6, size=5, share_removed=1 / 6),
                                    dict(criterion="inflow", size_before=5, size=4, share_removed=1 / 5)]
    assert rec["pool_size"] == dict(min=3, median=3.0, max=3) and rec["pool_size_untimed"] == 4
    assert rec["share_removed_by_timing"] == 0.25 and rec["timing_applied"] and rec["endorser_ranks"] is None
    with pytest.raises(ValueError, match="only 3 candidate endorsers credible at snapshot 5 for 4"):
        G.arm_picker(G.RunProtocol(attack_edges=4, pool_eps=ALPHA_EPS), "whitewash", 0, inputs, [3])(0, 5)


def test_an_endorser_below_theta0_admits_no_fake_under_capped_btp():
    # Criterion (e) of ruling R43 keeps the accounts whose reference capped B(u) reaches theta0. A fake's trust arrives
    # only through its region's attack edges, and B(u) bounds what one endorsement from u carries over the K steps, so
    # an endorser below theta0 cannot lift any fake to Tier >= 1 under capped BTP. On Bitcoin-Alpha every BFS Tier >= 2
    # account below theta0, made the only endorser of a one-fake region, admits nobody; the largest B(u) admits it.
    bg = background()
    n = bg.D.n
    protocol, btp, inputs = _alpha_setting("capped", 1)
    B, theta0 = inputs.reference_inflow, inputs.reference_theta0
    assert np.array_equal(B, tiers.endorser_inflow_scores(bg.Tr, n, alpha_calibration()))
    pool, sizes = G.common_pool(inputs, bg.edge_t)
    other = ~np.isin(np.arange(n), bg.edge_t)
    before = np.flatnonzero((inputs.bfs_levels >= 2) & (inputs.reference_levels >= 2) & other)
    assert np.array_equal(pool, before[B[before] >= theta0]) and dict(sizes)["not_target"] == len(before)
    low = np.flatnonzero((inputs.bfs_levels >= 2) & other & (B < theta0))
    assert len(low)
    cred = btp(bg.Tr, n)
    for u in low.tolist() + [int(np.argmax(B))]:
        cred_inj = _injected(bg, cred, bg.edge_t[:1], +1, 1, [[u]], tier_fn=btp)[2]
        assert (cred_inj[n] > 0) == (B[u] >= theta0), u


def test_ranked_endorsers_take_consecutive_ranks_with_ties_to_the_smaller_id():
    # Ruling R44 (replacing decision D-b): group i (from 0) takes the accounts ranked i g + 1 .. (i + 1) g by B(u)
    # among the accounts other than the arm's targets, ties to the smaller id; the record holds each group's ranks.
    inflow = np.array([0.5, 0.9, 0.9, 0.1, 0.9, 0.3])
    picker = G.endorser_picker("max_trust", 2, None, inflow=inflow, exclude=[2])
    assert picker.ranking.tolist() == [1, 4, 0, 5, 3]                # 1, 2 and 4 tie; 2 is a target
    assert [picker(i, None) for i in range(2)] == [[1, 4], [0, 5]]
    with pytest.raises(ValueError, match=r"^group 2: only 5 candidate endorsers for ranks 5 to 6 \(ruling R44\)$"):
        picker(2, None)
    # Bitcoin-Alpha under BFS tiers: the reference capped B(u) ranks the accounts.
    bg = background()
    n, g = bg.D.n, 2
    protocol, tier_fn, inputs = _alpha_setting("bfs", g, "max_trust")
    ranking = [u for u in tiers_report.top_accounts(inputs.reference_inflow, n).tolist() if u not in set(bg.edge_t)]
    picker = G.arm_picker(protocol, "whitewash", 0, inputs, bg.edge_t)
    assert picker.ranking.tolist() == ranking and inputs.rank_inflow is inputs.reference_inflow
    _, _, cred_inj, _, Tr_inj = _injected(bg, None, bg.edge_t, +1, 3, picker, tier_fn=tier_fn)
    rec = G.admission_record(cred_inj, Tr_inj, n, 15, 3, protocol, tier_fn, picker, inputs.join)
    assert rec["endorsers"] == [ranking[g * i:g * (i + 1)] for i in range(15)] and rec["endorsers"][0] == ALPHA_MAX_B[:2]
    assert rec["endorser_ranks"] == [[g * i + 1, g * i + 2] for i in range(15)]
    assert rec["pool_size"] is None and rec["pool_criteria"] is None


def test_the_uncapped_ranking_follows_the_runs_own_inflow():
    # Ruling R44: BFS and capped runs rank by the reference capped B(u), an uncapped run by B(u) under its own uncapped
    # calibration (c_e = inf). The cap makes the strongest endorsers tie at K c_e, where the smaller id goes first, so
    # the rankings differ where the inflows differ.
    capped, uncapped = np.array([2.0, 2.0, 2.0, 1.0, 0.5]), np.array([3.0, 9.0, 5.0, 1.0, 0.5])
    assert G.endorser_picker("max_trust", 1, None, inflow=capped).ranking.tolist() == [0, 1, 2, 3, 4]
    assert G.endorser_picker("max_trust", 1, None, inflow=uncapped).ranking.tolist() == [1, 2, 0, 3, 4]
    bg = background()
    n, groups = bg.D.n, {}
    for name in ("bfs", "capped", "uncapped"):
        protocol, _, inputs = _alpha_setting(name, 1, "max_trust")
        own = protocol.calibrate(bg.Tr, n) if name == "uncapped" else protocol.reference_calibration(bg.Tr, n)
        assert np.array_equal(inputs.rank_inflow, tiers.endorser_inflow_scores(bg.Tr, n, own))
        picker = G.arm_picker(protocol, "whitewash", 0, inputs, bg.edge_t)
        assert picker.ranking.tolist() == [u for u in tiers_report.top_accounts(inputs.rank_inflow, n).tolist()
                                           if u not in set(bg.edge_t)]
        groups[name] = [picker(i, None) for i in range(15)]
    assert _alpha_setting("uncapped", 1, "max_trust")[0].calibrate(bg.Tr, n).edge_cap_units is None
    assert groups["bfs"] == groups["capped"] != groups["uncapped"]
