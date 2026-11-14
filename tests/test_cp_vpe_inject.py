"""Regression tests for the CPPR inject-then-erase workload and the credibility gate.

  I1  background requests keep their edges - after the fakes (and attack edges) are inserted, every
                                             background request still erases the same training edge
  I2  snapshot gate (ablation)             - the per-snapshot gate normalizes tier weights over a node's
                                             in-edges
  I3  count gate (CPPR's gate, Eq. 15)     - tier weights are divided by the in-edge count, so Tier-0
                                             edges keep weight 0.01 even when alone
Task 3.2 (upgrade M-A in CPPR, ruling G1, protocol options), on Bitcoin-Alpha without training:
  I4  the default gate of the CLI and of run_single_seed is count (G1)
  I5  without attack edges the injected graph is byte-identical to the old function's (frozen copy)
  I6  attack edges come from distinct honest Tier >= 2 accounts, at the group's snap_f, before its edges
  I7  attack edges are erased with the fakes and never pooled
  I8  legacy BFS lifts the whole ring or star to Tier >= 1 with one attack edge from a member of D
  I9  an endorser that joins D after snap_f lifts only the fake it rates (legacy's single pass)
  I10 BTP admits at most floor(K g edge_cap_units) fakes per group, where legacy admits all
  I11 the frozen calibration serves cred_inj: 50 fakes change neither theta0, K nor the seeds (R32)
  I12 star topology: the hub endorses every other fake and receives the attack edges (R33)
  I13 max_trust ranks the accounts by B(u) (R33b), ties to the smaller id; group i takes the next g (R44)
  I14 random endorsers: distinct, Tier >= 2, never one of the arm's targets, reproducible
  I15 the control arm has no fake and only the background requests
  I16 --hold-test-snapshot: Plain Erasure keeps the original test snapshot; every arm is scored on it
Ruling R39 (random endorsers must already be in legacy's trusted set D at their attack edge; R43 extends it to BTP):
  I17 legacy_pass replays legacy's endorsement pass: join snapshots, and senders in D when a row is read
  I18 BFS + random: timely endorsers let legacy lift every group's whole region; all are recorded in D
  I19 BTP + random: the pool is the common one, timing included; late joiners of D are never drawn (R43)
  I20 a toy graph with a known late joiner: the pool excludes it; passed explicitly, it lifts only the
      fake it rates and is recorded as not in D; a timely pool smaller than g raises
  I21 the premises of R39's condition are checked: a graph whose rows are not in snapshot order is refused
Task 4.1b (Phase 4 hardening):
  I22 random endorsers leave the stored rng untouched: its state, each group's snap_f and the bad-mouth targets
      are those of a build without attack edges (the arm-subset and --methods invariance rests on this)
  I23 the pre-flight check raises the endorser pickers' ValueErrors before any pick: untimed pool, the timely pool
      at the first snapshot a group can draw (every tier method, R43), and the ranked candidates (R44)
  I24 that first snapshot is the lowest snap_f the graph builder draws, and the timely pool is smallest there
  I25 each injected arm's sybil_admission records every group's snap_f in group order, next to the endorsers: the
      snapshot of the group's own injected rows and the stored rng's draw, with or without attack edges
Task 4.1c (rulings R43, R44; --pool-eps, the reference capped calibration): I26 to I30 are in
tests/test_cp_vpe_endorsers.py, which uses this file's helpers.
Ruling R56 (BTP's credible-negatives guard):
  I31 the run's BTP tier function counts negative ratings from trusted raters only unless told otherwise: the
      bad-mouth arm's fakes, without attack edges, move no honest tier; under neg_from="all" they demote targets

Run from the repository root (CPU is enough):
    CUDA_VISIBLE_DEVICES="" <ps python> -m pytest -q tests/test_cp_vpe_inject.py
"""
import dataclasses
import functools
import inspect
import math

import numpy as np
import pytest
import torch

from keeper.analysis import defense_extra as DE
from keeper.analysis import tiers_report
from keeper.cppr import cp_vpe_gpu as G  # noqa: E402
from keeper.cppr import vpe as V
from keeper.cppr.vpe_frontier import workload  # noqa: E402
from keeper.credibility import tiers
from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import signed as S
from tests import cp_vpe_legacy as O
from tests.cppr_common import (ALPHA_EPS, ALPHA_MAX_B, alpha_calibration, background, legacy_join_snapshots,
                               rng_with_first_snapshot)

INF = float("inf")
ALPHA_UNCAPPED_EPS = 0.05     # the uncapped BTP's eps in these tests (a test parameter, not a registered value)


@functools.lru_cache(maxsize=None)
def _alpha_setting(name, g, select="random"):
    """A protocol of Bitcoin-Alpha with g attack edges under BFS tiers, capped BTP or uncapped BTP, the pool's reference
    calibration at ALPHA_EPS (with attack edges only), its tier function and its EndorserInputs, all computed by the
    library."""
    bg = background()
    common = dict(attack_edges=g, endorser_select=select, pool_eps=ALPHA_EPS if g else None)
    protocol = dict(bfs=lambda: G.RunProtocol(**common),
                    capped=lambda: G.RunProtocol(tiers="btp", btp_eps=ALPHA_EPS, **common),
                    uncapped=lambda: G.RunProtocol(tiers="btp", btp_eps=ALPHA_UNCAPPED_EPS, btp_edge_cap=None,
                                                   **common))[name]()
    cal = protocol.calibrate(bg.Tr, bg.D.n)
    tier_fn = G.CredibilityTiers("bfs") if cal is None else G.CredibilityTiers("btp", cal)
    inputs = G.endorser_inputs(protocol, bg.Tr, bg.D.n, cal, protocol.reference_calibration(bg.Tr, bg.D.n))
    return protocol, tier_fn, inputs


def _background(ds="bitcoinalpha", seed=0, rho=0.02):
    D = C.load(ds)
    Gr = C.full_graph(D)
    tt = D.tt
    Tr = C.train_split(Gr, tt)
    cohort, neg, _ = C.pick_targets(Tr, max_neg=3, strict=True)
    honest_rng = np.random.default_rng(int(1e6 * rho) + 17 * seed)
    reqs, edge_t, _ = workload(Tr, cohort, neg, 15, 5, rho, 3000 + seed, honest_rng)
    honest = [q for q in reqs if q.tag == "honest"]
    cred = G.compute_credibility_levels(Tr, D.n, seeded=True)
    return D, Gr, tt, Tr, cred, edge_t, honest


@pytest.mark.parametrize("attack_edges", [0, 2])
def test_background_requests_keep_their_edges_after_injection(attack_edges):
    D, Gr, tt, Tr, cred, edge_t, honest = _background()
    rng = np.random.default_rng(3000)
    endorsers = (G.select_endorsers("random", len(edge_t), attack_edges, cred, exclude=edge_t,
                                    rng=np.random.default_rng(5)) if attack_edges else None)
    for sign_val, ring in ((+1, True), (-1, False)):
        _, G_inj, _, reqs = G.build_inject_then_erase_graph(
            D, Gr, tt, cred, edge_t, sign_val, 3, honest, rng, sybil_ring=ring, seeded=True, endorsers=endorsers)
        Tr_inj = C.train_split(G_inj, tt)
        moved = [q for q in reqs if q.tag == "honest"]
        assert len(moved) == len(honest)
        want = np.concatenate([Tr.rid[q.rows] for q in honest])
        got = np.concatenate([Tr_inj.rid[q.rows] for q in moved])
        assert np.array_equal(got, want), f"{np.mean(got == want):.1%} of background rows kept"


def _toy_edges():
    # node 3 receives three edges from Tier-0 node 0; node 4 receives one edge
    # from Tier-0 node 0 and one from Tier-3 node 1.
    row = torch.tensor([0, 0, 0, 0, 1])
    col = torch.tensor([3, 3, 3, 4, 4])
    node_w = torch.tensor([0.01, 1.00, 0.70, 0.40, 0.40])
    return node_w, row, col


def test_snapshot_gate_normalizes_over_snapshot_in_edges():
    node_w, row, col = _toy_edges()
    w = G.gate_weights(node_w, row, col, num_nodes=5, norm="snapshot").squeeze(-1)
    assert torch.allclose(w[:3], torch.full((3,), 1 / 3))
    assert torch.allclose(w[3:], torch.tensor([0.01 / 1.01, 1.00 / 1.01]))


def test_count_gate_keeps_tier0_weight_when_alone():
    node_w, row, col = _toy_edges()
    w = G.gate_weights(node_w, row, col, num_nodes=5, norm="count").squeeze(-1)
    assert torch.allclose(w[:3], torch.full((3,), 0.01 / 3))
    assert torch.allclose(w[3:], torch.tensor([0.01 / 2, 1.00 / 2]))


# ================================================================== Task 3.2
def test_default_gate_is_count():
    # Ruling G1: count is CPPR's gate (Eq. 15) and snapshot the ablation. The stored cpgate_count_* runs passed
    # --gate-norm count and the cp_vpe_* runs relied on the old default, so they need --gate-norm snapshot now.
    args = G.build_parser().parse_args([])
    assert args.gate_norm == "count"
    assert G.build_parser().parse_args(["--gate-norm", "snapshot"]).gate_norm == "snapshot"
    assert inspect.signature(G.run_single_seed).parameters["gate_norm"].default == "count"
    assert G.protocol_from_args(args) == G.RunProtocol() and G.RunProtocol().is_legacy()


def _arrays_equal(a, b, names):
    """The named arrays of a and b agree in dtype, shape and bytes."""
    for name in names:
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            return False
    return True


def _graphs_equal(a, b):
    """Same arrays (dtype, shape and bytes) and node count."""
    return _arrays_equal(a, b, ("edges", "labels", "time", "snap", "rid")) and int(a.n) == int(b.n)


def _datasets_equal(a, b):
    return (_arrays_equal(a, b, ("edges", "labels", "time")) and (a.name, a.n, a.T, a.meta) == (b.name, b.n, b.T, b.meta)
            and a.cfg is b.cfg)


def _requests_equal(a, b):
    if len(a) != len(b):
        return False
    for q, r in zip(a, b):
        if (q.account, q.tag, q.target) != (r.account, r.tag, r.target):
            return False
        if (q.rows is None) != (r.rows is None):
            return False
        if q.rows is not None and (np.asarray(q.rows).dtype != np.asarray(r.rows).dtype
                                   or np.asarray(q.rows).tobytes() != np.asarray(r.rows).tobytes()):
            return False
    return True


@pytest.mark.parametrize("sign_val,ring", [(+1, True), (-1, False)])
def test_without_attack_edges_the_graph_is_byte_identical_to_the_old_function(sign_val, ring):
    # The frozen copy of commit ee67679 is the oracle. Defaults (no endorsers, ring topology, legacy tiers) and an
    # explicit legacy tier function must both reproduce it, and consume the same random draws.
    bg = background()
    cred = G.compute_credibility_levels(bg.Tr, bg.D.n, seeded=True)
    rng_old = np.random.default_rng(3000)
    want = O.build_inject_then_erase_graph(bg.D, bg.G, bg.tt, cred, bg.edge_t, sign_val, 3, bg.honest, rng_old,
                                           sybil_ring=ring, seeded=True)
    for kwargs in (dict(), dict(endorsers=None, topology="ring", tier_fn=G.CredibilityTiers("bfs"))):
        rng_new = np.random.default_rng(3000)
        got = G.build_inject_then_erase_graph(bg.D, bg.G, bg.tt, cred, bg.edge_t, sign_val, 3, bg.honest, rng_new,
                                              sybil_ring=ring, seeded=True, **kwargs)
        assert _datasets_equal(got[0], want[0]) and _graphs_equal(got[1], want[1])
        assert got[2].dtype == want[2].dtype and got[2].tobytes() == want[2].tobytes()
        assert _requests_equal(got[3], want[3])
        assert rng_new.bit_generator.state == rng_old.bit_generator.state


def _group_rows(Tr_inj, n_honest, i, k_fake):
    """Row positions of group i in the injected training list: its own edges (sent by its fakes) and the
    attack edges (sent by other accounts to its fakes), and the group's fakes."""
    fakes = np.arange(n_honest + i * k_fake, n_honest + (i + 1) * k_fake)
    src, dst = Tr_inj.edges[:, 0], Tr_inj.edges[:, 1]
    own = np.flatnonzero(np.isin(src, fakes))
    attack = np.flatnonzero(np.isin(dst, fakes) & ~np.isin(src, fakes))
    return own, attack, fakes


def test_attack_edges_come_from_tier2_accounts_at_the_group_snapshot_before_its_edges():
    bg = background()
    n, k, g = bg.D.n, 3, 2
    cred = G.compute_credibility_levels(bg.Tr, n, seeded=True)
    endorsers = G.select_endorsers("random", len(bg.edge_t), g, cred, exclude=bg.edge_t, rng=np.random.default_rng(7))
    _, G_inj, _, _ = G.build_inject_then_erase_graph(bg.D, bg.G, bg.tt, cred, bg.edge_t, +1, k, bg.honest,
                                                     np.random.default_rng(3000), endorsers=endorsers)
    Tr_inj = C.train_split(G_inj, bg.tt)
    assert len(Tr_inj.edges) == len(bg.Tr.edges) + len(bg.edge_t) * (2 * k + g)
    for i in range(len(bg.edge_t)):
        own, attack, fakes = _group_rows(Tr_inj, n, i, k)
        assert len(attack) == g and len(set(endorsers[i])) == g
        assert Tr_inj.edges[attack, 0].tolist() == list(endorsers[i])          # in selection order
        assert (Tr_inj.edges[attack, 0] < n).all() and (cred[Tr_inj.edges[attack, 0]] >= G.ENDORSER_MIN_TIER).all()
        assert Tr_inj.edges[attack, 1].tolist() == [int(fakes[a % k]) for a in range(g)]   # ring: round-robin
        assert not S.neg_mask(Tr_inj.labels[attack]).any()                    # positive endorsements
        snap_f = Tr_inj.snap[own[0]]
        assert (Tr_inj.snap[own] == snap_f).all() and (Tr_inj.snap[attack] == snap_f).all()
        assert attack.max() < own.min()                                        # before the group's own edges


def _injected(bg, cred, targets, sign_val, k, endorsers, topology="ring", tier_fn=None, seed=3000):
    D_inj, G_inj, cred_inj, reqs = G.build_inject_then_erase_graph(
        bg.D, bg.G, bg.tt, cred, targets, sign_val, k, bg.honest, np.random.default_rng(seed),
        sybil_ring=sign_val > 0, endorsers=endorsers, topology=topology, tier_fn=tier_fn)
    return D_inj, G_inj, cred_inj, reqs, C.train_split(G_inj, bg.tt)


@pytest.mark.parametrize("sign_val", [+1, -1])
def test_attack_edges_are_erased_with_the_fakes_and_never_pooled(sign_val):
    bg = background()
    n, k = bg.D.n, 3
    cred = G.compute_credibility_levels(bg.Tr, n, seeded=True)
    endorsers = G.select_endorsers("random", len(bg.edge_t), 2, cred, exclude=bg.edge_t, rng=np.random.default_rng(9))
    D_inj, G_inj, cred_inj, reqs, Tr_inj = _injected(bg, cred, bg.edge_t, sign_val, k, endorsers)
    fakes = np.arange(n, n + k * len(bg.edge_t))
    attack = np.isin(Tr_inj.edges[:, 1], fakes) & (Tr_inj.edges[:, 0] < n)
    assert attack.sum() == 2 * len(bg.edge_t)
    erase, to_pool, accounts = V._erased_sets(Tr_inj, reqs)
    assert set(fakes.tolist()) <= set(accounts)
    assert erase[attack].all() and not to_pool[attack].any()
    rng = np.random.default_rng(0)
    res_cp, _ = G.apply_cp_vpe(D_inj, G_inj, bg.tt, reqs, cred_inj, G.CPVPEConfig(1, INF, 10), rng)
    res_vpe = V.apply_vpe(D_inj, G_inj, bg.tt, reqs, V.VPEConfig(1, INF, 10), rng)
    for res in (res_cp, res_vpe):
        assert not np.isin(res.Tr.edges, fakes).any()                          # nothing by or about a fake stays
        assert not any(p["target"] in set(fakes.tolist()) for p in res.pooled)


def test_legacy_lifts_the_whole_region_through_one_attack_edge_from_a_member_of_D():
    # The max-B(u) endorsers are seeds, hence members of the legacy endorsed set D from the start. The attack edge
    # is read before the region's edges, so legacy_bfs admits every fake of a ring or a star; without attack edges
    # it admits none.
    bg = background()
    n, k = bg.D.n, 10
    cred = G.compute_credibility_levels(bg.Tr, n, seeded=True)
    inflow = tiers.endorser_inflow_scores(bg.Tr, n, alpha_calibration())
    endorsers = G.select_endorsers("max_trust", 3, 1, cred, inflow=inflow, exclude=bg.edge_t[:3])
    assert set(endorsers[0]) <= set(tiers.seed_set(bg.Tr, n).tolist())
    for topology in ("ring", "star"):
        _, _, cred_inj, _, _ = _injected(bg, cred, bg.edge_t[:3], +1, k, endorsers, topology)
        assert (cred_inj[n:] >= 1).all(), topology
    _, _, cred_inj, _, _ = _injected(bg, cred, bg.edge_t[:3], +1, k, None)
    assert not cred_inj[n:].any()


def test_an_endorser_that_joins_D_after_the_attack_lifts_only_the_fake_it_rates():
    # Legacy's pass reads each rating once, in time order: a rating sent before its sender joins D admits nobody.
    # A Tier >= 2 endorser that joins D after snap_f therefore makes only the fake it rates count as endorsed (its
    # sender is in D at the end), and the rest of the ring stays in Tier 0.
    bg = background()
    n, k, snap_f = bg.D.n, 5, 5
    cred = G.compute_credibility_levels(bg.Tr, n, seeded=True)
    join = legacy_join_snapshots(bg.Tr, n)
    late = [u for u in np.flatnonzero(cred >= 2) if snap_f < join[u] < 10 ** 6 and u not in bg.edge_t]
    assert late, "Bitcoin-Alpha has Tier >= 2 accounts that join D after snapshot 5"
    seed = rng_with_first_snapshot(snap_f, bg.tt)
    _, _, cred_inj, _, Tr_inj = _injected(bg, cred, bg.edge_t[:1], +1, k, [[int(late[0])]], seed=seed)
    own, _, fakes = _group_rows(Tr_inj, n, 0, k)
    assert (Tr_inj.snap[own] == snap_f).all()
    assert cred_inj[fakes[0]] >= 1 and not cred_inj[fakes[1:]].any()


@pytest.mark.parametrize("topology", ["ring", "star"])
@pytest.mark.parametrize("g", [1, 2])
def test_btp_admits_at_most_the_per_edge_bound_where_legacy_admits_every_fake(g, topology):
    # Ruling R37: under KEEPER's cap at most floor(K g edge_cap_units) fakes of a region reached by g attack edges
    # reach Tier >= 1 (24 g on Bitcoin-Alpha). Regions of 50 fakes and the max-B(u) endorsers, the worst case for
    # BTP; legacy_bfs on the same graph admits all 50 of every group.
    bg = background()
    n, k, targets = bg.D.n, 50, bg.edge_t[:3]
    cal = alpha_calibration()
    btp = G.CredibilityTiers("btp", cal)
    cred = btp(bg.Tr, n)
    endorsers = G.select_endorsers("max_trust", len(targets), g, cred,
                                   inflow=tiers.endorser_inflow_scores(bg.Tr, n, cal), exclude=targets)
    _, G_inj, cred_inj, _, Tr_inj = _injected(bg, cred, targets, +1, k, endorsers, topology, tier_fn=btp)
    bound = math.floor(cal.K * g * cal.edge_cap_units + 1e-9)
    assert bound == 24 * g < k
    admitted = (cred_inj[n:].reshape(len(targets), k) > 0).sum(1)
    assert (admitted <= bound).all() and admitted.max() > 0
    summary = G.sybil_admission(cred_inj, n, len(targets), k, g, endorsers, btp)
    assert summary["admitted_per_group"] == admitted.tolist() and summary["guarantee_per_group"] == bound
    assert summary["n_fakes"] == k * len(targets) and summary["n_admitted"] == int(admitted.sum())
    legacy = G.compute_credibility_levels(Tr_inj, n + k * len(targets), seeded=True)
    assert (legacy[n:] > 0).all()


def test_the_frozen_calibration_serves_the_injected_graph():
    # Ruling R32: seeds, K and theta0 come from the honest graph only. Fifty fakes in one region change neither
    # (they count in n_active only); recalibrating on the attacked graph would lower theta0.
    bg = background()
    n, k = bg.D.n, 50
    cal = alpha_calibration()
    btp = G.CredibilityTiers("btp", cal)
    cred = btp(bg.Tr, n)
    endorsers = G.select_endorsers("max_trust", 1, 1, cred, inflow=tiers.endorser_inflow_scores(bg.Tr, n, cal),
                                   exclude=bg.edge_t[:1])
    _, _, cred_inj, _, Tr_inj = _injected(bg, cred, bg.edge_t[:1], +1, k, endorsers, tier_fn=btp)
    levels, info = btp.levels_info(Tr_inj, n + k)
    assert (info["method"], info["K"], info["theta0"], info["n_ref"], info["n_seeds"]) == (
        "btp", cal.K, cal.theta0, cal.n_ref, cal.n_seeds)
    assert info["n_active"] == cal.n_ref + k and np.array_equal(levels, cred_inj)
    assert np.array_equal(cred_inj, tiers.btp_tiers(Tr_inj, n + k, calibration=cal)[0])
    assert np.array_equal(cred, tiers.btp_tiers(bg.Tr, n, calibration=cal)[0])
    recal = tiers.btp_tiers(Tr_inj, n + k, ALPHA_EPS, edge_cap_units=cal.edge_cap_units)[2]
    assert recal["theta0"] < cal.theta0 and recal["n_ref"] == cal.n_ref + k
    assert not set(tiers.seed_set(Tr_inj, n + k).tolist()) - set(cal.seeds)   # no fake can be a seed either
    with pytest.raises(ValueError):
        G.CredibilityTiers("btp")                                              # BTP needs a frozen calibration
    with pytest.raises(ValueError):
        G.CredibilityTiers("bfs", cal)


@pytest.mark.parametrize("neg_from", ["trusted", "all"])
def test_btp_tier_function_applies_the_credible_negatives_guard_by_default(neg_from):
    # I31, ruling R56. Three fakes per target rate it negatively (the bad-mouth arm, no attack edge), so the fakes hold
    # no trust. The targets are the 15 trusted-cohort accounts in Tier >= 1 with the fewest positive ratings. Under the
    # default guard the fakes' ratings count for nothing and every honest tier stays; under the ablation "all" they
    # count, and targets lose eligibility.
    bg = background()
    n, k = bg.D.n, 3
    cal = alpha_calibration()
    btp = G.CredibilityTiers("btp", cal) if neg_from == "trusted" else G.CredibilityTiers("btp", cal, neg_from)
    assert btp.neg_from == neg_from
    cred = btp(bg.Tr, n)
    assert np.array_equal(cred, tiers.btp_tiers(bg.Tr, n, calibration=cal, neg_from=neg_from)[0])
    pos_in = np.bincount(bg.Tr.edges[~S.neg_mask(bg.Tr.labels), 1], minlength=n)
    cohort = bg.trusted[cred[bg.trusted] >= 1]
    targets = cohort[np.argsort(pos_in[cohort], kind="stable")[:15]]
    _, _, cred_inj, _, Tr_inj = _injected(bg, cred, targets, -1, k, None, tier_fn=btp)
    levels, info = btp.levels_info(Tr_inj, n + k * len(targets))
    assert info["neg_from"] == neg_from and np.array_equal(levels, cred_inj) and not cred_inj[n:].any()
    if neg_from == "trusted":
        assert np.array_equal(cred_inj[:n], cred)
    else:
        assert ((cred_inj[targets] == 0) & (cred[targets] > 0)).any()   # some target leaves the eligible set
    with pytest.raises(ValueError):
        G.CredibilityTiers("btp", cal, "none")


def test_star_topology_puts_the_hub_between_the_attack_edges_and_the_other_fakes():
    bg = background()
    n, k, g = bg.D.n, 4, 2
    cred = G.compute_credibility_levels(bg.Tr, n, seeded=True)
    endorsers = G.select_endorsers("random", len(bg.edge_t), g, cred, exclude=bg.edge_t, rng=np.random.default_rng(3))
    for sign_val in (+1, -1):
        _, _, _, _, Tr_inj = _injected(bg, cred, bg.edge_t, sign_val, k, endorsers, "star")
        for i, v in enumerate(bg.edge_t):
            own, attack, fakes = _group_rows(Tr_inj, n, i, k)
            hub = int(fakes[0])
            assert Tr_inj.edges[attack, 1].tolist() == [hub] * g
            pairs = {(int(a), int(b)) for a, b in Tr_inj.edges[own]}
            internal = {pr for pr in pairs if pr[1] != v}
            assert {(int(f), int(v)) for f in fakes} <= pairs
            assert internal == ({(hub, int(f)) for f in fakes[1:]} if sign_val > 0 else set())
            assert not S.neg_mask(Tr_inj.labels[own[np.isin(Tr_inj.edges[own, 1], fakes)]]).any()


def test_max_trust_picks_the_accounts_with_the_largest_inflow():
    # Ruling R33b: B(u) of keeper.credibility.tiers.endorser_inflow_scores; ties go to the smaller id. Ruling R44
    # (replacing decision D-b's shared endorsers): group i takes the accounts ranked i g + 1 .. (i + 1) g.
    levels = np.array([0, 2, 3, 1])
    inflow = np.array([0.1, 0.3, 0.3, 0.2])
    assert G.select_endorsers("max_trust", 2, 2, levels, inflow=inflow) == [[1, 2], [3, 0]]
    assert G.select_endorsers("max_trust", 1, 2, levels, inflow=inflow, exclude=[1]) == [[2, 3]]
    assert G.select_endorsers("max_trust", 1, 3, levels, inflow=inflow, exclude=[2]) == [[1, 3, 0]]
    with pytest.raises(ValueError):
        G.select_endorsers("max_trust", 1, 2, levels)                          # needs the inflow scores
    # On Bitcoin-Alpha it agrees with the Task 3.1 report's selection.
    bg = background()
    cal = alpha_calibration()
    inflow = tiers.endorser_inflow_scores(bg.Tr, bg.D.n, cal)
    got = G.select_endorsers("max_trust", 4, 5, G.compute_credibility_levels(bg.Tr, bg.D.n), inflow=inflow)
    assert got[0] == ALPHA_MAX_B and sum(got, []) == tiers_report.top_accounts(inflow, 20).tolist()
    assert got[0] == tiers_report.max_trust_endorsers(bg.Tr, bg.D.n, cal, 5).tolist()
    excluded = G.select_endorsers("max_trust", 1, 5, None, inflow=inflow, exclude=[34, 100])
    assert excluded[0][:3] == [73, 118, 137] and not {34, 100} & set(excluded[0])


def test_random_endorsers_are_distinct_tier2_accounts_outside_the_targets():
    levels = np.array([0, 2, 3, 1, 2, 2, 3])
    got = G.select_endorsers("random", 50, 3, levels, exclude=[5, 6], rng=np.random.default_rng(1))
    assert len(got) == 50 and all(len(e) == 3 == len(set(e)) and set(e) <= {1, 2, 4} for e in got)
    assert len({tuple(sorted(e)) for e in got}) == 1 and len({tuple(e) for e in got}) > 1   # one pool, shuffled
    again = G.select_endorsers("random", 50, 3, levels, exclude=[5, 6], rng=np.random.default_rng(1))
    assert got == again
    assert all(len(set(e)) == 2 for e in G.select_endorsers("random", 20, 2, levels, rng=np.random.default_rng(2)))
    with pytest.raises(ValueError):
        G.select_endorsers("random", 1, 4, levels, exclude=[5, 6], rng=np.random.default_rng(1))   # pool of 3
    with pytest.raises(ValueError):
        G.select_endorsers("random", 1, 1, levels)                             # needs a generator
    with pytest.raises(ValueError):
        G.select_endorsers("adaptive", 1, 1, levels, rng=np.random.default_rng(1))
    assert G.select_endorsers("random", 3, 0, levels, rng=np.random.default_rng(1)) == [[], [], []]


def test_control_arm_has_no_fake_and_only_the_background_requests():
    bg = background()
    cred = G.compute_credibility_levels(bg.Tr, bg.D.n, seeded=True)
    D_c, G_c, cred_c, reqs_c = G.build_control_graph(bg.D, bg.G, cred, bg.honest)
    assert D_c.n == bg.D.n and G_c.n == bg.D.n and _graphs_equal(G_c, bg.G)
    assert np.array_equal(cred_c, cred)
    assert len(reqs_c) == len(bg.honest) and all(a is b for a, b in zip(reqs_c, bg.honest))
    assert {q.tag for q in reqs_c} == {"honest"} and all(q.account is None for q in reqs_c)


def test_hold_test_snapshot_keeps_one_test_set_for_every_arm():
    bg = background()
    n, tt = bg.D.n, bg.tt
    test_rids = bg.G.rid[bg.G.snap == tt]
    erase_rids, _ = V.plain_erasure(bg.D, bg.G, tt, bg.reqs)
    # Default: exact retraining re-cuts the snapshots of the erased data, so the test snapshot shrinks and some
    # original test edges move into training. This is why the arms' test sets differ without the flag.
    Dp0, Gp0, Trp0 = G.plain_erasure_graph(bg.D, bg.G, tt, erase_rids)
    Dp_ref, ridp = C.erased_dataset(bg.D, erase_rids)
    assert _graphs_equal(Gp0, C.full_graph(Dp_ref, rid=ridp)) and _graphs_equal(Trp0, C.train_split(Gp0, tt))
    assert np.isin(Trp0.rid, test_rids).any()
    # Held: the erased rows leave and every other edge keeps its snapshot, as in Plain VPE and CPPR.
    Dp1, Gp1, Trp1 = G.plain_erasure_graph(bg.D, bg.G, tt, erase_rids, hold_test_snapshot=True)
    keep = ~np.isin(bg.G.rid, erase_rids)
    assert _graphs_equal(Gp1, bg.G.subset(keep)) and Dp1.n == n and np.array_equal(Dp1.edges, Gp1.edges)
    assert np.array_equal(np.sort(Gp1.rid[Gp1.snap == tt]), np.sort(test_rids))
    assert not np.isin(Trp1.rid, test_rids).any() and _graphs_equal(Trp1, C.train_split(Gp1, tt))
    # Every arm is scored on the original test rows with the original Task 1 and Task 2 masks.
    cred = G.compute_credibility_levels(bg.Tr, n, seeded=True)
    rng = np.random.default_rng(0)
    res_vpe = V.apply_vpe(bg.D, bg.G, tt, bg.reqs, V.VPEConfig(1, INF, 10), rng)
    res_cp, _ = G.apply_cp_vpe(bg.D, bg.G, tt, bg.reqs, cred, G.CPVPEConfig(1, INF, 10), rng)
    D_ww, G_ww, cred_ww, reqs_ww, _ = _injected(bg, cred, bg.edge_t, +1, 3, None)
    res_ww, _ = G.apply_cp_vpe(D_ww, G_ww, tt, reqs_ww, cred_ww, G.CPVPEConfig(1, INF, 10), rng)
    arms = [(Dp0, Gp0), (Dp1, Gp1), (res_vpe.D, res_vpe.G), (res_cp.D, res_cp.G), (res_ww.D, res_ww.G)]
    rows0, t1_0, t2_0 = C.task_masks(bg.G, tt)
    for D_a, G_a in arms:
        D_e, G_e = G.evaluation_graphs(True, bg.D, bg.G, D_a, G_a)
        rows, t1, t2 = C.task_masks(G_e, D_e.tt)
        assert np.array_equal(G_e.rid[rows], bg.G.rid[rows0]) and np.array_equal(t1, t1_0)
        assert np.array_equal(t2, t2_0) and np.array_equal(G_e.edges[rows], bg.G.edges[rows0])
        D_own, G_own = G.evaluation_graphs(False, bg.D, bg.G, D_a, G_a)
        assert D_own is D_a and G_own is G_a                                   # default: the arm's own graph
    rows_p, _, _ = C.task_masks(Gp0, tt)
    assert not np.array_equal(np.sort(Gp0.rid[rows_p]), np.sort(bg.G.rid[rows0]))


# ================================================================== Ruling R39
def test_legacy_pass_replays_the_endorsement_pass():
    # join[v] is the snapshot of the row that admitted v to legacy's trusted set D (-1 for a seed). The accounts that
    # ever join are keeper.analysis.defense_extra.endorsed_set's D, and the snapshots equal an independent replay's.
    bg = background()
    n = bg.D.n
    join, none = G.legacy_pass(bg.Tr, n)
    assert join.dtype == np.int64 and none.shape == (0,)
    seeds, members = DE.endorsed_set(bg.Tr, n)
    assert set(np.flatnonzero(join == -1).tolist()) == seeds
    assert set(np.flatnonzero(join != G.NEVER_JOINS).tolist()) == members
    oracle = legacy_join_snapshots(bg.Tr, n)
    assert np.array_equal(np.where(join == G.NEVER_JOINS, -2, join), np.where(oracle == 10 ** 6, -2, oracle))
    # A sender that joined in an earlier snapshot is in D when its row is read, and one in D has joined by then.
    rows = np.arange(0, len(bg.Tr.edges), 37)
    _, in_D = G.legacy_pass(bg.Tr, n, rows)
    senders, snaps = bg.Tr.edges[rows, 0], bg.Tr.snap[rows]
    assert in_D.dtype == bool and len(in_D) == len(rows) and in_D.any() and not in_D.all()
    assert in_D[join[senders] < snaps].all() and (join[senders][in_D] <= snaps[in_D]).all()


@pytest.mark.parametrize("topology", ["ring", "star"])
@pytest.mark.parametrize("g", [1, 2])
def test_timely_random_endorsers_let_legacy_lift_every_region(g, topology):
    # Under BFS tiers a group's random endorsers come from the common pool (ruling R43), whose members joined D in the
    # honest base pass by the group's snap_f (ruling R39), so they are in D when the pass reads the attack edges.
    # Legacy then lifts every fake of every group, and every endorser is recorded as in D.
    bg = background()
    n, k, groups = bg.D.n, 10, len(bg.edge_t)
    protocol, tier_fn, inputs = _alpha_setting("bfs", g)
    picker = G.arm_picker(protocol, "whitewash", 0, inputs, bg.edge_t)
    assert picker.join is inputs.join and picker.selection == "random"
    _, _, cred_inj, _, Tr_inj = _injected(bg, inputs.bfs_levels, bg.edge_t, +1, k, picker, topology)
    assert (cred_inj[n:] >= 1).all()
    rec = G.admission_record(cred_inj, Tr_inj, n, groups, k, protocol, tier_fn, picker, inputs.join)
    assert rec["n_admitted"] == k * groups and rec["endorsers_in_D_at_attack"] == [[True] * g] * groups
    untimed, _ = G.common_pool(inputs, bg.edge_t)
    for i, endorsers in enumerate(rec["endorsers"]):
        own, _, _ = _group_rows(Tr_inj, n, i, k)
        assert len(set(endorsers)) == g and set(endorsers) <= set(untimed.tolist())
        assert (inputs.join[endorsers] <= Tr_inj.snap[own[0]]).all()
    sizes = rec["pool_size"]
    assert rec["timing_applied"] and rec["pool_size_untimed"] == len(untimed)
    assert g <= sizes["min"] <= sizes["median"] <= sizes["max"] <= len(untimed)
    assert 0 < rec["share_removed_by_timing"] < 1


def test_btp_random_endorsers_draw_from_the_timely_common_pool():
    # Ruling R43 replaces R33's per-method pools and R39's exemption of BTP: a BTP run draws from the same common pool
    # as a BFS run, restricted at each group's snap_f by legacy's join snapshots, so late joiners of legacy's D are
    # never drawn and the record reports timing as applied.
    bg = background()
    n, groups = bg.D.n, len(bg.edge_t)
    protocol, btp, inputs = _alpha_setting("capped", 1)
    picker = G.arm_picker(protocol, "whitewash", 0, inputs, bg.edge_t)
    pool, _ = G.common_pool(inputs, bg.edge_t)
    assert picker.join is inputs.join and np.array_equal(picker.pool, pool)
    late = set(pool[inputs.join[pool] > 5].tolist())
    assert late and not {u for _ in range(200) for u in picker(0, 5)} & late
    _, _, cred_inj, _, Tr_inj = _injected(bg, inputs.reference_levels, bg.edge_t, +1, 3, picker, tier_fn=btp)
    rec = G.admission_record(cred_inj, Tr_inj, n, groups, 3, protocol, btp, picker, inputs.join)
    assert rec["timing_applied"] and rec["pool_size_untimed"] == len(pool)
    assert rec["pool_size"]["max"] <= len(pool) and 0 < rec["share_removed_by_timing"] < 1
    assert rec["endorsers_in_D_at_attack"] == [[True]] * groups


# A toy platform. Snapshot 0: accounts 5 and 6 rate 0-4, which become seeds, as does 10 (rated in snapshot 1).
# Snapshot 2: the seeds rate one another. Snapshot 5: 0-3 rate 12, which joins D there. Snapshot 6: 0-3 rate 11,
# which joins D only then. Legacy tiers: 11 and 12 (four trusted ratings each) take Tier 3, and 0-4 and 10 Tier 1.
TOY_ROWS = ([(r, s, +1, 0) for s in range(5) for r in (5, 6)] + [(0, 10, +1, 1)]
            + [(1, 0, +1, 2), (0, 1, +1, 2), (3, 2, +1, 2), (2, 3, +1, 2), (0, 4, +1, 2)]
            + [(u, 12, +1, 5) for u in range(4)] + [(u, 11, +1, 6) for u in range(4)] + [(7, 8, +1, 7)])
TOY_N, TOY_T = 13, 8                                   # snapshots 0-6 train; snapshot 7 is held out


def _toy_platform():
    rows = np.array(TOY_ROWS, dtype=np.int64)
    labels = S.one_hot_from_sign(rows[:, 2])
    D = C.Dataset("toy", rows[:, :2], labels, rows[:, 3] * 100, TOY_N, TOY_T, {}, meta={})
    return D, C.Graph(rows[:, :2], labels, rows[:, 3] * 100, rows[:, 3], TOY_N), TOY_T - 1


def _toy_inputs(Tr, **changes):
    """The toy's EndorserInputs with tiers set explicitly: the legacy tiers and join snapshots, reference tiers equal to
    the legacy ones, and every account's B(u) above theta0, so that criteria (b) and (e) remove nobody."""
    levels = G.compute_credibility_levels(Tr, TOY_N, seeded=True)
    inputs = G.EndorserInputs(bfs_levels=levels, reference_levels=levels, reference_inflow=np.ones(TOY_N),
                              reference_theta0=0.5, join=G.legacy_pass(Tr, TOY_N)[0], rank_inflow=np.ones(TOY_N))
    return dataclasses.replace(inputs, **changes)


def test_a_late_joiner_is_left_out_of_the_bfs_pool_and_lifts_only_its_fake_when_forced():
    D, Gt, tt = _toy_platform()
    Tr = C.train_split(Gt, tt)
    cred = G.compute_credibility_levels(Tr, TOY_N, seeded=True)
    assert cred.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 3, 3]
    join = G.legacy_pass(Tr, TOY_N)[0]
    assert join[[0, 1, 2, 3, 4, 10, 11, 12]].tolist() == [-1] * 6 + [6, 5]
    assert (join[5:10] == G.NEVER_JOINS).all()
    bfs = G.RunProtocol(attack_edges=1, pool_eps=ALPHA_EPS)
    picker = G.arm_picker(bfs, "whitewash", 0, _toy_inputs(Tr), [10])
    assert picker.pool.tolist() == [11, 12]
    assert all(picker(0, 5) == [12] for _ in range(20))          # 12 joined in snapshot 5 itself, 11 after it
    assert {u for _ in range(50) for u in picker(0, 6)} == {11, 12}
    with pytest.raises(ValueError, match="credible"):              # never filled silently
        G.arm_picker(G.RunProtocol(attack_edges=2, pool_eps=ALPHA_EPS), "whitewash", 0, _toy_inputs(Tr), [10])(0, 5)
    seed = rng_with_first_snapshot(5, tt)                          # snap_f = 5
    for endorsers, lifted, in_D in (([[11]], 1, False), ([[12]], 3, True), (picker, 3, True)):
        _, G_inj, cred_inj, _ = G.build_inject_then_erase_graph(D, Gt, tt, cred, [10], +1, 3, [],
                                                                 np.random.default_rng(seed), endorsers=endorsers)
        Tr_inj = C.train_split(G_inj, tt)
        assert int((cred_inj[TOY_N:] > 0).sum()) == lifted and cred_inj[TOY_N] > 0   # the rated fake always
        rec = G.admission_record(cred_inj, Tr_inj, TOY_N, 1, 3, bfs, G.CredibilityTiers("bfs"),
                                 picker if callable(endorsers) else None, join)
        assert rec["endorsers_in_D_at_attack"] == [[in_D]]


def test_the_ordering_premises_of_r39_are_checked():
    # join <= snap_f guarantees membership only if the honest rows keep their order and precede the injected rows of
    # their snapshot. honest_rows_lead checks both; with endorsers, a graph whose rows are not in snapshot order is
    # refused. Without endorsers the stored behaviour (no check) is kept.
    assert G.honest_rows_lead(np.array([0, 1, 2, 3]), np.array([0, 1, 1, 1]), 3)
    assert not G.honest_rows_lead(np.array([0, 1, 3, 2]), np.array([0, 1, 1, 1]), 3)   # injected before honest
    assert not G.honest_rows_lead(np.array([1, 0, 2]), np.array([0, 0, 1]), 3)         # honest rows reordered
    D, Gt, tt = _toy_platform()
    cred = G.compute_credibility_levels(C.train_split(Gt, tt), TOY_N, seeded=True)
    backwards = Gt.subset(np.arange(len(Gt.edges))[::-1])                             # snapshots descending
    with pytest.raises(ValueError, match="snapshot order"):
        G.build_inject_then_erase_graph(D, backwards, tt, cred, [10], +1, 3, [], np.random.default_rng(0),
                                        endorsers=[[12]])
    G.build_inject_then_erase_graph(D, backwards, tt, cred, [10], +1, 3, [], np.random.default_rng(0))


# ================================================================== Task 4.1b
def _group_snaps(Tr_inj, n_honest, n_groups, k_fake):
    """The snap_f of each target group: the snapshot of the group's own edges."""
    return [int(Tr_inj.snap[_group_rows(Tr_inj, n_honest, i, k_fake)[0][0]]) for i in range(n_groups)]


@pytest.mark.parametrize("method,topology", [("bfs", "ring"), ("capped", "star")])
def test_random_endorsers_leave_the_stored_stream_untouched(method, topology):
    # The driver's arm-subset and --methods invariance needs every draw from the stored rng to be the same whatever
    # the options. Random endorsers draw from their arm's own stream (arm_rng), so building both injected graphs with
    # attack edges leaves rng's state, each group's snap_f and the bad-mouth targets as a build without them does.
    bg = background()
    n, g, k = bg.D.n, 2, 3
    protocol, tier_fn, inputs = _alpha_setting(method, g)
    cred = tier_fn(bg.Tr, n)
    runs = []
    for attack in (True, False):
        def picker(arm, targets):
            return G.arm_picker(protocol, arm, 0, inputs, targets) if attack else None

        rng = np.random.default_rng(3000)
        _, G_ww, _, _ = G.build_inject_then_erase_graph(bg.D, bg.G, bg.tt, cred, bg.edge_t, +1, k, bg.honest, rng,
                                                        endorsers=picker("whitewash", bg.edge_t), topology=topology,
                                                        tier_fn=tier_fn)
        after_whitewash = rng.bit_generator.state
        bm = rng.choice(bg.trusted, size=min(len(bg.edge_t), len(bg.trusted)), replace=False).tolist()
        _, G_bm, _, _ = G.build_inject_then_erase_graph(bg.D, bg.G, bg.tt, cred, bm, -1, k, bg.honest, rng,
                                                        sybil_ring=False, endorsers=picker("badmouth", bm),
                                                        topology=topology, tier_fn=tier_fn)
        runs.append(dict(after_whitewash=after_whitewash, bm=bm, end=rng.bit_generator.state,
                         snaps=[_group_snaps(C.train_split(Gi, bg.tt), n, len(t), k)
                                for Gi, t in ((G_ww, bg.edge_t), (G_bm, bm))],
                         n_edges=[len(G_ww.edges), len(G_bm.edges)]))
    with_attack, without = runs
    assert with_attack["n_edges"] == [m + g * len(bg.edge_t) for m in without["n_edges"]]   # the attack edges exist
    assert with_attack["after_whitewash"] == without["after_whitewash"] and with_attack["end"] == without["end"]
    assert with_attack["bm"] == without["bm"] and with_attack["snaps"] == without["snaps"]


def test_the_preflight_raises_the_pickers_errors_before_any_pick():
    # Before any training the driver checks each injected arm's endorsers: for random ones, g accounts of the common
    # pool before timing and in the timely pool at the first snapshot a group can draw (5); for ranked ones, g
    # candidates per group. On the toy the pool without target 10 is {11, 12}, and only 12 has joined D by snapshot 5.
    D, Gt, tt = _toy_platform()
    Tr = C.train_split(Gt, tt)
    inputs = _toy_inputs(Tr)

    def check(protocol, exclude=(10,), n_groups=1):
        G.preflight_endorser_pools(protocol, 0, inputs, dict(whitewash=list(exclude)), dict(whitewash=n_groups))

    check(G.RunProtocol(attack_edges=1, pool_eps=ALPHA_EPS))
    with pytest.raises(ValueError, match=r"^before training, whitewash arm, pool without its targets \(1\): only 1 "
                                         r"candidate endorsers credible at snapshot 5 for 2 attack edges \(ruling R39\)$"):
        check(G.RunProtocol(attack_edges=2, pool_eps=ALPHA_EPS))   # the error a group drawn at snapshot 5 would raise
    with pytest.raises(ValueError, match=r"^before training, whitewash arm, pool without its targets \(1\): only 2 "
                                         r"candidate endorsers for 3 attack edges per group$"):
        check(G.RunProtocol(attack_edges=3, pool_eps=ALPHA_EPS))   # the untimed pool's error
    with pytest.raises(ValueError, match=r"^before training, badmouth arm, pool without the accounts its targets are "
                                         r"drawn from \(2\): only 0 candidate endorsers credible at snapshot 5 for 1 "):
        G.preflight_endorser_pools(G.RunProtocol(attack_edges=1, pool_eps=ALPHA_EPS), 0, inputs,
                                   dict(badmouth=[10, 12]), dict(badmouth=1),
                                   drawn_from=("badmouth",))      # the wording for a superset of the targets
    with pytest.raises(ValueError, match="credible at snapshot 5"):
        check(G.RunProtocol(attack_edges=1, pool_eps=ALPHA_EPS), exclude=(10, 12))   # nobody timely is left
    # Ruling R43: one pool for every tier method, timing included, so a BTP run is checked as a BFS run is.
    with pytest.raises(ValueError, match="credible at snapshot 5 for 2 attack edges"):
        check(G.RunProtocol(tiers="btp", btp_eps=ALPHA_EPS, attack_edges=2, pool_eps=ALPHA_EPS))
    # Ruling R44: ranked max-B(u) endorsers need g candidates outside the targets for every group (12 on the toy).
    max_b = G.RunProtocol(endorser_select="max_trust", attack_edges=4, pool_eps=ALPHA_EPS)
    check(max_b, n_groups=3)
    with pytest.raises(ValueError, match=r"^before training, whitewash arm, pool without its targets \(1\): only 12 "
                                         r"candidate endorsers for 4 groups of 4 ranked max-B\(u\) endorsers "
                                         r"\(ruling R44\)$"):
        check(max_b, n_groups=4)
    # Nothing to check without attack edges.
    check(G.RunProtocol(attack_edges=0), exclude=range(TOY_N))


def test_the_timely_pool_is_smallest_at_the_first_snapshot_a_group_can_draw():
    # The graph builder draws each group's snap_f from [FIRST_FAKE_SNAPSHOT, tt), and legacy's trusted set D only
    # grows along the pass, so the timely pool at FIRST_FAKE_SNAPSHOT is contained in every later one.
    bg = background()
    n = bg.D.n
    cred = G.compute_credibility_levels(bg.Tr, n, seeded=True)

    class LowestSnapshot:
        """Stands in for rng: records each integers(lo, hi) call and returns lo."""
        def __init__(self):
            self.calls = []

        def integers(self, lo, hi):
            self.calls.append((lo, hi))
            return lo

    rec = LowestSnapshot()
    _, G_inj, _, _ = G.build_inject_then_erase_graph(bg.D, bg.G, bg.tt, cred, bg.edge_t[:2], +1, 3, bg.honest, rec)
    assert G.FIRST_FAKE_SNAPSHOT == 5 and rec.calls == [(G.FIRST_FAKE_SNAPSHOT, bg.tt)] * 2
    assert _group_snaps(C.train_split(G_inj, bg.tt), n, 2, 3) == [G.FIRST_FAKE_SNAPSHOT] * 2
    protocol, _, inputs = _alpha_setting("bfs", 1)
    picker = G.arm_picker(protocol, "badmouth", 0, inputs, bg.trusted)
    pools = [set(picker.candidates(s).tolist()) for s in range(G.FIRST_FAKE_SNAPSHOT, bg.tt)]
    assert all(pools[0] <= pool for pool in pools) and len(pools[0]) < len(pools[-1])


@pytest.mark.parametrize("attack_edges", [0, 2])
def test_each_injected_arm_records_its_groups_snapshots(attack_edges):
    # sybil_admission stores each group's snap_f in group order, next to the endorsers, so that a later run of the
    # same seed (E5) can be checked to inject at the same snapshots: one entry per group, the snapshot of the group's
    # own injected rows, and the stored rng's draw. Without attack edges too (the E3 runs).
    bg = background()
    n, k = bg.D.n, 3
    cred = G.compute_credibility_levels(bg.Tr, n, seeded=True)
    protocol, tier_fn, inputs = _alpha_setting("bfs", attack_edges)
    join = None if inputs is None else inputs.join
    rng, replay = np.random.default_rng(3000), np.random.default_rng(3000)
    targets_by_arm = dict(whitewash=bg.edge_t)
    for arm, sign in (("whitewash", +1), ("badmouth", -1)):
        if arm == "badmouth":                                    # drawn after the whitewash graph, as in the driver
            size = min(len(bg.edge_t), len(bg.trusted))
            targets_by_arm["badmouth"] = rng.choice(bg.trusted, size=size, replace=False).tolist()
            assert replay.choice(bg.trusted, size=size, replace=False).tolist() == targets_by_arm["badmouth"]
        targets = targets_by_arm[arm]
        draws = [int(replay.integers(G.FIRST_FAKE_SNAPSHOT, bg.tt)) for _ in targets]
        picker = G.arm_picker(protocol, arm, 0, inputs, targets)
        _, G_inj, cred_inj, _ = G.build_inject_then_erase_graph(bg.D, bg.G, bg.tt, cred, targets, sign, k, bg.honest,
                                                                rng, sybil_ring=sign > 0, endorsers=picker,
                                                                tier_fn=tier_fn)
        Tr_inj = C.train_split(G_inj, bg.tt)
        rec = G.admission_record(cred_inj, Tr_inj, n, len(targets), k, protocol, tier_fn, picker, join)
        assert rec["snap_f"] == draws == _group_snaps(Tr_inj, n, len(targets), k) and len(draws) == len(targets)
        keys = list(rec)
        assert keys.index("snap_f") == keys.index("endorsers") + 1
