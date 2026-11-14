"""Task 3.3: the witness levels of a stream, the witness attack and Property 1's per-record bound.

- Witness levels: the causal BTP tiers under the calibration frozen on the clean stream (ruling R32), with the
  clean stream a required argument and the credible-negatives guard by default (ruling R56).
- The attack: Sybils bad-mouth honest warners after the seed window (ruling R34), endorsed by accounts credible at
  the attack-edge snapshot (ruling R39), round-robin, optionally linked in a ring.
- Property 1 (keeper.scem.property1): the bound by hand against defense_extra's closed form; test (b), attack minus
  comparator <= bound per record, over the policy grid; the indirect paths that lie outside the bound; and a
  seeded randomized test over streams that meet every clause of the stated scope.
Shared helpers are in tests/scem_common.py. CPU only.
"""
import math

import numpy as np
import pytest

from keeper.analysis import defense_extra as DE
from keeper.credibility import tiers
from keeper.scem import policies as RP
from keeper.scem import property1 as P1
from keeper.scem import scem as SC
from keeper.scem import witness_attack as WA
from keeper.scem import witness_cred as WC
from tests.scem_common import N_NODES, PSI_NEG, SYBILS, TOL, W0, W, closed_form, engine, sybil_stream


# ------------------------------------------------------------------ witness levels of a stream (R32)


EPS = 0.5           # toy eps; callers pass each network's eps


def honest_stream(seed=3, n=40, E=600, K=21):
    """A random honest stream, 85% positive, sorted by snapshot."""
    rng = np.random.default_rng(seed)
    u = rng.integers(0, n, E)
    v = (u + rng.integers(1, n, E)) % n
    snap = np.sort(rng.integers(0, K, E))
    sign = np.where(rng.random(E) < 0.15, -1, 1)
    return np.column_stack([u, v]), sign, snap, n


def test_witness_levels_are_the_causal_btp_tiers_under_the_clean_calibration():
    edges, sign, snap, n = honest_stream()
    graph = WC.stream_graph(edges, sign, snap)
    levels = WC.witness_levels(edges, sign, snap, n, EPS, clean=(edges, sign, snap))
    cal = tiers.calibrate_btp_causal(graph, n, EPS, T=21, edge_cap_units=tiers.KEEPER_EDGE_CAP_UNITS)
    assert levels.shape == (21, n) and (levels[-1] >= 2).sum() >= 5
    assert np.array_equal(levels, tiers.causal_tiers(graph, n, calibration=cal))
    assert WC.clean_calibration(edges, sign, snap, n, EPS) == cal and cal.edge_cap_units == 2.0
    uncapped = tiers.calibrate_btp_causal(graph, n, EPS, T=21)
    assert np.array_equal(WC.witness_levels(edges, sign, snap, n, EPS, clean=(edges, sign, snap), edge_cap_units=None),
                          tiers.causal_tiers(graph, n, calibration=uncapped))
    assert np.array_equal(WC.level_weights(levels), tiers.causal_weights(graph, n, calibration=cal))


def test_witness_levels_require_the_clean_stream_explicitly():
    """R32 cannot be broken by forgetting `clean`: it is a required keyword, for a clean stream too."""
    edges, sign, snap, n = honest_stream()
    with pytest.raises(TypeError):
        WC.witness_levels(edges, sign, snap, n, EPS)


def test_witness_levels_take_the_credible_negatives_guard_by_default():
    """Ruling R56. witness_levels counts negative ratings from trusted raters only, unless told otherwise
    (neg_from="all", the ablation). Five fresh Sybils per victim, without attack edges, bad-mouth the warners: no
    honest witness tier moves in any row under the default, and some do under the ablation."""
    edges, sign, snap, n = honest_stream()
    clean = (edges, sign, snap)
    graph = WC.stream_graph(*clean)
    cal = WC.clean_calibration(*clean, n, EPS)
    for neg_from in ("trusted", "all"):
        want = tiers.causal_tiers(graph, n, calibration=cal, neg_from=neg_from)
        got = (WC.witness_levels(*clean, n, EPS, clean=clean) if neg_from == "trusted"
               else WC.witness_levels(*clean, n, EPS, clean=clean, neg_from=neg_from))
        assert np.array_equal(got, want), neg_from
    victims = [int(i) for i in np.flatnonzero(sign < 0) if 8 <= snap[i] <= 15]
    base = WC.witness_levels(*clean, n, EPS, clean=clean)
    att = WA.inject_adversarial_witnesses(edges, sign, snap, n, victims, 5, 1, 0, np.random.default_rng(0), base, 21)
    default = WC.witness_levels(att.edges, att.sign, att.snap, att.n_nodes, EPS, clean=clean)
    assert np.array_equal(default[:, :n], base) and not default[:, n:].any()
    ablation = WC.witness_levels(att.edges, att.sign, att.snap, att.n_nodes, EPS, clean=clean, neg_from="all")
    assert not np.array_equal(ablation[:, :n], WC.witness_levels(*clean, n, EPS, clean=clean, neg_from="all"))


# ------------------------------------------------------------------ the witness attack (R34, R39)


ATTACK_ROWS = [(1, 0, +1, 0), (2, 0, +1, 1), (3, 4, -1, 2), (5, 6, -1, 6), (7, 8, -1, 15), (9, 10, +1, 15),
               (0, 11, +1, 20)]

ATTACK_N = 12

CREDIBLE = (0, 1, 2, 5, 6, 9, 10, 11)                     # tier >= 2 in every row of the hand-made levels


def attack_toy():
    rows = np.asarray(ATTACK_ROWS, dtype=np.int64)
    levels = np.ones((21, ATTACK_N), dtype=np.int64)
    levels[:, list(CREDIBLE)] = [2, 3, 2, 3, 2, 3, 2, 3]
    return rows[:, :2], rows[:, 2], rows[:, 3], levels


def inject(victims=(2, 3, 4), k=3, lag=1, g=2, seed=7, **kw):
    edges, sign, snap, levels = attack_toy()
    args = dict(edges=edges, sign=sign, snap=snap, n_nodes=ATTACK_N, victims=list(victims), k=k, lag=lag, g=g,
                rng=np.random.default_rng(seed), base_levels=levels, K_snap=21)
    args.update(kw)
    return WA.inject_adversarial_witnesses(**args)


def test_warnings_about_selects_the_negative_ratings_of_the_targets():
    edges, sign, _, _ = attack_toy()
    assert WA.warnings_about(edges, sign, [4, 8, 10, 0]).tolist() == [2, 4]


def test_witness_attack_appends_sybils_attack_edges_and_badmouthing():
    edges, sign, snap, levels = attack_toy()
    att = inject()
    assert WA.InjectedStream._fields[:6] == ("edges", "sign", "snap", "n_nodes", "sybil_ids", "attack_edge_ids")
    assert att.victims.tolist() == [3, 4] and att.skipped_seed_window.tolist() == [2]
    assert att.skipped_beyond_stream.tolist() == []
    assert att.n_nodes == ATTACK_N + 2 * 3 and att.sybil_ids.tolist() == [[12, 13, 14], [15, 16, 17]]
    assert len(att.edges) == len(edges) + 2 * (2 + 3) and att.ring_edge_ids.shape == (2, 0)   # no ring by default
    assert np.array_equal(att.edges[:len(edges)], edges) and np.array_equal(att.sign[:len(edges)], sign)
    assert np.array_equal(att.snap[:len(edges)], snap)          # original records keep their ids
    assert att.t_attack.tolist() == [6, 15] and att.t_badmouth.tolist() == [7, 16]
    for i, rec in enumerate(att.victims):
        w, tgt = edges[rec]
        bad = att.badmouth_edge_ids[i]
        assert att.edges[bad, 0].tolist() == att.sybil_ids[i].tolist() and (att.edges[bad, 1] == w).all()
        assert (att.sign[bad] == -1).all() and (att.snap[bad] == att.t_badmouth[i]).all()
        ae = att.attack_edge_ids[i]
        assert att.edges[ae, 0].tolist() == att.endorsers[i].tolist()
        assert att.edges[ae, 1].tolist() == [att.sybil_ids[i][a % 3] for a in range(2)]
        assert (att.sign[ae] == 1).all() and (att.snap[ae] == att.t_attack[i]).all()
        assert len(set(att.endorsers[i].tolist())) == 2 and not {w, tgt} & set(att.endorsers[i].tolist())
        assert att.endorser_levels[i].tolist() == levels[att.t_attack[i], att.endorsers[i]].tolist()
        assert (att.endorser_levels[i] >= 2).all()
        assert att.pool_size[i] == len(set(CREDIBLE) - {int(w), int(tgt)})


@pytest.mark.parametrize("k, g", [(3, 2), (3, 5), (1, 2)])
def test_attack_edges_go_round_robin_to_the_first_g_sybils(k, g):
    """Endorser j endorses Sybil (j mod k) + 1, so g <= k attack edges reach Sybils 1..g."""
    att = inject(k=k, g=g)
    for i in range(len(att.victims)):
        targets = att.edges[att.attack_edge_ids[i], 1].tolist()
        assert targets == [int(att.sybil_ids[i][j % k]) for j in range(g)]
        if g <= k:
            assert targets == att.sybil_ids[i][:g].tolist()


def test_sybil_ring_rows_sit_at_the_attack_edge_snapshot():
    edges, _, _, _ = attack_toy()
    att = inject(k=3, g=2, sybil_ring=True)
    assert att.ring_edge_ids.shape == (2, 3) and len(att.edges) == len(edges) + 2 * (2 + 3 + 3)
    for i, sybils in enumerate(att.sybil_ids):
        ring = att.ring_edge_ids[i]
        assert att.edges[ring].tolist() == [[sybils[0], sybils[1]], [sybils[1], sybils[2]], [sybils[2], sybils[0]]]
        assert (att.sign[ring] == 1).all() and (att.snap[ring] == att.t_attack[i]).all()
        assert att.t_attack[i] >= 5 and att.t_badmouth[i] > att.t_attack[i]   # R34: after t_seed, before bad-mouthing
        assert att.attack_edge_ids[i].max() < ring.min() and ring.max() < att.badmouth_edge_ids[i].min()
    single = inject(k=1, g=2, sybil_ring=True)                 # one Sybil: no ring and no self-loop
    assert single.ring_edge_ids.shape == (2, 0) and not (single.edges[:, 0] == single.edges[:, 1]).any()
    default, off = inject(), inject(sybil_ring=False)
    for field in default._fields:
        assert np.array_equal(np.asarray(getattr(default, field)), np.asarray(getattr(off, field))), field


def test_kappa_farming_moves_only_the_ring_to_the_first_snapshot_after_the_seed_window():
    """Task 4.4 option (E1b's kappa-farming variant): the ring sits at t_seed + 1; every other field and row is the
    default ring attack's, and kappa_farming=False is the default. It needs a ring."""
    ring = inject(k=3, g=2, sybil_ring=True)
    farm = inject(k=3, g=2, sybil_ring=True, kappa_farming=True)
    ring_rows = farm.ring_edge_ids.ravel()
    assert (farm.snap[ring_rows] == 5).all() and (ring.snap[ring_rows] == ring.t_attack.repeat(3)).all()
    other = np.setdiff1d(np.arange(len(farm.snap)), ring_rows)
    assert np.array_equal(farm.snap[other], ring.snap[other])
    for field in farm._fields:
        if field != "snap":
            assert np.array_equal(np.asarray(getattr(farm, field)), np.asarray(getattr(ring, field))), field
    explicit_off = inject(k=3, g=2, sybil_ring=True, kappa_farming=False)
    assert np.array_equal(explicit_off.snap, ring.snap)
    with pytest.raises(ValueError, match="sybil_ring"):
        inject(kappa_farming=True)


def isolated_warnings():
    """Three warners, each warning its own target once (records 0, 1 and 2, at snapshots 7, 9 and 11). No
    other rating touches them, so each victim's revocation comes from its own Sybils alone."""
    rows = np.array([(0, 1, -1, 7), (2, 3, -1, 9), (4, 5, -1, 11)], dtype=np.int64)
    return rows[:, :2], rows[:, 2], rows[:, 3], 6


@pytest.mark.parametrize("ring", [False, True])
def test_mb_sybil_ring_without_attack_edges_keeps_its_sybils_in_tier0(ring):
    """M-B, g = 0. Without an attack edge the ring cannot lift a Sybil out of Tier 0, so each Sybil weighs
    w_0 * kappa. Without the ring kappa is 0 and the attack changes nothing. With the ring each Sybil holds
    the durable mass of its ring rating, released over the one snapshot before the bad-mouthing (P(2, 0.5)), so
    kappa > 0 and the attack revokes the closed-form share of 5 witnesses at w_0 * kappa: about 0.1%, within
    Property 1's bound (kappa_max = 1) but above 1e-12."""
    edges, sign, snap, n = isolated_warnings()
    clean = (edges, sign, snap)
    base = WC.witness_levels(*clean, n, EPS, clean=clean, T=21)
    att = WA.inject_adversarial_witnesses(edges, sign, snap, n, [0, 1, 2], 5, 1, 0, np.random.default_rng(0), base,
                                          21, sybil_ring=ring)
    levels = WC.witness_levels(att.edges, att.sign, att.snap, att.n_nodes, EPS, clean=clean, T=21)
    assert not levels[:, n:].any()
    before = engine(np.column_stack([edges, sign, snap]), n, witness_cred=base)["records"]["revoked"]
    after = engine(np.column_stack([att.edges, att.sign, att.snap]), att.n_nodes,
                   witness_cred=levels)["records"]["revoked"][:len(edges)]
    diff = after - before
    if not ring:
        assert np.all(diff <= TOL)
    else:
        p = float(SC.erlang_Q_P_np(2, 0.5 * 1.0)[1])
        kappa = (p + 1.0) / (p + 2.0) * (1.0 - math.exp(-p / 2.0))
        want = closed_form(1, [W0 * kappa] * len(SYBILS))
        assert np.all(np.abs(diff - want) < TOL) and want > 1e-4
        assert np.all(diff <= closed_form(1, [W0 * 1.0] * len(SYBILS)))


@pytest.mark.parametrize("lag", [0, 1, 3, 4, 5, 6, 14, 20])
def test_sybils_act_only_after_the_seed_window(lag):
    """Ruling R34: attack edges at max(t_k + lag - 1, t_seed + 1), bad-mouthing at t_k + lag >= that + 1;
    a victim whose bad-mouthing would fall at or before t_seed + 1, or beyond the stream, is skipped."""
    _, _, snap, _ = attack_toy()
    att = inject(lag=lag)
    t_b = {rec: int(snap[rec]) + lag for rec in (2, 3, 4)}
    assert att.victims.tolist() == [rec for rec, t in t_b.items() if 5 < t < 21]
    assert att.skipped_seed_window.tolist() == [rec for rec, t in t_b.items() if t <= 5]
    assert att.skipped_beyond_stream.tolist() == [rec for rec, t in t_b.items() if t >= 21]
    assert (att.t_attack == np.maximum(snap[att.victims] + lag - 1, 5)).all()
    assert (att.t_badmouth == snap[att.victims] + lag).all() and (att.t_badmouth >= att.t_attack + 1).all()


def test_witness_attack_is_deterministic_given_the_rng():
    a, b, c = inject(seed=11), inject(seed=11), inject(seed=12, k=1)
    for field in a._fields:
        assert np.array_equal(np.asarray(getattr(a, field)), np.asarray(getattr(b, field))), field
    assert c.sybil_ids.shape == (2, 1) and (c.edges[c.attack_edge_ids.reshape(-1), 1] ==
                                            np.repeat(c.sybil_ids[:, 0], 2)).all()   # a mod k = 0


def test_endorsers_are_drawn_uniformly_from_the_credible_pool():
    edges, sign, snap, _ = attack_toy()
    levels = np.zeros((21, ATTACK_N), dtype=np.int64)
    levels[:, [5, 6, 9, 10, 11]] = 2                  # the victim 5 -> 6 at 6 leaves the pool {9, 10, 11}
    counts = np.zeros(ATTACK_N, dtype=np.int64)
    for seed in range(600):
        att = inject(victims=[3], g=1, seed=seed, base_levels=levels)
        counts[att.endorsers[0, 0]] += 1
    assert counts[[9, 10, 11]].sum() == 600 and counts[[9, 10, 11]].min() >= 150


@pytest.mark.parametrize("lag", [1, 2])
def test_endorsers_must_be_credible_at_the_attack_edge_snapshot(lag):
    """R39 reads base_levels at t_a = t_k + lag - 1 for the warning at t_k = 6: row 6 at lag 1, row 7 at lag 2,
    where t_a differs from t_k. Account 9 is credible only in row t_a; account 10 in every other row, t_k, t_b and
    the last row included. Reading any row other than t_a would draw account 10."""
    t_a = 6 + lag - 1
    levels = np.zeros((21, ATTACK_N), dtype=np.int64)
    levels[:, 10] = 3
    levels[t_a, 10] = 0
    levels[t_a, 9] = 2
    att = inject(victims=[3], g=1, lag=lag, base_levels=levels)
    assert att.t_attack.tolist() == [t_a] and att.t_badmouth.tolist() == [t_a + 1]
    assert att.endorsers.tolist() == [[9]] and att.endorser_levels.tolist() == [[2]] and att.pool_size.tolist() == [1]


def test_too_few_credible_endorsers_raise():
    edges, sign, snap, _ = attack_toy()
    levels = np.zeros((21, ATTACK_N), dtype=np.int64)
    levels[:, [5, 6, 9]] = 3                          # one eligible account besides W and T
    assert inject(victims=[3], g=1, base_levels=levels).endorsers.tolist() == [[9]]
    with pytest.raises(ValueError, match="fewer than g"):
        inject(victims=[3], g=2, base_levels=levels)


def test_witness_attack_input_is_checked():
    _, _, _, levels = attack_toy()
    for bad in (dict(victims=[0]), dict(victims=[3, 3]), dict(victims=[99]), dict(k=0), dict(lag=-1),
                dict(g=-1), dict(K_snap=0), dict(base_levels=levels[:20]), dict(base_levels=levels[:, :11]),
                dict(n_nodes=5)):
        with pytest.raises(ValueError):
            inject(**bad)
    with pytest.raises(TypeError):
        inject(rng=7)


@pytest.mark.parametrize("ring", [False, True])
def test_attack_edges_reach_the_tiers_before_the_badmouthing(ring):
    """On an honest stream under the frozen clean calibration: without attack edges the Sybils stay in Tier 0
    in every row, ring or not; with them, a Sybil can enter Tier >= 1 only from row t_a + 1 = t_b on, so row
    t_b already weighs its bad-mouthing by the tier the attack edges bought (R34)."""
    edges, sign, snap, n = honest_stream()
    clean = (edges, sign, snap)
    base = WC.witness_levels(*clean, n, EPS, clean=clean)
    victims = [int(i) for i in np.flatnonzero(sign < 0) if 8 <= snap[i] <= 15]
    for g in (0, 2):
        att = WA.inject_adversarial_witnesses(edges, sign, snap, n, victims, 5, 1, g, np.random.default_rng(0),
                                              base, 21, sybil_ring=ring)
        levels = WC.witness_levels(att.edges, att.sign, att.snap, att.n_nodes, EPS, clean=clean)
        cal = WC.clean_calibration(*clean, att.n_nodes, EPS)
        graph = WC.stream_graph(att.edges, att.sign, att.snap)
        assert np.array_equal(levels, tiers.causal_tiers(graph, att.n_nodes, calibration=cal))
        sybil_rows = [levels[:, s] for s in att.sybil_ids.reshape(-1)]
        if g == 0:
            assert not np.any(sybil_rows)
        else:
            for i, sybils in enumerate(att.sybil_ids):
                assert not levels[: att.t_attack[i] + 1, sybils].any()
            reached = [levels[att.t_badmouth[i], s] for i, row in enumerate(att.sybil_ids) for s in row]
            assert max(reached) >= 1                  # the channel exists on this stream


def test_witness_attack_end_to_end_on_an_honest_stream():
    """Attack minus clean revoked share of the victim warnings, per victim. Stored rule: the Sybils revoke a
    large part on average (26% here), though not of every warning: a warner whose live positive evidence
    outweighs the Sybils' 1.0 has no hazard (7 of 38 here). M-B with g = 0: Tier-0 Sybils add no evidence, and
    what remains (0.1% on average) is second order. Their negative ratings no longer count in the BTP distrust
    guard (ruling R56), so no honest tier moves, but they still enter the warner's durable tallies, which shifts
    some reliabilities."""
    edges, sign, snap, n = honest_stream()
    clean = (edges, sign, snap)
    base = WC.witness_levels(*clean, n, EPS, clean=clean)
    victims = [int(i) for i in np.flatnonzero(sign < 0) if 8 <= snap[i] <= 15]
    att = WA.inject_adversarial_witnesses(edges, sign, snap, n, victims, 5, 1, 0, np.random.default_rng(0), base, 21)
    levels = WC.witness_levels(att.edges, att.sign, att.snap, att.n_nodes, EPS, clean=clean)
    assert np.array_equal(levels[:, :n], base)                    # R56: the Sybils move no honest tier
    rows_clean = np.column_stack([edges, sign, snap])
    rows_att = np.column_stack([att.edges, att.sign, att.snap])
    for mb in (False, True):
        before = engine(rows_clean, n, witness_cred=base if mb else None)["records"]["revoked"][att.victims]
        out = engine(rows_att, att.n_nodes, witness_cred=levels if mb else None)
        after = out["records"]["revoked"][att.victims]
        assert out["max_conservation_err"] <= TOL and out["monotone_verified"]
        if mb:
            assert np.mean(after - before) <= 0.01
        else:
            assert np.mean(after - before) > 0.1


# ------------------------------------------------------------------ Property 1, per record (keeper.scem.property1)


@pytest.mark.parametrize("lag, live", [(1, 6), (12, 1)])
def test_property1_bound_by_hand(lag, live):
    """Five fresh Sybils at the stored floor 0.20 warn W: their evidence adds X = 5 * 0.20 = 1 for H = 6
    snapshots, so each channel's hazard rises by at most gamma_sigma * L_h * X. The bound is the revoked share of
    W's warning under exactly that added hazard (defense_extra's engine-rule closed form): above what the engine
    revokes, below the reviewer's 1 - exp(-gamma_max L_h sum_t X) over the `live` open snapshots. At lag 12 the
    evidence matures at age 13 and meets only the warning's closing segment [13, A)."""
    rows, k = sybil_stream(lag=lag)
    arr = np.asarray(rows, dtype=np.int64)
    bad = np.arange(len(rows) - len(SYBILS), len(rows))
    out = engine(rows, N_NODES)
    bound = P1.property1_bound(arr[:, :2], arr[:, 2], arr[:, 3], bad, out["records"]["witness_kappa"], K_snap=21)
    want = sum(p * DE.revoked_engine_rules(lag, g * 1.0 * 1.0) for p, g in zip(PSI_NEG, DE.ENGINE["gamma"]))
    assert abs(bound[k] - want) < TOL
    assert out["records"]["revoked"][k] < bound[k] < 1.0 - math.exp(-1.0 * 1.0 * live * 1.0)
    others = np.setdiff1d(np.arange(len(rows)), np.flatnonzero(arr[:, 0] == W))
    assert not bound[others].any()                                    # only W's records are exposed
    with pytest.raises(ValueError):
        P1.property1_bound(arr[:, :2], arr[:, 2], arr[:, 3], bad, out["records"]["witness_kappa"], K_snap=21,
                           policy="retroactive")


def test_routing_is_the_engines_and_defense_extras():
    """property1's ROUTING is the engine's channel mixing. A rating warned once by a fresh witness loses the psi-mix
    of the per-channel closed forms, for either sign; defense_extra pins the same psi."""
    assert P1.ROUTING == {1: DE.ENGINE["psi"]["positive"], -1: DE.ENGINE["psi"]["negative"]}
    lam = DE.witness_hazard([0.20])["lam"]
    for sign in (1, -1):
        out = engine([(0, 1, sign, 5), (2, 0, -1, 6)], 3)
        want = sum(p * DE.revoked_engine_rules(1, x) for p, x in zip(P1.ROUTING[sign], lam))
        assert abs(out["records"]["revoked_haz"][0] - want) < TOL
    assert WA.property1_bound is P1.property1_bound              # re-exported where callers already import it


def community_and_warnings(seed=5, n_comm=12, prior=0):
    """A credible community (accounts 0..11) that endorses itself in snapshots 0 to 4, so that BTP finds seeds
    and Tier >= 2 endorsers, and three warners outside it (12 -> 13 at snapshot 7, 14 -> 15 at 9 and 16 -> 17
    at 11) that no later rating touches. With `prior`, community accounts 0..prior-1 rate each warner positively
    at snapshot 0, so that a warner holds durable consensus mass >= 2 when the Sybils bad-mouth it (V2 live).
    Sorted by snapshot; returns the warnings' ids too."""
    rng = np.random.default_rng(seed)
    u = np.repeat(np.arange(n_comm), 3)
    v = (u + rng.integers(1, n_comm, len(u))) % n_comm
    rows = np.column_stack([u, v, np.ones_like(u), rng.integers(0, 5, len(u))])
    raters = np.asarray([(r, w, 1, 0) for w in (12, 14, 16) for r in range(prior)], dtype=np.int64).reshape(-1, 4)
    rows = np.concatenate([rows, raters])
    rows = np.concatenate([rows, [(12, 13, -1, 7), (14, 15, -1, 9), (16, 17, -1, 11)]])
    rows = rows[np.argsort(rows[:, 3], kind="stable")]
    return rows[:, :2], rows[:, 2], rows[:, 3], n_comm + 6, np.flatnonzero(rows[:, 2] < 0)


def reviewer_form(att, weights, rec, A, K=21, L_f=1, H=6):
    """1 - exp(-gamma_max L_h sum_t sum_j min(kappa_hat_j, cap_c)) over the record's open snapshots t (t - t_k < A)
    at which witness j's evidence about its sender is live (gamma_max = L_h = cap_c = 1)."""
    sender, t_k = att.edges[rec, 0], int(att.snap[rec])
    total = sum(min(weights[j], 1.0) for j in att.badmouth_edge_ids.reshape(-1) if att.edges[j, 1] == sender
                for t in range(int(att.snap[j]) + L_f, int(att.snap[j]) + L_f + H) if t_k <= t < K and t - t_k < A)
    return 1.0 - math.exp(-total)


BOUND_POLICIES = [("arrival", {}), ("erlang", dict(N=1, phi=0.25)), ("erlang", {}), ("erlang", dict(N=3, phi=0.75)),
                  ("erlang", dict(N=5, phi=1.25)), ("quarantine", {}), ("no_commit", {})]


@pytest.mark.parametrize("prior", [0, 4])
@pytest.mark.parametrize("policy, kw", BOUND_POLICIES)
@pytest.mark.parametrize("mb", [True, False])
@pytest.mark.parametrize("k, g, ring", [(5, 0, False), (5, 0, True), (5, 2, False), (5, 2, True), (50, 2, True),
                                        (50, 0, True)])
def test_witness_evidence_stays_within_the_property1_bound(k, g, ring, mb, policy, kw, prior):
    """(b) Attack minus comparator, per record, is at most property1_bound, under every release policy. The
    comparator is the attacked stream without its bad-mouthing edges: it keeps the attack edges and the ring, and
    it reuses the attacked stream's witness levels. Recomputed levels could differ: once attack edges make a Sybil
    trusted, the BTP distrust guard counts its bad-mouthing against the warner (ruling R56 ignores only untrusted
    raters). On this stream nothing else reaches the warners.
    With prior raters, each warner holds consensus mass >= 2, so V2 self-surprise raises the Sybils' hazard. The
    guarantee then covers every record not sent by a witness; the ring, sent by the Sybils, is revoked more. (Under
    the stored rule the ring's own positive evidence offsets V2 until it expires.)"""
    edges, sign, snap, n, victims = community_and_warnings(prior=prior)
    clean = (edges, sign, snap)
    base = WC.witness_levels(*clean, n, EPS, clean=clean, T=21)
    att = WA.inject_adversarial_witnesses(edges, sign, snap, n, victims, k, 1, g, np.random.default_rng(3), base, 21,
                                          sybil_ring=ring)
    levels = WC.witness_levels(att.edges, att.sign, att.snap, att.n_nodes, EPS, clean=clean, T=21) if mb else None
    keep = np.ones(len(att.edges), dtype=bool)
    keep[att.badmouth_edge_ids.reshape(-1)] = False
    rows = np.column_stack([att.edges, att.sign, att.snap])
    attacked = engine(rows, att.n_nodes, witness_cred=levels, policy=policy, **kw)["records"]
    comparator = engine(rows[keep], att.n_nodes, witness_cred=levels, policy=policy, **kw)["records"]
    bound = P1.property1_bound(att.edges, att.sign, att.snap, att.badmouth_edge_ids.reshape(-1),
                               attacked["witness_kappa"], K_snap=21, policy=policy, **kw)
    delta = attacked["revoked"][keep] - comparator["revoked"]
    witness_sent = np.isin(att.edges[keep, 0], att.sybil_ids.reshape(-1))
    covered = ~witness_sent if prior else np.ones(len(delta), dtype=bool)
    assert np.all(delta[covered] <= bound[keep][covered] + TOL)
    if prior and ring and mb and policy != "arrival":         # under M-B the ring's own evidence weighs 0
        assert delta[witness_sent].max() > 1e-6               # V2 self-surprise is live: the ring loses more
    A = RP.release_policy(policy, **kw).A
    for rec in victims:                       # never looser than the reviewer's 1 - exp(-gamma_max L_h sum min)
        assert bound[rec] <= reviewer_form(att, attacked["witness_kappa"], rec, A) + 1e-15
    # Only records sent by a bad-mouthed account are exposed: the warnings and, once prior raters make a warner
    # credible, any attack edge it sends for another victim. Fresh Tier-0 Sybils weigh 0, and under arrival nothing
    # is revocable, so then nothing is exposed.
    weighted = attacked["witness_kappa"][att.badmouth_edge_ids.reshape(-1)].any() and policy != "arrival"
    exposed = bound[keep] > 0.0
    subjects = np.unique(att.edges[att.badmouth_edge_ids.reshape(-1), 1])
    assert not (exposed & ~np.isin(att.edges[keep, 0], subjects)).any()
    assert exposed[victims].all() if weighted else not exposed.any()
    if weighted:
        assert delta[victims].max() > 1e-6                    # the attack does revoke, so the check bites


def out_of_scope_case(case):
    """(rows, number of comparator rows, n_nodes, witness levels, id of the record shown, accounts whose records lie
    outside the bound). W = 0 warns T = 1; five Sybils bad-mouth W in the last rows, which the comparator drops.
    - "subject rates later" (the reviewer's probe): W warns T at 7 and X = 2 at 12, and X rates Y = 3 at 14. Every
      account is in Tier 0, so the Sybils (5..9, at 8) weigh 0.
    - "receiver rates later": W sends nothing more; the Sybils (10..14, at 8) are in Tier 3, weight 0.20 each. T
      endorses Z = 2 at 12, Tier-3 accounts 3 and 4 warn Z at 12, and Z rates Y = 5 at 14.
    In the other cases the Sybils are 11..15, in Tier 0, and bad-mouth W at 7; W warns T at 6.
    - "witness self-surprise" (the re-review's probe): accounts 7..10 rate W at 0, so W holds a positive consensus
      of mass >= 2 when bad-mouthed; the Sybils form a ring at 6.
    - "witness rated later": the same, plus honest warnings of Sybil 11 by 2 and 3 at 6, account 4's endorsement
      of 11 at 12, and 4's rating of 5 at 14.
    - "witness's receiver rates later": accounts 7..10 rate W at 0; the Sybils rate Y = 2 at 6, before the
      bad-mouthing; six accounts (16..21) warn Z = 3 at 9; Y endorses Z at 10; Z rates U = 5 at 8.
    - "W rated later" (clause 2): W has no prior raters; X = 2 rates Y1 = 3 at 10 and Y2 = 4 at 11, then rates W
      at 12."""
    sybils, t_b = range(11, 16), 7
    if case == "subject rates later":
        base = [(0, 1, -1, 7), (0, 2, -1, 12), (2, 3, -1, 14)]
        sybils, n, t_b, shown, exempt = range(5, 10), 10, 8, 2, {2}
    elif case == "receiver rates later":
        base = [(0, 1, -1, 7), (1, 2, 1, 12), (3, 2, -1, 12), (4, 2, -1, 12), (2, 5, 1, 14)]
        sybils, n, t_b, shown, exempt = range(10, 15), 15, 8, 4, {2}
    elif case in ("witness self-surprise", "witness rated later"):
        n, base = 16, [(r, 0, 1, 0) for r in (7, 8, 9, 10)] + [(0, 1, -1, 6)]
        if case == "witness rated later":
            base += [(2, 11, -1, 6), (3, 11, -1, 6)]
        base += [(s, 11 + (i + 1) % 5, 1, 6) for i, s in enumerate(sybils)]
        if case == "witness rated later":
            base += [(4, 11, 1, 12), (4, 5, 1, 14)]
            shown, exempt = len(base) - 1, {*sybils, 4}
        else:
            shown, exempt = len(base) - 5, set(sybils)         # the ring's first record, 11 -> 12
    elif case == "witness's receiver rates later":
        n, base = 22, [(r, 0, 1, 0) for r in (7, 8, 9, 10)] + [(0, 1, -1, 6)] + [(s, 2, 1, 6) for s in sybils]
        base += [(q, 3, -1, 9) for q in range(16, 22)] + [(2, 3, 1, 10), (3, 5, 1, 8)]
        shown, exempt = len(base) - 1, {*sybils, 3}
    else:
        n, base, shown, exempt = 16, [(0, 1, -1, 6), (2, 3, 1, 10), (2, 4, -1, 11), (2, 0, 1, 12)], 1, {2}
    if case == "subject rates later":
        levels = np.zeros((21, n), dtype=np.int64)               # every account in Tier 0
    elif case == "receiver rates later":
        levels = np.zeros((21, n), dtype=np.int64)
        levels[:, [3, 4, *sybils]] = 3
    else:
        levels = np.full((21, n), 2, dtype=np.int64)
        levels[:, list(sybils)] = 0
    rows = np.asarray(base + [(s, 0, -1, t_b) for s in sybils], dtype=np.int64)
    return rows, len(base), n, levels, shown, exempt


@pytest.mark.parametrize("case, excess", [("subject rates later", 2.129e-3), ("receiver rates later", 1.504e-4),
                                          ("witness self-surprise", 4.009e-2), ("witness rated later", 4.180e-4),
                                          ("witness's receiver rates later", 6.295e-4), ("W rated later", 2.916e-2)])
def test_indirect_paths_lie_outside_the_property1_bound(case, excess):
    """Documented out of scope (property1's docstring; E1 reports the bound as a descriptive diagnostic, ruling
    R58). The shown record's bound is 0, yet the attack raises its revoked share.
    - The Sybils' released mass raises W's kappa from 0 (its confidence factor counts negative mass too), so W's
      later warning registers evidence, and X's rating of Y loses more; W's warning itself stays within its bound.
    - W's warning about T is revoked more, so T's tallies and kappa shrink, T's later endorsement weighs less, and
      Z's rating of Y loses more.
    - Each Sybil's rating contradicts W's consensus, so V2 self-surprise raises the Sybil's hazard and its ring
      record loses more: a witness-sent record.
    - That ring record feeds Sybil 11's consensus, which account 4's later rating of 11 reads through V2, so 4's
      rating of 5 loses more.
    - The witnesses' earlier ratings of Y release less under their raised hazard, so Y's tallies and kappa shrink,
      Y's later endorsement of Z weighs less, and Z's rating of U loses more.
    - Clause 2: X's later rating of W reads, through V2, W's consensus, which the Sybils' negative mass has turned,
      so X's ratings lose more.
    Every other record stays within the bound. The M-B rule and the comparator's levels are the attacked run's."""
    rows, n_base, n, levels, shown, exempt = out_of_scope_case(case)
    attacked = engine(rows, n, witness_cred=levels)["records"]
    comparator = engine(rows[:n_base], n, witness_cred=levels)["records"]
    bound = P1.property1_bound(rows[:, :2], rows[:, 2], rows[:, 3], np.arange(n_base, len(rows)),
                               attacked["witness_kappa"], K_snap=21)[:n_base]
    delta = attacked["revoked"][:n_base] - comparator["revoked"]
    assert bound[shown] == 0.0 and delta[shown] == pytest.approx(excess, rel=1e-3)
    covered = ~np.isin(rows[:n_base, 0], list(exempt))
    assert np.all(delta[covered] <= bound[covered] + TOL)


# ------------------------------------------------------------------ the scope, randomized (fix round 3)
SEARCH_POLICIES = [("erlang", {}), ("erlang", dict(N=1, phi=0.25)), ("erlang", dict(N=5, phi=1.25)), ("arrival", {}),
                   ("quarantine", {}), ("no_commit", {})]
VARIED = dict(L_f=(1, 1, 2, 3), H=(3, 6, 6, 10), cap_c=(0.5, 1.0, 1.0), c_self=(0.25, 0.5, 1.0),
              supersession_impulse=(0.1, 0.35, 0.7), m_0=(1.0, 2.0, 4.0))


def clause_stream(rng, n_h=14, K=21):
    """A random stream that meets every clause of property1_bound's scope. W = 0 warns T = 1 at t_k, and k Sybils
    bad-mouth W at t_b. Up to and including t_b anything goes among honest accounts, ratings by and of W included;
    in 60% of the streams 3 to 6 accounts rate W positively at 0 to 2 (V2 live). Honest accounts rate Sybils, the
    Sybils form a ring and rate one another and honest accounts. After t_b, only accounts that are not W, not a
    witness and received no rating from W or a witness send or receive ratings. Returns (rows, n_nodes, ids of the
    bad-mouthing, Sybils)."""
    def pick(values):
        return values[int(rng.integers(len(values)))]

    def sign(p_neg):
        return -1 if rng.random() < p_neg else 1

    k, lag, t_k = int(rng.integers(2, 6)), int(rng.integers(0, 3)), int(rng.integers(6, 12))
    t_b = min(t_k + lag, K - 3)
    sybils = list(range(n_h, n_h + k))
    rows = [(*rng.choice(n_h, 2, replace=False), sign(0.3), rng.integers(0, t_b + 1))
            for _ in range(int(rng.integers(30, 70)))]
    if rng.random() < 0.6:
        rows += [(r, 0, 1, rng.integers(0, 3)) for r in rng.choice(np.arange(2, n_h), int(rng.integers(3, 7)),
                                                                   replace=False)]
    rows.append((0, 1, -1, t_k))
    rows += [(rng.integers(0, n_h), pick(sybils), sign(0.2), rng.integers(5, t_b + 1))
             for _ in range(int(rng.integers(0, 6)))]
    rows += [(s, sybils[(i + 1) % k], 1, max(t_b - 1, 5)) for i, s in enumerate(sybils)]
    rows += [(*rng.choice(sybils, 2, replace=False), sign(0.3), rng.integers(5, t_b + 1))
             for _ in range(int(rng.integers(0, 4)))]
    rows += [(pick(sybils), rng.integers(2, n_h), sign(0.25), rng.integers(3, t_b + 1))
             for _ in range(int(rng.integers(0, 8)))]
    first_bad = len(rows)
    rows += [(s, 0, -1, t_b) for s in sybils]
    pre = np.asarray(rows, dtype=np.int64)
    touched = {0} | set(pre[np.isin(pre[:, 0], [0, *sybils]), 1].tolist())
    free = [a for a in range(n_h) if a not in touched]
    if len(free) >= 2:
        rows += [(*rng.choice(free, 2, replace=False), sign(0.35), rng.integers(t_b + 1, K))
                 for _ in range(int(rng.integers(25, 70)))]
    return np.asarray(rows, dtype=np.int64), n_h + k, np.arange(first_bad, first_bad + k), sybils


def test_property1_bound_holds_on_random_streams_that_meet_every_clause():
    """A randomized guard on the stated scope. 300 seeded streams meet every clause; they vary the honest history,
    the prior raters of W, ratings in the bad-mouthing snapshot, witness -> honest ratings, the release policy, M-B
    and the Sybils' tiers, and the engine constants L_f, H, cap_c, c_self, supersession_impulse, m_0 and
    closed_only_reads. No record that a witness did not send exceeds its bound; a failure prints the stream. The
    check bites: the bound is positive and attained in part in many records, and witnesses' own records move."""
    bites = witness_moves = 0
    for seed in range(300):
        rng = np.random.default_rng((20261027, seed))
        rows, n, bad, sybils = clause_stream(rng)
        policy, kw = SEARCH_POLICIES[int(rng.integers(len(SEARCH_POLICIES)))]
        kw = dict(kw, closed_only_reads=bool(rng.random() < 0.25),
                  **{name: values[int(rng.integers(len(values)))] for name, values in VARIED.items()})
        levels = None
        if rng.random() < 0.6:
            levels = rng.integers(0, 4, size=(21, n))
            levels[:, sybils] = rng.integers(0, 4, size=(21, 1)) * (rng.random() < 0.5)
        keep = np.ones(len(rows), dtype=bool)
        keep[bad] = False
        attacked = engine(rows, n, witness_cred=levels, policy=policy, **kw)["records"]
        comparator = engine(rows[keep], n, witness_cred=levels, policy=policy, **kw)["records"]
        bound = P1.property1_bound(rows[:, :2], rows[:, 2], rows[:, 3], bad, attacked["witness_kappa"], K_snap=21,
                                   policy=policy, **kw)[keep]
        delta = attacked["revoked"][keep] - comparator["revoked"]
        witness_sent = np.isin(rows[keep, 0], sybils)
        over = (delta > bound + TOL) & ~witness_sent
        assert not over.any(), (seed, policy, kw, rows.tolist(), np.flatnonzero(keep)[over].tolist())
        bites += int(((delta > 1e-6) & (bound > 1e-9) & ~witness_sent).sum())
        witness_moves += int((np.abs(delta[witness_sent]) > 1e-6).sum())
    assert bites > 100 and witness_moves > 100
