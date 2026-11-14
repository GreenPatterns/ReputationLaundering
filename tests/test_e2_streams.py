"""E2 streams (Task 4.5; keeper.scem.e2_streams): the draws and the silent-sender stream. CPU only.

The draws replay the registered order from default_rng(5000 + seed); witnesses are eligible at t_s + lag and never the
target; silent senders rate their target once and are rated by their two witnesses, nothing else; one stream holds both
kinds of target, each with its direction's sign; the original rows are untouched.
"""
import numpy as np
import pytest

from keeper.laundering import core as C
from keeper.laundering import signed as S
from keeper.scem import e2_streams as E

N_NODES, T_TRAIN = 120, 21
N_SENDERS = 2 * E.N_TARGETS * E.SENDERS_PER_TARGET


def synthetic_graph(n=N_NODES, T=T_TRAIN + 1, seed=0) -> C.Graph:
    """A graph over snapshots 0..T-1: positive background among accounts 40..n-1 in every snapshot, a distrust-majority
    cohort 0..19 (two distrust in-edges, at most one trust in-edge) and a trusted cohort 20..39 (three trust in-edges)."""
    rng = np.random.default_rng(seed)
    rows = []
    background = np.arange(40, n)
    for t in range(T):
        rows += [(*rng.choice(background, 2, replace=False), 1, t) for _ in range(60)]
    for v in range(20):
        rows += [(u, v, -1, rng.integers(T - 1)) for u in rng.choice(background, 2, replace=False)]
        rows += [(rng.choice(background), v, 1, rng.integers(T - 1))] if v % 2 else []
    for v in range(20, 40):
        rows += [(u, v, 1, rng.integers(T - 1)) for u in rng.choice(background, 3, replace=False)]
    rows.sort(key=lambda r: r[3])
    a = np.array(rows, dtype=np.int64)
    return C.Graph(a[:, :2], S.one_hot_from_sign(a[:, 2]), 1000 * a[:, 3] + np.arange(len(a)) % 7, a[:, 3], n)


def synthetic_levels(n=N_NODES, T=T_TRAIN) -> np.ndarray:
    """[T, n] tiers: accounts 0..39 (every possible target) are eligible in every row, the others in rows that depend
    on the account, so a witness's row matters."""
    levels = np.zeros((T, n), dtype=np.int64)
    levels[:, :40] = 3
    others = np.arange(40, n)
    for t in range(T):
        levels[t, others[(others * 7 + t) % 5 == 0]] = 2
    return levels


@pytest.fixture(scope="module")
def Tr():
    return C.train_split(synthetic_graph(), T_TRAIN)


@pytest.fixture(scope="module")
def plan(Tr):
    return E.draw(Tr, synthetic_levels(), seed=7)


def test_draw_replays_the_registered_order(Tr, plan):
    levels = synthetic_levels()
    rng = np.random.default_rng(5000 + 7)
    cohort, neg, pos = C.pick_targets(Tr, max_neg=3, strict=True)
    assert np.array_equal(plan.whitewash, rng.choice(cohort, 15, replace=False))
    assert np.array_equal(plan.badmouth, rng.choice(E.trusted_cohort(neg, pos), 15, replace=False))
    assert np.array_equal(plan.t_s, rng.integers(5, 15, size=30))
    for lag in (1, 3):                                           # lag 1's witnesses, then lag 3's
        for i, (v, t) in enumerate(zip(plan.targets, plan.t_s)):
            pool = np.flatnonzero(levels[t + lag] >= 2)
            for j in range(3):
                assert np.array_equal(plan.witnesses[lag][i, j], rng.choice(pool[pool != v], 2, replace=False))


def test_draw_is_deterministic_given_the_seed(Tr, plan):
    again = E.draw(Tr, synthetic_levels(), seed=7)
    assert again.sha256() == plan.sha256() and again.to_dict() == plan.to_dict()
    assert E.draw(Tr, synthetic_levels(), seed=8).sha256() != plan.sha256()


def test_targets_come_from_their_cohorts(Tr, plan):
    cohort, neg, pos = C.pick_targets(Tr, max_neg=3, strict=True)
    assert set(plan.whitewash) <= set(cohort) and len(set(plan.whitewash)) == 15
    assert len(set(plan.badmouth)) == 15
    assert all(len(pos.get(v, [])) >= 3 and not neg.get(v) for v in plan.badmouth)
    assert plan.t_s.min() >= 5 and plan.t_s.max() <= 14 and len(plan.t_s) == 30


def test_witnesses_are_eligible_at_t_s_plus_lag_and_never_the_target(plan):
    levels = synthetic_levels()
    for lag in (1, 3):
        w = plan.witnesses[lag]
        assert w.shape == (30, 3, 2)
        rows = (plan.t_s + lag)[:, None, None]
        assert (levels[rows, w] >= 2).all()                       # eligible at t_s + lag
        assert (w != plan.targets[:, None, None]).all()           # every target is eligible: the exclusion bites
        assert (w[..., 0] != w[..., 1]).all()                     # without replacement
    assert not np.array_equal(plan.witnesses[1], plan.witnesses[3])


@pytest.mark.parametrize("lag", (1, 3))
def test_silent_senders_do_nothing_else(Tr, plan, lag):
    G, meta = E.attack_stream(Tr, plan, lag)
    senders = np.asarray(meta["senders"])
    assert senders.shape == (30, 3) and senders.min() == Tr.n and G.n == Tr.n + N_SENDERS
    sign = 1 - 2 * S.class_ids(G.labels)
    for i, v in enumerate(plan.targets):
        for j, s in enumerate(senders[i]):
            out = np.flatnonzero(G.edges[:, 0] == s)
            assert len(out) == 1 and G.edges[out[0], 1] == v                         # one rating, of its target,
            assert sign[out[0]] == plan.direction[i] and G.snap[out[0]] == plan.t_s[i]  # in the attack's sign, at t_s
            inc = np.flatnonzero(G.edges[:, 1] == s)
            assert sorted(G.edges[inc, 0]) == sorted(plan.witnesses[lag][i, j])       # rated by its two witnesses,
            assert (sign[inc] == -1).all() and (G.snap[inc] == plan.t_s[i] + lag).all()  # negatively, at t_s + lag
    assert meta["n_injected"] == N_SENDERS * 3 == int((G.rid < 0).sum())


def test_one_stream_holds_both_kinds_with_the_direction_sign(Tr, plan):
    G, _ = E.attack_stream(Tr, plan, 1)
    assert plan.kinds == ["whitewash"] * 15 + ["badmouth"] * 15
    assert plan.direction.tolist() == [1] * 15 + [-1] * 15
    injected = G.rid < 0
    sign = 1 - 2 * S.class_ids(G.labels)
    to_ww = injected & np.isin(G.edges[:, 1], plan.whitewash)
    to_bm = injected & np.isin(G.edges[:, 1], plan.badmouth)
    assert to_ww.sum() == 45 and (sign[to_ww] == 1).all()
    assert to_bm.sum() == 45 and (sign[to_bm] == -1).all()


def test_the_original_rows_are_untouched_and_the_stream_is_sorted(Tr, plan):
    G, _ = E.attack_stream(Tr, plan, 3)
    keep = G.rid >= 0
    for name in ("edges", "labels", "time", "snap", "rid"):
        assert np.array_equal(getattr(G, name)[keep], getattr(Tr, name))
    assert (np.diff(G.snap) >= 0).all()
    for k in np.flatnonzero(~keep):
        assert G.time[k] == Tr.time[Tr.snap == G.snap[k]].max()


def test_attack_stream_needs_a_registered_lag(Tr, plan):
    with pytest.raises(ValueError, match="lag must be one of"):
        E.attack_stream(Tr, plan, 2)


def test_plan_digest_covers_every_draw(plan):
    w = {lag: a.copy() for lag, a in plan.witnesses.items()}
    w[3][0, 0, 0] += 1
    assert E.Plan(plan.whitewash, plan.badmouth, plan.t_s, w).sha256() != plan.sha256()


def test_bitcoin_alpha_witnesses_keep_their_tier_in_the_attack_stream():
    """Real data: the clean stream's levels drive the draw; under the attack stream's levels (frozen clean calibration,
    the credible-negatives guard) every witness is still Tier >= 2 at t_s + lag and no honest tier moves."""
    D = C.load("bitcoinalpha")
    Tr_a = C.train_split(C.full_graph(D), D.tt)
    clean = E.stream_levels(Tr_a, Tr_a, 0.02, D.tt)
    p = E.draw(Tr_a, clean, seed=5)
    G, _ = E.attack_stream(Tr_a, p, 1)
    attacked = E.stream_levels(G, Tr_a, 0.02, D.tt)
    assert attacked.shape == (D.tt, D.n + N_SENDERS)
    assert np.array_equal(attacked[:, :D.n], clean)
    assert (attacked[(p.t_s + 1)[:, None, None], p.witnesses[1]] >= 2).all()
