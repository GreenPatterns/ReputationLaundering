"""Task 3.4, M-D (approved 2026-10-26): the calibrated cold start, keeper.analysis.cold_start (experiment E8).

    r~(v) = (n_v r(v) + m r_prior) / (n_v + m),   p~ = (n_v p + m r_prior) / (n_v + m) for test edges into v,

with r_prior the 10th percentile of the active accounts' proxies. Every expected value below is computed by hand
in its comment. Percentiles follow keeper.laundering.metrics.platform_percentile (the share strictly below), and
the test MCC follows keeper.laundering.core.test_metrics (p < 0.5 predicts distrust). Each target is ranked on its
own post-U1 platform when its population is given (probe4's definition), else against the shared population. The
low-degree MCC covers every held-out edge into an account with at most two incoming training ratings, accounts
unseen in training included. CPU only, no model.
"""
import math

import numpy as np
import pytest

from keeper.analysis import cold_start as CS
from keeper.laundering import core as C
from keeper.laundering import metrics as M
from keeper.laundering import signed as S


# ------------------------------------------------------------------ the formula and its inputs
def test_calibrate_by_hand():
    assert float(CS.calibrate(0.9, 1, 1, 0.3)) == pytest.approx(0.6)         # (0.9 + 0.3) / 2
    assert float(CS.calibrate(0.9, 0, 4, 0.3)) == pytest.approx(0.3)         # no rating: the prior
    assert float(CS.calibrate(0.9, 3, 1, 0.3)) == pytest.approx(0.75)        # (2.7 + 0.3) / 4
    got = CS.calibrate([0.9, 0.9, 0.9], [0, 1, 3], 2, 0.3)                   # 0.6/2, (0.9+0.6)/3, (2.7+0.6)/5
    assert got == pytest.approx([0.3, 0.5, 0.66])


def test_prior_is_the_tenth_percentile_of_the_active_proxies():
    assert CS.MS == (1, 2, 4) and CS.PRIOR_PERCENTILE == 10.0
    assert CS.prior(np.linspace(0.0, 1.0, 11)) == pytest.approx(0.1)        # 0.0, 0.1, ..., 1.0
    assert CS.prior([0.0, 1.0]) == pytest.approx(0.1)                        # linear interpolation
    assert CS.prior([0.4] * 5) == pytest.approx(0.4)


def test_incoming_counts_by_hand():
    assert CS.incoming_counts([1, 1, 3], 5).tolist() == [0, 2, 0, 1, 0]
    assert CS.incoming_counts([], 3).tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        CS.incoming_counts([1, 5], 5)


def test_platform_percentiles_are_the_scalar_convention_bit_for_bit():
    rng = np.random.default_rng(0)
    population = np.round(rng.uniform(0, 1, 500), 2)                          # many ties
    scores = np.concatenate([population[:50], rng.uniform(-0.1, 1.1, 50), [population.min(), population.max()]])
    want = [M.platform_percentile(s, population) for s in scores]
    assert CS.platform_percentiles(scores, population).tolist() == want


# ------------------------------------------------------------------ laundering percentile
# Active accounts 2, 5, 6, 8, 9 with proxies 0.2, 0.4, 0.6, 0.8, 0.5 and 5, 5, 5, 5, 3 incoming ratings; r_prior = 0.3.
# At m = 1 the others calibrate to 1.3/6 = 0.2167, 2.3/6 = 0.3833, 3.3/6 = 0.55, 4.3/6 = 0.7167; account 9 to
# 1.8/4 = 0.45 before its deletion.
ACTIVE = np.array([2, 5, 6, 8, 9])
PROXY = np.array([0.2, 0.4, 0.6, 0.8, 0.5])
N_IN = np.array([0, 0, 5, 0, 0, 5, 5, 0, 5, 3])                            # per account id 0..9


def test_laundering_percentile_by_hand():
    # Target 9 after U1: proxy 0.95 with no rating left, so r~ = 0.3 and only 0.2167 lies below it: 1/5.
    # Target 2 after a deletion: proxy 0.9 on 2 ratings, r~ = (1.8 + 0.3)/3 = 0.7; below it 0.3833, 0.55, 0.45: 3/5.
    got = CS.laundering_percentiles(ACTIVE, PROXY, N_IN, [9, 2], [0.95, 0.9], [0, 2], m=1, r_prior=0.3)
    assert got.tolist() == pytest.approx([0.2, 0.6])
    # With one rating left, target 9 has r~ = (0.95 + 0.3)/2 = 0.625; below it 0.2167, 0.3833, 0.55: 3/5.
    one = CS.laundering_percentiles(ACTIVE, PROXY, N_IN, [9], [0.95], [1], m=1, r_prior=0.3)
    assert one.tolist() == pytest.approx([0.6])
    # Uncalibrated (probe4's pct_after): 0.95 among 0.2, 0.4, 0.6, 0.8, 0.95 is 4/5; 0.9 among 0.9, 0.4, 0.6, 0.8,
    # 0.5 is 4/5.
    raw = CS.percentiles_after_deletion(ACTIVE, PROXY, [9, 2], [0.95, 0.9])
    assert raw.tolist() == pytest.approx([0.8, 0.8])


# Each target's post-U1 platform (probe4 re-scores every active account after the target's U1). Row 0 belongs to
# target 9: rater 2's proxy moved from 0.2 to 0.35 and account 8's from 0.8 to 0.97. Row 1 belongs to target 2:
# account 8's proxy moved from 0.8 to 0.5.
POST_U1 = np.array([[0.35, 0.4, 0.6, 0.97, 0.95],
                    [0.9, 0.4, 0.6, 0.5, 0.5]])


def test_laundering_percentile_on_each_targets_post_u1_platform_by_hand():
    # Target 9 (r~ = 0.3): rater 2 now calibrates to (1.75 + 0.3)/6 = 0.3417, past the target, so no account lies
    # below it: 0/5, where the shared population gives 1/5. Target 2 (r~ = 0.7): account 8 now calibrates to
    # (2.5 + 0.3)/6 = 0.4667, below the target; with 0.3833, 0.55 and 0.45 that makes 4/5, where the shared gives 3/5.
    kw = dict(m=1, r_prior=0.3)
    got = CS.laundering_percentiles(ACTIVE, PROXY, N_IN, [9, 2], [0.95, 0.9], [0, 2], target_population=POST_U1, **kw)
    assert got.tolist() == pytest.approx([0.0, 0.8])
    shared = CS.laundering_percentiles(ACTIVE, PROXY, N_IN, [9, 2], [0.95, 0.9], [0, 2], **kw)
    assert shared.tolist() == pytest.approx([0.2, 0.6])
    # Uncalibrated: 0.95 among 0.35, 0.4, 0.6, 0.97, 0.95 is 3/5 (account 8 moved past it); 0.9 among 0.9, 0.4, 0.6,
    # 0.5, 0.5 is 4/5. The shared population gives 4/5 and 4/5.
    raw = CS.percentiles_after_deletion(ACTIVE, PROXY, [9, 2], [0.95, 0.9], target_population=POST_U1)
    assert raw.tolist() == pytest.approx([0.6, 0.8])


def test_post_u1_populations_are_checked():
    kw = dict(m=1, r_prior=0.3)
    with pytest.raises(ValueError, match="target_population"):    # rows swapped: row 0 holds 0.5 for target 9
        CS.laundering_percentiles(ACTIVE, PROXY, N_IN, [9, 2], [0.95, 0.9], [0, 2], target_population=POST_U1[::-1],
                                  **kw)
    with pytest.raises(ValueError, match="shape"):
        CS.laundering_percentiles(ACTIVE, PROXY, N_IN, [9, 2], [0.95, 0.9], [0, 2],
                                  target_population=POST_U1[:, :4], **kw)
    with pytest.raises(ValueError, match="finite"):
        CS.percentiles_after_deletion(ACTIVE, PROXY, [9], [0.95], target_population=[[0.35, 0.4, 0.6, np.nan, 0.95]])


def test_laundering_inputs_are_checked():
    with pytest.raises(ValueError, match="not an active account"):
        CS.laundering_percentiles(ACTIVE, PROXY, N_IN, [3], [0.9], [0], m=1, r_prior=0.3)
    with pytest.raises(ValueError, match="unique"):
        CS.percentiles_after_deletion([2, 2], [0.1, 0.2], [2], [0.5])
    with pytest.raises(ValueError):
        CS.laundering_percentiles(ACTIVE, PROXY[:4], N_IN, [9], [0.9], [0], m=1, r_prior=0.3)
    with pytest.raises(ValueError):
        CS.laundering_percentiles(ACTIVE, PROXY, N_IN, [9, 2], [0.9], [0, 1], m=1, r_prior=0.3)


# ------------------------------------------------------------------ newcomer rank cost
def test_newcomer_rank_cost_by_hand():
    # Proxies 0.9, 0.8, 0.5, 0.3, 0.1 on 0, 1, 10, 10, 10 incoming ratings; r_prior = 0.25, m = 1:
    # r~ = 0.25, 1.05/2 = 0.525, 5.25/11 = 0.4773, 3.25/11 = 0.2955, 1.25/11 = 0.1136.
    # Newcomer 0 falls from 4/5 (under r) to 1/5 (under r~), newcomer 1 rises from 3/5 to 4/5: mean change -0.2.
    ids, proxy, n_in = np.arange(5), [0.9, 0.8, 0.5, 0.3, 0.1], np.array([0, 1, 10, 10, 10])
    got = CS.newcomer_rank_cost(ids, proxy, n_in, m=1, r_prior=0.25)
    assert got["n"] == 2
    assert got["mean_change"] == pytest.approx(-0.2)
    assert got["mean_pct"] == pytest.approx(0.7) and got["mean_pct_calibrated"] == pytest.approx(0.5)
    only_0 = CS.newcomer_rank_cost(ids, proxy, n_in, m=1, r_prior=0.25, max_in=0)
    assert only_0["n"] == 1 and only_0["mean_change"] == pytest.approx(-0.6)
    none = CS.newcomer_rank_cost(ids, proxy, np.full(5, 10), m=1, r_prior=0.25)
    assert none["n"] == 0 and none["mean_change"] is None                    # undefined, never 0


# ------------------------------------------------------------------ low-degree test MCC
# Accounts 0, 1, 2, 3 hold 0, 1, 2, 5 incoming training ratings; r_prior = 0.3, m = 1.
LOW_N_IN = np.array([0, 1, 2, 5])
DST = np.array([0, 1, 2, 2, 3])
Y = np.array([1, 0, 1, 0, 1])                                              # 1 = distrust
P = np.array([0.7, 0.9, 0.4, 0.8, 0.9])                                    # the model's P(trust)


def test_low_degree_mcc_by_hand():
    # The edge into account 3 (5 ratings) is left out. p~ = 0.3, (0.9+0.3)/2 = 0.6, (0.8+0.3)/3 = 0.367,
    # (1.6+0.3)/3 = 0.633. Under p < 0.5, p predicts 0, 0, 1, 0 against y = 1, 0, 1, 0: TP = 1, FN = 1, TN = 2,
    # FP = 0, so MCC = 2/sqrt(1*2*2*3) = 0.5774. p~ predicts 1, 0, 1, 0: MCC 1.
    got = CS.low_degree_mcc(DST, Y, P, LOW_N_IN, m=1, r_prior=0.3)
    assert got["n"] == 4
    assert got["base"]["mcc"] == pytest.approx(2 / math.sqrt(12))
    assert got["calibrated"]["mcc"] == pytest.approx(1.0)
    assert got["delta_mcc"] == pytest.approx(1.0 - 2 / math.sqrt(12))
    # The base metric is keeper.laundering.core.test_metrics' rule on the same edges.
    assert got["base"] == C._cls_metrics(Y[:4], P[:4])
    # An edge into a well-rated account does not count.
    assert CS.low_degree_mcc(DST, Y, np.r_[P[:4], 0.1], LOW_N_IN, m=1, r_prior=0.3) == got


def test_low_degree_mcc_is_none_when_one_class_remains():
    got = CS.low_degree_mcc(DST, np.r_[0, 0, 0, 0, 1], P, LOW_N_IN, m=1, r_prior=0.3)
    assert got["n"] == 4 and got["base"]["mcc"] is None and got["delta_mcc"] is None


def toy_stream():
    """Accounts 0-7; training snapshots 0 and 1, held-out snapshot 2 (rows 6-11). Incoming training ratings:
    account 1 holds 3, account 2 holds 2, account 5 holds 1, the others none; 6 and 7 never appear in training."""
    rows = np.array([(0, 1, +1, 0), (2, 1, +1, 0), (3, 1, -1, 0), (0, 2, +1, 0), (4, 5, +1, 1), (5, 2, -1, 1),
                     (0, 2, -1, 2), (4, 1, +1, 2), (4, 3, -1, 2), (1, 6, +1, 2), (6, 5, +1, 2), (7, 3, -1, 2)])
    return C.Graph(rows[:, :2], S.one_hot_from_sign(rows[:, 2]), np.arange(len(rows)), rows[:, 3], 8)


def test_low_degree_set_is_every_held_out_edge_into_an_account_with_at_most_two_ratings():
    G = toy_stream()
    rows, dst, y = CS.held_out_snapshot(G, 2)
    held, t1, _ = C.task_masks(G, 2)
    assert rows.tolist() == held.tolist() == [6, 7, 8, 9, 10, 11]
    assert dst.tolist() == [2, 1, 3, 6, 5, 3] and y.tolist() == [1, 0, 1, 0, 0, 1]
    n_in = CS.incoming_counts(C.train_split(G, 2).edges[:, 1], G.n)
    assert n_in.tolist() == [0, 3, 2, 0, 0, 1, 0, 0]
    # The set is rows 6, 8, 9, 10 and 11; row 7 rates account 1, which holds 3. Rows 9, 10 and 11 have an endpoint
    # unseen in training, so Task 1 leaves them out, but they count here. On y = 1, 1, 0, 0, 1, p predicts 1, 0, 0, 0,
    # 0: TP 1, FN 2, TN 2, FP 0, MCC = 2/sqrt(1*3*2*4). p~ = 0.3, 0.3, 0.3, (0.95 + 0.3)/2 = 0.625, 0.3 predicts
    # 1, 1, 1, 0, 1: TP 3, FP 1, TN 1, FN 0, MCC = 3/sqrt(4*3*2*1).
    p = np.array([0.3, 0.9, 0.6, 0.8, 0.95, 0.7])
    got = CS.low_degree_mcc(dst, y, p, n_in, m=1, r_prior=0.3)
    assert got["n"] == 5 and int((~t1 & (n_in[dst] <= 2)).sum()) == 3
    assert got["base"]["mcc"] == pytest.approx(2 / math.sqrt(24))
    assert got["calibrated"]["mcc"] == pytest.approx(3 / math.sqrt(24))
    assert got["delta_mcc"] == pytest.approx(1 / math.sqrt(24))


# ------------------------------------------------------------------ the E8 report and validation
def test_evaluate_runs_every_metric_for_each_m():
    active_ids, proxy = np.arange(4), np.array([0.3, 0.6, 0.9, 0.5])
    report = CS.evaluate(active_ids, proxy, LOW_N_IN, target_ids=[0], target_proxy=[0.95], target_n=[0],
                         test_dst=DST, test_y=Y, test_p=P)
    r_prior = CS.prior(proxy)                                                 # 10th percentile: 0.36
    assert report["r_prior"] == pytest.approx(0.36) and report["r_prior"] == r_prior
    assert report["uncalibrated"]["laundering_pct"] == pytest.approx([0.75])  # 0.95 above 0.6, 0.9, 0.5
    assert sorted(report["by_m"]) == ["1", "2", "4"]
    for m in CS.MS:
        got = report["by_m"][str(m)]
        assert got["laundering_pct"] == CS.laundering_percentiles(active_ids, proxy, LOW_N_IN, [0], [0.95], [0],
                                                                  m, r_prior).tolist()
        assert got["newcomer"] == CS.newcomer_rank_cost(active_ids, proxy, LOW_N_IN, m, r_prior)
        assert got["low_degree"] == CS.low_degree_mcc(DST, Y, P, LOW_N_IN, m, r_prior)


def test_evaluate_ranks_on_post_u1_populations_and_keeps_the_base_elsewhere():
    active_ids, proxy = np.arange(4), np.array([0.3, 0.6, 0.9, 0.5])
    rows = np.array([[0.95, 0.99, 0.9, 0.5]])              # target 0's platform after U1: account 1 moved to 0.99
    kw = dict(target_ids=[0], target_proxy=[0.95], target_n=[1], test_dst=DST, test_y=Y, test_p=P)
    report = CS.evaluate(active_ids, proxy, LOW_N_IN, target_population=rows, **kw)
    shared = CS.evaluate(active_ids, proxy, LOW_N_IN, **kw)
    assert report["laundering_population"] == "per-target post-U1" and shared["laundering_population"] == "shared"
    assert report["r_prior"] == shared["r_prior"] == CS.prior(proxy)            # 0.36, from the base proxies
    assert report["uncalibrated"]["laundering_pct"] == pytest.approx([0.5])      # 0.95 above 0.9 and 0.5 only
    # m = 1: the target's r~ = (0.95 + 0.36)/2 = 0.655. In the shared population accounts 1 and 3 lie below it
    # ((0.6 + 0.36)/2 = 0.48, (2.5 + 0.36)/6 = 0.477): 2/4. On its post-U1 platform account 1 calibrates to
    # (0.99 + 0.36)/2 = 0.675, above it: 1/4.
    assert report["by_m"]["1"]["laundering_pct"] == pytest.approx([0.25])
    assert shared["by_m"]["1"]["laundering_pct"] == pytest.approx([0.5])
    for m in CS.MS:
        got, ref = report["by_m"][str(m)], shared["by_m"][str(m)]
        assert got["laundering_pct"] == CS.laundering_percentiles(active_ids, proxy, LOW_N_IN, [0], [0.95], [1], m,
                                                                  report["r_prior"], target_population=rows).tolist()
        assert got["laundering_pct"] != ref["laundering_pct"]                    # the two definitions differ here
        assert got["newcomer"] == ref["newcomer"] and got["low_degree"] == ref["low_degree"]


@pytest.mark.parametrize("kw", [dict(m=0), dict(m=-1), dict(n=[-1]), dict(score=[float("nan")]),
                                dict(r_prior=float("inf"))])
def test_calibration_inputs_are_checked(kw):
    args = dict(score=[0.5], n=[1], m=1, r_prior=0.3) | kw
    with pytest.raises(ValueError):
        CS.calibrate(**args)
