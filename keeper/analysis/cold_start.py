"""Calibrated cold start (M-D, approved 2026-10-26): the Regime-1 mitigation baseline, not a KEEPER component.

Experiment E8 (protocol Section 2.10) shrinks each score toward a low prior, in proportion to the evidence behind it:

    r~(v) = (n_v r(v) + m r_prior) / (n_v + m),        m in {1, 2, 4},

where r(v) is account v's proxy from the model, n_v the number of its live incoming training ratings (after U1 a
target keeps n_after of them), and r_prior the 10th percentile of the active accounts' proxies. A test edge into v has
its probability p shrunk the same way: p~ = (n_v p + m r_prior) / (n_v + m). A target that U1 leaves without ratings
falls to r_prior whatever its post-deletion proxy. That is the mitigation, and genuine newcomers pay its cost.

Metrics, for one dataset and seed:
* laundering percentile: each target's percentile of r~ among all active accounts on its own post-U1 platform, with
  the convention of keeper.laundering.metrics.platform_percentile (the share of accounts strictly below), as probe4
  ranks it. U1 deletes the edges u -> v, which moves each rater u's proxy as well, so the exact definition needs every
  account's post-U1 proxy for each target (target_population); without it, one shared population approximates it;
* newcomer rank cost: the mean percentile change, r~ against r, of the active accounts with at most one incoming
  training rating, on the base proxies;
* low-degree test MCC: the test MCC on every held-out edge whose destination holds at most two incoming training
  ratings, zero included (held_out_snapshot: Task 1 and Task 2 rows alike, and accounts unseen in training), under
  the threshold rule of keeper.laundering.core.test_metrics (p < 0.5 predicts distrust), for p and for p~.

The inputs are arrays (account ids, proxies, counts, probabilities, labels), not models, so the analysis runs on the
CPU from the E9a dumps. An undefined quantity is None, as in keeper.laundering.metrics.
"""
import numpy as np

from keeper.laundering import core as C
from keeper.laundering import metrics as M
from keeper.laundering import signed as S

MS = (1, 2, 4)                  # the prior weights m
PRIOR_PERCENTILE = 10.0         # r_prior: this percentile of the active accounts' proxies
NEWCOMER_MAX_IN = 1             # newcomers: at most this many incoming training ratings
LOW_DEGREE_MAX_IN = 2           # the low-degree accounts of the test MCC
POPULATION_ATOL = 1e-6          # a post-U1 population's entry for its own target must match target_proxy this closely


def _finite(x, name):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite")
    return x


def _same_length(**arrays):
    lengths = {name: len(a) for name, a in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"lengths differ: {lengths}")


def _counts_of(n_in, ids):
    """n_in[ids], the ids checked against the range of the count array."""
    n_in = np.asarray(n_in)
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size and (ids.min() < 0 or ids.max() >= len(n_in)):
        raise ValueError(f"account ids must lie in [0, {len(n_in)}), the range of n_in")
    return n_in[ids]


def _positions(active_ids, target_ids):
    """Each target's position in active_ids, which must be unique and hold every target."""
    index = {int(a): i for i, a in enumerate(np.asarray(active_ids, dtype=np.int64).tolist())}
    if len(index) != len(active_ids):
        raise ValueError("active_ids must be unique")
    pos = []
    for v in np.asarray(target_ids, dtype=np.int64).tolist():
        if v not in index:
            raise ValueError(f"target {v} is not an active account")
        pos.append(index[v])
    return np.asarray(pos, dtype=np.int64)


def _post_u1_rows(target_population, pos, target_proxy, n_active):
    """target_population as a [targets, active accounts] array, checked: finite, that shape, and each row holding its
    target's post-deletion proxy at the target's own position, so that rows out of step with target_ids or active_ids
    fail here instead of ranking a target on another target's platform."""
    rows = _finite(target_population, "target_population")
    if rows.shape != (len(pos), n_active):
        raise ValueError(f"target_population must have shape (targets, active accounts) = {(len(pos), n_active)}, "
                         f"got {rows.shape}")
    own = rows[np.arange(len(pos)), pos]
    off = np.abs(own - target_proxy) > POPULATION_ATOL
    if off.any():
        k = int(np.argmax(off))
        raise ValueError(f"row {k} of target_population holds {own[k]!r} for its target, not the target's post-deletion "
                         f"proxy {target_proxy[k]!r}: rows must follow target_ids and columns active_ids")
    return rows


def _rank_after_deletion(rows, pos, target_score):
    """platform_percentile of each target's score within its row, the target's own entry set to that score."""
    out = np.empty(len(pos))
    for k, (j, s) in enumerate(zip(pos.tolist(), target_score)):
        population = np.array(rows[k], dtype=np.float64)          # a copy
        population[j] = s
        out[k] = M.platform_percentile(s, population)
    return out


def held_out_snapshot(G, tt):
    """Every edge of the held-out test snapshot tt, as (row ids, destinations, class ids): the rows that
    keeper.laundering.core.task_masks(G, tt) starts from, Task 1 and Task 2 rows and the rest alike. Unlike
    keeper.laundering.threshold.held_out, it keeps edges with an endpoint unseen in training. low_degree_mcc takes the
    model's P(trust) on these rows, in this order."""
    rows = np.where(np.asarray(G.snap) == tt)[0]
    return rows, np.asarray(G.edges)[rows, 1], S.class_ids(np.asarray(G.labels)[rows])


def incoming_counts(dst, n_nodes):
    """n_v for every account id 0..n_nodes-1: how many of the given ratings, by destination dst, v received.
    Pass the live training ratings, e.g. keeper.laundering.core.train_split(G, tt).edges[:, 1]."""
    dst = np.asarray(dst, dtype=np.int64)
    if dst.size and (dst.min() < 0 or dst.max() >= n_nodes):
        raise ValueError(f"destinations must lie in [0, {n_nodes})")
    return np.bincount(dst, minlength=n_nodes)


def prior(active_proxy, q=PRIOR_PERCENTILE):
    """r_prior: the q-th percentile of the active accounts' proxies (numpy's default linear interpolation)."""
    return float(np.percentile(_finite(active_proxy, "active_proxy"), q))


def calibrate(score, n, m, r_prior):
    """(n score + m r_prior) / (n + m), elementwise: r~ of account proxies, or p~ of test-edge probabilities."""
    score = _finite(score, "score")
    n = _finite(n, "n")
    if (n < 0).any():
        raise ValueError("the counts n must be non-negative")
    if not (np.isfinite(m) and m > 0):
        raise ValueError(f"the prior weight m must be positive and finite, got {m!r}")
    if not np.isfinite(r_prior):
        raise ValueError(f"r_prior must be finite, got {r_prior!r}")
    return (n * score + m * r_prior) / (n + m)


def platform_percentiles(scores, population):
    """keeper.laundering.metrics.platform_percentile(s, population) for every s in scores, vectorised: the share of
    the population strictly below s, equal to the scalar function bit for bit (tests/test_cold_start.py)."""
    population = np.sort(_finite(population, "population"))
    if not len(population):
        raise ValueError("the population is empty")
    return np.searchsorted(population, _finite(scores, "scores"), side="left") / len(population)


def percentiles_after_deletion(active_ids, active_score, target_ids, target_score, target_population=None):
    """Each target's percentile among the active accounts after its own U1, with its own entry set to its
    post-deletion score.

    target_population ([targets, active accounts], optional): row i holds every active account's score on target i's
    post-U1 platform, aligned with active_ids. That is probe4's pct_after, which re-scores every active account on the
    target's post-U1 embedding (U1 deletes the edges u -> v, so each rater u's proxy moves too). Without it, every
    target is ranked against active_score with only its own entry replaced: the shared-population approximation,
    exact only when U1 moves no other account's score.
    """
    active_score = _finite(active_score, "active_score")
    target_score = _finite(target_score, "target_score")
    _same_length(active_ids=active_ids, active_score=active_score)
    _same_length(target_ids=target_ids, target_score=target_score)
    pos = _positions(active_ids, target_ids)
    rows = (np.broadcast_to(active_score, (len(pos), len(active_score))) if target_population is None
            else _post_u1_rows(target_population, pos, target_score, len(active_score)))
    return _rank_after_deletion(rows, pos, target_score)


def laundering_percentiles(active_ids, active_proxy, n_in, target_ids, target_proxy, target_n, m, r_prior,
                           target_population=None):
    """Each target's percentile of r~ among all active accounts after its U1.

    Every account's r~ comes from its proxy and its live incoming training ratings: the target from its stored
    post-deletion proxy and its n_after (target_proxy, target_n), each other account from its proxy and its n_in count,
    which U1 leaves unchanged (it deletes in-edges of the target only). The other accounts' proxies are those of target
    i's post-U1 platform, row i of target_population ([targets, active accounts], aligned with active_ids: probe4's
    definition), or, without it, active_proxy for every target: the shared-population approximation, exact only when
    U1 moves no other account's proxy. n_in holds the live incoming training ratings per account id
    (incoming_counts); r_prior is a platform constant, from the base proxies (prior, as evaluate computes it).
    """
    _same_length(active_ids=active_ids, active_proxy=active_proxy)
    _same_length(target_ids=target_ids, target_proxy=target_proxy, target_n=target_n)
    counts = _counts_of(n_in, active_ids)
    target_proxy = _finite(target_proxy, "target_proxy")
    pos = _positions(active_ids, target_ids)
    if target_population is None:
        tilde = np.broadcast_to(calibrate(active_proxy, counts, m, r_prior), (len(pos), len(counts)))
    else:
        tilde = calibrate(_post_u1_rows(target_population, pos, target_proxy, len(counts)), counts, m, r_prior)
    return _rank_after_deletion(tilde, pos, calibrate(target_proxy, target_n, m, r_prior))


def newcomer_rank_cost(active_ids, active_proxy, n_in, m, r_prior, max_in=NEWCOMER_MAX_IN):
    """The mean percentile change, r~ against r, of the active accounts with at most max_in incoming training
    ratings, each percentile taken among all active accounts; negative means the newcomers lose rank. Returns n (the
    newcomers), mean_change, and their mean percentiles under r (mean_pct) and under r~ (mean_pct_calibrated)."""
    proxy = _finite(active_proxy, "active_proxy")
    _same_length(active_ids=active_ids, active_proxy=proxy)
    n = _counts_of(n_in, active_ids)
    tilde = calibrate(proxy, n, m, r_prior)
    new = n <= max_in
    before = platform_percentiles(proxy[new], proxy)
    after = platform_percentiles(tilde[new], tilde)
    return dict(n=int(new.sum()), mean_change=M.mean_or_none(after - before),
                mean_pct=M.mean_or_none(before), mean_pct_calibrated=M.mean_or_none(after))


def low_degree_mcc(test_dst, test_y, test_p, n_in, m, r_prior, max_in=LOW_DEGREE_MAX_IN):
    """The test MCC on the low-degree edge set, for the model's p and for p~.

    The set: every edge of the held-out test snapshot whose destination holds at most max_in incoming training
    ratings, zero included. It takes Task 1 and Task 2 rows alike and keeps accounts unseen in training (n_v = 0),
    which the Task 1 MCC of keeper.laundering.core.test_metrics leaves out. Pass all the snapshot's edges:
    held_out_snapshot(G, tt) gives their destinations (test_dst) and class ids (test_y, 1 = distrust), and test_p is
    the model's P(trust) on the same rows (keeper.laundering.core.p_trust). n_in holds the incoming training ratings
    per account id (incoming_counts of the training split). The rule is test_metrics' own (its _cls_metrics: p < 0.5
    predicts distrust; the MCC is None when the set holds one class only). Returns n (the edges in the set), base and
    calibrated ({mcc, auc, n} each) and delta_mcc, calibrated minus base.
    """
    y = np.asarray(test_y, dtype=np.int64)
    p = _finite(test_p, "test_p")
    _same_length(test_dst=test_dst, test_y=y, test_p=p)
    n = _counts_of(n_in, test_dst)
    low = n <= max_in
    base = C._cls_metrics(y[low], p[low])
    cal = C._cls_metrics(y[low], calibrate(p[low], n[low], m, r_prior))
    delta = None if base["mcc"] is None or cal["mcc"] is None else cal["mcc"] - base["mcc"]
    return dict(n=int(low.sum()), base=base, calibrated=cal, delta_mcc=delta)


def evaluate(active_ids, active_proxy, n_in, target_ids, target_proxy, target_n, test_dst, test_y, test_p, ms=MS,
             target_population=None):
    """The three metrics for each m, for one dataset and seed (E8). r_prior and the newcomer rank cost use the base
    active_proxy. The laundering percentiles rank each target on its post-U1 platform when target_population is given
    (see laundering_percentiles), else on the shared population, and laundering_population records which. The
    'uncalibrated' entry ranks the targets' post-deletion proxies the same way, for comparison. JSON-ready."""
    r_prior = prior(active_proxy)
    raw = percentiles_after_deletion(active_ids, active_proxy, target_ids, target_proxy, target_population)
    report = dict(r_prior=r_prior, prior_percentile=PRIOR_PERCENTILE, newcomer_max_in=NEWCOMER_MAX_IN,
                  low_degree_max_in=LOW_DEGREE_MAX_IN,
                  laundering_population="shared" if target_population is None else "per-target post-U1",
                  uncalibrated=dict(laundering_pct=raw.tolist()), by_m={})
    for m in ms:
        report["by_m"][str(m)] = dict(
            laundering_pct=laundering_percentiles(active_ids, active_proxy, n_in, target_ids, target_proxy,
                                                  target_n, m, r_prior, target_population).tolist(),
            newcomer=newcomer_rank_cost(active_ids, active_proxy, n_in, m, r_prior),
            low_degree=low_degree_mcc(test_dst, test_y, test_p, n_in, m, r_prior))
    return report
