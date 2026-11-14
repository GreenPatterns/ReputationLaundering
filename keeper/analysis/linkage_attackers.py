"""E6's linkage attackers and the G3 metric on the per-verdict dumps (Task 4.6; protocol Section 2.8).

A Table holds one run's candidate rows of one method's direct-arm pooled verdicts (keeper.cppr.dumps.VERDICT_COLUMNS),
multi-member pools only; a single-member pool is counted, never scored. An item is a (pooled verdict, true rater) pair:
a row with label 1. Its within-pool AUC is the share of the other pool members (other true raters included) that score
below the true rater, ties counted 1/2; with average ranks r (0-based) within the pool of n members it is r / (n - 1).
G3 of a seed and attacker is the mean item AUC. Top-1 splits ties: an item scores 1/k when the true rater is among the
k members at the pool's maximum, else 0 (the expected hit rate of a random tie-break); chance is 1 / pool size.

Attackers (ATTACKERS):
  handcrafted   keeper.cppr.linkage.linkage_attack's combined score: each of structure, timing and model_prob is oriented
                with the sign learned on the other half of the run's verdicts (verdict index parity; + when its mean
                item AUC there is >= 0.5), turned into within-pool rank percentiles (average ranks), and averaged.
  <signal>      each of SIGNALS alone, oriented by the sign of its mean item AUC on the training seeds (leave one
                seed out).
  learned       HistGradientBoostingClassifier(random_state=0, early_stopping=False) on FEATURES, fitted on the
                candidate rows of the training seeds (label = true rater), scoring the held-out seed.
corrected_t is the Nadeau-Bengio corrected t over the per-seed values; nested_bootstrap_max resamples seeds, then the
pooled verdicts within each drawn seed, and recomputes the per-seed maximum over attackers.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

FEATURES = ("structure", "timing", "model_prob", "degree", "activity", "activity_window", "candidate_tier",
            "pool_size", "pool_tier")
SIGNALS = FEATURES[:6]
HANDCRAFTED_SIGNALS = ("structure", "timing", "model_prob")
ATTACKERS = ("handcrafted",) + SIGNALS + ("learned",)
LOSO_ATTACKERS = SIGNALS + ("learned",)          # need at least one training seed
TIMING = FEATURES.index("timing")
PD_QUANTILES = (0.1, 0.5, 0.9)
PERMUTATION_SEED = 20261026                      # keeper.analysis.intervals.BOOT_SEED


@dataclass(frozen=True)
class Table:
    """One run's direct-arm candidate rows of one method, multi-member pools only, sorted by verdict and candidate.
    X holds FEATURES (NaN for a missing pool_tier); n_single_items counts the true raters of single-member pools."""
    verdict: np.ndarray
    label: np.ndarray
    X: np.ndarray
    n_single_items: int
    n_single_verdicts: int

    @property
    def n_verdicts(self) -> int:
        return len(np.unique(self.verdict))


def table_from_rows(columns, rows, arm: str, method: str) -> tuple:
    """(Table, failure messages) from verdict-dump rows of ``arm`` and ``method``. A verdict must list each candidate
    once, hold pool_size rows with one pool_id and target, have at least one true rater, and carry finite features in a
    multi-member pool (None in a single-member one)."""
    col = {c: i for i, c in enumerate(columns)}
    by_verdict = {}
    for row in rows:
        if row[col["arm"]] == arm and row[col["method"]] == method:
            by_verdict.setdefault(int(row[col["verdict"]]), []).append(row)
    failures, kept, n_single_items, n_single = [], [], 0, 0
    for v, members in sorted(by_verdict.items()):
        heads = {(r[col["pool_id"]], r[col["target"]], r[col["sign"]], r[col["pool_size"]]) for r in members}
        cands = [int(r[col["candidate"]]) for r in members]
        n_true = sum(int(r[col["label"]]) for r in members)
        if len(heads) != 1 or len(set(cands)) != len(cands) or members[0][col["pool_size"]] != len(members) \
                or n_true < 1:
            failures.append(f"{arm}/{method} verdict {v}: inconsistent pool rows (pool head, candidates, pool_size "
                            f"or no true rater)")
            continue
        if len(members) == 1:
            n_single, n_single_items = n_single + 1, n_single_items + n_true
            continue
        feats = [[r[col[f]] for f in FEATURES] for r in sorted(members, key=lambda r: int(r[col["candidate"]]))]
        values = np.array([[np.nan if x is None and f == "pool_tier" else x for f, x in zip(FEATURES, fr)]
                           for fr in feats], dtype=float)
        if not np.all(np.isfinite(values[:, :FEATURES.index("pool_tier")])):
            failures.append(f"{arm}/{method} verdict {v}: a multi-member pool has a missing or non-finite feature")
            continue
        labels = [int(r[col["label"]]) for r in sorted(members, key=lambda r: int(r[col["candidate"]]))]
        kept.append((v, values, labels))
    if not kept:
        return Table(np.zeros(0, np.int64), np.zeros(0, bool), np.zeros((0, len(FEATURES))), n_single_items,
                     n_single), failures
    verdict = np.concatenate([np.full(len(lab), v, np.int64) for v, _, lab in kept])
    return Table(verdict, np.concatenate([np.asarray(lab, bool) for _, _, lab in kept]),
                 np.concatenate([x for _, x, _ in kept]), n_single_items, n_single), failures


# ------------------------------------------------------------------------ within-pool ranks and the G3 items
def group_ranks(group: np.ndarray, score: np.ndarray) -> tuple:
    """(rank, size, top_share) per row: the 0-based average rank of the row's score within its group (ties averaged),
    the group's size, and 1/k when the row ties k rows at its group's maximum (else 0)."""
    n = len(group)
    if n == 0:
        return np.zeros(0), np.zeros(0, np.int64), np.zeros(0)
    order = np.lexsort((score, group))
    g, s = group[order], score[order]
    new_group = np.r_[True, g[1:] != g[:-1]]
    new_run = new_group | np.r_[True, s[1:] != s[:-1]]
    run_id = np.cumsum(new_run) - 1
    run_start = np.flatnonzero(new_run)
    run_end = np.r_[run_start[1:], n]                       # exclusive
    group_id = np.cumsum(new_group) - 1
    group_start = np.flatnonzero(new_group)
    group_end = np.r_[group_start[1:], n]
    rank_sorted = (run_start + run_end - 1)[run_id] / 2.0 - group_start[group_id]
    run_len = (run_end - run_start)[run_id]
    top_sorted = np.where(run_end[run_id] == group_end[group_id], 1.0 / run_len, 0.0)
    rank, size, top = np.empty(n), np.empty(n, np.int64), np.empty(n)
    rank[order], size[order], top[order] = rank_sorted, (group_end - group_start)[group_id], top_sorted
    return rank, size, top


def item_metrics(t: Table, score: np.ndarray) -> dict:
    """Per item (rows with label 1, in row order): within-pool AUC, tie-split top-1, chance 1 / pool size, verdict."""
    rank, size, top = group_ranks(t.verdict, score)
    m = t.label
    return dict(auc=rank[m] / (size[m] - 1), top1=top[m], chance=1.0 / size[m], verdict=t.verdict[m])


def g3(t: Table, score: np.ndarray) -> float:
    """G3 linkage AUC of one seed: the mean item AUC (NaN without an item)."""
    auc = item_metrics(t, score)["auc"]
    return float(auc.mean()) if len(auc) else float("nan")


def _sign(mean_auc: float) -> float:
    return 1.0 if not np.isfinite(mean_auc) or mean_auc >= 0.5 else -1.0


def handcrafted_score(t: Table) -> tuple:
    """(combined score per row, [orientation of the even half, of the odd half]) as linkage_attack computes them:
    the half with even verdict index takes the signs learned on the odd half, and vice versa."""
    odd = (t.verdict % 2).astype(bool)
    raw = {k: item_metrics(t, t.X[:, FEATURES.index(k)]) for k in HANDCRAFTED_SIGNALS}
    score, orientation = np.zeros(len(t.verdict)), []
    for half in (False, True):                                 # evaluate the half, fit on the other
        rows = odd == half
        fit = {k: raw[k]["auc"][(raw[k]["verdict"] % 2).astype(bool) != half] for k in HANDCRAFTED_SIGNALS}
        sign = {k: _sign(v.mean() if len(v) else float("nan")) for k, v in fit.items()}
        orientation.append(sign)
        pct = []
        for k in HANDCRAFTED_SIGNALS:
            rank, size, _ = group_ranks(t.verdict, sign[k] * t.X[:, FEATURES.index(k)])
            pct.append(rank / np.maximum(size - 1, 1))
        score[rows] = np.mean(np.stack(pct, axis=1), axis=1)[rows]
    return score, orientation


# ------------------------------------------------------------------------ leave-one-seed-out attackers
def loso_signal_scores(tables: dict, feature: str) -> dict:
    """seed -> (oriented score per row, sign): each seed's sign is that of the mean item AUC of the raw signal over
    the other seeds' items; None for a seed when no other seed has an item."""
    j = FEATURES.index(feature)
    raw = {s: item_metrics(t, t.X[:, j])["auc"] for s, t in tables.items()}
    out = {}
    for s, t in tables.items():
        train = [raw[o] for o in tables if o != s and len(raw[o])]
        if not train:
            out[s] = None
            continue
        sign = _sign(float(np.concatenate(train).mean()))
        out[s] = (sign * t.X[:, j], sign)
    return out


def _model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=0, early_stopping=False)


def _proba(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def loso_learned(tables: dict, importance: bool = True) -> dict:
    """seed -> None (no training seed) or dict(score per row of the held-out seed, n_train_rows, and, with
    ``importance``, permutation_importance: {feature: drop of the held-out G3 when the feature is permuted over the
    held-out rows}, timing_pd: {quantile: mean predicted probability with timing set to that quantile of the training
    rows}). The learner is fitted on the other seeds' rows only."""
    out = {}
    for s, t in tables.items():
        train = [o for o in tables if o != s and len(tables[o].verdict)]
        if not train or not len(t.verdict):
            out[s] = None
            continue
        X = np.concatenate([tables[o].X for o in train])
        y = np.concatenate([tables[o].label for o in train]).astype(np.int64)
        if len(np.unique(y)) < 2:
            out[s] = None
            continue
        model = _model().fit(X, y)
        score = _proba(model, t.X)
        res = dict(score=score, n_train_rows=int(len(y)), train_seeds=sorted(train))
        if importance:
            base = g3(t, score)
            drops = {}
            for j, f in enumerate(FEATURES):
                Xp = t.X.copy()
                Xp[:, j] = np.random.default_rng([PERMUTATION_SEED, int(s), j]).permutation(Xp[:, j])
                drops[f] = base - g3(t, _proba(model, Xp))
            res["permutation_importance"] = drops
            res["timing_pd"] = {}
            for q in PD_QUANTILES:
                Xq = t.X.copy()
                Xq[:, TIMING] = float(np.quantile(X[:, TIMING], q))
                res["timing_pd"][str(q)] = float(_proba(model, Xq).mean())
        out[s] = res
    return out


def attacker_scores(tables: dict, importance: bool = True) -> tuple:
    """(scores, details): scores[seed][attacker] = score per row or None (undefined); details holds the handcrafted
    orientations, the single-signal signs and the learned attacker's extras per seed."""
    scores = {s: {} for s in tables}
    details = dict(handcrafted_orientation={}, signal_sign={k: {} for k in SIGNALS}, learned={})
    for s, t in tables.items():
        scores[s]["handcrafted"], details["handcrafted_orientation"][s] = handcrafted_score(t)
    for k in SIGNALS:
        for s, got in loso_signal_scores(tables, k).items():
            scores[s][k] = None if got is None else got[0]
            details["signal_sign"][k][s] = None if got is None else got[1]
    for s, got in loso_learned(tables, importance).items():
        scores[s]["learned"] = None if got is None else got["score"]
        details["learned"][s] = None if got is None else {k: v for k, v in got.items() if k != "score"}
    return scores, details


# ------------------------------------------------------------------------ statistics
def corrected_t(values, null: float, n_test_over_train: float | None = None, alpha: float = 0.05) -> dict:
    """Nadeau-Bengio corrected t over the per-fold values (one per held-out seed): variance factor 1 + J n_test /
    n_train (J folds; leave one seed out gives n_test / n_train = 1 / (J - 1) unless given), so the standard error is
    sqrt(factor * sd^2 / J). p_below is the one-sided p-value of mean < null (df = J - 1); ci95 the two-sided
    (1 - alpha) interval of the mean, clipped to [0, 1]."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    J = len(v)
    if J < 2:
        return dict(n_folds=J, mean=float(v.mean()) if J else None, sd=None, factor=None, se=None, t=None, df=None,
                    p_below=None, ci95=None)
    ratio = 1.0 / (J - 1) if n_test_over_train is None else float(n_test_over_train)
    factor = 1.0 + J * ratio
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    se = float(np.sqrt(factor * sd ** 2 / J))
    if se == 0:
        t, p = (float("-inf"), 0.0) if mean < null else (float("inf") if mean > null else float("nan"), 1.0)
    else:
        t = (mean - null) / se
        p = float(student_t.cdf(t, J - 1))
    half = float(student_t.ppf(1 - alpha / 2, J - 1)) * se
    return dict(n_folds=J, mean=mean, sd=sd, n_test_over_train=ratio, factor=factor, se=se, t=t, df=J - 1,
                p_below=p, ci95=[max(0.0, mean - half), min(1.0, mean + half)])


def verdict_sums(items: dict, attackers) -> np.ndarray:
    """[n_verdicts, len(attackers) + 1]: per pooled verdict, the sum of its items' AUC under each attacker and its
    number of items (the bootstrap unit is the pooled verdict; G3 stays the mean over items)."""
    first = items[attackers[0]]["verdict"]
    _, inv = np.unique(first, return_inverse=True)
    out = np.zeros((inv.max() + 1 if len(inv) else 0, len(attackers) + 1))
    for j, a in enumerate(attackers):
        np.add.at(out[:, j], inv, items[a]["auc"])
    np.add.at(out[:, -1], inv, 1.0)
    return out


def max_ratio_stat(batch: np.ndarray) -> np.ndarray:
    """[B, n, k + 1] verdict rows -> [B]: the maximum over the k attackers of sum(AUC) / sum(items)."""
    sums = batch.sum(axis=1)
    return np.max(sums[:, :-1] / sums[:, -1:], axis=1)


def nested_bootstrap_max(per_seed: dict, n_boot: int, seed, chunk: int = 200) -> np.ndarray:
    """Replicates of the mean over seeds of max_ratio_stat: seeds resampled, then each drawn seed's pooled verdicts
    (the rows of per_seed[seed]); evaluated ``chunk`` replicates at a time to bound memory."""
    seeds = sorted(per_seed)
    arrays = [per_seed[s] for s in seeds]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        pick = rng.integers(0, len(seeds), size=(b, len(seeds)))
        vals = np.empty(pick.shape)
        for i, a in enumerate(arrays):
            where = pick == i
            count = int(where.sum())
            if count:
                vals[where] = max_ratio_stat(a[rng.integers(0, len(a), size=(count, len(a)))])
        out[start:start + b] = vals.mean(axis=1)
    return out
