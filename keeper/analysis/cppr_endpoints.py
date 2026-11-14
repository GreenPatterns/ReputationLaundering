"""Endpoints of the E3/E4 confirmatory analysis (Task 4.2b): per-target cells, intervals, p-values and readings.

Pure functions; keeper.analysis.cppr_confirm reads the runs and calls them. The rules are the protocol's
(docs/preregistration.md, Sections 1, 3 and 4).

Cells. A run's target dump is indexed as (arm, method, kind) -> {target: row}, with the columns of
keeper.cppr.dumps.TARGET_COLUMNS (gain = r_after - r0). An endpoint maps one network's runs to cells
seed -> {target: [values]}, the layout of keeper.analysis.bootstrap: one value for a difference, (numerator,
denominator) for a ratio. A pair is the same seed and target in every arm it uses; a target that some arm lacks is
dropped and counted (Section 3).
  H1(a)  direct arm, kind edge: gain(plain) - gain(cppr).
  H1(b)  cut = 1 - mean(gain_adj(cppr)) / mean(gain(plain)), with gain_adj(cppr) = gain(cppr, direct) - gain(cppr,
         control, kind edge); Plain Erasure is not control-adjusted.
  H2w    whitewash target: (vpe_ww - vpe_ctl) - (cppr_ww - cppr_ctl); suppression = 1 - mean(cppr_ww - cppr_ctl) /
         mean(vpe_ww - vpe_ctl).
  H2b    bad-mouth victim, in losses (minus gains), with the control rows of kind trusted: the same two quantities.
  H3b    per seed: the contribution under BFS minus the contribution under capped BTP. A method's contribution is the
         seed's mean control-adjusted CPPR whitewash gain in its best-response cell, the g = 1 E4 cell with the largest
         mean gain over the other seeds (leave one seed out; see loso_endpoints).
A mean is the mean of per-seed means, every seed weighted equally; a ratio is the ratio of seed-averaged means.

Intervals and tests (Section 4). The headline 95% interval is the hull of the crossed bootstrap percentile interval
(seeds and target ids resampled independently, missing cells allowed) and the seed-t interval (df = seeds - 1;
linearized for a ratio), clipped to the support; a 0/1 field at exactly 0 or 1 takes Clopper-Pearson over its distinct
targets. A difference's p-value is max(one-sided seed-t p, (1 + #{replicates <= 0}) / (B + 1)); a cut's or
suppression's, against c = 0.8, is max(linearized one-sided seed-t p of (ratio - c), (1 + #{replicates < c}) /
(B + 1)); B counts the defined replicates (10,000 when none is undefined). A component with results for fewer than 80%
of its seeds is not evaluable. A hypothesis' p-value is the largest of its components' (intersection-union). The
nested bootstrap gives the same quantities as a sensitivity (sensitivity_nested).
"""
import math
import warnings
from fractions import Fraction
from functools import partial
from typing import NamedTuple

import numpy as np
from scipy.stats import t as student_t

from keeper.analysis.bootstrap import (ALPHA, column_means_stat, combine_mean, combine_ratio, crossed_replicates,
                                       mean_stat, nested_replicates, per_seed_values, percentile_interval,
                                       ratio_seed_t_interval, seed_t_interval)
from keeper.analysis.intervals import clip_to_support, headline_interval
from keeper.cppr.protocol import CAP_TOL

THRESHOLD = 0.8                  # removal threshold of H1(b), H2w(b) and H2b(b) (Section 1)
MIN_SEED_SHARE = Fraction(4, 5)  # a component with results for fewer than 80% of its seeds is not evaluable
OVERSHOOT_CUT = 1.2              # CPPR's overshoot, a secondary outcome: a target's cut above 1.2
DIFFERENCE_RULE = "95% interval > 0"
NOT_EVALUABLE = "not evaluable"

DIRECT = {m: ("direct", m, "edge") for m in ("plain", "vpe", "cppr")}
ACCOUNT = {m: ("direct", m, "account") for m in ("plain", "vpe", "cppr")}
WHITEWASH = {m: ("whitewash", m, "edge") for m in ("vpe", "cppr")}
BADMOUTH = {m: ("badmouth", m, "trusted") for m in ("vpe", "cppr")}
CONTROL_EDGE = {m: ("control", m, "edge") for m in ("vpe", "cppr")}
CONTROL_TRUSTED = {m: ("control", m, "trusted") for m in ("vpe", "cppr")}


def threshold_rule(threshold: float = THRESHOLD) -> str:
    return f"95% interval lower bound >= {threshold:g}"


class Paired(NamedTuple):
    """Cells seed -> {target: [values]}, and the number of targets dropped because one of their arms lacked them."""
    cells: dict
    dropped: int = 0


# ------------------------------------------------------------------------ cells
def gains(index: dict, key: tuple) -> dict:
    """{target: gain} of one (arm, method, kind) of a run's indexed dump; empty if the run has none."""
    return {t: float(row["gain"]) for t, row in index.get(key, {}).items()}


def over_seeds(by_seed: dict, keys: tuple, fn) -> Paired:
    """Cells of fn(gain under keys[0], gain under keys[1], ...) over the targets present under every key of their
    seed (``by_seed``: seed -> indexed dump); a target present under some keys but not all is dropped and counted."""
    cells, dropped = {}, 0
    for seed in sorted(by_seed):
        tables = [gains(by_seed[seed], key) for key in keys]
        common = set.intersection(*(set(t) for t in tables))
        dropped += len(set().union(*tables) - common)
        if common:
            cells[seed] = {t: [float(x) for x in fn(*(table[t] for table in tables))] for t in sorted(common)}
    return Paired(cells, dropped)


def h1_cells(by_seed: dict) -> dict:
    """H1 on the capped-BTP E3 runs: a, the difference; b, (gain_adj(cppr), gain(plain)) for the cut."""
    return dict(a=over_seeds(by_seed, (DIRECT["plain"], DIRECT["cppr"]), lambda p, c: [p - c]),
                b=over_seeds(by_seed, (DIRECT["cppr"], CONTROL_EDGE["cppr"], DIRECT["plain"]),
                             lambda c, k, p: [c - k, p]))


def h2w_cells(by_seed: dict) -> dict:
    """H2w: a, the difference of control-adjusted whitewash gains; b, (CPPR's, Plain VPE's) for the suppression."""
    keys = (WHITEWASH["vpe"], CONTROL_EDGE["vpe"], WHITEWASH["cppr"], CONTROL_EDGE["cppr"])
    return dict(a=over_seeds(by_seed, keys, lambda v, vc, c, cc: [(v - vc) - (c - cc)]),
                b=over_seeds(by_seed, keys, lambda v, vc, c, cc: [c - cc, v - vc]))


def h2b_cells(by_seed: dict) -> dict:
    """H2b as H2w, in losses (minus gains), with the control arm scored on the bad-mouth victims (kind trusted)."""
    keys = (BADMOUTH["vpe"], CONTROL_TRUSTED["vpe"], BADMOUTH["cppr"], CONTROL_TRUSTED["cppr"])

    def in_losses(fn):
        return lambda v, vc, c, cc: fn(-v, -vc, -c, -cc)
    return dict(a=over_seeds(by_seed, keys, in_losses(lambda v, vc, c, cc: [(v - vc) - (c - cc)])),
                b=over_seeds(by_seed, keys, in_losses(lambda v, vc, c, cc: [c - cc, v - vc])))


def account_cells(by_seed: dict) -> dict:
    """Account-level direct laundering: each method's gain, and Plain Erasure's minus CPPR's. Not control-adjusted:
    the control arm scores no account-level target."""
    out = {f"gain_{m}": over_seeds(by_seed, (ACCOUNT[m],), lambda g: [g]) for m in ("plain", "vpe", "cppr")}
    out["plain_minus_cppr"] = over_seeds(by_seed, (ACCOUNT["plain"], ACCOUNT["cppr"]), lambda p, c: [p - c])
    return out


def attack_minus_control_cells(by_seed: dict) -> dict:
    """Each attack arm's gain minus the control arm's on the same targets (bad-mouth: signed gains, not losses)."""
    def minus(a, k):
        return [a - k]
    return dict(direct_cppr=over_seeds(by_seed, (DIRECT["cppr"], CONTROL_EDGE["cppr"]), minus),
                whitewash_vpe=over_seeds(by_seed, (WHITEWASH["vpe"], CONTROL_EDGE["vpe"]), minus),
                whitewash_cppr=over_seeds(by_seed, (WHITEWASH["cppr"], CONTROL_EDGE["cppr"]), minus),
                badmouth_vpe=over_seeds(by_seed, (BADMOUTH["vpe"], CONTROL_TRUSTED["vpe"]), minus),
                badmouth_cppr=over_seeds(by_seed, (BADMOUTH["cppr"], CONTROL_TRUSTED["cppr"]), minus))


def overshoot_cells(by_seed: dict) -> tuple:
    """(0/1 cells, targets left out): 1 when a target's cut 1 - gain_adj(cppr) / gain(plain) exceeds OVERSHOOT_CUT,
    i.e. retention pushes it below its base reputation by more than 20% of its laundering gain. A cut measures a
    reduction only for a positive Plain Erasure gain, so the other targets are left out and counted."""
    full = h1_cells(by_seed)["b"]
    cells, left_out = {}, 0
    for seed, rows in full.cells.items():
        kept = {t: [float(1.0 - adj / plain > OVERSHOOT_CUT)] for t, (adj, plain) in rows.items() if plain > 0}
        left_out += len(rows) - len(kept)
        if kept:
            cells[seed] = kept
    return Paired(cells, full.dropped), left_out


def adjusted_gains(e4_index: dict, e3_index: dict) -> tuple:
    """({target: control-adjusted CPPR whitewash gain}, problems) of one E4 run against the E3 run with the same seed
    and tier setting: gain(cppr, whitewash, edge) of the E4 run minus gain(cppr, control, edge) of the E3 run. Both
    runs start from the same base model, so each target's r0 must be identical in the two (exact equality). A target
    whose r0 differs, or that one of the runs lacks, is a problem and is left out."""
    ww, ctl = e4_index.get(WHITEWASH["cppr"], {}), e3_index.get(CONTROL_EDGE["cppr"], {})
    problems = [f"target {t} is scored in the {'E4 run' if t in ww else 'E3 control arm'} only"
                for t in sorted(set(ww) ^ set(ctl))]
    adjusted = {}
    for t in sorted(set(ww) & set(ctl)):
        if ww[t]["r0"] != ctl[t]["r0"]:
            problems.append(f"target {t}: r0 {ww[t]['r0']!r} in the E4 run differs from r0 {ctl[t]['r0']!r} in the "
                            "E3 control arm")
        else:
            adjusted[t] = float(ww[t]["gain"]) - float(ctl[t]["gain"])
    return adjusted, problems


def difference_of(first: dict, second: dict) -> Paired:
    """Cells of first - second for the (seed, target) pairs in both (seed -> {target: value}); others dropped."""
    cells, dropped = {}, 0
    for seed in sorted(set(first) | set(second)):
        a, b = first.get(seed, {}), second.get(seed, {})
        common = set(a) & set(b)
        dropped += len(set(a) ^ set(b))
        if common:
            cells[seed] = {t: [a[t] - b[t]] for t in sorted(common)}
    return Paired(cells, dropped)


def method_cell_cells(adjusted: dict, methods: tuple, cells: tuple) -> Paired:
    """H3b's cells, seed -> {target: row}: row[m * len(cells) + c] = adjusted[(methods[m], cells[c])][seed][target],
    NaN where that run or target is missing."""
    seeds = sorted({s for key, by_seed in adjusted.items() if key[0] in methods and key[1] in cells for s in by_seed})
    out = {}
    for seed in seeds:
        tables = [adjusted.get((m, c), {}).get(seed, {}) for m in methods for c in cells]
        rows = {t: [table.get(t, math.nan) for table in tables] for t in sorted(set().union(*tables))}
        if rows:
            out[seed] = rows
    return Paired(out)


def restrict(paired: Paired, exclude) -> tuple:
    """(paired without the targets in ``exclude``, the number of (seed, target) pairs removed)."""
    exclude, cells, removed = set(exclude), {}, 0
    for seed, rows in paired.cells.items():
        kept = {t: v for t, v in rows.items() if t not in exclude}
        removed += len(rows) - len(kept)
        if kept:
            cells[seed] = kept
    return Paired(cells, paired.dropped), removed


# ------------------------------------------------------------------------ H3b: leave-one-seed-out best response
def loso_endpoints(cell_means: np.ndarray, drawn: np.ndarray) -> tuple:
    """Leave-one-seed-out best responses of H3b.

    cell_means [B, S, methods, cells] holds each drawn seed's mean gain per method and cell (NaN: no value); drawn
    [B, S] holds the drawn seeds' identities. Per method, a drawn seed's best-response cell is the cell with the
    largest mean over the drawn seeds that differ from it, so a seed drawn twice never chooses with its own data; ties
    go to the first cell. Returns (endpoint [B, S], the contribution of method 0 minus that of method 1; choice
    [B, S, methods], -1 where no other seed has a value; contribution [B, S, methods])."""
    finite = np.isfinite(cell_means)
    other = (drawn[:, :, None] != drawn[:, None, :]).astype(float)
    total = np.einsum("bij,bjmc->bimc", other, np.where(finite, cell_means, 0.0))
    count = np.einsum("bij,bjmc->bimc", other, finite.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        loso = np.where(count > 0, total / count, np.nan)
    has = np.isfinite(loso).any(axis=3)
    choice = np.argmax(np.where(np.isfinite(loso), loso, -np.inf), axis=3)
    contribution = np.where(has, np.take_along_axis(cell_means, choice[..., None], axis=3)[..., 0], np.nan)
    return contribution[..., 0] - contribution[..., 1], np.where(has, choice, -1), contribution


def h3b_combine(per_seed: np.ndarray, drawn: np.ndarray, n_methods: int = 2) -> np.ndarray:
    """Seed-aware combine for keeper.analysis.bootstrap: per replicate, the mean over the drawn seeds of the H3b
    endpoint, from the per-seed cell means [B, S, methods * cells]; NaN when no drawn seed has an endpoint."""
    b, s, width = per_seed.shape
    endpoint = loso_endpoints(per_seed.reshape(b, s, n_methods, width // n_methods), drawn)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(endpoint, axis=1)


def _one_seed(drawn: np.ndarray) -> np.ndarray:
    return np.all(drawn == drawn[:, :1], axis=1)


# ------------------------------------------------------------------------ p-values and readings
def p_seed_t(values, null: float = 0.0):
    """One-sided seed-level t p-value of mean(values) > null (df = n - 1); None with fewer than two values."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return None
    excess, sd = v.mean() - null, v.std(ddof=1)
    if sd == 0:
        return 0.0 if excess > 0 else 1.0
    return float(student_t.sf(excess / (sd / math.sqrt(len(v))), len(v) - 1))


def p_ratio_seed_t(num, den, threshold: float):
    """One-sided linearized seed-t p-value of 1 - mean(num) / mean(den) > threshold, with the residuals of
    keeper.analysis.bootstrap.ratio_seed_t_interval (df = n - 1); None with fewer than two seeds or mean(den) = 0."""
    pairs = [(a, b) for a, b in zip(num, den)
             if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 2:
        return None
    a, b = (np.asarray(x, dtype=float) for x in zip(*pairs))
    if b.mean() == 0:
        return None
    ratio = a.mean() / b.mean()
    se = ((a - ratio * b) / b.mean()).std(ddof=1) / math.sqrt(len(a))
    excess = (1.0 - ratio) - threshold
    if se == 0:
        return 0.0 if excess > 0 else 1.0
    return float(student_t.sf(excess / se, len(a) - 1))


def p_bootstrap(replicates, threshold: float = 0.0, strict: bool = False) -> float:
    """(1 + #{replicates <= threshold}) / (B + 1), with < when strict; B counts the defined replicates."""
    r = np.asarray(replicates, dtype=float)
    r = r[np.isfinite(r)]
    hits = int(np.sum(r < threshold) if strict else np.sum(r <= threshold))
    return (1 + hits) / (len(r) + 1)


def difference_decision(ci, evaluable: bool) -> dict:
    """Rule "interval > 0": met when the lower bound is above 0, contradicted when the upper bound is below 0."""
    if not evaluable or ci is None:
        return dict(decision=NOT_EVALUABLE, met=False, contradicted=False)
    met, contradicted = ci[0] > 0, ci[1] < 0
    return dict(decision="met" if met else "contradicted" if contradicted else "not met", met=bool(met),
                contradicted=bool(contradicted))


def threshold_decision(ci, threshold: float, evaluable: bool) -> dict:
    """Rule "lower bound >= threshold", read three ways (Section 1): met when the lower bound reaches the threshold,
    inconclusive when only the upper bound does, partial when the upper bound is below it. Contradicted only when the
    interval lies wholly below 0, the method then doing worse than nothing (ruling R62(b)); a partial reading alone is
    not a contradiction."""
    if not evaluable or ci is None:
        return dict(decision=NOT_EVALUABLE, reading=None, met=False, contradicted=False)
    reading = "met" if ci[0] >= threshold else "inconclusive" if ci[1] >= threshold else "partial"
    return dict(decision=reading, reading=reading, met=reading == "met", contradicted=bool(ci[1] < 0))


# ------------------------------------------------------------------------ components
def _counts(paired: Paired, expected: int, seeds=None) -> dict:
    """Seeds (those with a result), targets and pairs of a component, and whether it is evaluable."""
    cells = paired.cells
    seeds = sorted(cells) if seeds is None else list(seeds)
    return dict(n_seeds=len(seeds), n_seeds_expected=int(expected), seeds=seeds,
                n_targets=len(set().union(*(set(c) for c in cells.values()))) if cells else 0,
                n_pairs=sum(len(c) for c in cells.values()), n_pairs_dropped=int(paired.dropped),
                evaluable=bool(expected) and Fraction(len(seeds), int(expected)) >= MIN_SEED_SHARE)


def _replicates(groups: dict, name: str, boot, stat, combine, seed_aware: bool = False) -> tuple:
    """(point, crossed replicates, nested replicates) on the streams of (dataset, name) of ``boot`` (a
    keeper.analysis.intervals.Boot). With seed_aware, a replicate that drew one distinct seed only is left out: a
    leave-one-seed-out choice needs a second seed (probability S^(1 - S): 0.16% with 5 seeds)."""
    point, crossed, drawn_c = crossed_replicates(groups, stat, combine, boot.n_boot, boot.seed_for(name, "crossed"),
                                                 seed_aware)
    _, nested, drawn_n = nested_replicates(groups, stat, combine, boot.n_boot, boot.seed_for(name, "nested"),
                                           seed_aware)
    if seed_aware:
        crossed, nested = crossed[~_one_seed(drawn_c)], nested[~_one_seed(drawn_n)]
    return point, crossed, nested


def _interval(point: float, replicates, seed_t, support, groups, unit) -> dict:
    """One design's headline interval: the hull with the seed-t interval (Clopper-Pearson for a 0/1 field at 0 or 1),
    clipped to the support (ruling R31)."""
    info = {}
    boot_ci = percentile_interval(replicates, ALPHA, info)
    ci, source = headline_interval(point, boot_ci, seed_t, groups, unit)
    clipped, changed = clip_to_support(ci, support)
    if changed:
        source += f", clipped to the support {list(support)}"
    return dict(ci95=clipped, ci95_bootstrap=boot_ci, ci95_source=source, ci95_unclipped=ci if changed else None,
                bootstrap_undefined_share=info["undefined_share"], n_replicates=int(len(replicates)))


def _tests(ci, p_t, replicates, evaluable: bool, threshold=None) -> dict:
    """p-values and decision of a difference (threshold None) or of a ratio against threshold."""
    if threshold is None:
        p_b, decision = p_bootstrap(replicates, 0.0), difference_decision(ci, evaluable)
    else:
        p_b, decision = p_bootstrap(replicates, threshold, strict=True), threshold_decision(ci, threshold, evaluable)
    return dict(p_value=None if p_t is None else max(p_t, p_b), p_seed_t=p_t, p_bootstrap=p_b, evaluable=evaluable,
                **decision)


def _no_test(out: dict, confirmatory: bool, reason: str) -> dict:
    """A component without intervals (too few seeds, or a refused bootstrap): reported, never evaluable."""
    out.update(evaluable=False, not_evaluable_reason=reason, ci95=None)
    if confirmatory:
        out.update(p_value=None, p_seed_t=None, p_bootstrap=None, decision=NOT_EVALUABLE, met=False,
                   contradicted=False, sensitivity_nested=dict(evaluable=False, p_value=None, decision=NOT_EVALUABLE,
                                                               met=False, contradicted=False))
    return out


def _finish(out: dict, cells: dict, name: str, boot, stat, combine, seed_t, p_t, confirmatory: bool, support=None,
            unit=None, threshold=None, seed_aware: bool = False) -> dict:
    """Bootstrap the component, then add its intervals and, if confirmatory, its p-values and decision under the
    crossed (headline) and nested (sensitivity_nested) designs. A refused bootstrap (more than 1% of replicates
    undefined) leaves the component without a headline interval, so it is not evaluable; its seed-t interval stays."""
    out.update(ci95_seed_t=seed_t, ci95_support=None if support is None else list(support),
               bootstrap_streams={design: boot.seed_for(name, design) for design in ("crossed", "nested")})
    try:
        point, crossed, nested = _replicates(cells, name, boot, stat, combine, seed_aware)
        main = _interval(point, crossed, seed_t, support, cells, unit)
        alt = _interval(point, nested, seed_t, support, cells, unit)
    except FloatingPointError as err:
        return _no_test(out, confirmatory, f"bootstrap refused: {err}")
    out.update(estimate=point, **main, ci95_nested=alt["ci95_bootstrap"])
    if confirmatory:
        out.update(_tests(main["ci95"], p_t, crossed, out["evaluable"], threshold))
        out["sensitivity_nested"] = dict(ci95=alt["ci95"], ci95_bootstrap=alt["ci95_bootstrap"],
                                         **_tests(alt["ci95"], p_t, nested, out["evaluable"], threshold))
    return out


def difference_component(paired: Paired, name: str, boot, expected: int, *, confirmatory: bool = True,
                         support=None, unit=None) -> dict:
    """A difference (or a mean) over targets: estimate, intervals and, if confirmatory, p-values and decision (rule:
    95% interval > 0). ``name`` keys the random streams of ``boot`` (keeper.analysis.intervals.Boot); ``expected`` is
    the number of registered seeds; ``unit`` marks a 0/1 field (Clopper-Pearson at 0 or 1)."""
    cells = paired.cells
    out = dict(kind="difference", **({"rule": DIFFERENCE_RULE} if confirmatory else {}), **_counts(paired, expected))
    per_seed = per_seed_values(cells, mean_stat) if cells else []
    out.update(estimate=float(np.mean(per_seed)) if per_seed else None, by_seed=dict(zip(out["seeds"], per_seed)))
    if len(per_seed) < 2:
        return _no_test(out, confirmatory, "fewer than two seeds")
    return _finish(out, cells, name, boot, mean_stat, combine_mean, seed_t_interval(per_seed),
                   p_seed_t(per_seed) if confirmatory else None, confirmatory, support, unit)


def _cut(per_seed: np.ndarray) -> np.ndarray:
    return 1.0 - combine_ratio(per_seed)


def ratio_component(paired: Paired, name: str, boot, expected: int, *, threshold: float = THRESHOLD,
                    confirmatory: bool = True) -> dict:
    """1 - mean(numerator) / mean(denominator) over seed-averaged means (the cut of H1(b); the suppression of H2w and
    H2b), with the ratio intervals: the crossed bootstrap of the ratio of seed-averaged means and the linearized
    seed-t interval of keeper.analysis.bootstrap.ratio_seed_t_interval. Rule: lower bound >= threshold, read three
    ways. A ratio with a signed numerator and denominator has no bounded support, so it is not clipped."""
    cells = paired.cells
    out = dict(kind="ratio", **({"rule": threshold_rule(threshold)} if confirmatory else {}),
               **_counts(paired, expected))
    pairs = per_seed_values(cells, column_means_stat) if cells else []
    num, den = [p[0] for p in pairs], [p[1] for p in pairs]
    out.update(estimate=None if not pairs or np.mean(den) == 0 else float(1.0 - np.mean(num) / np.mean(den)),
               numerator_mean=float(np.mean(num)) if pairs else None,
               denominator_mean=float(np.mean(den)) if pairs else None,
               by_seed={s: None if d == 0 else 1.0 - n / d for s, n, d in zip(out["seeds"], num, den)},
               by_seed_numerator=dict(zip(out["seeds"], num)), by_seed_denominator=dict(zip(out["seeds"], den)))
    if len(pairs) < 2:
        return _no_test(out, confirmatory, "fewer than two seeds")
    ratio_t = ratio_seed_t_interval(num, den)
    seed_t = None if ratio_t is None else [1.0 - ratio_t[1], 1.0 - ratio_t[0]]
    return _finish(out, cells, name, boot, column_means_stat, _cut, seed_t,
                   p_ratio_seed_t(num, den, threshold) if confirmatory else None, confirmatory, threshold=threshold)


def h3b_component(paired: Paired, name: str, boot, expected: int, *, methods: tuple, cell_labels: tuple,
                  confirmatory: bool = True) -> dict:
    """The leave-one-seed-out best-response difference of H3b over ``cell_labels`` (cells from method_cell_cells):
    per seed, method 0's contribution minus method 1's. The seed-t interval and p-value use the per-seed endpoints;
    each bootstrap replicate recomputes every cell mean on its resampled targets, then the choices, contributions and
    difference (h3b_combine). The in-sample choice (best cell over all seeds) is reported for comparison."""
    cells = {s: rows for s, rows in paired.cells.items() if np.isfinite(np.asarray(list(rows.values()))).any()}
    out = dict(kind="best-response difference", **({"rule": DIFFERENCE_RULE} if confirmatory else {}),
               methods=list(methods), cells=list(cell_labels))
    seeds, n_m, n_c = sorted(cells), len(methods), len(cell_labels)
    if not seeds:
        out.update(_counts(Paired({}, paired.dropped), expected), estimate=None, by_seed={})
        return _no_test(out, confirmatory, "no seed has a value")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.array(per_seed_values(cells, column_means_stat), dtype=float).reshape(1, len(seeds), n_m, n_c)
        in_sample = np.nanmean(means[0], axis=0)
    endpoint, choice, contribution = loso_endpoints(means, np.arange(len(seeds))[None])
    defined = np.isfinite(endpoint[0])
    label = lambda c: None if c < 0 else cell_labels[c]  # noqa: E731
    out.update(_counts(Paired(cells, paired.dropped), expected, seeds=[s for s, d in zip(seeds, defined) if d]),
               seeds_with_cells=seeds,
               in_sample_choice={m: None if np.isnan(in_sample[i]).all() else cell_labels[int(np.nanargmax(in_sample[i]))]
                                 for i, m in enumerate(methods)},
               cell_means={m: {c: None if np.isnan(in_sample[i, j]) else float(in_sample[i, j])
                               for j, c in enumerate(cell_labels)} for i, m in enumerate(methods)},
               by_seed={s: dict(endpoint=float(endpoint[0, k]) if defined[k] else None,
                                choice={m: label(int(choice[0, k, i])) for i, m in enumerate(methods)},
                                contribution={m: None if np.isnan(contribution[0, k, i]) else float(contribution[0, k, i])
                                              for i, m in enumerate(methods)})
                        for k, s in enumerate(seeds)})
    values = [float(v) for v in endpoint[0][defined]]
    out["estimate"] = float(np.mean(values)) if values else None
    if len(values) < 2:
        return _no_test(out, confirmatory, "fewer than two seeds with an endpoint")
    return _finish(out, cells, name, boot, column_means_stat, partial(h3b_combine, n_methods=n_m),
                   seed_t_interval(values), p_seed_t(values) if confirmatory else None, confirmatory, seed_aware=True)


# ------------------------------------------------------------------------ hypotheses and H3a
def network_summary(components: dict) -> dict:
    """One network's reading of a hypothesis: evaluable when every component is; the interval rule met when every
    component meets its rule; the p-value is the largest component p-value (intersection-union)."""
    evaluable = all(c.get("evaluable") and c.get("p_value") is not None for c in components.values())
    met = evaluable and all(c.get("met") for c in components.values())
    return dict(evaluable=evaluable, interval_rule_met=met,
                p_value=max(c["p_value"] for c in components.values()) if evaluable else None,
                contradicted=sorted(n for n, c in components.items() if c.get("contradicted")),
                status=NOT_EVALUABLE if not evaluable else "interval rule met" if met else "interval rule not met")


def hypothesis_summary(networks: dict, registered: tuple) -> dict:
    """A hypothesis over the registered networks (its rule holds "on each network"): evaluable when every network was
    analysed and is evaluable; its p-value is the largest network p-value, for the decisions step's Holm correction."""
    missing = [ds for ds in registered if ds not in networks]
    evaluable = not missing and all(n["evaluable"] for n in networks.values())
    met = evaluable and all(n["interval_rule_met"] for n in networks.values())
    return dict(evaluable=evaluable, interval_rule_met=met, networks_missing=missing,
                p_value=max(n["p_value"] for n in networks.values()) if evaluable else None,
                contradicted={ds: n["contradicted"] for ds, n in networks.items() if n["contradicted"]},
                status_before_holm=(NOT_EVALUABLE if not evaluable else
                                    "interval rule met on every network" if met else "interval rule not met"))


def h3a_check(label: str, admission: dict, *, K: int, attack_edges: int, edge_cap_units: float, arms: tuple) -> tuple:
    """(groups checked, violations) of one capped-BTP run: in every arm and group, the fakes in Tier >= 1
    (admitted_per_group) are at most guarantee_per_group, which must equal floor(K g edge_cap_units) (H3a)."""
    bound = math.floor(K * attack_edges * edge_cap_units + CAP_TOL)
    n_groups, violations = 0, []

    def flag(arm, reason, group=None, admitted=None, guarantee=None):
        violations.append(dict(run=label, arm=arm, group=group, admitted=admitted, guarantee=guarantee, bound=bound,
                               reason=reason))
    for arm in arms:
        record = admission.get(arm)
        if record is None:
            flag(arm, "no admission record for this arm")
            continue
        guarantee = record.get("guarantee_per_group")
        if guarantee is None:
            flag(arm, "no guarantee_per_group recorded in a capped-BTP run")
            continue
        if guarantee != bound:
            flag(arm, f"recorded guarantee {guarantee} differs from floor(K g edge_cap_units) = {bound}",
                 guarantee=guarantee)
        for group, admitted in enumerate(record.get("admitted_per_group") or []):
            n_groups += 1
            if admitted > min(guarantee, bound):
                flag(arm, "fakes in Tier >= 1 exceed the guarantee", group, int(admitted), int(guarantee))
    return n_groups, violations
