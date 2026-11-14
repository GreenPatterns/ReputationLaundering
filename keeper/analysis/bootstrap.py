"""Confidence intervals over seeds and targets: crossed and nested bootstraps, seed-t and exact intervals.

The probe draws the same targets in every seed, so seeds and targets are crossed factors (ruling R29). One
crossed replicate resamples the seeds with replacement and, independently, the target set once; every
resampled seed then uses that same target resample. One nested replicate (ruling R21) resamples the seeds
and then, inside each drawn seed, that seed's targets independently. A target that scores high in every
seed moves the crossed statistic in all seeds at once, whereas the nested replicate averages its draws over
seeds, so the nested interval is narrower when targets carry most of the variance. Both intervals are
percentile intervals around the statistic on the data.

With five seeds the percentile bootstrap understates the seed variance, so ruling R30 adds a t-interval
over the per-seed statistics (:func:`seed_t_interval`, :func:`ratio_seed_t_interval`); the analysis
reports the hull of the two. A proportion of exactly 0 or 1 gets the exact Clopper-Pearson interval
instead, because every resample repeats it.

crossed_replicates and nested_replicates return the replicates themselves, with the drawn seeds, for bootstrap p-values
and for statistics that compare the drawn seeds (a leave-one-seed-out choice, Task 4.2b); the two bootstrap functions
are their percentile step.

Data: seed -> {target: value or row} ("cells"; a target absent from a seed is a missing cell, e.g. a target
outside a subgroup whose membership changes with the seed), or seed -> list. The crossed design needs
cells keyed by target whenever a seed has more than one target; it never aligns a list by position. A
per-seed statistic maps a batch of target sets [B, n, n_cols] to [B] (or [B, k]) and skips missing cells
(NaN); a combiner maps per-seed values [B, S] (or [B, S, k]) to [B] and skips a seed with no cell.
"""
import warnings

import numpy as np
from scipy.stats import beta, t as student_t

N_BOOT = 10_000
ALPHA = 0.05
CHUNK = 1_000                     # crossed replicates evaluated together (memory: CHUNK * S * T * cols)
MAX_UNDEFINED = 0.01              # refuse an interval when more than 1% of its replicates are undefined
DESIGNS = ("crossed", "nested")


def mean_stat(batch: np.ndarray) -> np.ndarray:
    """Per-seed mean of column 0."""
    return np.nanmean(batch[:, :, 0], axis=1)


def median_stat(batch: np.ndarray) -> np.ndarray:
    """Per-seed median of column 0."""
    return np.nanmedian(batch[:, :, 0], axis=1)


def ratio_stat(batch: np.ndarray) -> np.ndarray:
    """Per-seed ratio of means, column 0 over column 1 (amplification: model gain over yardstick shift)."""
    return np.nanmean(batch[:, :, 0], axis=1) / np.nanmean(batch[:, :, 1], axis=1)


def column_means_stat(batch: np.ndarray) -> np.ndarray:
    """Per-seed mean of every column, [B, n_cols]; pairs with :func:`combine_ratio`."""
    return np.nanmean(batch, axis=1)


def combine_mean(per_seed: np.ndarray) -> np.ndarray:
    """Mean over seeds of a scalar per-seed statistic."""
    return np.nanmean(per_seed, axis=1)


def combine_ratio(per_seed: np.ndarray) -> np.ndarray:
    """Ratio of seed-averaged means: mean over seeds of column 0 over mean over seeds of column 1."""
    return np.nanmean(per_seed[:, :, 0], axis=1) / np.nanmean(per_seed[:, :, 1], axis=1)


# ------------------------------------------------------------------------ data layout
def to_cells(groups) -> dict:
    """seed -> {target: 1-D row}; a list is keyed by position; seeds without a cell are dropped."""
    out = {}
    for seed in sorted(groups):
        g = groups[seed]
        items = g.items() if isinstance(g, dict) else enumerate(g)
        cells = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in items}
        if cells:
            out[seed] = cells
    if not out:
        raise ValueError("bootstrap: every seed is empty")
    return out


def is_keyed(groups) -> bool:
    """True if targets can be aligned across seeds: cells keyed by target, or one value per seed."""
    return all(isinstance(g, dict) for g in groups.values()) or all(len(g) <= 1 for g in groups.values())


def resolve_design(groups, design: str) -> str:
    """design="auto": crossed for aligned data, else the nested fallback (targets cannot be aligned)."""
    if design == "auto":
        return "crossed" if is_keyed(groups) else "nested"
    if design not in DESIGNS:
        raise ValueError(f"design must be 'auto' or one of {DESIGNS}, got {design!r}")
    return design


def _seed_arrays(cells: dict) -> list:
    """Each seed's rows as a 2-D array, in target order (the nested design ignores target identity)."""
    return [np.stack([c[k] for k in sorted(c)]) for c in cells.values()]


def _cell_array(groups) -> np.ndarray:
    """[S, T, n_cols] array over the seeds and the union of targets, NaN for a missing cell."""
    if not is_keyed(groups):
        raise ValueError("crossed bootstrap: a seed has several targets but they are not keyed by target; "
                         "pass seed -> {target: value} (a list is never aligned by position)")
    cells = to_cells(groups)
    targets = sorted(set().union(*(c.keys() for c in cells.values())))
    width = len(next(iter(next(iter(cells.values())).values())))
    X = np.full((len(cells), len(targets), width), np.nan)
    index = {k: j for j, k in enumerate(targets)}
    for i, c in enumerate(cells.values()):
        for k, row in c.items():
            X[i, index[k]] = row
    return X


def per_seed_values(groups, stat=mean_stat) -> list:
    """The per-seed statistic on the data, in seed order; None for an empty seed."""
    out = []
    for seed in sorted(groups):
        g = groups[seed]
        if not len(g):
            out.append(None)
            continue
        val = stat(_seed_arrays(to_cells({seed: g}))[0][None])[0]
        out.append(float(val) if np.ndim(val) == 0 else [float(x) for x in val])
    return out


def n_units(groups, unit: str) -> int:
    """Distinct units behind a 0/1 field: "targets" (keys over all seeds) or "evaluations" (all cells)."""
    if unit == "targets":
        return len(set().union(*(set(g.keys() if isinstance(g, dict) else range(len(g)))
                                 for g in groups.values())))
    if unit == "evaluations":
        return sum(len(g) for g in groups.values())
    raise ValueError(f"unit must be 'targets' or 'evaluations', got {unit!r}")


# ------------------------------------------------------------------------ bootstraps
def _undefined(vals: np.ndarray, has: np.ndarray) -> np.ndarray:
    """Replicates with a drawn seed that has cells but no finite statistic (0/0), or with no cell at all."""
    finite = np.isfinite(vals) if vals.ndim == 2 else np.all(np.isfinite(vals), axis=2)
    return np.any(has & ~finite, axis=1) | ~np.any(has, axis=1)


def percentile_interval(boot, alpha: float = ALPHA, info: dict | None = None) -> list:
    """[lo, hi]: the percentile (1 - alpha) interval of the defined (finite) replicates. ``info`` receives the share
    of undefined replicates; more than MAX_UNDEFINED of them refuses the interval (FloatingPointError)."""
    boot = np.asarray(boot, dtype=float)
    defined = np.isfinite(boot)
    share = float(1.0 - defined.mean())
    if info is not None:
        info["undefined_share"] = share
    if share > MAX_UNDEFINED:
        raise FloatingPointError(f"bootstrap: {share:.1%} of replicates are undefined "
                                 f"(more than {MAX_UNDEFINED:.0%})")
    lo, hi = np.percentile(boot[defined], [50 * alpha, 100 - 50 * alpha])
    return [float(lo), float(hi)]


def _finish(point: float, boot: np.ndarray, alpha: float, info: dict | None) -> tuple:
    lo, hi = percentile_interval(boot, alpha, info)
    return point, lo, hi


def _combined(combine, per_seed: np.ndarray, drawn: np.ndarray, seed_aware: bool) -> np.ndarray:
    return combine(per_seed, drawn) if seed_aware else combine(per_seed)


def nested_replicates(groups, stat=mean_stat, combine=combine_mean, n_boot: int = N_BOOT, seed=0,
                      seed_aware: bool = False) -> tuple:
    """(point, replicates, drawn) of the nested design (ruling R21), before the percentile step.

    ``replicates`` [n_boot] holds NaN for an undefined replicate; ``drawn`` [n_boot, S] gives each replicate's seeds
    as positions in sorted seed order. With ``seed_aware``, combine is called as combine(per_seed, drawn) and decides
    alone which replicates are undefined: a statistic that compares the drawn seeds with each other (a
    leave-one-seed-out choice) needs their identities, and a missing cell is then not an undefined replicate.
    """
    arrays = _seed_arrays(to_cells(groups))
    n_seeds = len(arrays)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a 0/0 per-seed ratio is counted, not hidden
        on_data = np.stack([stat(a[None]) for a in arrays], axis=1)
        point = float(_combined(combine, on_data, np.arange(n_seeds)[None], seed_aware)[0])
        rng = np.random.default_rng(seed)
        pick = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
        per_seed = None
        for i, a in enumerate(arrays):
            where = pick == i
            count = int(where.sum())
            if count == 0:
                continue
            vals = stat(a[rng.integers(0, len(a), size=(count, len(a)))])
            if per_seed is None:
                per_seed = np.empty((n_boot, n_seeds) + vals.shape[1:])
            per_seed[where] = vals
        boot = _combined(combine, per_seed, pick, seed_aware)
    if not seed_aware:
        boot[_undefined(per_seed, np.ones((n_boot, n_seeds), bool))] = np.nan
    return point, boot, pick


def nested_bootstrap(groups, stat=mean_stat, combine=combine_mean, n_boot: int = N_BOOT, seed=0,
                     alpha: float = ALPHA, info: dict | None = None) -> tuple:
    """(point, lo, hi) with seeds resampled, then targets within each drawn seed (ruling R21)."""
    point, boot, _ = nested_replicates(groups, stat, combine, n_boot=n_boot, seed=seed)
    return _finish(point, boot, alpha, info)


def crossed_replicates(groups, stat=mean_stat, combine=combine_mean, n_boot: int = N_BOOT, seed=0,
                       seed_aware: bool = False) -> tuple:
    """(point, replicates, drawn) of the crossed design (ruling R29), before the percentile step; ``replicates``,
    ``drawn`` and ``seed_aware`` as in :func:`nested_replicates`. The draws are crossed_bootstrap's, in its order."""
    X = _cell_array(groups)
    n_seeds, n_targets = X.shape[:2]
    drawn = np.empty((n_boot, n_seeds), dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a resampled seed may lack a cell
        point = float(_combined(combine, stat(X)[None], np.arange(n_seeds)[None], seed_aware)[0])
        rng = np.random.default_rng(seed)
        boot = np.empty(n_boot)
        for start in range(0, n_boot, CHUNK):
            b = min(CHUNK, n_boot - start)
            seeds = rng.integers(0, n_seeds, size=(b, n_seeds))
            targets = rng.integers(0, n_targets, size=(b, n_targets))
            sub = X[seeds[:, :, None], targets[:, None, :]]      # [b, S, T, n_cols]
            vals = stat(sub.reshape(b * n_seeds, n_targets, -1))
            vals = vals.reshape((b, n_seeds) + vals.shape[1:])
            chunk = _combined(combine, vals, seeds, seed_aware)
            if not seed_aware:
                chunk[_undefined(vals, ~np.all(np.isnan(sub[..., 0]), axis=2))] = np.nan
            boot[start:start + b] = chunk
            drawn[start:start + b] = seeds
    return point, boot, drawn


def crossed_bootstrap(groups, stat=mean_stat, combine=combine_mean, n_boot: int = N_BOOT, seed=0,
                      alpha: float = ALPHA, info: dict | None = None) -> tuple:
    """(point, lo, hi) with seeds and, independently, the target set resampled once (ruling R29)."""
    point, boot, _ = crossed_replicates(groups, stat, combine, n_boot=n_boot, seed=seed)
    return _finish(point, boot, alpha, info)


def hier_bootstrap(groups, stat=mean_stat, combine=combine_mean, n_boot: int = N_BOOT, seed=0,
                   alpha: float = ALPHA, *, design: str, info: dict | None = None) -> tuple:
    """(point, lo, hi): the statistic on the data and its percentile (1 - alpha) interval.

    ``design`` is required: "crossed" (ruling R29) needs targets aligned across seeds (cells keyed by
    target, or one value per seed); "nested" (ruling R21) needs neither. A library default could switch
    designs silently, which changes the interval. ``seed`` is anything numpy.random.default_rng accepts.
    ``info`` receives the design that ran and the share of undefined replicates, which are dropped; more
    than MAX_UNDEFINED of them refuses the interval.
    """
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, got {design!r} (only hier_bootstrap_mean "
                         "accepts 'auto')")
    if info is not None:
        info["design"] = design
    run = crossed_bootstrap if design == "crossed" else nested_bootstrap
    return run(groups, stat, combine, n_boot=n_boot, seed=seed, alpha=alpha, info=info)


def hier_bootstrap_mean(data, n_boot: int = N_BOOT, seed=0, design: str = "auto",
                        info: dict | None = None) -> tuple:
    """(mean of per-seed means, lo, hi).

    ``data`` maps seed -> {target: value} (crossed, ruling R29) or seed -> values. design="auto" runs the
    crossed design for aligned data and falls back to the nested one (ruling R21) for unkeyed lists of
    several values per seed, whose targets cannot be aligned; ``info["design"]`` records which ran. Pass
    design="crossed" to require alignment.
    """
    return hier_bootstrap(data, mean_stat, combine_mean, n_boot=n_boot, seed=seed,
                          design=resolve_design(data, design), info=info)


# ------------------------------------------------------------------------ seed-t and exact intervals
def _t_quantile(n: int, alpha: float) -> float:
    return float(student_t.ppf(1 - alpha / 2, n - 1))


def seed_t_interval(values, alpha: float = ALPHA) -> list | None:
    """t-interval over per-seed statistics (df = n - 1) centred on their mean; None with fewer than 2."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) < 2:
        return None
    half = _t_quantile(len(v), alpha) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean() - half), float(v.mean() + half)]


def ratio_seed_t_interval(num, den, alpha: float = ALPHA) -> list | None:
    """t-interval for R = mean(num) / mean(den) over seeds, centred on R.

    Linearized: the per-seed residuals (num_s - R den_s) / mean(den) average to zero, and their standard
    error is that of R (df = n - 1). None with fewer than 2 seeds.
    """
    pairs = [(a, b) for a, b in zip(num, den) if a is not None and b is not None]
    if len(pairs) < 2:
        return None
    a, b = (np.asarray(x, dtype=float) for x in zip(*pairs))
    ratio = a.mean() / b.mean()
    half = _t_quantile(len(a), alpha) * ((a - ratio * b) / b.mean()).std(ddof=1) / np.sqrt(len(a))
    return [float(ratio - half), float(ratio + half)]


def hull(*intervals) -> list | None:
    """Smallest interval containing every given interval (None entries skipped)."""
    given = [iv for iv in intervals if iv is not None]
    return None if not given else [min(iv[0] for iv in given), max(iv[1] for iv in given)]


def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> list:
    """Exact binomial (Clopper-Pearson) interval for k successes in n trials."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"clopper_pearson: need 0 <= k <= n and n > 0, got k={k}, n={n}")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]
