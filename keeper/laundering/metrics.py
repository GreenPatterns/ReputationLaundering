"""Proposal §6 metrics, plus amendments A2 (counting baseline) and A3 (strata).

Conventions:
* An undefined quantity (0/0, empty stratum) is ``None``, never 0.
* "gain" is the change in the target's Panel reputation, P(trust | T), averaged
  over the frozen probe sources (the laundering-gain oracle).
* Strata (A3): ``pos`` means the target has at least 1 trust in-edge, so it has
  the positive history the threat model assumes it keeps; ``nopos`` means none.
"""
import numpy as np

STRATA = ("all", "pos", "nopos")


def safe_div(a, b):
    return None if b in (0, None) or a is None else float(a) / float(b)


def mean_or_none(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if len(x) else None


# ------------------------------------------------------ counting baseline (A2)
def counting_score(n_pos, n_neg, kind: str = "laplace"):
    """Non-learned reputation from in-edge counts.

    laplace (PRIMARY, A2): (n_pos + 1) / (n_pos + n_neg + 2), defined everywhere
        and bounded like a probability.
    raw (secondary): n_pos / (n_pos + n_neg), with 0/0 scored as 1.0, the
        convention of the original probe (a target with no ratings left reads as
        fully trusted).
    """
    n_pos = np.asarray(n_pos, float)
    n_neg = np.asarray(n_neg, float)
    if kind == "laplace":
        return (n_pos + 1.0) / (n_pos + n_neg + 2.0)
    if kind == "raw":
        tot = n_pos + n_neg
        return np.where(tot > 0, n_pos / np.maximum(tot, 1.0), 1.0)
    raise ValueError(kind)


def counting_gain(n_pos, n_neg, n_neg_erased, kind: str = "laplace"):
    """Counting-score change when ``n_neg_erased`` distrust in-edges are erased."""
    n_neg = np.asarray(n_neg, float)
    after = n_neg - np.asarray(n_neg_erased, float)
    return counting_score(n_pos, after, kind) - counting_score(n_pos, n_neg, kind)


# ------------------------------------------------------------------ flip rates
def flip_rates(r0, r_new, threshold: float = 0.5) -> dict:
    """§6 flip rate and its conditional variant.

    literal (PRIMARY, the proposal's definition): the fraction of distrust-
        majority targets whose prediction crosses to trust, over ALL targets.
    conditional (secondary): the same crossings over the targets that were
        below the threshold before erasure.
    """
    r0 = np.asarray(r0, float)
    r_new = np.asarray(r_new, float)
    below = r0 < threshold
    crossed = below & (r_new >= threshold)
    return dict(threshold=float(threshold), n=int(len(r0)), n_below=int(below.sum()),
                n_crossed=int(crossed.sum()),
                literal=safe_div(int(crossed.sum()), len(r0)),
                conditional=safe_div(int(crossed.sum()), int(below.sum())))


# ----------------------------------------------------------------- percentile
def platform_percentile(score: float, all_scores: np.ndarray) -> float:
    """T's trust percentile among ALL active nodes scored by the SAME model.

    Ranking against the distrust-majority cohort alone saturates at 1.0 after
    laundering (audit M3/TM-11); the platform-wide rank is what a user sees.
    """
    all_scores = np.asarray(all_scores, float)
    return float((all_scores < score).mean())


# ------------------------------------------------------- amplification, share
def amplification(model_gains, count_gains):
    """Mean learned gain divided by mean counting gain (both over the same targets)."""
    return safe_div(mean_or_none(model_gains), mean_or_none(count_gains))


def indirect_share(indirect_gains, direct_gains):
    """Mean indirect (C3) gain over mean direct gain, both under the SAME operator.

    Targets without any C3 candidate contribute an indirect gain of 0: the
    channel is genuinely unavailable to the attacker there.
    """
    return safe_div(mean_or_none(indirect_gains), mean_or_none(direct_gains))


# ------------------------------------------------------------------- strata
def stratum_of(n_pos: int) -> str:
    return "pos" if n_pos >= 1 else "nopos"


def by_stratum(values, strata) -> dict:
    """Mean of ``values`` per stratum, with counts. ``None`` values are skipped."""
    values = list(values)
    strata = list(strata)
    out = {}
    for h in STRATA:
        sel = [v for v, s in zip(values, strata) if (h == "all" or s == h) and v is not None]
        out[h] = dict(mean=mean_or_none(sel), n=len(sel),
                      sd=float(np.std(sel, ddof=1)) if len(sel) > 1 else None)
    return out


def post_stratified_mean(values, strata, stratum_sizes: dict):
    """Cohort-level mean from a stratified sample: sum_h (N_h / N) * mean_h.

    The probe samples the positive-history stratum at a higher rate than its
    cohort share (A3), so a plain sample mean would over-weight it.
    """
    N = sum(stratum_sizes.values())
    tot, used = 0.0, 0
    for h in ("pos", "nopos"):
        vals = [v for v, s in zip(values, strata) if s == h and v is not None]
        if stratum_sizes.get(h, 0) == 0:
            continue
        if not vals:
            return None                      # a populated stratum has no data
        tot += stratum_sizes[h] / N * float(np.mean(vals))
        used += stratum_sizes[h]
    return float(tot) if used == N else None


def post_stratified_flip(r0, r_new, strata, stratum_sizes: dict,
                         threshold: float = 0.5, z: float = 1.96) -> dict:
    """Cohort flip rate (literal §6 definition) estimated from a stratified sample.

    p_hat = sum_h W_h p_h with W_h = N_h / N, and the stratified-sampling variance
    sum_h W_h^2 p_h (1 - p_h) / n_h (finite-population corrected). Used when
    exact retraining of the whole cohort is infeasible (Epinions), and reported
    next to the full-cohort count wherever both exist.
    """
    r0 = np.asarray(r0, float)
    r_new = np.asarray(r_new, float)
    strata = np.asarray(strata)
    N = sum(stratum_sizes.values())
    est, var, per = 0.0, 0.0, {}
    for h, Nh in stratum_sizes.items():
        sel = strata == h
        nh = int(sel.sum())
        if Nh == 0:
            continue
        if nh == 0:
            return dict(estimate=None, ci95=None, per_stratum=per,
                        note=f"stratum {h} has cohort members but no sampled target")
        ph = float(((r0[sel] < threshold) & (r_new[sel] >= threshold)).mean())
        Wh = Nh / N
        fpc = (Nh - nh) / (Nh - 1) if Nh > 1 else 0.0
        est += Wh * ph
        var += Wh * Wh * ph * (1 - ph) / nh * fpc
        per[h] = dict(p=ph, n=nh, N=int(Nh))
    se = float(np.sqrt(var))
    return dict(estimate=float(est), ci95=[max(0.0, est - z * se), min(1.0, est + z * se)],
                se=se, per_stratum=per, threshold=float(threshold))


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion (None if n == 0)."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]
