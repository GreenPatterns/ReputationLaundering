"""Re-analysis of the stored laundering results: edgeless reset, percentiles, yardsticks and CIs (Task 2.1).

No model is trained or run. The module reads the stored probe4, baselines and calibration records in
results/laundering/ and rebuilds each dataset's pre-deletion training graph with the dataset loader of
keeper.laundering.core, which is data processing only (CPU). It ports the archival review scripts
docs/reviews/2026-10-26/scripts/review_{probe4,isolated,isolated2,amp_split}.py, which stay unchanged,
and adds a learned no-message-passing yardstick (ruling R20), partial deletion, the logit terms, the
maximum utility change and hierarchical-bootstrap confidence intervals (ruling R21).

Conventions, as in docs/reviews/2026-10-26/review_evidence.md:
* An evaluation is one (seed, sampled target). Calibration targets are excluded, except in the SGCN
  share, which pools all 50 targets per seed as the paper's SGCN rows do.
* U1 is inference-only unlinking (operator "unlink"), U2 exact retraining (operator "exact_retrain").
* Amplification is the U2 gain over a yardstick's shift: a ratio of means within each seed, averaged
  over seeds. The U1/U2 shares are ratios of five-seed means, as in the paper.

Run from the repository root, CPU only:
  CUDA_VISIBLE_DEVICES="" python -m keeper.analysis.laundering_extra [--force]
It writes results/analysis/laundering_extra.json and results/analysis/laundering_extra.md.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import socket
import sys
import warnings
from dataclasses import dataclass

import numpy as np

from keeper.analysis.bootstrap import (N_BOOT, column_means_stat, combine_ratio, hier_bootstrap_mean,
                                       median_stat, per_seed_values, ratio_stat)
from keeper.analysis.intervals import BOOT_SEED, NON_NEGATIVE, PROPORTION, Block, Boot, ci_design_for
from keeper.analysis.provenance import git_state
from keeper.laundering import core as C
from keeper.laundering import signed as S

__all__ = ["laplace_shift", "raw_shift", "base_rate_shift", "hier_bootstrap_mean", "edgeless_after_c2",
           "escaped_from_bottom", "degree_table", "edge_degree_counts", "degrees_after_c2", "DegreeLogReg",
           "fit_degree_logreg", "panel_score", "c2_logreg_shift", "target_rows", "partial_deletion_values",
           "Boot", "ci_design_for", "amplification_block", "analyse_dataset", "main"]

DATASETS = ("bitcoinalpha", "bitcoinotc", "wiki-rfa", "epinions-dated")
SEEDS = tuple(range(5))
MAX_NEG = 3                       # cohort: strict distrust majority on at most 3 distrust in-edges (probe4)
N_PROBE, PANEL_SEED = 256, 1234   # the proxy's frozen probe panel (probe4: C.Panel(Tr, 256, seed=1234))
PRIOR_STRENGTH = 2.0              # base-rate prior: Beta(s p, s (1 - p)) with s = 2
ESCAPE_LEVELS = {"escape_bottom5": 0.05, "escape_bottom10": 0.10}
BOTTOM_QUARTILE = 0.25
RATER_TERM_BAR = 0.003            # the paper's bound on the rater term of the logit
COUNT_GAIN_TOL = 1e-12
LOGREG_MAX_ITER = 1000
LAUNDERING = "results/laundering"
ANALYSIS = os.path.join(C.ROOT, "results", "analysis")
CENSUS_FILES = ("results/laundering/dataset_census.json", "results/legacy/dataset_census.json")
FEATURES = ("log1p pos_in(v)", "log1p neg_in(v)", "log1p pos_out(u)", "log1p neg_out(u)")
MODULE_FILES = ("keeper/analysis/laundering_extra.py", "keeper/analysis/laundering_extra_md.py",
                "keeper/analysis/bootstrap.py", "keeper/analysis/intervals.py", "keeper/analysis/provenance.py")


# ------------------------------------------------------------------------ counting yardsticks
def _out(a):
    return float(a) if np.ndim(a) == 0 else a


def base_rate_shift(npos, nneg, p, strength: float = PRIOR_STRENGTH):
    """Change of (npos + s p) / (npos + nneg + s) when all nneg distrust in-edges are deleted."""
    npos, nneg = np.asarray(npos, float), np.asarray(nneg, float)
    a = npos + strength * p
    return _out(a / (npos + strength) - a / (npos + nneg + strength))


def laplace_shift(npos, nneg):
    """Laplace (Beta(1, 1)) yardstick: (npos + 1) / (npos + nneg + 2), all distrust deleted."""
    return base_rate_shift(npos, nneg, 0.5, 2.0)


def _raw_score(npos, nneg):
    tot = npos + nneg
    return np.where(tot > 0, npos / np.maximum(tot, 1.0), 1.0)


def raw_shift(npos, nneg):
    """Raw fraction npos / (npos + nneg), with 0/0 read as 1.0 (metrics.counting_score), all deleted."""
    npos, nneg = np.asarray(npos, float), np.asarray(nneg, float)
    return _out(_raw_score(npos, 0.0 * nneg) - _raw_score(npos, nneg))


# ------------------------------------------------------------------------ graph facts
def edgeless_after_c2(edges, is_neg, nodes, n_nodes: int) -> np.ndarray:
    """For each node, True if deleting all its distrust in-edges (channel C2) leaves it no edge.

    Kept after C2: the node's out-edges (a distrust self-loop is also one of its distrust in-edges, so it
    goes) and its trust in-edges. Without self-loops this is review_isolated.py's test: no out-edge and
    no trust in-edge in the pre-deletion training graph.
    """
    edges = np.asarray(edges, dtype=np.int64)
    is_neg = np.asarray(is_neg, bool)
    src, dst = edges[:, 0], edges[:, 1]
    out_kept = np.bincount(src[~(is_neg & (src == dst))], minlength=n_nodes)
    in_kept = np.bincount(dst[~is_neg], minlength=n_nodes)
    nodes = np.asarray(nodes, dtype=np.int64)
    return (out_kept[nodes] == 0) & (in_kept[nodes] == 0)


def escaped_from_bottom(pct_before, pct_after, q: float) -> np.ndarray:
    """Escape indicators of the targets that start below platform percentile q: ending at q or above."""
    before, after = np.asarray(pct_before, float), np.asarray(pct_after, float)
    return after[before < q] >= q


# ------------------------------------------------------------------ logistic-regression yardstick
def degree_table(edges, is_pos, n_nodes: int) -> np.ndarray:
    """[n_nodes, 4] signed degrees: trust in, distrust in, trust out, distrust out."""
    edges = np.asarray(edges, dtype=np.int64)
    w = np.asarray(is_pos, float)
    src, dst = edges[:, 0], edges[:, 1]
    return np.column_stack([np.bincount(dst, weights=w, minlength=n_nodes),
                            np.bincount(dst, weights=1.0 - w, minlength=n_nodes),
                            np.bincount(src, weights=w, minlength=n_nodes),
                            np.bincount(src, weights=1.0 - w, minlength=n_nodes)])


def edge_degree_counts(edges, is_pos, n_nodes: int, loo: str = "in_and_out") -> np.ndarray:
    """[E, 4] training counts (d+in(v), d-in(v), d+out(u), d-out(u)) for each edge u -> v.

    The edge's own label must not reach its features. loo="in_and_out" (ruling R28, the headline) leaves
    it out of v's in-degrees and of u's out-degrees; loo="in" (ruling R20, kept as a sensitivity
    variant) leaves it out of v's in-degrees only, so u's out-degrees still carry it.
    """
    if loo not in ("in", "in_and_out"):
        raise ValueError(f"loo must be 'in' or 'in_and_out', got {loo!r}")
    edges = np.asarray(edges, dtype=np.int64)
    pos = np.asarray(is_pos, float)
    neg = 1.0 - pos
    deg = degree_table(edges, is_pos, n_nodes)
    src, dst = edges[:, 0], edges[:, 1]
    own_out = 1.0 if loo == "in_and_out" else 0.0
    counts = np.column_stack([deg[dst, 0] - pos, deg[dst, 1] - neg,
                              deg[src, 2] - own_out * pos, deg[src, 3] - own_out * neg])
    assert (counts >= 0).all(), "leave-one-out produced a negative degree"
    return counts


def degrees_after_c2(deg: np.ndarray, edges, is_neg, v: int) -> np.ndarray:
    """A copy of the degree table with v's distrust in-edges deleted, at both of their endpoints."""
    edges = np.asarray(edges, dtype=np.int64)
    rows = np.where((edges[:, 1] == v) & np.asarray(is_neg, bool))[0]
    out = np.array(deg, dtype=float, copy=True)
    out[v, 1] -= len(rows)
    np.subtract.at(out[:, 3], edges[rows, 0], 1.0)
    return out


@dataclass
class DegreeLogReg:
    """P(trust | u -> v) = sigmoid(coef . log1p(counts) + intercept) over the four signed degrees."""
    coef: np.ndarray
    intercept: float
    loo: str = "in_and_out"
    n_train: int = 0
    n_iter: int = 0
    max_iter: int = LOGREG_MAX_ITER

    def prob(self, counts) -> np.ndarray:
        z = np.log1p(np.asarray(counts, float)) @ np.asarray(self.coef, float) + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def summary(self) -> dict:
        return dict(features=list(FEATURES), coef=[float(x) for x in self.coef],
                    intercept=float(self.intercept), loo=self.loo, n_train=int(self.n_train),
                    n_iter=int(self.n_iter), solver="lbfgs", l2_C=1.0)


def fit_degree_logreg(edges, is_pos, n_nodes: int, loo: str = "in_and_out",
                      max_iter: int = LOGREG_MAX_ITER) -> DegreeLogReg:
    """Fit the yardstick on the training edges (scikit-learn, L2 with C = 1, lbfgs); refuse non-convergence."""
    from sklearn.linear_model import LogisticRegression
    X = np.log1p(edge_degree_counts(edges, is_pos, n_nodes, loo))
    y = np.asarray(is_pos, dtype=np.int64)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=max_iter).fit(X, y)
    n_iter = int(lr.n_iter_[0])
    if n_iter >= max_iter:
        raise RuntimeError(f"degree logistic regression did not converge in {max_iter} iterations")
    return DegreeLogReg(coef=lr.coef_[0].copy(), intercept=float(lr.intercept_[0]), loo=loo,
                        n_train=len(y), n_iter=n_iter, max_iter=max_iter)


def panel_score(model: DegreeLogReg, deg: np.ndarray, v: int, sources) -> float:
    """Mean P(trust | u -> v) over the probe sources u, the self-pair u = v dropped (as C.Panel does)."""
    src = np.asarray(sources, dtype=np.int64)
    src = src[src != v]
    counts = np.column_stack([np.full(len(src), deg[v, 0]), np.full(len(src), deg[v, 1]),
                              deg[src, 2], deg[src, 3]])
    return float(model.prob(counts).mean())


def c2_logreg_shift(model: DegreeLogReg, deg: np.ndarray, edges, is_neg, v: int, sources) -> float:
    """Yardstick shift of v: its panel score after its distrust in-edges leave the degrees, minus before."""
    after = degrees_after_c2(deg, edges, is_neg, v)
    return panel_score(model, after, v, sources) - panel_score(model, deg, v, sources)


# ------------------------------------------------------------------------ probe4 records
def _check_count_gain(t: dict) -> None:
    want = dict(laplace=laplace_shift(t["n_pos"], t["n_neg"]), raw=raw_shift(t["n_pos"], t["n_neg"]))
    for kind, value in want.items():
        if abs(t["count_gain"][kind] - value) > COUNT_GAIN_TOL:
            raise ValueError(f"target {t['target']}: stored count_gain[{kind}] = {t['count_gain'][kind]} "
                             f"differs from the ported yardstick {value}")


def _flat_row(t: dict, edgeless) -> dict:
    ops = t["direct"]["ops"]
    u1, u1x, u2 = ops["unlink"], ops["unlink_frozen_x"], ops["exact_retrain"]
    _check_count_gain(t)
    placebo = ((t.get("placebo") or {}).get("ops") or {}).get("exact_retrain")
    curve = t.get("direct_greedy_curve") or []
    return dict(target=int(t["target"]), n_neg=int(t["n_neg"]), n_pos=int(t["n_pos"]),
                stratum=t["stratum"], edgeless=int(t["target"]) in edgeless, r0=t["r0"],
                pct_before=t["pct_before"], g_u1=u1["gain"], g_u2=u2["gain"], g_u1_frozen=u1x["gain"],
                rep_u1=u1["rep"], rep_u2=u2["rep"], pct_u1=u1["pct_after"], pct_u2=u2["pct_after"],
                d_src_u1=u1["d_src"], d_tgt_u1=u1["d_tgt"], dmcc_u2=u2["d_t1_mcc"],
                dmcc_placebo_u2=None if placebo is None else placebo["d_t1_mcc"],
                lap=t["count_gain"]["laplace"], raw=t["count_gain"]["raw"],
                curve={int(c["k"]): c["unlink_gain"] for c in curve},
                curve_short=sum(1 for c in curve if c["cost"]["edges"] != c["k"]))


def target_rows(record: dict, calibration, edgeless) -> list:
    """One flat row per sampled target of a probe4 record, calibration targets excluded."""
    return [_flat_row(t, edgeless) for t in record["targets"] if t["target"] not in calibration]


PARTIAL_KEYS = (("k1_of_2", 2, 1), ("k1_of_3", 3, 1), ("k2_of_3", 3, 2))


def partial_deletion_values(rows: list) -> dict:
    """Unlink gains after deleting k' of a target's k warnings (greedy order, stored curves), by target.

    The curve's gains come from attack.Scorer, which keeps the trained input features; the full-deletion
    reference full_of_k is therefore the stored unlink_frozen_x gain of the same targets.
    """
    out = {key: {} for key, _, _ in PARTIAL_KEYS}
    out.update(full_of_2={}, full_of_3={})
    for r in rows:
        k = r["n_neg"]
        if k < 2:
            continue
        if k > 3 or sorted(r["curve"]) != list(range(1, k)):
            raise ValueError(f"target {r['target']}: greedy curve has k' = {sorted(r['curve'])}, "
                             f"expected 1..{k - 1}")
        for key, kk, kp in PARTIAL_KEYS:
            if kk == k:
                out[key][r["target"]] = r["curve"][kp]
        out[f"full_of_{k}"][r["target"]] = r["g_u1_frozen"]
    return out


# ------------------------------------------------------------------------ bootstrap plumbing
def column(rows_by_seed: dict, key: str, where=None) -> dict:
    """seed -> {target: row[key]} over the rows that pass ``where``: cells for the crossed bootstrap."""
    return {s: {r["target"]: r[key] for r in rows if where is None or where(r)}
            for s, rows in rows_by_seed.items()}


def columns(rows_by_seed: dict, fns, where=None) -> dict:
    """seed -> {target: [f(row) for f in fns]} over the rows that pass ``where``."""
    return {s: {r["target"]: [f(r) for f in fns] for r in rows if where is None or where(r)}
            for s, rows in rows_by_seed.items()}


def _g2(r):
    return r["g_u2"]


# ------------------------------------------------------------------------ output blocks
def _spread(rows: list) -> dict:
    sel = [r for r in rows if r["edgeless"]]
    if not sel:
        return dict(n=0)
    rep, r0 = np.array([r["rep_u1"] for r in sel]), np.array([r["r0"] for r in sel])
    return dict(n=len(sel), mean=float(rep.mean()), sd=float(rep.std()), min=float(rep.min()),
                max=float(rep.max()), r0_min=float(r0.min()), r0_max=float(r0.max()))


def edgeless_block(graph: dict, rows: dict, boot: Boot) -> dict:
    """Cohort census of targets left edgeless by C2, and the cold-start proxy the sampled ones reset to."""
    b = Block("edgeless", boot)
    n, cohort = len(graph["edgeless"]), len(graph["cohort"])
    b.exact("n", n)
    b.exact("share", n / cohort)
    sel = lambda r: r["edgeless"]                                       # noqa: E731
    b.stat("cold_start_proxy_mean", column(rows, "rep_u1", sel), support=PROPORTION)
    b.stat("cold_start_pct_mean", column(rows, "pct_u1", sel), support=PROPORTION)
    b.stat("cold_start_proxy_u2_mean", column(rows, "rep_u2", sel), support=PROPORTION)
    b.extra.update(cohort=cohort, n_without_out_edges=graph["n_without_out"],
                   cold_start_proxy=b.by_seed["cold_start_proxy_mean"],
                   cold_start_pct=b.by_seed["cold_start_pct_mean"],
                   n_sampled=[len(v) for v in column(rows, "target", sel).values()],
                   spread=[_spread(rows[s]) for s in sorted(rows)])
    return b.to_dict()


def _escape_counts(rows: dict, q: float) -> dict:
    below, escaped = {}, {}
    for rs in rows.values():
        for r in rs:
            if r["pct_before"] < q:
                below[r["target"]] = below.get(r["target"], 0) + 1
                escaped[r["target"]] = escaped.get(r["target"], 0) + int(r["pct_u1"] >= q)
    always = sum(1 for v in below if escaped[v] == below[v])
    return dict(level=q, n_below=sum(below.values()), n_escaped=sum(escaped.values()),
                n_targets_below=len(below), n_targets_always_escaped=always)


def _escape_cells(rows: list, q: float) -> dict:
    """target -> 1.0 if it left the bottom q under U1, over one seed's targets that started below q."""
    below = [r for r in rows if r["pct_before"] < q]
    flags = escaped_from_bottom([r["pct_before"] for r in below], [r["pct_u1"] for r in below], q)
    return {r["target"]: float(f) for r, f in zip(below, flags)}


def percentile_block(rows: dict, boot: Boot) -> dict:
    """Platform percentile before and after deletion, and escape from the bottom 5% and 10% under U1."""
    b = Block("percentile", boot)
    b.stat("before_mean", column(rows, "pct_before"), support=PROPORTION)        # percentiles as fractions
    b.stat("after_mean_u1", column(rows, "pct_u1"), support=PROPORTION)
    b.stat("after_mean_u2", column(rows, "pct_u2"), support=PROPORTION)
    b.stat("after_median_u1", column(rows, "pct_u1"), stat=median_stat, support=PROPORTION)
    for field, q in ESCAPE_LEVELS.items():
        b.stat(field, {s: _escape_cells(rs, q) for s, rs in rows.items()}, unit="targets", support=PROPORTION)
        b.extra.setdefault("escape_counts", {})[field] = _escape_counts(rows, q)
    b.stat("frac_after_below25", {s: {r["target"]: float(r["pct_u1"] < BOTTOM_QUARTILE) for r in rs}
                                  for s, rs in rows.items()}, unit="targets", support=PROPORTION)
    med = b.by_seed["after_median_u1"]
    b.extra["after_median_u1_range"] = [min(med), max(med)]
    return b.to_dict()


COHORT_FIELDS = (("gain_u1", "g_u1"), ("gain_u2", "g_u2"), ("pct_before", "pct_before"),
                 ("pct_after_u1", "pct_u1"), ("pct_after_u2", "pct_u2"))


def post_stratified_stat(w_pos: float):
    """A statistic for Block.stat: the per-seed mean of column 0 in the positive-history stratum (column 1 is 1)
    and in the other stratum (column 1 is 0), weighted by the cohort shares w_pos and 1 - w_pos. This is the
    probe's keeper.laundering.metrics.post_stratified_mean (amendment A3: the draw over-samples the first)."""
    def stat(batch: np.ndarray) -> np.ndarray:
        value, pos = batch[:, :, 0], batch[:, :, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # a replicate without one stratum is NaN
            m_pos = np.nanmean(np.where(pos == 1.0, value, np.nan), axis=1)
            m_nop = np.nanmean(np.where(pos == 0.0, value, np.nan), axis=1)
        return w_pos * m_pos + (1.0 - w_pos) * m_nop
    return stat


def cohort_weighted_block(rows: dict, w_pos: float, boot: Boot) -> dict:
    """Cohort-level gains and platform percentiles from the stratified sample, and the percentiles per stratum.

    w_pos is the cohort share of targets with at least one endorsement (stratum "pos"). The sample holds about as
    many of them as of the others, so the plain means of the percentile block over-weight them."""
    b = Block("cohort_weighted", boot)
    for field, key in COHORT_FIELDS:
        is_pct = field.startswith("pct")
        b.stat(field, columns(rows, [lambda r, k=key: r[k], lambda r: float(r["stratum"] == "pos")]),
               stat=post_stratified_stat(w_pos), support=PROPORTION if is_pct else None)
        for h in ("pos", "nopos") if is_pct else ():
            b.stat(f"{field}_{h}", column(rows, key, lambda r, h=h: r["stratum"] == h), support=PROPORTION)
    b.extra["cohort_share_pos"] = w_pos
    return b.to_dict()


def _crossed(rows: list, tau: float, gain: str) -> int:
    return sum(1 for r in rows if r["r0"] < tau <= r["r0"] + r[gain])


def tau_block(records: dict, rows: dict, boot: Boot) -> dict:
    """MCC-optimal edge threshold per seed (base.tau) and the targets below and crossing it."""
    taus = {s: [records[s]["base"]["tau"]] for s in sorted(records)}
    fcs = {s: ((records[s].get("full_cohort") or {}).get("flip_tau") or {}).get("all") for s in sorted(records)}
    has_cohort = all(fc is not None for fc in fcs.values())
    b = Block("tau_mcc", boot)
    b.stat("mean", taus, support=PROPORTION)
    if has_cohort:                  # a mean count of cohort targets lies in [0, cohort size]
        b.stat("cohort_below_tau_mean", {s: [fc["n_below"]] for s, fc in fcs.items()},
               support=(0.0, float(fcs[0]["n"])))
    b.extra.update(per_seed=[t[0] for t in taus.values()],
                   cohort_below_tau=[fc["n_below"] for fc in fcs.values()] if has_cohort else None,
                   cohort_crossed_tau_u2=[fc["n_crossed"] for fc in fcs.values()] if has_cohort else None,
                   cohort_n=fcs[0]["n"] if has_cohort else None,
                   range=[min(t[0] for t in taus.values()), max(t[0] for t in taus.values())],
                   sample_below_tau=[sum(r["r0"] < taus[s][0] for r in rows[s]) for s in sorted(rows)],
                   sample_crossed_tau_u1=[_crossed(rows[s], taus[s][0], "g_u1") for s in sorted(rows)],
                   sample_crossed_tau_u2=[_crossed(rows[s], taus[s][0], "g_u2") for s in sorted(rows)],
                   n_eval=[len(rows[s]) for s in sorted(rows)])
    return b.to_dict()


def amplification_block(rows: dict, p_pos: float, logreg_shift: dict, boot: Boot,
                        sensitivity: dict | None = None) -> dict:
    """U2 gain over four yardstick shifts: Laplace, raw fraction, base-rate prior and degree logreg."""
    b = Block("amplification", boot)
    yard = {"laplace": lambda r: r["lap"], "raw": lambda r: r["raw"],
            "base_rate": lambda r: base_rate_shift(r["n_pos"], r["n_neg"], p_pos),
            "logreg": lambda r: logreg_shift[r["target"]]}
    for name, shift in yard.items():
        b.stat(name, columns(rows, [_g2, shift]), stat=ratio_stat, support=None)   # signed gain: no clip
    b.extra["mean_shift"] = {name: float(np.mean(per_seed_values(columns(rows, [shift]))))
                             for name, shift in yard.items()}
    b.extra["mean_gain_u2"] = float(np.mean(per_seed_values(column(rows, "g_u2"))))
    for name, shifts in (sensitivity or {}).items():       # e.g. logreg_target_loo (ruling R28)
        b.stat(name, columns(rows, [_g2, lambda r, s=shifts: s[r["target"]]]), stat=ratio_stat, support=None)
    return b.to_dict()


AMP_GROUPS = {"edgeless": lambda r: r["edgeless"], "edge_keeping": lambda r: not r["edgeless"],
              "pos": lambda r: r["stratum"] == "pos", "nopos": lambda r: r["stratum"] == "nopos"}


def amp_split_block(rows: dict, boot: Boot) -> dict:
    """Laplace amplification (U2) by group: edgeless after C2 or not, positive history or not."""
    b, gain = Block("amp_split", boot), Block("amp_split_gain_u2", boot)
    for name, sel in AMP_GROUPS.items():
        b.stat(name, columns(rows, [_g2, lambda r: r["lap"]], sel), stat=ratio_stat, support=None)
        gain.stat(name, column(rows, "g_u2", sel), support=None)
    b.extra["n"] = {name: [sum(1 for r in rows[s] if sel(r)) for s in sorted(rows)]
                    for name, sel in AMP_GROUPS.items()}
    b.extra["gain_u2"] = gain.to_dict()
    return b.to_dict()


def partial_block(rows: dict, boot: Boot) -> dict:
    """Mean unlink gain after deleting 1 of 2, 1 of 3 and 2 of 3 warnings, and its share of full deletion."""
    vals = {s: partial_deletion_values(rs) for s, rs in rows.items()}
    b, share = Block("partial_deletion_u1", boot), Block("partial_deletion_share", boot)
    for key in ("k1_of_2", "k1_of_3", "k2_of_3", "full_of_2", "full_of_3"):
        b.stat(key, {s: v[key] for s, v in vals.items()}, support=None)              # signed gains
    for key, k, _ in PARTIAL_KEYS:          # paired per target: partial gain over its full-deletion gain
        share.stat(key, {s: {t: [g, v[f"full_of_{k}"][t]] for t, g in v[key].items()} for s, v in vals.items()},
                   stat=ratio_stat, support=None)
    b.extra.update(share_of_full=share.to_dict(),
                   n_targets={f"n_neg_{k}": [len(vals[s][f"full_of_{k}"]) for s in sorted(vals)]
                              for k in (2, 3)},
                   n_curve_points=sum(len(r["curve"]) for rs in rows.values() for r in rs),
                   n_curve_points_short=sum(r["curve_short"] for rs in rows.values() for r in rs))
    return b.to_dict()


def shares_block(rows: dict, sgcn: dict | None, boot: Boot) -> dict:
    """U1 gain over U2 gain as a ratio of five-seed means, for LocalGDTE and for SGCN."""
    b = Block("shares", boot)
    b.stat("u1_over_u2", columns(rows, [lambda r: r["g_u1"], _g2]), stat=column_means_stat,
           combine=combine_ratio, shown=ratio_stat, support=None)                   # ratio of signed gains
    if sgcn is None:
        b.exact("sgcn_u1_over_u2", None, "no SGCN baseline runs for this dataset")
    else:
        b.stat("sgcn_u1_over_u2", sgcn, stat=column_means_stat, combine=combine_ratio, shown=ratio_stat,
               support=None)
    return b.to_dict()


def logit_block(rows: dict, boot: Boot) -> dict:
    """U1 shifts of the additive head's target term b.z_T and rater (probe-source) term a.z_u."""
    b = Block("logit", boot)
    b.stat("target_term_shift", column(rows, "d_tgt_u1"), support=None)          # signed logit shifts
    b.stat("rater_term_mean", column(rows, "d_src_u1"), support=None)
    b.stat("rater_term_mean_abs", {s: {t: abs(x) for t, x in v.items()}
                                   for s, v in column(rows, "d_src_u1").items()}, support=NON_NEGATIVE)
    over = {s: {r["target"]: float(abs(r["d_src_u1"]) > RATER_TERM_BAR) for r in rs} for s, rs in rows.items()}
    b.exact("n_over_0.003", int(sum(sum(v.values()) for v in over.values())), "count (see frac_over_0.003)")
    b.stat("frac_over_0.003", over, unit="targets", support=PROPORTION)
    above = {t for v in over.values() for t, x in v.items() if x}
    b.extra["targets_over_0.003"] = dict(n_targets=len(set().union(*(v.keys() for v in over.values()))),
                                         n_targets_any_over=len(above))
    b.extra["n_eval"] = sum(len(rs) for rs in rows.values())
    return b.to_dict()


def _extreme(rows: dict, key: str) -> dict:
    vals = [(abs(r[key]), r[key], s, r["target"]) for s, rs in rows.items() for r in rs if r[key] is not None]
    if not vals:
        return dict(n=0, value=None, seed=None, target=None)
    top = max(vals)
    return dict(n=len(vals), value=top[1], seed=top[2], target=top[3])


def dmcc_block(rows: dict, boot: Boot) -> dict:
    """Largest single-retrain test-MCC change (T1) under C2 and under the random-edge placebo, both U2."""
    b = Block("max_dmcc", boot)
    for field, key in (("c2_u2", "dmcc_u2"), ("placebo_u2", "dmcc_placebo_u2")):
        ext = _extreme(rows, key)
        b.extra.setdefault("argmax", {})[field] = ext
        if ext["n"] == 0:
            b.exact(field, None, "no evaluations")
            continue
        b.exact(field, abs(ext["value"]), "maximum over evaluations")
        b.stat(f"mean_{field}", column(rows, key, lambda r, k=key: r[k] is not None), support=None)
    return b.to_dict()


def r0_block(records: dict, rows: dict, boot: Boot) -> dict:
    """Mean proxy before deletion: over the sampled targets, and over the full cohort (base)."""
    b = Block("r0", boot)
    b.stat("sample_mean", column(rows, "r0"), support=PROPORTION)
    b.stat("cohort_mean", {s: [records[s]["base"]["r0_cohort_mean"]] for s in sorted(records)}, support=PROPORTION)
    return b.to_dict()


# ------------------------------------------------------------------------ inputs and graphs
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class Inputs:
    """Every file the analysis reads, by repository-relative path, with its sha256 (provenance)."""

    def __init__(self, root: str = C.ROOT):
        self.root, self.sha256 = root, {}

    def path(self, rel: str) -> str:
        full = os.path.join(self.root, rel)
        if rel not in self.sha256:
            self.sha256[rel] = sha256_file(full)
        return full

    def json(self, rel: str):
        with open(self.path(rel)) as f:
            return json.load(f)

    def exists(self, rel: str) -> bool:
        return os.path.exists(os.path.join(self.root, rel))


def load_graph(ds: str, inputs: Inputs) -> dict:
    """Pre-deletion training graph, cohort, edgeless census and probe panel, through keeper.laundering.core."""
    from keeper.laundering.threshold import held_out
    D = C.load(ds)
    for rel in [f"configs/{ds}.json"] + [D.cfg[k] for k in ("data_path", "edge_path") if D.cfg.get(k)]:
        inputs.path(rel)
    G = C.full_graph(D)
    Tr = C.train_split(G, D.tt)
    cohort, neg, pos = C.pick_targets(Tr, max_neg=MAX_NEG, strict=True)
    is_neg = S.neg_mask(Tr.labels)
    edgeless = set(cohort[edgeless_after_c2(Tr.edges, is_neg, cohort, D.n)].tolist())
    outdeg = np.bincount(Tr.edges[:, 0], minlength=D.n)
    review = {int(v) for v in cohort if outdeg[int(v)] == 0 and not pos.get(int(v))}
    if edgeless != review:                       # review_isolated.py / review_amp_split.py definition
        raise AssertionError(f"{ds}: edgeless census differs from the review's definition")
    npos = np.array([len(pos.get(int(v), [])) for v in cohort])
    nneg = np.array([len(neg[int(v)]) for v in cohort])
    return dict(n=D.n, Tr=Tr, cohort=cohort, is_neg=is_neg, edgeless=edgeless,
                n_without_out=int((outdeg[cohort] == 0).sum()), cohort_npos=npos, cohort_nneg=nneg,
                p_pos=float(1.0 - is_neg.mean()), heldout=held_out(G, D),
                panel_sources=C.Panel(Tr, n_probe=N_PROBE, seed=PANEL_SEED).sources)


def logreg_yardstick(graph: dict, targets, loo: str) -> tuple:
    """(model summary with held-out AUC, {target: C2 shift}) for the degree logistic regression."""
    from sklearn.metrics import roc_auc_score
    Tr, is_neg, n = graph["Tr"], graph["is_neg"], graph["n"]
    model = fit_degree_logreg(Tr.edges, ~is_neg, n, loo)
    deg = degree_table(Tr.edges, ~is_neg, n)
    shifts = {int(v): c2_logreg_shift(model, deg, Tr.edges, is_neg, int(v), graph["panel_sources"])
              for v in targets}
    e_h, y_h = graph["heldout"]
    p = model.prob(np.column_stack([deg[e_h[:, 1], 0], deg[e_h[:, 1], 1], deg[e_h[:, 0], 2],
                                    deg[e_h[:, 0], 3]]))
    summary = dict(model.summary(), auc_heldout=float(roc_auc_score(1 - y_h, p)), n_heldout=int(len(y_h)))
    return summary, shifts


def laplace_crossing(graph: dict) -> dict:
    """Census check: the Laplace score of every cohort target starts below 0.5 and ends at 0.5 or above."""
    npos, nneg = graph["cohort_npos"], graph["cohort_nneg"]
    pre, post = (npos + 1.0) / (npos + nneg + 2.0), (npos + 1.0) / (npos + 2.0)
    below = pre < 0.5
    crossed = int((below & (post >= 0.5)).sum())
    return dict(n_cohort=int(len(npos)), n_below_half_before=int(below.sum()), n_crossed=crossed,
                conditional_flip_rate=crossed / int(below.sum()) if below.any() else None)


def census_positive_share(ds: str, inputs: Inputs) -> tuple:
    """(1 - distrust_share, source) from the dataset census, in review_probe4.py's lookup order."""
    for rel in CENSUS_FILES:
        if inputs.exists(rel):
            for row in inputs.json(rel):
                if row.get("ds") == ds and "distrust_share" in row:
                    return 1.0 - float(row["distrust_share"]), rel
    return None, None


def base_rate_amp(rows: dict, p: float) -> float:
    """Base-rate amplification (seed mean of per-seed ratios of means) for a given trust share p."""
    shift = lambda r: base_rate_shift(r["n_pos"], r["n_neg"], p)          # noqa: E731
    return float(np.mean(per_seed_values(columns(rows, [_g2, shift]), ratio_stat)))


def load_records(ds: str, inputs: Inputs) -> dict:
    records = {s: inputs.json(f"{LAUNDERING}/probe4_{ds}_s{s}.json") for s in SEEDS}
    for s, rec in records.items():
        if rec["ds"] != ds or rec["seed"] != s or rec.get("only_target") is not None:
            raise ValueError(f"probe4_{ds}_s{s}.json is not the full run of ({ds}, seed {s})")
        if rec["n_probe"] != N_PROBE or rec["budget"] != MAX_NEG:
            raise ValueError(f"probe4_{ds}_s{s}.json: panel {rec['n_probe']} or budget {rec['budget']} "
                             f"differs from the {N_PROBE}-account panel and budget {MAX_NEG} assumed here")
    return records


def load_sgcn(ds: str, inputs: Inputs, records: dict) -> dict | None:
    """seed -> {target: [U1 gain, U2 gain]} of SGCN over all 50 sampled targets; None without baselines."""
    rels = {s: f"{LAUNDERING}/baselines_{ds}_s{s}.json" for s in SEEDS}
    if not all(inputs.exists(rel) for rel in rels.values()):
        return None
    out = {}
    for s, rel in rels.items():
        b = inputs.json(rel)
        if b["sample"] != records[s]["sample"]:
            raise ValueError(f"{rel}: sample differs from probe4_{ds}_s{s}.json")
        out[s] = {t["target"]: [t["sgcn"]["direct"]["unlink"], t["sgcn"]["direct"]["exact_retrain"]]
                  for t in b["targets"]}
    return out


# ------------------------------------------------------------------------ one dataset
def analyse_dataset(ds: str, inputs: Inputs, n_boot: int = N_BOOT) -> dict:
    graph = load_graph(ds, inputs)
    cal = set(inputs.json(f"{LAUNDERING}/calibration_{ds}_s0.json")["calibration_targets"])
    records = load_records(ds, inputs)
    design, reason = ci_design_for({s: rec["sample"] for s, rec in records.items()})
    boot = Boot(ds, n_boot, design=design)
    rows = {s: target_rows(rec, cal, graph["edgeless"]) for s, rec in records.items()}
    targets = sorted({r["target"] for rs in rows.values() for r in rs})
    lr_model, lr_shift = logreg_yardstick(graph, targets, "in_and_out")      # ruling R28: the headline
    tl_model, tl_shift = logreg_yardstick(graph, targets, "in")              # ruling R20: target-only LOO
    amp = amplification_block(rows, graph["p_pos"], lr_shift, boot, {"logreg_target_loo": tl_shift})
    p_census, census_src = census_positive_share(ds, inputs)
    amp.update(
        base_rate_prior=dict(p=graph["p_pos"], p_source="trust share of the pre-deletion training edges",
                             strength=PRIOR_STRENGTH, p_census=p_census, census_source=census_src,
                             amp_with_p_census=None if p_census is None else base_rate_amp(rows, p_census)),
        logreg_model=lr_model, laplace_crossing=laplace_crossing(graph))
    amp["logreg_target_loo_model"] = tl_model
    sample = records[0]["sample"]
    return dict(
        cohort=len(graph["cohort"]), sample_size=len(sample), n_seeds=len(records), ci_design=design,
        ci_design_reason=reason,
        n_eval=[len(rows[s]) for s in sorted(rows)], calibration_excluded=sorted(cal & set(sample)),
        edgeless=edgeless_block(graph, rows, boot), percentile=percentile_block(rows, boot),
        cohort_weighted=cohort_weighted_block(rows, float((graph["cohort_npos"] >= 1).mean()), boot),
        tau_mcc=tau_block(records, rows, boot), amplification=amp, amp_split=amp_split_block(rows, boot),
        partial_deletion_u1=partial_block(rows, boot),
        shares=shares_block(rows, load_sgcn(ds, inputs, records), boot), logit=logit_block(rows, boot),
        max_dmcc=dmcc_block(rows, boot), r0=r0_block(records, rows, boot),
        ci_undefined_replicates=dict(sorted(boot.undefined.items())))


# ------------------------------------------------------------------------ method, provenance, CLI
METHOD = dict(
    evaluations="one (seed, sampled target); calibration targets excluded except in shares.sgcn_u1_over_u2",
    operators="U1 = unlink (inference-only, input features recomputed); U2 = exact_retrain",
    ci95="Every headline field has a 95% interval ci95, and ci95_source says how it was made (ruling R30). "
         "By default ci95 is the hull of two intervals, keeping the lower of the two lower bounds and the "
         "higher of the two upper bounds. The first, ci95_bootstrap, is the percentile interval of the "
         "two-way crossed bootstrap of ruling R29: each of 10,000 replicates resamples the seeds with "
         "replacement and, independently, the target set once, and every resampled seed uses the same "
         "resampled targets. The second, ci95_seed_t, is a t-interval over the per-seed statistics with "
         "n_seeds - 1 degrees of freedom, centred on their mean, which is the point estimate; for a ratio "
         "of seed-averaged means (the U1/U2 shares) it is centred on that ratio and uses the linearized "
         "per-seed residuals. The hull is needed because with five seeds the percentile bootstrap "
         "understates the seed variance, so for fields that vary only across seeds the t-interval is in "
         "effect the headline. A 0/1-valued field whose point estimate is exactly 0 or 1 has a degenerate "
         "bootstrap and t-interval; its ci95 is instead the Clopper-Pearson interval over the distinct "
         "targets, because the same targets recur in every seed: for the escape rates and the "
         "bottom-quartile share a target counts once, and for logit.frac_over_0.003 a target counts as "
         "exceeding 0.003 if any of its evaluations does. ci95_nested is the nested bootstrap of ruling "
         "R21 (targets resampled within each drawn seed), kept as a sensitivity; a dataset whose seeds "
         "drew different targets "
         "uses it as its bootstrap (see ci_design). A replicate is undefined when a drawn seed that has "
         "cells gives a 0/0 statistic or when no drawn seed has a cell; undefined replicates are dropped "
         "and counted in ci_undefined_replicates, and an interval with more than 1% undefined replicates "
         "is refused. Finally each headline interval is clipped to the mathematical support of its field, "
         "which is declared where the field is computed and recorded in ci95_support (ruling R31): "
         "proportions, escape rates, percentiles (stored as fractions), mean proxies and the MCC-optimal "
         "threshold to [0, 1]; a mean count to [0, its largest possible value] (cohort_below_tau_mean to "
         "[0, cohort size]); a mean magnitude (rater_term_mean_abs) to [0, inf). Signed quantities (gains, "
         "logit shifts, test-MCC changes) and ratios with a signed numerator (amplifications, U1/U2 "
         "shares, partial-deletion shares) are not clipped; a ratio of non-negative quantities would be "
         "clipped at 0 from below. When clipping changes an interval, the unclipped one is kept in "
         "ci95_unclipped and ci95_source says so.",
    ci95_not_applicable="edgeless.n and edgeless.share are cohort censuses; max_dmcc.c2_u2 and "
                        "placebo_u2 are maxima; logit['n_over_0.003'] is a count (its fraction has a CI)",
    null_fields="epinions-dated: tau_mcc.cohort_below_tau (its probe4 runs have no full-cohort retrain; "
                "sample_below_tau is given) and shares.sgcn_u1_over_u2 (no SGCN baseline runs)",
    edgeless="cohort target with no training edge left once C2 deletes all its distrust in-edges",
    cold_start="per seed, mean stored U1 proxy (and platform percentile) of the sampled edgeless targets; "
               "that this equals the encoder's score for an account with no edge would need model "
               "inference, which this module does not run (ruling R22)",
    percentile="platform percentile among active accounts (pct_before, pct_after); escape_bottomX: "
               "share of targets starting below X that end at X or above under U1, mean over seeds",
    amplification="per seed mean U2 gain / mean yardstick shift, averaged over seeds; yardsticks: "
                  "Laplace (d+ + 1)/(d + 2); raw d+/d (0/0 = 1); base-rate prior (d+ + 2p)/(d + 2) with p "
                  "the training trust share; logreg: P(+ | u -> v) on log1p signed degrees, fit on the "
                  "training edges with the edge's own label left out of v's in-degrees and u's "
                  "out-degrees (ruling R28; logreg_target_loo leaves it out of v's in-degrees only, ruling "
                  "R20), scored over the 256 probe accounts with the self-pair dropped; its shift removes "
                  "the deleted warnings from both endpoints' degrees",
    shares="ratio of five-seed means of U1 gain over U2 gain (LocalGDTE without calibration targets; "
           "SGCN over all 50 sampled targets per seed)",
    partial_deletion_u1="direct_greedy_curve unlink gains (attack.Scorer, trained input features kept) "
                        "after k' of k warnings; share_of_full divides by the same targets' unlink_frozen_x "
                        "gain, per seed as a ratio of means",
    logit="U1 shift of the target term b.z_T (d_tgt) and the rater term a.z_u (d_src), per target means "
          "over the probe panel, stored by probe4",
    max_dmcc="largest |test T1 MCC change| of a single U2 retrain, under C2 and under the placebo",
    tau_mcc="base.tau, the MCC-optimal edge threshold of the base model (threshold.optimal_threshold)")


def provenance(inputs: Inputs, n_boot: int) -> dict:
    import scipy
    import sklearn
    return dict(
        **git_state(),                                          # commit, dirty flag, uncommitted paths
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        host=socket.gethostname(), argv=list(sys.argv), python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), n_boot=int(n_boot),
        boot_seed=BOOT_SEED, code_sha256=C.CODE_SHA256_AT_IMPORT,
        module_sha256={rel: sha256_file(os.path.join(C.ROOT, rel)) for rel in MODULE_FILES},
        inputs_sha256=dict(sorted(inputs.sha256.items())))


def parse(argv=None):
    ap = argparse.ArgumentParser(description="Re-analysis of the stored laundering results (CPU only).")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default=os.path.join(ANALYSIS, "laundering_extra.json"))
    ap.add_argument("--md", default=os.path.join(ANALYSIS, "laundering_extra.md"))
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    from keeper.analysis.laundering_extra_md import render_markdown
    a = parse(argv)
    out_path = C.result_path(os.path.abspath(a.out), a.force)      # refuse before the work, not after it
    md_path = C.result_path(os.path.abspath(a.md), a.force)
    inputs = Inputs()
    result = {}
    for ds in a.datasets:
        print(f"[{ds}] analysing", flush=True)
        result[ds] = analyse_dataset(ds, inputs, a.n_boot)
    result["method"] = METHOD
    result["provenance"] = provenance(inputs, a.n_boot)
    for path in (out_path, md_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    C.dump(result, out_path, force=a.force)
    text = render_markdown(json.loads(json.dumps(C._finite(result), default=C._json_default)))
    with open(md_path, "w") as f:
        f.write(text)
    print(f"wrote {out_path} and {md_path}")
    return result


if __name__ == "__main__":
    main()
