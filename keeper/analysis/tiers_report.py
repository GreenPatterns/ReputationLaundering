"""BTP credibility tiers on the three CPPR networks (Task 3.1, upgrade M-A; Task 3.1b, ruling R56): the eps
choice, the tier mix and synthetic Sybil regions under a frozen calibration, for four variants, with the demotion
census of ruling R56.

The variants cross two choices. The propagation is either the approved uncapped BTP or BTP with KEEPER's
per-endorsement transfer cap (ruling R37), edge_cap_units = keeper.credibility.tiers.KEEPER_EDGE_CAP_UNITS = 2
(ruling R38). The eligibility guard counts negative ratings either from trusted raters only (neg_from =
"trusted", ruling R56) or from every rater ("all", Task 3.1's rule). KEEPER's default is "capped" (the cap and the
credible-negatives guard); "uncapped", "capped_neg_all" and "uncapped_neg_all" are the ablations, the last being
Task 3.1's approved rule. No model is trained or run. Each network's pre-deletion training graph comes from
keeper.laundering.core (load, full_graph, train_split with the dataset's tt); CPU only.
1. Legacy coverage: keeper.analysis.defense_extra.tier_coverage, as in Task 2.2 (seeds S0, endorsed set D,
   active accounts, tier mix of keeper.credibility.tiers.legacy_bfs).
2. Per variant, eps by keeper.credibility.tiers.choose_eps, the preregistered honest-data rule: the largest
   eps in (1, 0.5, 0.2, 0.1, 0.05) whose BTP Tier >= 1 set (with the variant's cap and guard) keeps at least
   90% of the legacy Tier >= 1 set. The grid extends to 0.02, 0.01 and 0.005 if no value of it qualifies.
3. Per variant, the calibration (ruling R32) frozen on the honest graph at the chosen eps: calibrate_btp
   (seeds, K, theta0 = eps / n_ref, cap = edge_cap_units * theta0) and calibrate_btp_causal (one per prefix).
4. Per variant, BTP on the honest graph: btp_tiers' info, the tier mix over the active accounts, the overlap
   with the legacy Tier >= 1 set, the causal tiers of the tt training snapshots, and the wall-clock times.
5. Per variant, synthetic Sybil regions (rulings R33, R33b, R37). s fakes (new accounts) and g attack edges from
   honest accounts, appended after the honest edges in the last training snapshot, attack edges first.
   - Topologies: a ring (attack edge a reaches fake a mod s, and fake_j endorses fake_(j+1 mod s)), and a
     star (every attack edge reaches the hub fake_0, which endorses every other fake).
   - Endorser selections: random draws from the honest accounts in the legacy endorsed set D whose legacy
     and BTP tiers are both >= 2, and max_trust (ruling R33b), the honest accounts with the largest B(u),
     the trust one new endorsement from u would carry over K steps (endorser_inflow_scores).
   - Each case applies the frozen calibration and records the fakes admitted to Tier >= 1 by legacy_bfs and
     by btp_tiers, in total and per attack edge, and the fakes' trust. It also records the bound of
     bound_sybil_trust over the iteration history (capped terms for the capped variant), the cap
     floor(bound / theta0), and, for the capped variant, the constant floor(K g edge_cap_units).
   - A summary row whose best admission equals its region size is limited by that size: a larger region
     might admit more, so its admission is only a lower bound ("at least").
   - A last block recalibrates on the attacked graph (the uncalibrated path).
6. Per variant, the attacker's best star: one attack edge from each of the 5 accounts with the largest B(u)
   into stars of every size from 2 to floor(K edge_cap_units) + 4 (capped: exact for stars) or of the sparse
   sizes PROBE_SIZES_UNCAPPED (uncapped: a star admitted whole is a lower bound).
7. Cap-size sensitivity, under each guard: retention over eps for edge_cap_units in (1, 2, 4, 8), the eps the
   rule picks for each, and the per-edge constant floor(K edge_cap_units).
8. Ruling R56. guard_change compares each guarded variant's honest tiers with its all-raters twin's. The
   demotion census attacks 100 honest Tier >= 2 accounts per method, one per stream: k in (1, 5, 10) fresh
   Sybils each rate the victim negatively once. It reports the share of victims whose tier drops under legacy BFS
   and under every BTP variant, with victims from each method's own Tier >= 2 accounts ("own") and from the
   accounts in Tier >= 2 under every method ("common"); under the credible-negatives guard the share is 0 by
   construction.
9. calibration_table: one row per network and variant with eps, K, n_ref, theta0, the guarantee and the best
   star, the numbers the preregistration takes.

Run from the repository root, CPU only:
  CUDA_VISIBLE_DEVICES="" python -m keeper.credibility.tiers --report [--force]
It writes results/analysis/tiers_report.json.
"""
import argparse
import datetime
import functools
import math
import os
import platform
import socket
import sys
import time
from types import SimpleNamespace

import numpy as np

from keeper.analysis import defense_extra as DE
from keeper.analysis.laundering_extra import Inputs, sha256_file
from keeper.analysis.provenance import git_state
from keeper.credibility import tiers
from keeper.laundering import core as C
from keeper.laundering import signed as S

__all__ = ["NETWORKS", "VARIANTS", "tier_mix", "overlap", "calibration_summary", "with_injected", "sybil_rows",
           "sybil_case", "endorser_pool", "max_trust_endorsers", "sybil_test", "sybil_summary", "star_size_probe",
           "variant_role", "variant_block", "cap_sensitivity", "census_methods", "demotion_census", "guard_change",
           "network_block", "calibration_table", "analyse", "main"]

NETWORKS = ("bitcoinalpha", "bitcoinotc", "wiki-rfa")      # the three CPPR networks
# (name, edge_cap_units, neg_from). "capped" is KEEPER's default (rulings R38, R56); the others are ablations, and
# "uncapped_neg_all" is Task 3.1's approved rule.
VARIANTS = (("uncapped", None, "trusted"), ("capped", tiers.KEEPER_EDGE_CAP_UNITS, "trusted"),
            ("uncapped_neg_all", None, "all"), ("capped_neg_all", tiers.KEEPER_EDGE_CAP_UNITS, "all"))
N_TIERS = 4
RING_G = (1, 2, 5)               # attack edges per Sybil region
RING_S = (3, 10, 50, 350)        # fakes per Sybil region
RING_DRAWS = 5                   # random endorser draws per network and variant
RING_SEED = 20261027             # numpy default_rng seed of the draws
SELECTIONS = ("random", "max_trust")
TOPOLOGIES = ("ring", "star")
BOUND_TOL = 1e-15                # absolute floating-point slack of the trust-bound check
CAP_TOL = 1e-9                   # slack when flooring a fake count, so rounding cannot drop an exact integer
CAP_SENSITIVITY_UNITS = (1.0, 2.0, 4.0, 8.0)                          # cap sizes of the sensitivity block
PROBE_ENDORSERS = 5              # star-size probe: the accounts with the largest B(u) (ruling R33b)
PROBE_SIZES_UNCAPPED = (2, 5, 10, 20, 50, 100, 200, 350, 500, 750, 1000, 1500, 2000)   # sparse: lower bounds
EPS_SENSITIVITY_GRID = tiers.EPS_GRID_EXTENDED + (0.002, 0.001, 0.0005, 0.0001)
CENSUS_K = (1, 5, 10)            # demotion census (ruling R56): fresh Sybils per victim, each rating it once, negatively
CENSUS_VICTIMS = 100             # victims per method, one per attacked stream (E1's one-record batches)
CENSUS_SEED = 20261056           # numpy default_rng seed of the victim draw
CENSUS_POOLS = ("own", "common")  # victims from each method's own Tier >= 2 accounts, or from those of every method
ANALYSIS = os.path.join(C.ROOT, "results", "analysis")
PREREGISTRATION = os.path.join(C.ROOT, "docs", "preregistration.md")
MODULE_FILES = ("keeper/credibility/tiers.py", "keeper/analysis/tiers_report.py",
                "keeper/analysis/defense_extra.py", "keeper/analysis/laundering_extra.py",
                "keeper/analysis/provenance.py")


# ------------------------------------------------------------------------ tier mix, overlap, calibration
def tier_mix(levels: np.ndarray, active: np.ndarray) -> dict:
    """Tier counts and shares over the active accounts, and the number in Tier >= 1."""
    counts = np.bincount(levels[active], minlength=N_TIERS)
    return dict(tier_counts=counts.tolist(), tier_shares=(counts / len(active)).tolist(),
                n_tier1plus=int(counts[1:].sum()))


def overlap(legacy_levels: np.ndarray, btp_levels: np.ndarray) -> dict:
    """The legacy and BTP Tier >= 1 sets compared; retention is the share of the legacy set BTP keeps."""
    legacy, btp = legacy_levels > 0, btp_levels > 0
    n_legacy, both = int(legacy.sum()), int((legacy & btp).sum())
    return dict(n_legacy_tier1=n_legacy, n_btp_tier1=int(btp.sum()), n_both=both,
                n_legacy_only=int((legacy & ~btp).sum()), n_btp_only=int((~legacy & btp).sum()),
                retention=both / n_legacy if n_legacy else None)


def calibration_summary(cal: tiers.BTPCalibration, neg_from: str) -> dict:
    """The frozen calibration and the guard it is applied with (ruling R56)."""
    return dict(eps=cal.eps, K=cal.K, theta0=cal.theta0, n_ref=cal.n_ref, n_seeds=cal.n_seeds, t_seed=cal.t_seed,
                edge_cap_units=cal.edge_cap_units, edge_cap=cal.edge_cap, neg_from=neg_from)


# ------------------------------------------------------------------------ synthetic Sybil regions
def with_injected(Tr, rows, snap: int) -> SimpleNamespace:
    """Tr's edges followed by the injected (u, v, sign) rows, in their order, all in snapshot ``snap``."""
    r = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    return SimpleNamespace(
        edges=np.concatenate([np.asarray(Tr.edges, dtype=np.int64).reshape(-1, 2), r[:, :2]]),
        labels=np.concatenate([np.asarray(Tr.labels, dtype=np.float32), S.one_hot_from_sign(r[:, 2])]),
        snap=np.concatenate([np.asarray(Tr.snap, dtype=np.int64), np.full(len(r), snap, dtype=np.int64)]))


def sybil_rows(attackers, fakes, topology: str) -> list:
    """The attack edges, then the Sybil edges, all positive. Ring: attackers[a] -> fakes[a mod s], and
    fake_j -> fake_(j+1 mod s). Star: every attacker -> the hub fakes[0], and the hub -> every other fake."""
    s = len(fakes)
    if topology == "ring":
        rows = [(int(u), int(fakes[a % s]), 1) for a, u in enumerate(attackers)]
        return rows + [(int(fakes[j]), int(fakes[(j + 1) % s]), 1) for j in range(s)]
    if topology == "star":
        rows = [(int(u), int(fakes[0]), 1) for u in attackers]
        return rows + [(int(fakes[0]), int(fakes[j]), 1) for j in range(1, s)]
    raise ValueError(f"unknown topology {topology!r}")


def sybil_case(Tr, n: int, attackers, s: int, topology: str, snap: int, calibration=None, eps=None,
               edge_cap_units=None, with_legacy: bool = True, neg_from: str = tiers.KEEPER_NEG_FROM) -> dict:
    """One Sybil region of s fakes endorsed by ``attackers``: its admission to Tier >= 1 under legacy_bfs
    and btp_tiers (in total and per attack edge), its trust, the bound over the iteration history, the count
    cap floor(bound / theta0) and, under a per-endorsement cap, the constant floor(K g edge_cap_units). With
    ``calibration`` the frozen honest calibration is applied; without one, eps is required and the attacked
    graph calibrates itself (the uncalibrated path) with edge_cap_units. btp_tiers applies the guard neg_from
    (ruling R56). with_legacy=False skips the legacy pass (its fields are then None)."""
    fakes = np.arange(n, n + s, dtype=np.int64)
    rows = sybil_rows(attackers, fakes, topology)
    G, n_all, g = with_injected(Tr, rows, snap), n + s, len(attackers)
    levels, trust, info = tiers.btp_tiers(G, n_all, eps, calibration=calibration, edge_cap_units=edge_cap_units,
                                          neg_from=neg_from)
    units = calibration.edge_cap_units if calibration is not None else edge_cap_units
    endorse = tiers.endorsement_edges(G)
    seeds = np.asarray(calibration.seeds if calibration is not None else tiers.seed_set(G, n_all), dtype=np.int64)
    if np.isin(seeds, fakes).any():
        raise AssertionError("a fake is a seed, so the bound does not apply")
    history = tiers.btp_trust(n_all, endorse, seeds, info["K"], return_history=True, edge_cap=info["edge_cap"])
    if not np.array_equal(history[-1], trust):
        raise AssertionError("btp_trust and btp_tiers disagree on the same graph")
    attack = np.asarray(rows[:g], dtype=np.int64)[:, :2]
    bound = tiers.bound_sybil_trust(history, attack, np.bincount(endorse[:, 0], minlength=n_all),
                                    edge_cap=info["edge_cap"])
    sybil_trust = float(trust[fakes].sum())
    ratio = bound / info["theta0"]
    cap = math.floor(ratio + CAP_TOL)
    constant = None if units is None else math.floor(info["K"] * g * units + CAP_TOL)
    legacy = int((tiers.legacy_bfs(G, n_all)[fakes] > 0).sum()) if with_legacy else None
    admitted = int((levels[fakes] > 0).sum())
    return dict(g=g, s=int(s), topology=topology, attackers=[int(u) for u in attackers], calibrated=calibration
                is not None, legacy_admitted=legacy, legacy_per_edge=None if legacy is None else legacy / g,
                btp_admitted=admitted,
                btp_per_edge=admitted / g, btp_trusted=int((trust[fakes] >= info["theta0"]).sum()),
                sybil_trust=sybil_trust, bound=bound, theta0=info["theta0"], edge_cap=info["edge_cap"],
                bound_over_theta0=ratio, cap=cap, cap_per_edge=cap / g, cap_constant=constant,
                K=info["K"], n_active=info["n_active"], n_ref=info["n_ref"],
                bound_holds=bool(sybil_trust <= bound + BOUND_TOL), cap_holds=bool(admitted <= cap),
                cap_constant_holds=None if constant is None else bool(cap <= constant))


def endorser_pool(Tr, n: int, legacy_levels: np.ndarray, btp_levels: np.ndarray) -> np.ndarray:
    """Honest accounts in the legacy endorsed set D whose legacy and BTP tiers are both >= 2, sorted.

    Members of D make legacy_bfs admit a whole region by one rating, so both methods face the same edges.
    """
    _, members = DE.endorsed_set(Tr, n)
    in_D = np.zeros(n, dtype=bool)
    in_D[sorted(members)] = True
    return np.flatnonzero(in_D & (legacy_levels >= 2) & (btp_levels >= 2))


def top_accounts(scores: np.ndarray, k: int) -> np.ndarray:
    """The k accounts with the largest scores; ties go to the smaller account id."""
    ids = np.arange(len(scores))
    return ids[np.lexsort((ids, -np.asarray(scores)))[:k]]


def max_trust_endorsers(Tr, n: int, cal: tiers.BTPCalibration, k: int) -> np.ndarray:
    """The k honest accounts with the largest B(u) (ruling R33b): the trust one new endorsement would carry
    over K steps, keeper.credibility.tiers.endorser_inflow_scores under the frozen calibration ``cal``."""
    return top_accounts(tiers.endorser_inflow_scores(Tr, n, cal), k)


def sybil_summary(cases: list) -> list:
    """One row per (selection, topology, g, s): extremes over the draws and whether every check held."""
    groups = {}
    for c in cases:
        groups.setdefault((c["selection"], c["topology"], c["g"], c["s"]), []).append(c)
    rows = []
    for (selection, topology, g, s), cs in groups.items():
        constants = [c["cap_constant"] for c in cs if c["cap_constant"] is not None]
        rows.append(dict(
            selection=selection, topology=topology, g=g, s=s, n_cases=len(cs),
            legacy_admitted_min=min(c["legacy_admitted"] for c in cs),
            legacy_per_edge_min=min(c["legacy_per_edge"] for c in cs),
            btp_admitted_mean=float(np.mean([c["btp_admitted"] for c in cs])),
            btp_admitted_max=max(c["btp_admitted"] for c in cs), btp_per_edge_max=max(c["btp_per_edge"] for c in cs),
            limited_by_region_size=max(c["btp_admitted"] for c in cs) == s,
            cap_min=min(c["cap"] for c in cs), cap_max=max(c["cap"] for c in cs),
            cap_per_edge_max=max(c["cap_per_edge"] for c in cs), bound_max=max(c["bound"] for c in cs),
            cap_constant=max(constants) if constants else None,
            all_bounds_hold=all(c["bound_holds"] for c in cs), all_caps_hold=all(c["cap_holds"] for c in cs),
            all_cap_constants_hold=all(c["cap_constant_holds"] for c in cs) if constants else None))
    return rows


def sybil_test(Tr, n: int, cal: tiers.BTPCalibration, legacy_levels: np.ndarray, btp_levels: np.ndarray,
               ring_g=RING_G, ring_s=RING_S, draws: int = RING_DRAWS, seed: int = RING_SEED,
               neg_from: str = tiers.KEEPER_NEG_FROM) -> dict:
    """Every Sybil case of one network under the frozen calibration ``cal`` and the guard ``neg_from``, then the
    uncalibrated block.

    Each random draw and the max_trust selection pick max(ring_g) endorsers; a case with g attack edges uses
    the first g, with both topologies and every region size.
    """
    snap = int(np.max(Tr.snap))
    pool = endorser_pool(Tr, n, legacy_levels, btp_levels)
    g_max = max(ring_g)
    if len(pool) < g_max:
        raise ValueError(f"only {len(pool)} candidate endorsers for g = {g_max}")
    rng = np.random.default_rng(seed)
    random_sets = [rng.choice(pool, size=g_max, replace=False) for _ in range(draws)]
    out_deg = np.bincount(tiers.endorsement_edges(Tr)[:, 0], minlength=n)
    inflow = tiers.endorser_inflow_scores(Tr, n, cal)
    strongest = top_accounts(inflow, g_max)
    endorser_sets = [("random", draw, a) for draw, a in enumerate(random_sets)] + [("max_trust", None, strongest)]
    cases = [dict(selection=selection, draw=draw,
                  **sybil_case(Tr, n, attackers[:g], s, topology, snap, calibration=cal, neg_from=neg_from))
             for selection, draw, attackers in endorser_sets
             for topology in TOPOLOGIES for g in ring_g for s in ring_s]
    growth = [sybil_case(Tr, n, strongest[:1], s, "ring", snap, eps=cal.eps, edge_cap_units=cal.edge_cap_units,
                         neg_from=neg_from)
              for s in ring_s]
    return dict(n_pool=int(len(pool)), snapshot=snap,
                endorsers=dict(random=[a.tolist() for a in random_sets], max_trust=strongest.tolist(),
                               max_trust_out_degree=out_deg[strongest].tolist(),
                               max_trust_inflow=inflow[strongest].tolist(),
                               max_trust_inflow_over_theta0=(inflow[strongest] / cal.theta0).tolist()),
                cases=cases, summary=sybil_summary(cases), uncalibrated_cap_growth=growth)


def best_star(Tr, n: int, cal: tiers.BTPCalibration, endorser: int, sizes,
              neg_from: str = tiers.KEEPER_NEG_FROM) -> dict:
    """One attack edge from ``endorser`` into the hub of a star of s fakes, for each s in ``sizes``, under the
    frozen calibration and the guard neg_from (no legacy pass). best_s and max_admitted give the best star (the
    smallest s on ties). lower_bound is true when that star was admitted whole and size best_s + 1 was not probed:
    a larger star might then be admitted too, so max_admitted is only a lower bound on this endorser's best
    response."""
    sizes = [int(x) for x in sizes]
    snap = int(np.max(Tr.snap))
    rows = [sybil_case(Tr, n, [int(endorser)], s, "star", snap, calibration=cal, with_legacy=False, neg_from=neg_from)
            for s in sizes]
    best = max(rows, key=lambda row: row["btp_admitted"])
    lower = best["btp_admitted"] == best["s"] and best["s"] + 1 not in sizes
    return dict(endorser=int(endorser), best_s=best["s"], max_admitted=best["btp_admitted"],
                lower_bound=bool(lower), rows=rows)


def star_size_probe(Tr, n: int, cal: tiers.BTPCalibration, top: int = PROBE_ENDORSERS, sizes=None,
                    neg_from: str = tiers.KEEPER_NEG_FROM) -> dict:
    """The attacker's best star: best_star for each of the ``top`` accounts with the largest B(u) (ruling
    R33b), under the guard neg_from. With a cap, every size from 2 to floor(K edge_cap_units) + 4, so each best
    size is exact for stars and never exceeds the constant. Uncapped, the sparse sizes PROBE_SIZES_UNCAPPED (the
    caps run to hundreds), so a best star admitted whole is labelled a lower bound. ``sizes`` overrides either
    grid."""
    inflow = tiers.endorser_inflow_scores(Tr, n, cal)
    endorsers = top_accounts(inflow, top)
    constant = None if cal.edge_cap_units is None else math.floor(cal.K * cal.edge_cap_units + CAP_TOL)
    if sizes is None:
        sizes = PROBE_SIZES_UNCAPPED if constant is None else range(2, constant + 5)
    sizes = [int(x) for x in sizes]
    per_endorser = [dict(best_star(Tr, n, cal, u, sizes, neg_from=neg_from), inflow=float(inflow[u]))
                    for u in endorsers]
    best = max(per_endorser, key=lambda entry: entry["max_admitted"])   # the higher B(u) on ties
    return dict(neg_from=neg_from, constant=constant, sizes=sizes, endorsers=[int(u) for u in endorsers],
                max_admitted=best["max_admitted"], argmax_endorser=best["endorser"], argmax_s=best["best_s"],
                lower_bound=best["lower_bound"], per_endorser=per_endorser)


# ------------------------------------------------------------------------ variants, networks
def variant_role(edge_cap_units, neg_from: str = tiers.KEEPER_NEG_FROM) -> str:
    keeper_cap, guarded = edge_cap_units == tiers.KEEPER_EDGE_CAP_UNITS, neg_from == "trusted"
    if keeper_cap and guarded:
        return "KEEPER default (rulings R38, R56)"
    if edge_cap_units is None and guarded:
        return "ablation: uncapped propagation (the approved M-A rule), credible-negatives guard"
    if keeper_cap:
        return "ablation: negative ratings from every rater (the guard before ruling R56)"
    if edge_cap_units is None:
        return "ablation: uncapped, negative ratings from every rater (Task 3.1's approved rule)"
    return f"capped, edge_cap_units = {edge_cap_units:g}, neg_from = {neg_from}"


def variant_block(Tr, n: int, legacy_levels: np.ndarray, edge_cap_units, keep: float = tiers.EPS_KEEP,
                  T_rows: int = tiers.CAUSAL_T, neg_from: str = tiers.KEEPER_NEG_FROM, **ring_kw) -> dict:
    """One variant on one training graph: its eps choice, frozen calibration, BTP tiers and Sybil test, all with
    the cap edge_cap_units and the guard neg_from (ruling R56)."""
    eps, table = tiers.choose_eps(Tr, n, tiers.EPS_GRID, keep, edge_cap_units=edge_cap_units,
                                  fallback_grid=tiers.EPS_GRID_EXTENDED, neg_from=neg_from)
    cal = tiers.calibrate_btp(Tr, n, eps, edge_cap_units=edge_cap_units)
    start = time.perf_counter()
    levels, _, info = tiers.btp_tiers(Tr, n, calibration=cal, neg_from=neg_from)
    seconds_btp = time.perf_counter() - start
    start = time.perf_counter()
    causal_cal = tiers.calibrate_btp_causal(Tr, n, eps, T=T_rows, edge_cap_units=edge_cap_units)
    seconds_calibrate = time.perf_counter() - start
    start = time.perf_counter()
    causal = tiers.causal_tiers(Tr, n, calibration=causal_cal, neg_from=neg_from)
    seconds_causal = time.perf_counter() - start
    active = np.unique(np.asarray(Tr.edges, dtype=np.int64))
    return dict(edge_cap_units=edge_cap_units, neg_from=neg_from, role=variant_role(edge_cap_units, neg_from),
                choose_eps=table, eps=eps, calibration=calibration_summary(cal, neg_from),
                btp=dict(info, **tier_mix(levels, active)), overlap=overlap(legacy_levels, levels),
                causal=dict(T=int(T_rows), tier1_counts=(causal > 0).sum(axis=1).tolist(), K=list(causal_cal.K),
                            theta0=list(causal_cal.theta0), n_ref=list(causal_cal.n_ref),
                            n_seeds=[row.n_seeds for row in causal_cal.rows]),
                sybil_test=sybil_test(Tr, n, cal, legacy_levels, levels, neg_from=neg_from, **ring_kw),
                star_size_probe=star_size_probe(Tr, n, cal, neg_from=neg_from),
                runtime_s=dict(btp_tiers=seconds_btp, calibrate_btp_causal=seconds_calibrate,
                               causal_tiers=seconds_causal))


def cap_sensitivity(Tr, n: int, keep: float = tiers.EPS_KEEP, units_grid=CAP_SENSITIVITY_UNITS,
                    eps_grid=EPS_SENSITIVITY_GRID, neg_from: str = tiers.KEEPER_NEG_FROM) -> list:
    """The honest-coverage cost of the per-endorsement cap under the guard neg_from: for each cap size,
    choose_eps' rows over eps_grid (retention of the legacy Tier >= 1 set), the eps the preregistered rule picks
    (the standard grid, extended downward if no value qualifies) and the per-edge constant floor(K
    edge_cap_units). Ruling R38 picks KEEPER's cap from it: the smallest value for which the rule holds on all
    three networks."""
    K = tiers.default_K(tiers.n_active(Tr))
    out = []
    for units in units_grid:
        _, table = tiers.choose_eps(Tr, n, eps_grid, keep, edge_cap_units=units, neg_from=neg_from)
        rule_eps, rule = tiers.choose_eps(Tr, n, tiers.EPS_GRID, keep, edge_cap_units=units,
                                          fallback_grid=tiers.EPS_GRID_EXTENDED, neg_from=neg_from)
        out.append(dict(edge_cap_units=units, neg_from=neg_from, rows=table["rows"], rule_eps=rule_eps,
                        rule_met=rule["met"], rule_extended=rule["extended"],
                        per_edge_constant=math.floor(K * units + CAP_TOL)))
    return out


# ------------------------------------------------------------------------ ruling R56: guard change, demotion census
def _frozen_btp_levels(G, n_nodes: int, calibration: tiers.BTPCalibration, neg_from: str) -> np.ndarray:
    return tiers.btp_tiers(G, n_nodes, calibration=calibration, neg_from=neg_from)[0]


def census_methods(Tr, n: int, specs) -> dict:
    """The tier functions of the demotion census, in order: legacy BFS, then one per BTP variant spec (name,
    edge_cap_units, neg_from, eps) under its calibration frozen on Tr. name -> (description, f(G, n_nodes))."""
    methods = {"bfs": (dict(method="bfs", variant=None, edge_cap_units=None, neg_from="all", eps=None),
                       tiers.legacy_bfs)}
    for name, units, neg_from, eps in specs:
        cal = tiers.calibrate_btp(Tr, n, eps, edge_cap_units=units)
        methods[name] = (dict(method="btp", variant=name, edge_cap_units=units, neg_from=neg_from, eps=eps),
                         functools.partial(_frozen_btp_levels, calibration=cal, neg_from=neg_from))
    return methods


def _census_rows(Tr, n: int, fn, base: np.ndarray, victims: np.ndarray, k_grid, snap: int) -> list:
    """One census row per k for the tier function fn (base: its honest levels): k fresh Sybils (accounts n .. n + k
    - 1) each rate one victim negatively once, appended after the honest edges in snapshot snap; one stream per
    victim."""
    rows = []
    for k in k_grid:
        s = np.zeros((len(victims), 4), dtype=np.int64)
        for i, v in enumerate(victims):
            levels = fn(with_injected(Tr, [(n + j, int(v), -1) for j in range(k)], snap), n + k)[:n]
            moved = levels != base
            s[i] = (levels[v] < base[v], levels[v] == 0, moved.sum() - moved[v], moved.any())
        rows.append(dict(k=int(k), n_victims=len(victims), n_dropped=int(s[:, 0].sum()),
                         share_dropped=int(s[:, 0].sum()) / len(victims), n_to_tier0=int(s[:, 1].sum()),
                         share_to_tier0=int(s[:, 1].sum()) / len(victims), collateral_total=int(s[:, 2].sum()),
                         collateral_max=int(s[:, 2].max()), n_streams_with_collateral=int((s[:, 2] > 0).sum()),
                         n_streams_changed=int(s[:, 3].sum())))
    return rows


def demotion_census(Tr, n: int, methods: dict, k_grid=CENSUS_K, n_victims: int = CENSUS_VICTIMS,
                    seed: int = CENSUS_SEED, pool: str = "own") -> dict:
    """The demotion census of ruling R56 (no training), for the tier functions ``methods`` (census_methods).

    Each method's victims are n_victims honest accounts drawn (default_rng(seed), without replacement) from its own
    Tier >= 2 accounts (pool "own": how many of the accounts it deems credible can free Sybils demote?) or from the
    accounts in Tier >= 2 under every method (pool "common": the same victims for every method, a paired
    comparison). For each k and victim, one attacked stream: k fresh Sybils each rate the victim negatively once
    (_census_rows). Per method and k, over the streams: the victims whose tier drops and those in Tier 0
    afterwards; the collateral, the other honest accounts whose tier moves (summed, maximised, and the streams with
    any); and the streams in which any honest tier moves. unchanged marks a method under which no honest tier moved
    in any stream."""
    if pool not in CENSUS_POOLS:
        raise ValueError(f"pool must be one of {CENSUS_POOLS}, got {pool!r}")
    snap = int(np.max(Tr.snap))
    base = {name: fn(Tr, n) for name, (_, fn) in methods.items()}
    common = np.flatnonzero(np.logical_and.reduce([levels >= 2 for levels in base.values()]))
    out = {}
    for name, (description, fn) in methods.items():
        candidates = np.flatnonzero(base[name] >= 2) if pool == "own" else common
        if len(candidates) < n_victims:
            raise ValueError(f"{name}: only {len(candidates)} candidate victims ({pool} pool) for {n_victims}")
        victims = np.sort(np.random.default_rng(seed).choice(candidates, size=n_victims, replace=False))
        rows = _census_rows(Tr, n, fn, base[name], victims, k_grid, snap)
        out[name] = dict(description, n_pool=int(len(candidates)), victims=victims.tolist(), rows=rows,
                         unchanged=all(row["n_streams_changed"] == 0 for row in rows))
    return dict(pool=pool, k_grid=[int(k) for k in k_grid], n_victims=int(n_victims), seed=int(seed), snapshot=snap,
                methods=out)


def guard_change(Tr, n: int, trusted_spec, all_spec) -> dict:
    """The honest tiers under the credible-negatives guard against the all-raters guard (ruling R56), for one cap:
    specs (name, edge_cap_units, neg_from, eps), each applied with its own eps under its calibration frozen on Tr.
    Counts over the union of the two Tier >= 1 sets; the share of that union whose tier differs
    (share_tier_changed) or whose Tier >= 1 membership differs; and transitions[a][b], the active accounts in tier
    a under "all" and tier b under "trusted"."""
    (name_t, units_t, neg_t, eps_t), (name_a, units_a, neg_a, eps_a) = trusted_spec, all_spec
    if (neg_t, neg_a) != ("trusted", "all") or units_t != units_a:
        raise ValueError("guard_change compares a 'trusted' spec with the 'all' spec of the same cap")
    t = _frozen_btp_levels(Tr, n, tiers.calibrate_btp(Tr, n, eps_t, edge_cap_units=units_t), "trusted")
    a = _frozen_btp_levels(Tr, n, tiers.calibrate_btp(Tr, n, eps_a, edge_cap_units=units_a), "all")
    union = (t > 0) | (a > 0)
    n_union = int(union.sum())
    trusted_only, all_only = int(((t > 0) & (a == 0)).sum()), int(((a > 0) & (t == 0)).sum())
    changed = int((union & (t != a)).sum())
    active = np.unique(np.asarray(Tr.edges, dtype=np.int64))
    return dict(trusted_variant=name_t, all_variant=name_a, edge_cap_units=units_t, eps_trusted=eps_t,
                eps_all=eps_a, same_eps=bool(eps_t == eps_a), n_tier1_trusted=int((t > 0).sum()),
                n_tier1_all=int((a > 0).sum()), n_union=n_union, n_both=int(((t > 0) & (a > 0)).sum()),
                n_trusted_only=trusted_only, n_all_only=all_only, n_tier_changed=changed,
                share_tier1_membership_changed=(trusted_only + all_only) / n_union if n_union else None,
                share_tier_changed=changed / n_union if n_union else None,
                transitions=[[int(((a[active] == i) & (t[active] == j)).sum()) for j in range(N_TIERS)]
                             for i in range(N_TIERS)])


def network_block(Tr, n: int, keep: float = tiers.EPS_KEEP, T_rows: int = tiers.CAUSAL_T, variants=VARIANTS,
                  census_kw=None, **ring_kw) -> dict:
    """Legacy coverage, every variant (name, edge_cap_units, neg_from) of one training graph, ruling R56's guard
    comparison (each guarded variant against the all-raters variant of the same cap) and demotion census for both
    victim pools (census_kw overrides its k_grid, n_victims or seed), and the cap-size sensitivity under each
    guard."""
    legacy_levels = tiers.legacy_bfs(Tr, n)
    blocks = {name: variant_block(Tr, n, legacy_levels, units, keep, T_rows, neg_from=neg_from, **ring_kw)
              for name, units, neg_from in variants}
    specs = [(name, units, neg_from, blocks[name]["eps"]) for name, units, neg_from in variants]
    change = {t[0]: guard_change(Tr, n, t, a) for t in specs if t[2] == "trusted"
              for a in specs if a[2] == "all" and a[1] == t[1]}
    methods = census_methods(Tr, n, specs)
    return dict(legacy=DE.tier_coverage(Tr, n), variants=blocks, guard_change=change,
                demotion_census={pool: demotion_census(Tr, n, methods, pool=pool, **(census_kw or {}))
                                 for pool in CENSUS_POOLS},
                cap_sensitivity={neg_from: cap_sensitivity(Tr, n, keep, neg_from=neg_from) for neg_from in tiers.NEG_FROM})


def calibration_table(networks: dict) -> list:
    """One row per network and variant: the calibration the preregistration takes (eps, K, n_ref, theta0, edge_cap,
    n_seeds), the eps rule's outcome (met, extended, retention, n_tier1), the guarantee floor(K edge_cap_units) per
    attack edge and the probe's best star, and, for a guarded variant, the share of honest Tier >= 1 accounts whose
    tier, or Tier >= 1 membership, differs from the all-raters variant of the same cap (guard_change)."""
    rows = []
    for ds, block in networks.items():
        for name, v in block["variants"].items():
            change, cal, probe = block["guard_change"].get(name), v["calibration"], v["star_size_probe"]
            rows.append(dict(network=ds, variant=name, role=v["role"], neg_from=v["neg_from"],
                             edge_cap_units=v["edge_cap_units"], eps=v["eps"], met=v["choose_eps"]["met"],
                             extended=v["choose_eps"]["extended"], retention=v["overlap"]["retention"],
                             n_tier1=v["overlap"]["n_btp_tier1"],
                             **{key: cal[key] for key in ("K", "n_ref", "theta0", "edge_cap", "n_seeds")},
                             guarantee_per_edge=probe["constant"], best_star=probe["max_admitted"],
                             best_star_lower_bound=probe["lower_bound"],
                             share_tier_changed_vs_all=None if change is None else change["share_tier_changed"],
                             share_tier1_membership_changed_vs_all=None if change is None
                             else change["share_tier1_membership_changed"]))
    return rows


def analyse(datasets, inputs: Inputs) -> dict:
    networks = {}
    for ds in datasets:
        print(f"[{ds}] BTP tiers, four variants: eps, frozen calibration, Sybil test, demotion census", flush=True)
        D = C.load(ds)
        for rel in [f"configs/{ds}.json"] + [D.cfg[k] for k in ("data_path", "edge_path") if D.cfg.get(k)]:
            inputs.path(rel)
        networks[ds] = network_block(C.train_split(C.full_graph(D), D.tt), D.n, T_rows=D.tt)
    return dict(calibration_table=calibration_table(networks), networks=networks)


SETTINGS = dict(variants=[dict(name=name, edge_cap_units=units, neg_from=neg_from, role=variant_role(units, neg_from))
                          for name, units, neg_from in VARIANTS],
                neg_from=list(tiers.NEG_FROM), keeper_neg_from=tiers.KEEPER_NEG_FROM,
                keeper_edge_cap_units=tiers.KEEPER_EDGE_CAP_UNITS,
                eps_grid=list(tiers.EPS_GRID), eps_grid_extended=list(tiers.EPS_GRID_EXTENDED), keep=tiers.EPS_KEEP,
                t_seed=tiers.T_SEED, laziness=tiers.LAZINESS, tier_weights=list(tiers.TIER_WEIGHTS),
                ring_g=list(RING_G), ring_s=list(RING_S), ring_draws=RING_DRAWS, ring_seed=RING_SEED,
                selections=list(SELECTIONS), topologies=list(TOPOLOGIES), bound_tol=BOUND_TOL, cap_tol=CAP_TOL,
                cap_sensitivity_units=list(CAP_SENSITIVITY_UNITS), eps_sensitivity_grid=list(EPS_SENSITIVITY_GRID),
                census_k=list(CENSUS_K), census_victims=CENSUS_VICTIMS, census_seed=CENSUS_SEED,
                census_pools=list(CENSUS_POOLS))

METHOD = dict(
    graph="Pre-deletion training graph of each CPPR network: keeper.laundering.core load, full_graph and "
          "train_split with the dataset's tt (snapshots 0 to tt - 1). Active accounts have a training edge "
          "at either end.",
    legacy="keeper.analysis.defense_extra.tier_coverage (Task 2.2): seeds S0 (more positive than negative "
           "in-edges in snapshots 0 to 4), the endorsed set D of the one-pass endorsement, and the tier mix "
           "of keeper.credibility.tiers.legacy_bfs over the active accounts.",
    variants="Four variants per network (settings.variants), crossing the propagation and the eligibility guard. "
             "Propagation: uncapped, the approved BTP rule, or capped, in which every endorsement carries at most "
             "edge_cap = edge_cap_units * theta0 per step (ruling R37), the sender keeping the rest, with "
             "edge_cap_units = keeper.credibility.tiers.KEEPER_EDGE_CAP_UNITS = 2 (ruling R38: the smallest value "
             "in (1, 2, 4, 8) for which the honest-data eps rule holds on all three networks). Guard (ruling R56, "
             "neg_from): the negative in-edges counted against eligibility come from trusted raters only "
             "('trusted', the credible-negatives guard) or from every rater ('all', the rule of Task 3.1). capped "
             "with 'trusted' is KEEPER's default; uncapped, capped_neg_all and uncapped_neg_all (Task 3.1's "
             "approved rule) are the ablations. Every block under variants is computed for that variant alone.",
    choose_eps="keeper.credibility.tiers.choose_eps with the variant's edge_cap_units and neg_from: for each eps "
               "in settings.eps_grid, btp_tiers on the training graph; retention is the share of the legacy "
               "Tier >= 1 accounts that BTP keeps in Tier >= 1. eps is the largest value with retention >= "
               "settings.keep (met). If no value qualifies, the values of settings.eps_grid_extended are added "
               "(extended true). If none qualifies, the smallest value evaluated (met false). Honest data only; "
               "no attack data enters the choice.",
    calibration="Ruling R32: keeper.credibility.tiers.calibrate_btp on the training graph at the chosen eps "
                "freezes the seeds S0 (snapshots <= t_seed), K = ceil(log2 n_ref), theta0 = eps / n_ref (n_ref "
                "= the training graph's active accounts) and edge_cap = edge_cap_units * theta0. Every "
                "attacked graph below uses these values unchanged.",
    btp="keeper.credibility.tiers.btp_tiers with the frozen calibration on the training graph (identical to "
        "calibrating on it): trust starts at 1/|S0| on each seed and spreads for K steps over the endorsement "
        "graph (pairs whose last rating is positive); each step an account sends half its trust, split "
        "equally over its endorsees (each endorsement capped at edge_cap in the capped variants), and keeps "
        "the rest; an account without endorsees keeps all. Endorsed: trust >= theta0 and positive in-edges "
        "from trusted raters outnumber the negative in-edges from trusted raters (neg_from 'trusted') or from "
        "every rater ('all'); rating edges, repeats included. Tiers 1 to 3: terciles of trust over the "
        "endorsed accounts (tie rule of legacy_levels). Tier mix over the active accounts.",
    overlap="Legacy (legacy_bfs) and BTP Tier >= 1 sets of the same graph; retention = n_both / "
            "n_legacy_tier1.",
    causal="keeper.credibility.tiers.calibrate_btp_causal (row t calibrated on the training edges of snapshots "
           "< t: K(t), theta0(t) = eps / n_ref(t), seeds, edge_cap(t) = edge_cap_units * theta0(t)) and "
           "causal_tiers with that calibration, T = tt rows. tier1_counts[t] counts the Tier >= 1 accounts "
           "of row t; theta0 of the empty row 0 is infinite (stored as null).",
    sybil_test="s fakes (new accounts n..n+s-1) and g attack edges, all positive and in the last training "
               "snapshot, appended after the honest edges with the attack edges first. ring: attack edge a "
               "reaches fake a mod s, and fake_j endorses fake_(j+1 mod s). star: every attack edge reaches "
               "the hub fake_0, which endorses every other fake. random: settings.ring_draws draws "
               "(default_rng(settings.ring_seed)) of max(ring_g) distinct accounts from the pool, the honest "
               "accounts in the legacy endorsed set D whose legacy tier and the variant's BTP tier are both "
               ">= 2. max_trust (ruling R33b): the max(ring_g) honest accounts with the largest B(u) = "
               "sum_{i<K} min(0.5 t_i(u) / (dout(u) + 1), edge_cap) over the honest iterates of the variant's "
               "frozen calibration (keeper.credibility.tiers.endorser_inflow_scores; edge_cap infinite "
               "uncapped; ties to the smaller id), recorded with B(u) and B(u) / theta0 under endorsers. A "
               "case with g attack edges uses the first "
               "g endorsers of its set. Every case applies the variant's frozen calibration and guard (the "
               "fakes receive no negative rating, so the guard changes their admission only through the "
               "variant's eps and the random pool). legacy_admitted "
               "and btp_admitted count fakes in Tier >= 1 under legacy_bfs and btp_tiers, and *_per_edge "
               "divides them by g; btp_trusted counts fakes with trust >= theta0; sybil_trust is the fakes' "
               "total trust; bound = keeper.credibility.tiers.bound_sybil_trust over the iteration history of "
               "the attacked graph (seeds, K and edge_cap of the calibration; each term min(0.5 t_i(u_a) / "
               "dout(u_a), edge_cap) when capped); cap = floor(bound / theta0 + settings.cap_tol) and "
               "cap_per_edge = cap / g; cap_constant = floor(K g edge_cap_units + settings.cap_tol) for the "
               "capped variants (null uncapped); bound_holds: sybil_trust <= bound + settings.bound_tol; "
               "cap_holds: btp_admitted <= cap; cap_constant_holds: cap <= cap_constant. In the summary, "
               "limited_by_region_size marks rows whose largest admission equals s, a lower bound. "
               "uncalibrated_cap_growth: the first max_trust endorser, ring, g = 1, with btp_tiers "
               "recalibrated on each attacked graph (calibration=None, at the chosen eps and the variant's "
               "edge_cap_units), so theta0 and K count the fakes.",
    star_size_probe="Every variant, under its frozen calibration and guard, without the legacy pass: one attack "
                    "edge from "
                    "each of the PROBE_ENDORSERS = 5 accounts with the largest B(u) (ruling R33b) into the hub of "
                    "a star of s fakes, for every s from 2 to floor(K edge_cap_units) + 4 (capped; best sizes "
                    "exact for stars) or for s in PROBE_SIZES_UNCAPPED (uncapped). per_endorser gives each "
                    "endorser's best star (best_s, max_admitted; smallest s on ties) and lower_bound, true when "
                    "that star was admitted whole and size best_s + 1 was not probed; max_admitted, "
                    "argmax_endorser, argmax_s and lower_bound summarise the best endorser.",
    cap_sensitivity="A sensitivity analysis of the cap size, keyed by guard (neg_from): 'all' is the evidence of "
                    "ruling R38, and 'trusted' repeats it under the guard of ruling R56. For each edge_cap_units in "
                    "settings.cap_sensitivity_units, keeper.credibility.tiers.choose_eps rows "
                    "over settings.eps_sensitivity_grid (retention with that cap), the eps the preregistered rule "
                    "picks (rule_eps: settings.eps_grid, extended to settings.eps_grid_extended if no value "
                    "qualifies; rule_met, rule_extended) and per_edge_constant = floor(K edge_cap_units), K of the "
                    "training graph: the most fakes one attack edge can bring into Tier >= 1.",
    guard_change="Ruling R56, honest data: for each cap (uncapped, capped), the guarded variant's tiers against the "
                 "all-raters variant's, each at its own eps under its calibration frozen on the training graph "
                 "(same_eps tells whether the two eps agree). n_union counts the accounts in Tier >= 1 under either; "
                 "n_trusted_only and n_all_only those in Tier >= 1 under one guard only; n_tier_changed those of the "
                 "union whose tier differs; share_tier_changed = n_tier_changed / n_union and "
                 "share_tier1_membership_changed = (n_trusted_only + n_all_only) / n_union. transitions[a][b] counts "
                 "the active accounts in tier a under 'all' and tier b under 'trusted'.",
    demotion_census="Ruling R56, no training, keyed by victim pool (settings.census_pools). Methods: legacy_bfs "
                    "(which counts every rater's negative ratings, neg_from 'all') and every variant, each under its "
                    "calibration frozen on the training graph at its own eps. Per method, settings.census_victims "
                    "victims are drawn without replacement (default_rng(settings.census_seed)) from its own Tier >= 2 "
                    "accounts (pool 'own': the share of the accounts it deems credible that free Sybils demote) or "
                    "from the accounts in Tier >= 2 under every method (pool 'common': the same victims for every "
                    "method, a paired comparison); n_pool counts the candidates. For each k in settings.census_k and "
                    "each victim, one attacked stream: k fresh Sybils "
                    "(accounts n .. n + k - 1, no attack edge) each rate the victim negatively once, appended after "
                    "the honest edges in the last training snapshot; tiers are recomputed on that stream with the "
                    "frozen calibration. Per method and k: n_dropped (victims whose tier drops) and share_dropped, "
                    "n_to_tier0 (victims in Tier 0 afterwards) and share_to_tier0, collateral_total and "
                    "collateral_max (other honest accounts whose tier moves, summed and maximised over the streams), "
                    "n_streams_with_collateral, and n_streams_changed (streams in which any honest tier moves). "
                    "unchanged: no honest tier moved in any stream; it must hold under 'trusted' (the property of "
                    "keeper.credibility.tiers, ruling R56).",
    calibration_table="One row per network and variant, taken from the blocks under networks: the frozen "
                      "calibration (eps, K, n_ref, theta0, edge_cap, n_seeds), the eps rule's outcome (met, "
                      "extended, retention, n_tier1), guarantee_per_edge = floor(K edge_cap_units) (null uncapped), "
                      "best_star = the star-size probe's max_admitted (best_star_lower_bound for the sparse "
                      "uncapped probe), and, for the guarded variants, share_tier_changed_vs_all and "
                      "share_tier1_membership_changed_vs_all from guard_change. The preregistration takes its "
                      "numbers from this table.",
    runtime_s="Wall-clock seconds (time.perf_counter) on provenance.host, one process, CPU: btp_tiers on the "
              "training graph, calibrate_btp_causal over the T prefixes, and causal_tiers over its T rows "
              "with that calibration. Not deterministic.")


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def provenance(inputs: Inputs) -> dict:
    return dict(
        **git_state(),                                          # commit, dirty flag, uncommitted paths
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        host=socket.gethostname(), cpu=_cpu_model(), argv=list(sys.argv), python=platform.python_version(),
        numpy=np.__version__, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        code_sha256=C.CODE_SHA256_AT_IMPORT,
        module_sha256={rel: sha256_file(os.path.join(C.ROOT, rel)) for rel in MODULE_FILES},
        inputs_sha256=dict(sorted(inputs.sha256.items())),
        preregistration_sha256=sha256_file(PREREGISTRATION) if os.path.exists(PREREGISTRATION) else None)


def parse(argv=None):
    ap = argparse.ArgumentParser(prog="python -m keeper.credibility.tiers",
                                 description="BTP credibility tiers report on the CPPR networks (CPU only).")
    ap.add_argument("--report", action="store_true", help="write the report (required)")
    ap.add_argument("--datasets", nargs="+", default=list(NETWORKS), choices=NETWORKS)
    ap.add_argument("--out", default=os.path.join(ANALYSIS, "tiers_report.json"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    a = ap.parse_args(argv)
    if not a.report:
        ap.error("pass --report to write the tiers report")
    return a


def main(argv=None) -> dict:
    a = parse(argv)
    out_path = C.result_path(os.path.abspath(a.out), a.force)      # refuse before the work, not after it
    inputs = Inputs()
    result = analyse(a.datasets, inputs)
    result["settings"] = SETTINGS
    result["method"] = METHOD
    result["provenance"] = provenance(inputs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    C.dump(result, out_path, force=a.force)
    print(f"wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
