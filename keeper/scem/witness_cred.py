"""Upgrade M-B: SCEM witnesses weighted by their causal BTP credibility tier (Task 3.3, design-M-B.md).

Engine rule (Eq. 11 with the credibility factor). A rating u -> v at snapshot t registers witness evidence
about v with the reliability
    t <= t_seed:               kap = max(kappa_u, 0.35)                (bootstrap: the stored rule)
    t > t_seed, tier >= 1:     kap = w[tier] * max(kappa_u, 0.20)
    t > t_seed, tier 0:        kap = w[0] * kappa_u                    (the floor is removed for Tier 0)
Here tier is row t of the causal BTP tiers (computed on snapshots < t, the convention of kappa(t_j^-)) and
w = TIER_WEIGHTS. Without witness levels the engine keeps its stored rule, max(kappa_u, 0.35 if t <= 4 else
0.20).

Module rule. The SCEM module multiplies its witness term by a [T, n] tensor. level_weights gives that tensor:
1 for t <= t_seed, else w[tier]. These are the weights of keeper.credibility.tiers.causal_weights.

Levels of a stream (ruling R32). witness_levels computes the causal tiers under a calibration frozen on the
clean stream (calibrate_btp_causal), and it does so for an attacked stream too. Injected accounts therefore
never become seeds and cannot move theta0(t) or K(t). Capped BTP is the default (edge_cap_units =
KEEPER_EDGE_CAP_UNITS, rulings R37 and R38), and edge_cap_units=None gives the uncapped ablation. The
credible-negatives guard is the default as well (neg_from = KEEPER_NEG_FROM = "trusted", ruling R56): only
trusted raters' negative ratings count against eligibility, so Sybils without attack edges cannot demote an
honest warner; neg_from="all" is the ablation. Callers pass the network's eps for the chosen variant: capped
0.02, 0.01 and 0.5 for Bitcoin-Alpha, Bitcoin-OTC and Wiki-RfA, uncapped 0.05, 0.05 and 0.5 (unchanged by
ruling R56, results/analysis/tiers_report.json).

Property 1 (bounded adversarial hazard). Each witness adds at most kap <= 1 to the evidence on its subject,
and a Tier-0 witness at most w[0] * kappa <= w[0] after t_seed. So N1 witnesses in Tier >= 1 and N0 in Tier 0
raise the subject's evidence by at most N1 + w[0] N0, and its hazard by at most the engine's hazard map of
L_h times that amount. This is the direct channel; keeper.scem.property1.property1_bound turns it into a
per-record bound on the revoked share and states the conditions under which it holds (keeper/scem/property1.py).
"""
from types import SimpleNamespace

import numpy as np

from keeper.credibility.tiers import (KEEPER_EDGE_CAP_UNITS, KEEPER_NEG_FROM, T_SEED, TIER_WEIGHTS,
                                      calibrate_btp_causal, causal_tiers)
from keeper.laundering.signed import one_hot_from_sign

SEED_FLOOR = 0.35      # the engine's witness floor up to t_seed (stored rule; kept by M-B)
FLOOR = 0.20           # the engine's witness floor after t_seed (kept by M-B for Tier >= 1 only)


def mb_witness_kappa(kappa: float, t: int, level: int, t_seed: int = T_SEED, tier_weights=TIER_WEIGHTS) -> float:
    """The witness reliability of a rating at snapshot t under M-B, from the sender's reliability kappa and
    its causal tier `level` (row t)."""
    if t <= t_seed:
        return max(kappa, SEED_FLOOR)
    if level >= 1:
        return tier_weights[level] * max(kappa, FLOOR)
    return tier_weights[0] * kappa


def check_witness_levels(levels, K_snap: int, n_nodes: int, n_tiers: int = len(TIER_WEIGHTS)) -> np.ndarray:
    """The engine's witness_cred as int64 after checks: two dimensions, at least K_snap rows and n_nodes
    columns, and integer tiers in [0, n_tiers)."""
    arr = np.asarray(levels)
    if arr.ndim != 2 or arr.shape[0] < K_snap or arr.shape[1] < n_nodes:
        raise ValueError(f"witness levels must have shape [T >= {K_snap}, >= {n_nodes}], got {list(arr.shape)}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"witness levels must be integer tiers, got dtype {arr.dtype}")
    if arr.size and (arr.min() < 0 or arr.max() >= n_tiers):
        raise ValueError(f"witness levels must lie in [0, {n_tiers}), got [{arr.min()}, {arr.max()}]")
    return arr.astype(np.int64, copy=False)


def check_witness_weights(weights, n_snapshots: int, n_nodes: int) -> None:
    """The SCEM module's witness_weights: two dimensions, at least n_snapshots rows, one column per node."""
    shape = tuple(weights.shape)
    if len(shape) != 2 or shape[0] < n_snapshots or shape[1] != n_nodes:
        raise ValueError(f"witness_weights must have shape [T >= {n_snapshots}, {n_nodes}], got {list(shape)}")


def stream_graph(edges, sign, snap) -> SimpleNamespace:
    """A rating stream as the graph object of keeper.credibility.tiers: edges [E, 2], one-hot labels, snap."""
    return SimpleNamespace(edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                           labels=one_hot_from_sign(np.asarray(sign)), snap=np.asarray(snap, dtype=np.int64))


def clean_calibration(edges, sign, snap, n_nodes: int, eps: float, *, edge_cap_units=KEEPER_EDGE_CAP_UNITS,
                      T=None, t_seed: int = T_SEED):
    """The per-snapshot BTP calibration frozen on a clean (honest) stream (ruling R32): calibrate_btp_causal
    with T rows, by default one per snapshot of the stream."""
    snap = np.asarray(snap, dtype=np.int64)
    if T is None:
        if not len(snap):
            raise ValueError("T is required for an empty stream")
        T = int(snap.max()) + 1
    return calibrate_btp_causal(stream_graph(edges, sign, snap), n_nodes, eps, T=T, t_seed=t_seed,
                                edge_cap_units=edge_cap_units)


def witness_levels(edges, sign, snap, n_nodes: int, eps: float, *, clean, edge_cap_units=KEEPER_EDGE_CAP_UNITS,
                   neg_from: str = KEEPER_NEG_FROM, T=None, t_seed: int = T_SEED) -> np.ndarray:
    """[T, n_nodes] int64 witness tiers of a stream, the engine's witness_cred. Row t holds the causal BTP tiers
    on snapshots < t, under the calibration frozen on the clean stream (ruling R32).

    clean = (edges, sign, snap) of the honest stream, required so that R32 cannot be broken by omission; for a
    clean stream, pass the stream itself. With an attacked stream, pass n_nodes of the attacked stream.
    edge_cap_units = KEEPER_EDGE_CAP_UNITS gives capped BTP (KEEPER's default), None the uncapped ablation; eps is
    the network's eps for that variant. neg_from is the eligibility guard (ruling R56): "trusted" (the default)
    counts only trusted raters' negative ratings, "all" every rater's. T defaults to the number of snapshots of
    the clean stream.
    """
    calibration = clean_calibration(*clean, n_nodes, eps, edge_cap_units=edge_cap_units, T=T, t_seed=t_seed)
    return causal_tiers(stream_graph(edges, sign, snap), n_nodes, t_seed=t_seed, calibration=calibration,
                        neg_from=neg_from)


def level_weights(levels, t_seed: int = T_SEED, tier_weights=TIER_WEIGHTS) -> np.ndarray:
    """[T, n] float64 witness weights for the SCEM module: 1 for t <= t_seed (bootstrap), else
    tier_weights[level]. For levels from witness_levels, this equals keeper.credibility.tiers.causal_weights."""
    weights = np.asarray(tier_weights, dtype=np.float64)[np.asarray(levels, dtype=np.int64)]
    weights[: t_seed + 1] = 1.0
    return weights
