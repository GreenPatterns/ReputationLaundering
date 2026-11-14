"""Credibility tiers of raters, in one module so that CPPR and SCEM share one definition.

A rater's tier c(u) in {0, 1, 2, 3} sets the weight TIER_WEIGHTS[c(u)] of its ratings in CPPR's credibility
gate. Tier 0 holds unendorsed accounts, possibly Sybils; Tiers 1 to 3 are the terciles of the endorsed
accounts.

Legacy tiers (ruling R6). legacy_levels is keeper.cppr.cp_vpe_gpu.compute_credibility_levels moved here
verbatim, both branches; TIER_WEIGHTS is that module's CRED_WEIGHTS. cp_vpe_gpu imports both back under
their old names, so its callers and every stored result are unchanged, and tests/test_credibility.py
checks the move against the original function at tag pre-revision-2026-10-26. legacy_bfs is the seeded
variant that CPPR uses:

* Seeds: accounts whose training in-edges in snapshots 0 to 4 are more often positive than negative.
* Endorsed set D: one pass over the training edges in their stored (time) order adds the target of every
  positive rating whose sender is already in D. It is not a closure: a rating read before its sender
  joins D admits nobody. One positive rating from a member of D admits its recipient, and a ring of
  accounts that rate each other positively afterwards follows it into D.
* Tiers: an account is endorsed when its positive in-edges from members of D outnumber all its negative
  in-edges. Tier 0 holds the others; Tiers 1 to 3 are the terciles, over the endorsed accounts, of the
  Laplace score (p + 1) / (p + n + 2) times log(1 + p), with p and n those two counts.

This module imports nothing from keeper.cppr, so SCEM and CPPR can both import it without a cycle. The
Sybil-bounded tiers of the revision (Task 3.1) are added here next to the legacy ones.

The tests named above were split on 2026-10-27: tests/test_credibility_legacy.py checks the move, and
tests/test_credibility_btp.py and tests/test_tiers_report.py check the BTP tiers and their report.

BTP tiers (Task 3.1, upgrade M-A). Budgeted trust propagation, in the style of SybilRank but over directed
endorsements, replaces the unbounded endorsement pass:

* Endorsement graph H: the pairs u -> v (u != v) whose last rating of v by u in scope is positive.
* Trust: 1/|S0| on each legacy seed. For K = ceil(log2 |V_active|) steps, every account keeps half its
  trust and splits the other half equally over its endorsees; an account without endorsees keeps all of
  it. Total trust stays 1.
* Tiers: an account is trusted when its trust reaches theta0 = eps / |V_active|. It is endorsed when it is
  trusted and its positive ratings from trusted accounts outnumber its negative ratings from trusted
  accounts (the distrust guard of Eq. 14, restricted to credible raters by ruling R56). Tiers 1 to 3 are
  the terciles of trust over the endorsed accounts.

Calibration (ruling R32). The seeds S0, K and |V_active| of theta0 form a calibration. calibrate_btp
freezes it on the honest graph, and calibrate_btp_causal freezes one per snapshot prefix.
btp_tiers(..., calibration=c) and causal_tiers(..., calibration=c) apply a frozen calibration unchanged to
any graph, fakes included. Without one, they calibrate on the graph they are given, which is right for
honest data only: on an attacked graph the fakes would count in |V_active| and could become seeds.

Trust enters a Sybil region only along honest -> Sybil endorsements, the attack edges. So, if no seed is a
Sybil, the region holds exactly 0 trust without attack edges, and with them at most bound_sybil_trust of
the iteration history. At most floor(bound / theta0) Sybils can then be trusted. Under a frozen
calibration, no Sybil is a seed, and theta0 and K do not depend on the fakes. The bound depends only on the
iterates of the attack edges' senders, so if the region endorses no honest account, the cap is the same
for every region size (tests/test_credibility_btp.py). It scales with the trust per out-edge of the
compromised endorsers. choose_eps fixes eps by the preregistered honest-data rule, and causal_tiers gives
the tiers of each snapshot from the edges before it.

Per-endorsement cap (ruling R37). With edge_cap_units, each endorsement carries at most c_e =
edge_cap_units * theta0 per step, and the sender keeps the rest. The guarantee needs three conditions:
* the calibration (seeds, K, theta0) is frozen on the honest graph;
* no Sybil is a seed;
* every endorsement into the Sybil region is counted as one of its g attack edges u_a -> s_a.

The region's trust after K steps is then at most sum_{i<K} sum_a min(t_i(u_a) / (2 dout(u_a)), c_e) <=
K g c_e, so at most floor(K g edge_cap_units) of its fakes reach Tier >= 1. This is a constant per attack
edge, whatever the endorsers' trust, the region's size or its topology. Without edge_cap_units the approved
uncapped rule runs unchanged; it is the ablation. KEEPER_EDGE_CAP_UNITS = 2 is KEEPER's cap (ruling R38).
endorser_inflow_scores gives each account's B(u), the trust one new endorsement from it would carry. The
strongest endorsers by B(u) are the attacker's best choice (ruling R33b).

Credible-negatives guard (ruling R56). Only credible accounts affect credibility. Positive ratings count in
the guard only from trusted raters, and with neg_from="trusted", KEEPER_NEG_FROM, so do negative ones.
neg_from="all" counts every rater's negative ratings, the rule of Task 3.1, kept as the ablation. Under a
frozen calibration, a set of accounts that receives ratings only from its own members and holds no seed
holds zero trust at every step, so its members are untrusted. Adding such accounts, in any number and with
ratings of either sign to anyone, then changes no other account's trust, eligibility or tier, and leaves
the tercile cut points fixed. The reason is threefold: accounts with zero trust send nothing, the other
accounts' out-degrees do not change, and neither count of the guard includes an untrusted rater. Under
"all", such free Sybils push honest accounts out of the eligible set by rating them negatively. The legacy
tiers count every rater's negative ratings as well and keep that weakness; keeper.analysis.tiers_report
measures it (the demotion census). The guard counts rating edges, repeats included, as pos_in does.
"""
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from keeper.laundering import signed as S

if TYPE_CHECKING:                     # the legacy signature annotates Tr as "C.Graph"
    from keeper.laundering import core as C  # noqa: F401

TIER_WEIGHTS = (0.01, 0.40, 0.70, 1.00)  # Level 0 (unendorsed/Sybil), Levels 1..3 (endorsed terciles)


def legacy_levels(Tr: "C.Graph", n_nodes: int, seeded: bool = True) -> np.ndarray:
    """Compute rater credibility level c(u) in {0, 1, 2, 3} from INCOMING training edges."""
    y_neg = S.class_ids(Tr.labels)
    pos_in = np.zeros(n_nodes, dtype=np.int64)
    neg_in = np.zeros(n_nodes, dtype=np.int64)

    if not seeded:
        for (u, v), is_neg in zip(Tr.edges, y_neg):
            if is_neg:
                neg_in[int(v)] += 1
            else:
                pos_in[int(v)] += 1
    else:
        early_pos = np.zeros(n_nodes, dtype=np.int64)
        early_neg = np.zeros(n_nodes, dtype=np.int64)
        for (u, v), is_neg, sn in zip(Tr.edges, y_neg, Tr.snap):
            if int(sn) <= 4:
                if is_neg:
                    early_neg[int(v)] += 1
                else:
                    early_pos[int(v)] += 1
        trusted_endorsers = set(np.where(early_pos > early_neg)[0].tolist())
        for (u, v), is_neg in zip(Tr.edges, y_neg):
            if not is_neg and int(u) in trusted_endorsers:
                trusted_endorsers.add(int(v))

        for (u, v), is_neg in zip(Tr.edges, y_neg):
            if is_neg:
                neg_in[int(v)] += 1
            elif int(u) in trusted_endorsers:
                pos_in[int(v)] += 1

    levels = np.zeros(n_nodes, dtype=np.int64)
    endorsed = np.where(pos_in > neg_in)[0]
    if len(endorsed) > 0:
        laplace = (pos_in[endorsed] + 1.0) / (pos_in[endorsed] + neg_in[endorsed] + 2.0)
        score = laplace * np.log1p(pos_in[endorsed])
        q1, q2 = np.quantile(score, [1.0 / 3.0, 2.0 / 3.0])
        for idx, u in enumerate(endorsed):
            sc = score[idx]
            if sc <= q1:
                levels[int(u)] = 1
            elif sc <= q2:
                levels[int(u)] = 2
            else:
                levels[int(u)] = 3
    return levels


def legacy_bfs(Tr: "C.Graph", n_nodes: int) -> np.ndarray:
    """The legacy seeded tiers that CPPR uses: legacy_levels(Tr, n_nodes, seeded=True)."""
    return legacy_levels(Tr, n_nodes, seeded=True)


# ================================================================================ BTP tiers (upgrade M-A)
T_SEED = 4                               # seeds: majority-positive in-edges in snapshots 0..T_SEED (legacy)
LAZINESS = 0.5                           # share of its trust an account with endorsees passes on per step
EPS_GRID = (1.0, 0.5, 0.2, 0.1, 0.05)    # candidate eps of theta0 = eps / |V_active| (preregistered rule)
EPS_GRID_EXTENDED = EPS_GRID + (0.02, 0.01, 0.005)   # R37: fallback for the capped variant
EPS_KEEP = 0.90                          # choose_eps: keep >= 90% of the legacy Tier >= 1 accounts
# Ruling R38 (honest data only, before any attack experiment): KEEPER's per-endorsement cap is the smallest
# edge_cap_units in (1, 2, 4, 8) for which the preregistered rule (keep >= 90% of the legacy Tier >= 1 set,
# eps grid extended to 0.005) holds on Bitcoin-Alpha, Bitcoin-OTC and Wiki-RfA. At 1, Bitcoin-OTC levels off
# at 86.2%. At 2 the rule picks eps = 0.02, 0.01, 0.5, and at most floor(2 K g) fakes reach Tier >= 1.
KEEPER_EDGE_CAP_UNITS = 2.0
# Ruling R56: whose negative ratings the distrust guard counts. "trusted" (KEEPER's default): only raters with trust
# >= theta0, the raters whose positive ratings count; "all": every rater (Task 3.1's rule, kept as the ablation).
NEG_FROM = ("trusted", "all")
KEEPER_NEG_FROM = "trusted"
CAUSAL_T = 21                            # causal_tiers rows: the training snapshots 0..20 of the CPPR networks
_MAX_KEY_BASE = 3_037_000_499            # floor(sqrt(2**63 - 1)): pair keys u * base + v fit in int64


def _signed_arrays(Tr, n_nodes=None) -> tuple:
    """(edges [E, 2] int64, is_neg [E] bool, snap [E] int64) of a graph with edges, labels and snap."""
    edges = np.asarray(Tr.edges, dtype=np.int64).reshape(-1, 2)
    is_neg = S.neg_mask(Tr.labels).reshape(-1) if len(edges) else np.zeros(0, dtype=bool)
    snap = np.asarray(Tr.snap, dtype=np.int64).reshape(-1)
    if not len(edges) == len(is_neg) == len(snap):
        raise ValueError(f"edges ({len(edges)}), labels ({len(is_neg)}) and snap ({len(snap)}) differ in length")
    if n_nodes is not None and len(edges) and (edges.min() < 0 or edges.max() >= n_nodes):
        raise ValueError(f"edge endpoints must lie in [0, {n_nodes})")
    return edges, is_neg, snap


def _in_scope(snap: np.ndarray, t_max, strict: bool) -> np.ndarray:
    """Edges in scope: all if t_max is None, else snapshot <= t_max (strict: < t_max)."""
    if t_max is None:
        return np.ones(len(snap), dtype=bool)
    return snap < t_max if strict else snap <= t_max


def _seeds(edges, is_neg, snap, n_nodes: int, t_seed: int) -> np.ndarray:
    early = snap <= t_seed
    early_pos = np.bincount(edges[early & ~is_neg, 1], minlength=n_nodes)
    early_neg = np.bincount(edges[early & is_neg, 1], minlength=n_nodes)
    return np.flatnonzero(early_pos > early_neg).astype(np.int64)


def _endorsements(edges, is_neg) -> np.ndarray:
    """One row (u, v), sorted, per pair u != v whose last row (in stored order) is a positive rating."""
    if not len(edges):
        return np.zeros((0, 2), dtype=np.int64)
    base = int(edges.max()) + 1
    if base > _MAX_KEY_BASE:
        raise ValueError(f"node ids up to {base - 1} overflow the int64 pair keys")
    key = edges[:, 0] * base + edges[:, 1]
    _, first_from_end = np.unique(key[::-1], return_index=True)
    last = len(key) - 1 - first_from_end               # each pair's last row, in ascending key order
    keep = ~is_neg[last] & (edges[last, 0] != edges[last, 1])
    return edges[last[keep]]


def _check_eps(eps) -> float:
    if eps is None:
        raise ValueError("eps is required: choose it with choose_eps (the preregistered honest-data rule)")
    eps = float(eps)
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be positive and finite, got {eps}")
    return eps


def _check_K(K) -> int:
    if isinstance(K, bool) or int(K) != K or K < 0:
        raise ValueError(f"K must be a non-negative integer, got {K!r}")
    return int(K)


def default_K(n_act: int) -> int:
    """K = ceil(log2 n_act), exact on integers; 0 for at most one active account."""
    return max(int(n_act) - 1, 0).bit_length()


def _check_count(value, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def _check_neg_from(neg_from) -> str:
    """One of NEG_FROM (ruling R56)."""
    if not (isinstance(neg_from, str) and neg_from in NEG_FROM):
        raise ValueError(f"neg_from must be one of {NEG_FROM}, got {neg_from!r}")
    return neg_from


def _check_units(edge_cap_units):
    """None (uncapped) or a positive, finite number of theta0 units per endorsement and step."""
    if edge_cap_units is None:
        return None
    units = float(edge_cap_units)
    if not (math.isfinite(units) and units > 0):
        raise ValueError(f"edge_cap_units must be positive and finite, got {edge_cap_units}")
    return units


@dataclass(frozen=True)
class BTPCalibration:
    """A BTP calibration frozen on honest data (ruling R32): the seeds S0, K, and theta0 = eps / n_ref.

    The operator calibrates once on the honest graph (calibrate_btp). btp_tiers(..., calibration=c) applies
    these values unchanged to any graph, so accounts that join later can neither become seeds nor move K or
    theta0. seeds holds sorted distinct account ids; n_ref is the honest graph's active-account count.
    edge_cap_units (ruling R37) caps each endorsement's flow per step at edge_cap = edge_cap_units * theta0;
    None leaves the propagation uncapped.
    """
    seeds: tuple = field(repr=False)
    K: int
    eps: float
    n_ref: int
    t_seed: int = T_SEED
    edge_cap_units: float | None = None

    def __post_init__(self):
        seeds = tuple(sorted({int(u) for u in self.seeds}))
        if len(seeds) != len(self.seeds) or (seeds and seeds[0] < 0):
            raise ValueError("seeds must be distinct non-negative account ids")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "K", _check_K(self.K))
        object.__setattr__(self, "eps", _check_eps(self.eps))
        object.__setattr__(self, "n_ref", _check_count(self.n_ref, "n_ref"))
        object.__setattr__(self, "t_seed", int(self.t_seed))
        object.__setattr__(self, "edge_cap_units", _check_units(self.edge_cap_units))

    @property
    def theta0(self) -> float:
        """eps / n_ref; infinite without active accounts, so that nobody is trusted."""
        return self.eps / self.n_ref if self.n_ref else math.inf

    @property
    def edge_cap(self) -> float | None:
        """edge_cap_units * theta0, the most trust one endorsement carries per step; None when uncapped."""
        return None if self.edge_cap_units is None else self.edge_cap_units * self.theta0

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)


@dataclass(frozen=True)
class BTPCausalCalibration:
    """Per-snapshot BTP calibrations frozen on honest data (ruling R32).

    rows[t] calibrates the honest strict prefix (snapshot < t): its seeds, K(t) and theta0(t) = eps /
    n_active_honest(t). causal_tiers(..., calibration=c) applies rows[t] to row t of any graph.
    """
    rows: tuple

    def __post_init__(self):
        rows = tuple(self.rows)
        if not rows or not all(isinstance(row, BTPCalibration) for row in rows):
            raise ValueError("rows must be a non-empty sequence of BTPCalibration")
        if len({(row.eps, row.t_seed, row.edge_cap_units) for row in rows}) != 1:
            raise ValueError("every row must share eps, t_seed and edge_cap_units")
        object.__setattr__(self, "rows", rows)

    @property
    def T(self) -> int:
        return len(self.rows)

    @property
    def edge_cap_units(self) -> float | None:
        return self.rows[0].edge_cap_units

    @property
    def eps(self) -> float:
        return self.rows[0].eps

    @property
    def t_seed(self) -> int:
        return self.rows[0].t_seed

    @property
    def K(self) -> tuple:
        return tuple(row.K for row in self.rows)

    @property
    def theta0(self) -> tuple:
        return tuple(row.theta0 for row in self.rows)

    @property
    def n_ref(self) -> tuple:
        return tuple(row.n_ref for row in self.rows)

    @property
    def seeds(self) -> tuple:
        """The last row's seeds: the honest S0 of snapshots <= t_seed when T > t_seed + 1."""
        return self.rows[-1].seeds


def _active_count(edges: np.ndarray, n_nodes: int) -> int:
    active = np.zeros(n_nodes, dtype=bool)
    active[edges.reshape(-1)] = True
    return int(active.sum())


def _calibrate(edges, is_neg, snap, n_nodes: int, eps: float, t_seed: int, K, edge_cap_units=None) -> BTPCalibration:
    """The calibration of the given (already scoped) edges: their seeds, K and active count."""
    n_ref = _active_count(edges, n_nodes)
    return BTPCalibration(seeds=tuple(_seeds(edges, is_neg, snap, n_nodes, t_seed).tolist()),
                          K=default_K(n_ref) if K is None else _check_K(K), eps=eps, n_ref=n_ref, t_seed=t_seed,
                          edge_cap_units=edge_cap_units)


def calibrate_btp(Tr_honest: "C.Graph", n_nodes: int, eps: float, t_seed: int = T_SEED, K=None,
                  edge_cap_units=None) -> BTPCalibration:
    """Freeze the BTP calibration on the honest graph: the seeds S0 of snapshots <= t_seed, K = ceil(log2
    n_ref) unless K is given, and theta0 = eps / n_ref, with n_ref the honest graph's active accounts. With
    edge_cap_units, each endorsement carries at most edge_cap_units * theta0 per step (ruling R37)."""
    edges, is_neg, snap = _signed_arrays(Tr_honest, n_nodes)
    return _calibrate(edges, is_neg, snap, n_nodes, _check_eps(eps), t_seed, K, edge_cap_units)


def calibrate_btp_causal(Tr_honest: "C.Graph", n_nodes: int, eps: float, T: int = CAUSAL_T,
                         t_seed: int = T_SEED, edge_cap_units=None) -> BTPCausalCalibration:
    """Freeze one calibration per snapshot t < T on the honest strict prefix (snapshot < t): its seeds,
    K(t) = ceil(log2 n_active_honest(t)) and theta0(t) = eps / n_active_honest(t); with edge_cap_units, the
    cap of row t is edge_cap_units * theta0(t)."""
    eps = _check_eps(eps)
    edges, is_neg, snap = _signed_arrays(Tr_honest, n_nodes)
    rows = []
    for t in range(int(T)):
        prefix = snap < t
        rows.append(_calibrate(edges[prefix], is_neg[prefix], snap[prefix], n_nodes, eps, t_seed, None,
                               edge_cap_units))
    return BTPCausalCalibration(tuple(rows))


def _check_calibration(calibration, eps, t_seed: int, K, edge_cap_units=None) -> BTPCalibration:
    """A given calibration fixes eps, K, t_seed and edge_cap_units; any of them passed as well must agree."""
    if not isinstance(calibration, BTPCalibration):
        raise TypeError(f"calibration must be a BTPCalibration, got {type(calibration).__name__}")
    if eps is not None and _check_eps(eps) != calibration.eps:
        raise ValueError(f"eps {eps} differs from the calibration's {calibration.eps}")
    if K is not None and _check_K(K) != calibration.K:
        raise ValueError(f"K {K} differs from the calibration's {calibration.K}")
    if t_seed != calibration.t_seed:
        raise ValueError(f"t_seed {t_seed} differs from the calibration's {calibration.t_seed}")
    if edge_cap_units is not None and _check_units(edge_cap_units) != calibration.edge_cap_units:
        raise ValueError(f"edge_cap_units {edge_cap_units} differs from the calibration's "
                         f"{calibration.edge_cap_units}")
    return calibration


def seed_set(Tr: "C.Graph", n_nodes: int, t_seed: int = T_SEED) -> np.ndarray:
    """The legacy seeds S0, sorted: accounts whose positive in-edges in snapshots <= t_seed outnumber their
    negative ones (the initialisation of legacy_levels)."""
    edges, is_neg, snap = _signed_arrays(Tr, n_nodes)
    return _seeds(edges, is_neg, snap, n_nodes, t_seed)


def endorsement_edges(Tr: "C.Graph", t_max=None, strict: bool = False) -> np.ndarray:
    """The endorsement graph H as [M, 2] int64 rows (u, v), sorted, one per pair: u != v and the last rating
    of v by u among the edges in scope (snapshot <= t_max, or < t_max if strict; all if t_max is None) is
    positive. "Last" is the stored order, which is time order."""
    edges, is_neg, snap = _signed_arrays(Tr)
    in_scope = _in_scope(snap, t_max, strict)
    return _endorsements(edges[in_scope], is_neg[in_scope])


def n_active(Tr: "C.Graph", t_max=None, strict: bool = False) -> int:
    """|V_active|: the distinct endpoints of the edges in scope."""
    edges, _, snap = _signed_arrays(Tr)
    return int(len(np.unique(edges[_in_scope(snap, t_max, strict)])))


def _check_endorsements(n_nodes: int, endorse) -> np.ndarray:
    endorse = np.asarray(endorse, dtype=np.int64).reshape(-1, 2)
    if len(endorse):
        if endorse.min() < 0 or endorse.max() >= n_nodes:
            raise ValueError(f"endorsement endpoints must lie in [0, {n_nodes})")
        if (endorse[:, 0] == endorse[:, 1]).any():
            raise ValueError("an account cannot endorse itself")
        if n_nodes > _MAX_KEY_BASE:
            raise ValueError(f"{n_nodes} accounts overflow the int64 pair keys")
        if len(np.unique(endorse[:, 0] * n_nodes + endorse[:, 1])) != len(endorse):
            raise ValueError("each endorsement pair must appear once (see endorsement_edges)")
    return endorse


def _check_edge_cap(edge_cap):
    """None (no cap) or a positive cap per endorsement and step; infinity never binds."""
    if edge_cap is None:
        return None
    edge_cap = float(edge_cap)
    if not edge_cap > 0:                               # rejects 0, negatives and NaN
        raise ValueError(f"edge_cap must be positive, got {edge_cap}")
    return edge_cap


def btp_trust(n_nodes: int, endorse, seeds, K: int, laziness: float = LAZINESS,
              return_history: bool = False, edge_cap=None):
    """Budgeted trust propagation on the endorsement graph ``endorse`` ([M, 2] unique pairs, no self-loops).

    t^(0) = 1/|S0| on each seed. At each of K steps every account keeps (1 - laziness) of its trust and
    splits ``laziness`` of it equally over its endorsees; an account without endorsees keeps all of it, so
    the total is conserved. Without seeds every value is 0. Returns t^(K), or with return_history the list
    of iterates t^(0), ..., t^(K). Time O(K * M), memory O(n_nodes) (O(K * n_nodes) with the history).

    ``edge_cap`` (ruling R37) caps each endorsement's flow per step at min(laziness * t(u) / dout(u),
    edge_cap). The sender keeps everything it does not send, so trust is still conserved and sinks keep all.
    edge_cap=None runs the approved uncapped rule unchanged; an infinite cap gives the same iterates bitwise.
    """
    endorse = _check_endorsements(n_nodes, endorse)
    seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
    if len(seeds) and (seeds.min() < 0 or seeds.max() >= n_nodes or len(np.unique(seeds)) != len(seeds)):
        raise ValueError(f"seeds must be distinct accounts in [0, {n_nodes})")
    K = _check_K(K)
    if not 0.0 <= laziness <= 1.0:
        raise ValueError(f"laziness must lie in [0, 1], got {laziness}")
    edge_cap = _check_edge_cap(edge_cap)
    trust = np.zeros(n_nodes, dtype=np.float64)
    if len(seeds):
        trust[seeds] = 1.0 / len(seeds)
    src, dst = endorse[:, 0], endorse[:, 1]
    out_deg = np.bincount(src, minlength=n_nodes)
    kept = np.where(out_deg > 0, 1.0 - laziness, 1.0)
    edge_share = laziness / out_deg[src]               # share of its sender's trust each endorsement carries
    if edge_cap is not None:                           # the same share per sender, to find the capped senders
        node_share = np.zeros(n_nodes, dtype=np.float64)
        node_share[out_deg > 0] = laziness / out_deg[out_deg > 0]
    history = [trust]
    for _ in range(K):
        flow = trust[src] * edge_share
        if edge_cap is None:
            trust = kept * trust + np.bincount(dst, weights=flow, minlength=n_nodes)
        else:
            capped = trust * node_share > edge_cap     # senders whose every endorsement hits the cap
            stays = kept * trust
            stays[capped] = trust[capped] - out_deg[capped] * edge_cap
            trust = stays + np.bincount(dst, weights=np.minimum(flow, edge_cap), minlength=n_nodes)
        history.append(trust)
    return history if return_history else trust


def tercile_levels(score: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Levels 1-3 from the terciles of score over the eligible accounts, with the tie rule of legacy_levels
    (score <= q1 -> 1, <= q2 -> 2, else 3); 0 for every other account."""
    levels = np.zeros(len(score), dtype=np.int64)
    idx = np.flatnonzero(eligible)
    if len(idx):
        sc = np.asarray(score, dtype=np.float64)[idx]
        q1, q2 = np.quantile(sc, [1.0 / 3.0, 2.0 / 3.0])
        levels[idx] = np.where(sc <= q1, 1, np.where(sc <= q2, 2, 3))
    return levels


def btp_tiers(Tr: "C.Graph", n_nodes: int, eps=None, t_max=None, t_seed: int = T_SEED, K=None,
              strict: bool = False, calibration=None, edge_cap_units=None, neg_from: str = KEEPER_NEG_FROM) -> tuple:
    """BTP credibility levels c(u) in {0, 1, 2, 3} from the edges in scope: (levels, trust, info).

    The calibration supplies the seeds, K, theta0 and the per-endorsement cap. With ``calibration`` (a
    BTPCalibration frozen on the honest graph, ruling R32) they are used unchanged, and eps, K, t_seed and
    edge_cap_units, if passed, must agree with it. Without one, the edges in scope calibrate themselves,
    which is right for honest data only: seeds from their snapshots <= t_seed, K = ceil(log2 n_active)
    unless given, theta0 = eps / n_active, and with edge_cap_units a cap of edge_cap_units * theta0 (ruling
    R37). eps is then required, positive and finite.

    Trust is btp_trust on the endorsement graph of the edges in scope. An account is trusted when trust >=
    theta0, and endorsed when it is trusted and its positive in-edges from trusted raters outnumber its
    negative in-edges (edges in scope, repeats counted). ``neg_from`` (ruling R56) says which negative in-edges
    count: those from trusted raters ("trusted", KEEPER's default), so that untrusted accounts cannot remove
    eligibility, or all of them ("all", the rule of Task 3.1, kept as the ablation). Levels are the terciles of
    trust over the endorsed accounts (tercile_levels). info = dict(K, theta0, n_active, n_ref, n_seeds,
    n_trusted, n_eligible, edge_cap, neg_from); n_active counts the accounts in scope and n_ref those of the
    calibration. Time O(E log E) to build the endorsement graph from the E edges in scope, then O(K * |E+|) to
    propagate.
    """
    neg_from = _check_neg_from(neg_from)
    edges, is_neg, snap = _signed_arrays(Tr, n_nodes)
    in_scope = _in_scope(snap, t_max, strict)
    edges, is_neg, snap = edges[in_scope], is_neg[in_scope], snap[in_scope]
    if calibration is None:
        calibration = _calibrate(edges, is_neg, snap, n_nodes, _check_eps(eps), t_seed, K, edge_cap_units)
    else:
        _check_calibration(calibration, eps, t_seed, K, edge_cap_units)
    trust = btp_trust(n_nodes, _endorsements(edges, is_neg), calibration.seeds, calibration.K,
                      edge_cap=calibration.edge_cap)
    trusted = trust >= calibration.theta0
    from_trusted = trusted[edges[:, 0]]
    pos_in = np.bincount(edges[~is_neg & from_trusted, 1], minlength=n_nodes)
    neg_in = np.bincount(edges[is_neg & from_trusted if neg_from == "trusted" else is_neg, 1], minlength=n_nodes)
    eligible = trusted & (pos_in > neg_in)
    info = dict(K=calibration.K, theta0=float(calibration.theta0), n_active=_active_count(edges, n_nodes),
                n_ref=calibration.n_ref, n_seeds=calibration.n_seeds, n_trusted=int(trusted.sum()),
                n_eligible=int(eligible.sum()), edge_cap=calibration.edge_cap, neg_from=neg_from)
    return tercile_levels(trust, eligible), trust, info


def choose_eps(Tr: "C.Graph", n_nodes: int, grid=EPS_GRID, keep: float = EPS_KEEP, edge_cap_units=None,
               fallback_grid=None, neg_from: str = KEEPER_NEG_FROM) -> tuple:
    """The preregistered honest-data rule: (eps, table).

    eps is the largest grid value whose BTP Tier >= 1 set keeps at least ``keep`` of the legacy Tier >= 1
    set (legacy_bfs) of the same graph. The BTP tiers use edge_cap_units (ruling R37), so a capped variant is
    calibrated with its cap; the cap scales with theta0. They apply the guard ``neg_from`` (ruling R56). If no
    grid value qualifies and ``fallback_grid`` is given, its other values are evaluated too and the largest
    qualifying value of both is taken. If none qualifies, eps is the smallest value evaluated and
    table["met"] is False. table = dict(keep, n_legacy_tier1, rows, met, eps, edge_cap_units, neg_from,
    extended), one row per evaluated value with eps, theta0, K, n_btp_tier1, n_overlap, retention (n_overlap /
    n_legacy_tier1) and meets_keep.
    """
    grid = tuple(_check_eps(e) for e in grid)
    if not grid:
        raise ValueError("the eps grid is empty")
    if not 0.0 < keep <= 1.0:
        raise ValueError(f"keep must lie in (0, 1], got {keep}")
    edge_cap_units = _check_units(edge_cap_units)
    neg_from = _check_neg_from(neg_from)
    legacy = legacy_bfs(Tr, n_nodes) > 0
    n_legacy = int(legacy.sum())
    if n_legacy == 0:
        raise ValueError("the legacy Tier >= 1 set is empty, so the retention is undefined")

    def row_for(eps: float) -> dict:
        levels, _, info = btp_tiers(Tr, n_nodes, eps, edge_cap_units=edge_cap_units, neg_from=neg_from)
        btp = levels > 0
        overlap = int((btp & legacy).sum())
        return dict(eps=eps, theta0=info["theta0"], K=info["K"], n_btp_tier1=int(btp.sum()), n_overlap=overlap,
                    retention=overlap / n_legacy, meets_keep=bool(overlap / n_legacy >= keep))

    rows = [row_for(eps) for eps in grid]
    extended = fallback_grid is not None and not any(row["meets_keep"] for row in rows)
    if extended:
        rows += [row_for(eps) for eps in (_check_eps(e) for e in fallback_grid) if eps not in grid]
    passing = [row["eps"] for row in rows if row["meets_keep"]]
    chosen = max(passing) if passing else min(row["eps"] for row in rows)
    return chosen, dict(keep=float(keep), n_legacy_tier1=n_legacy, rows=rows, met=bool(passing), eps=chosen,
                        edge_cap_units=edge_cap_units, neg_from=neg_from, extended=bool(extended))


def causal_tiers(Tr: "C.Graph", n_nodes: int, eps=None, T=None, t_seed: int = T_SEED,
                 calibration=None, edge_cap_units=None, neg_from: str = KEEPER_NEG_FROM) -> np.ndarray:
    """[T, n_nodes] int64 levels; row t is btp_tiers on the strict prefix (snapshot < t), the convention of
    kappa(t_j^-), so no row uses its own or a later snapshot. Each row costs one btp_tiers call, with the
    guard ``neg_from`` (ruling R56).

    With ``calibration`` (a BTPCausalCalibration frozen on the honest graph, ruling R32), row t uses
    calibration.rows[t], and T defaults to calibration.T. Without one, every prefix calibrates itself (honest
    data only, with the cap edge_cap_units * theta0(t) if edge_cap_units is given), eps is required and T
    defaults to CAUSAL_T.
    """
    neg_from = _check_neg_from(neg_from)
    if calibration is None:
        eps, T = _check_eps(eps), CAUSAL_T if T is None else int(T)
        rows = (None,) * T
    else:
        if not isinstance(calibration, BTPCausalCalibration):
            raise TypeError(f"calibration must be a BTPCausalCalibration, got {type(calibration).__name__}")
        if T is not None and int(T) != calibration.T:
            raise ValueError(f"T = {T} differs from the calibration's {calibration.T} rows")
        T, rows = calibration.T, calibration.rows
    levels = np.zeros((T, n_nodes), dtype=np.int64)
    for t in range(T):
        levels[t] = btp_tiers(Tr, n_nodes, eps, t_max=t, t_seed=t_seed, strict=True, calibration=rows[t],
                              edge_cap_units=edge_cap_units, neg_from=neg_from)[0]
    return levels


def causal_weights(Tr: "C.Graph", n_nodes: int, eps=None, T=None, t_seed: int = T_SEED,
                   calibration=None, edge_cap_units=None, neg_from: str = KEEPER_NEG_FROM) -> np.ndarray:
    """[T, n_nodes] float64 weights: 1.0 for t <= t_seed (bootstrap), else TIER_WEIGHTS[causal level].
    The arguments are those of causal_tiers."""
    levels = causal_tiers(Tr, n_nodes, eps, T, t_seed, calibration=calibration, edge_cap_units=edge_cap_units,
                          neg_from=neg_from)
    weights = np.asarray(TIER_WEIGHTS, dtype=np.float64)[levels]
    weights[: t_seed + 1] = 1.0
    return weights


def endorser_inflow_scores(Tr: "C.Graph", n_nodes: int, calibration: BTPCalibration,
                           laziness: float = LAZINESS) -> np.ndarray:
    """B(u) for every account u: the trust one new endorsement from u would carry over K steps (ruling R33b).

    B(u) = sum_{i<K} min(laziness * t_i(u) / (dout(u) + 1), c_e), from the honest iterates t_i of the frozen
    calibration (its seeds, K and cap c_e, with c_e = inf when uncapped) on Tr's endorsement graph. dout(u) +
    1 counts the new endorsement. B(u) is the attack-edge bound of bound_sybil_trust for one edge from u,
    taken on the honest iterates. Under a cap it is at most K c_e, and a seed that endorses nobody and keeps
    at least 2 c_e (dout + 1 = 1) reaches K c_e. The "max_trust" endorser selection ranks accounts by it.
    """
    if not isinstance(calibration, BTPCalibration):
        raise TypeError(f"calibration must be a BTPCalibration, got {type(calibration).__name__}")
    endorse = endorsement_edges(Tr)
    history = btp_trust(n_nodes, endorse, calibration.seeds, calibration.K, laziness, return_history=True,
                        edge_cap=calibration.edge_cap)
    share = laziness / (np.bincount(endorse[:, 0], minlength=n_nodes) + 1.0)
    cap = math.inf if calibration.edge_cap is None else calibration.edge_cap
    scores = np.zeros(n_nodes, dtype=np.float64)
    for t in history[:-1]:
        scores += np.minimum(t * share, cap)
    return scores


def bound_sybil_trust(history, attack_edges, out_deg, laziness: float = LAZINESS, edge_cap=None) -> float:
    """Bound on a Sybil region's trust after K steps: laziness * sum_{i<K} sum_a t_i(u_a) / dout(u_a).

    ``history`` holds the iterates t^(0..K) of btp_trust, ``attack_edges`` the [g, 2] honest -> Sybil
    endorsements (u_a, s_a) and ``out_deg`` the endorsement out-degrees of the same graph. Edges inside the
    region and edges leaving it carry no trust in, so the bound holds when ``attack_edges`` lists every
    endorsement into the region and the region contains no seed. With ``edge_cap`` (the cap the history was
    propagated with, ruling R37), each term is min(laziness * t_i(u_a) / dout(u_a), edge_cap), so the bound
    is at most K * g * edge_cap.
    """
    history = list(history)
    if not history:
        raise ValueError("history must hold at least t^(0)")
    senders = np.asarray(attack_edges, dtype=np.int64).reshape(-1, 2)[:, 0]
    deg = np.asarray(out_deg)[senders]
    if (deg < 1).any():
        raise ValueError("every attack edge's sender must have at least one endorsee")
    edge_cap = _check_edge_cap(edge_cap)
    if edge_cap is None:
        inverse = 1.0 / deg
        return float(laziness * sum(float(np.dot(t[senders], inverse)) for t in history[:-1]))
    share = laziness / deg
    return float(sum(float(np.minimum(t[senders] * share, edge_cap).sum()) for t in history[:-1]))


def main(argv=None):
    """CLI: python -m keeper.credibility.tiers --report [--force]; the report is keeper.analysis.tiers_report."""
    from keeper.analysis.tiers_report import main as report_main
    return report_main(argv)


if __name__ == "__main__":
    main()
