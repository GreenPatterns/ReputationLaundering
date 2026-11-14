"""The streams of experiment E2 (Task 4.5; docs/preregistration.md, Section 2.4): silent senders against SCEM.

Draws. default_rng(DRAW_ENTROPY + seed) draws, in this order:
  1. N_TARGETS whitewash targets, without replacement, from the cohort of keeper.laundering.core.pick_targets(Tr,
     max_neg=3, strict=True);
  2. N_TARGETS bad-mouth targets, without replacement, from the trusted cohort: accounts with at least 3 trust
     in-edges and no distrust in-edge (the cohort of keeper.cppr.cp_vpe_gpu);
  3. every target's snapshot t_s, uniform on the integers T_S_FIRST..T_S_LAST, in one call (whitewash targets first);
  4. for lag 1 and then for lag 3, each silent sender's two witnesses, without replacement, among the accounts whose
     causal capped tier at t_s + lag is at least WITNESS_MIN_TIER (row t_s + lag of the clean stream's levels: tiers
     on snapshots < t_s + lag), the target excluded. Senders come in target order, three per target.
The draws use the clean training graph and its levels only, so every run of a network and seed replays the same plan.

Stream. Per target, SENDERS_PER_TARGET fresh accounts, appended as node ids n, n + 1, ... (whitewash targets'
first), rate it once at t_s (+1 whitewash, -1 bad-mouth) and do nothing else; each silent sender is rated -1 by its
two witnesses at t_s + lag. One stream per lag holds both kinds of target. The clean twin is the training graph
unchanged. The injected rows take the last timestamp of their snapshot (as keeper.cppr.arm_graphs) and negative row
ids, and the stream is sorted stably by snapshot, so the original rows keep their order and lead every snapshot.
"""
import hashlib
from dataclasses import dataclass

import numpy as np

from keeper.laundering import core as C
from keeper.laundering import signed as S
from keeper.scem.witness_cred import witness_levels

DRAW_ENTROPY = 5000
N_TARGETS = 15                 # per kind
SENDERS_PER_TARGET = 3
WITNESSES_PER_SENDER = 2
T_S_FIRST, T_S_LAST = 5, 14    # t_s, inclusive
LAGS = (1, 3)
WITNESS_MIN_TIER = 2
TRUSTED_MIN_POS = 3            # the trusted cohort: >= 3 trust in-edges and no distrust in-edge
KINDS = ("whitewash", "badmouth")


@dataclass(frozen=True)
class Plan:
    """The draws of one network and seed. witnesses[lag] is [targets, SENDERS_PER_TARGET, WITNESSES_PER_SENDER]."""
    whitewash: np.ndarray
    badmouth: np.ndarray
    t_s: np.ndarray
    witnesses: dict

    @property
    def targets(self) -> np.ndarray:
        return np.concatenate([self.whitewash, self.badmouth])

    @property
    def direction(self) -> np.ndarray:
        """+1 for a whitewash target, -1 for a bad-mouth target: the sign of its silent senders' ratings."""
        return np.repeat(np.array([1, -1], dtype=np.int64), [len(self.whitewash), len(self.badmouth)])

    @property
    def kinds(self) -> list:
        return [KINDS[0]] * len(self.whitewash) + [KINDS[1]] * len(self.badmouth)

    def arrays(self) -> list:
        return [self.whitewash, self.badmouth, self.t_s, *(self.witnesses[lag] for lag in LAGS)]

    def sha256(self) -> str:
        h = hashlib.sha256()
        for a in self.arrays():
            h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
        return h.hexdigest()

    def to_dict(self) -> dict:
        return dict(whitewash=self.whitewash.tolist(), badmouth=self.badmouth.tolist(), t_s=self.t_s.tolist(),
                    witnesses={str(lag): self.witnesses[lag].tolist() for lag in LAGS}, sha256=self.sha256())


def signed(Tr) -> tuple:
    """(edges, sign, snap) of a graph: the rating stream of keeper.scem.witness_cred."""
    return Tr.edges, 1 - 2 * S.class_ids(Tr.labels), Tr.snap


def stream_levels(Tr_stream, Tr_clean, eps: float, T: int) -> np.ndarray:
    """[T, Tr_stream.n] causal capped BTP tiers of a stream under the calibration frozen on the clean stream (ruling
    R32), with witness_cred's defaults: the edge cap KEEPER_EDGE_CAP_UNITS and the guard KEEPER_NEG_FROM."""
    return witness_levels(*signed(Tr_stream), Tr_stream.n, eps, clean=signed(Tr_clean), T=T)


def trusted_cohort(neg: dict, pos: dict) -> np.ndarray:
    """Accounts with at least TRUSTED_MIN_POS trust in-edges and no distrust in-edge (in_edges' maps)."""
    return np.array(sorted(v for v, rows in pos.items() if len(rows) >= TRUSTED_MIN_POS and not neg.get(v)),
                    dtype=np.int64)


def _witnesses(rng, levels: np.ndarray, targets, t_s, lag: int) -> np.ndarray:
    out = np.empty((len(targets), SENDERS_PER_TARGET, WITNESSES_PER_SENDER), dtype=np.int64)
    for i, (v, t) in enumerate(zip(targets, t_s)):
        pool = np.flatnonzero(levels[t + lag] >= WITNESS_MIN_TIER)
        pool = pool[pool != v]
        for j in range(SENDERS_PER_TARGET):
            out[i, j] = rng.choice(pool, WITNESSES_PER_SENDER, replace=False)
    return out


def draw(Tr, levels: np.ndarray, seed: int) -> Plan:
    """The plan of a seed from the clean training graph Tr and its levels (stream_levels(Tr, Tr, ...))."""
    rng = np.random.default_rng(DRAW_ENTROPY + seed)
    cohort, neg, pos = C.pick_targets(Tr, max_neg=3, strict=True)
    whitewash = rng.choice(cohort, N_TARGETS, replace=False)
    badmouth = rng.choice(trusted_cohort(neg, pos), N_TARGETS, replace=False)
    targets = np.concatenate([whitewash, badmouth])
    t_s = rng.integers(T_S_FIRST, T_S_LAST + 1, size=len(targets))
    witnesses = {}
    for lag in LAGS:                   # lag 1 first, then lag 3
        witnesses[lag] = _witnesses(rng, levels, targets, t_s, lag)
    return Plan(whitewash, badmouth, t_s, witnesses)


def senders_of(plan: Plan, n_nodes: int) -> np.ndarray:
    """[targets, SENDERS_PER_TARGET] ids of the silent senders, appended after the n_nodes accounts."""
    return n_nodes + np.arange(len(plan.targets) * SENDERS_PER_TARGET).reshape(-1, SENDERS_PER_TARGET)


def attack_stream(Tr, plan: Plan, lag: int) -> tuple:
    """(the attacked training graph at `lag`, its metadata). Tr is the clean training graph."""
    if lag not in LAGS:
        raise ValueError(f"lag must be one of {LAGS}, got {lag!r}")
    senders = senders_of(plan, Tr.n)
    n_ratings = senders.size
    src = np.concatenate([senders.ravel(), plan.witnesses[lag].ravel()])
    dst = np.concatenate([np.repeat(plan.targets, SENDERS_PER_TARGET), np.repeat(senders.ravel(), WITNESSES_PER_SENDER)])
    sign = np.concatenate([np.repeat(plan.direction, SENDERS_PER_TARGET), -np.ones(n_ratings * WITNESSES_PER_SENDER,
                                                                                    dtype=np.int64)])
    snap = np.concatenate([np.repeat(plan.t_s, SENDERS_PER_TARGET),
                           np.repeat(plan.t_s + lag, SENDERS_PER_TARGET * WITNESSES_PER_SENDER)]).astype(np.int64)
    time = np.array([Tr.time[Tr.snap == s].max() if (Tr.snap == s).any() else 0 for s in snap], dtype=np.int64)
    G = C.Graph(np.concatenate([Tr.edges, np.stack([src, dst], axis=1)]),
                np.concatenate([Tr.labels, S.one_hot_from_sign(sign)]),
                np.concatenate([Tr.time, time]), np.concatenate([Tr.snap, snap]), Tr.n + n_ratings,
                np.concatenate([Tr.rid, -1 - np.arange(len(src), dtype=np.int64)]))
    G = G.subset(np.argsort(G.snap, kind="mergesort"))
    return G, dict(lag=lag, senders=senders.tolist(), n_nodes=int(G.n), n_rows=int(len(G.edges)),
                   n_injected=int(len(src)))
