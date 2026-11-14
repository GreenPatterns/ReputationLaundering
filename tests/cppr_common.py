"""Helpers shared by tests/test_cp_vpe_inject.py and tests/test_cp_vpe_driver.py (not collected).

background() rebuilds what keeper.cppr.cp_vpe_gpu.run_single_seed computes before its first model: the graph,
the training split, the target cohorts and the request workload. It is cached, so tests must not mutate it.
"""
import functools
import hashlib
from types import SimpleNamespace

import numpy as np
import torch

from keeper.credibility import tiers
from keeper.credibility.tiers import TIER_WEIGHTS
from keeper.cppr.vpe_frontier import workload
from keeper.laundering import core as C
from keeper.laundering import signed as S

ALPHA_EPS = 0.02          # the eps the preregistered rule picks on Bitcoin-Alpha under KEEPER's cap (ruling R38)
ALPHA_MAX_B = [34, 73, 100, 118, 137]   # its five largest-B(u) accounts (Task 3.1 report, Section I3)


@functools.lru_cache(maxsize=None)
def background(ds="bitcoinalpha", seed=0, rho=0.02):
    """The inputs of run_single_seed up to the workload (no model): graph, split, cohorts and requests."""
    D = C.load(ds)
    Gr = C.full_graph(D)
    tt = D.tt
    Tr = C.train_split(Gr, tt)
    cohort, neg, pos = C.pick_targets(Tr, max_neg=3, strict=True)
    trusted = np.array(sorted([v for v, p in pos.items() if len(p) >= 3 and len(neg.get(v, [])) == 0]),
                       dtype=np.int64)
    honest_rng = np.random.default_rng(int(1e6 * rho) + 17 * seed)
    reqs, edge_t, acc_t = workload(Tr, cohort, neg, 15, 5, rho, 3000 + seed, honest_rng)
    honest = [q for q in reqs if q.tag == "honest"]
    return SimpleNamespace(D=D, G=Gr, tt=tt, Tr=Tr, cohort=cohort, neg=neg, pos=pos, trusted=trusted, reqs=reqs,
                           edge_t=edge_t, acc_t=acc_t, honest=honest)


@functools.lru_cache(maxsize=None)
def alpha_calibration(eps=ALPHA_EPS, units=tiers.KEEPER_EDGE_CAP_UNITS):
    """The BTP calibration frozen on Bitcoin-Alpha's honest training graph (ruling R32)."""
    bg = background()
    return tiers.calibrate_btp(bg.Tr, bg.D.n, eps, edge_cap_units=units)


def legacy_join_snapshots(Tr, n_nodes):
    """The snapshot in which each account joins the legacy endorsed set D during its one pass over the edges:
    -1 for a seed, a large sentinel for an account that never joins. The pass is that of legacy_levels."""
    never = 10 ** 6
    seeds = set(tiers.seed_set(Tr, n_nodes).tolist())
    join = np.full(n_nodes, never, dtype=np.int64)
    join[list(seeds)] = -1
    members = set(seeds)
    for (u, v), is_neg, sn in zip(Tr.edges, S.neg_mask(Tr.labels), Tr.snap):
        if not is_neg and int(u) in members and int(v) not in members:
            members.add(int(v))
            join[int(v)] = int(sn)
    return join


def rng_with_first_snapshot(snap, tt, start=0):
    """A seed s whose numpy default_rng(s) draws ``snap`` first from integers(5, tt), the draw of snap_f."""
    for s in range(start, start + 10_000):
        if int(np.random.default_rng(s).integers(5, tt)) == snap:
            return s
    raise AssertionError(f"no seed draws snapshot {snap}")


# ------------------------------------------------------------------ a stand-in for a trained model (CPU)
class StubModel:
    """Stands in for a trained LocalGDTE on the CPU. Its embedding z and head are deterministic functions of the
    training inputs (graph, tiers, plain-VPE pool weights, gate and seed), so two runs that train on different
    inputs give different reputations and test metrics. No model is trained."""

    def __init__(self, Tr, cred, seed, tag, pool_w=None, anon=None):
        n = int(Tr.n)
        neg = S.neg_mask(Tr.labels)
        weight = np.asarray(TIER_WEIGHTS, dtype=np.float64)[np.asarray(cred, dtype=np.int64)[:n]]
        if pool_w is not None and anon is not None:
            for w_idx, a_id in enumerate(anon):
                if int(w_idx) in pool_w and int(a_id) < n:
                    weight[int(a_id)] = float(pool_w[int(w_idx)])
        f = np.stack([np.bincount(Tr.edges[~neg, 1], minlength=n), np.bincount(Tr.edges[neg, 1], minlength=n),
                      np.bincount(Tr.edges[:, 0], minlength=n), 3.0 * weight,
                      np.bincount(Tr.edges[:, 1], weights=Tr.snap, minlength=n) / 20.0,
                      np.full(n, 1.0 if tag == "count" else -1.0)], axis=1).astype(np.float64)
        self.z = torch.tensor(np.sign(f) * np.log1p(np.abs(f)) - 0.5, dtype=torch.float32)
        rng = np.random.default_rng(int(seed) + 7)
        self.regression_weights = torch.tensor(rng.normal(size=(2 * f.shape[1], 2)), dtype=torch.float32)

    def state_dict(self) -> dict:
        """The stand-in's weights, which fix its every output, as a model's state_dict (Task 4.1c's base cache)."""
        return {"z": self.z, "regression_weights": self.regression_weights}

    def load_state_dict(self, state: dict) -> None:
        self.z = state["z"].clone()
        self.regression_weights = state["regression_weights"].clone()


def fingerprint(*parts) -> str:
    """sha256 over arrays (dtype, shape and bytes), graphs (their arrays) and plain values, in order."""
    h = hashlib.sha256()

    def add(x):
        if x is None:
            h.update(b"None")
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            for k in sorted(x):
                add(k)
                add(x[k])
        elif hasattr(x, "edges") and hasattr(x, "snap"):          # a C.Graph
            for name in ("edges", "labels", "time", "snap", "rid"):
                add(np.asarray(getattr(x, name)))
            add(int(x.n))
        else:
            h.update(repr(x).encode())

    for part in parts:
        add(part)
    return h.hexdigest()
