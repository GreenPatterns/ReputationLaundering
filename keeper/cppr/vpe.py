"""Verdict-preserving erasure (VPE), the proposal's defence (§5b).

Standard erasure deletes the edge (u -> v, s, t), so the verdict s about v
disappears, and that is what laundering exploits. VPE changes WHAT is erased:
the rater's identity, not the verdict.

On erasure of (u -> v, s, t):
  1. u's identity link to the verdict is removed. For an edge-level request the
     edge leaves u's adjacency. For an account-level request u's node is removed
     entirely: all of u's edges, and the edges ABOUT u (u is then the subject,
     and erasing one's own identity forfeits one's history).
  2. The verdict (v, s, coarse time) is re-attached to a pooled anonymous node
     A_w shared by every erasure in time window w (``granularity`` training
     snapshots per window). The verdict is placed in the window's last training
     snapshot, so its time is coarsened to the window.
  3. The per-(window, target, sign) verdict count c is released as is when the
     window's pool (its distinct erased raters) has at least ``m_min`` members,
     and as round(c + Laplace(1/eps)), floored at 0, when the pool is smaller.
The platform then retrains exactly on the transformed data, so no parameter
encodes u's identity.

Formal note, stated plainly: the Laplace step makes each RELEASED count
eps-differentially private with respect to one rater's verdict on that target
(sensitivity 1). It does not hide which targets received a pooled verdict;
linkage of a pooled verdict back to u is measured empirically (linkage.py),
which is the proposal's frontier deliverable.
"""
from dataclasses import dataclass, field

import numpy as np

from keeper.laundering import core as C
from keeper.laundering import signed as S


@dataclass
class VPEConfig:
    granularity: int = 1          # training snapshots per pool window
    eps: float = float("inf")     # Laplace noise for small pools (inf = no noise)
    m_min: int = 0                # pools with fewer distinct raters get noise


@dataclass
class Request:
    """One erasure request: edge-level (``rows``) or account-level (``account``)."""
    rows: np.ndarray | None = None      # Tr row ids of the edges to erase
    account: int | None = None          # node id whose account is deleted
    tag: str = "honest"                 # "honest" | "launder"
    target: int | None = None


@dataclass
class VPEResult:
    D: "C.Dataset"
    G: "C.Graph"
    Tr: "C.Graph"
    anon_nodes: np.ndarray                       # ids of the A_w nodes
    pooled: list = field(default_factory=list)   # (window, v, sign, true_raters, c, c_released)
    pools: dict = field(default_factory=dict)    # window -> sorted erased rater ids
    removed_accounts: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


def _erased_sets(Tr: "C.Graph", requests):
    """Rows erased, verdicts to pool (row ids), and deleted accounts."""
    erase = np.zeros(len(Tr.edges), bool)
    to_pool = np.zeros(len(Tr.edges), bool)
    accounts = []
    for q in requests:
        if q.account is not None:
            a = int(q.account)
            accounts.append(a)
            issued = Tr.edges[:, 0] == a
            about = Tr.edges[:, 1] == a
            erase |= issued | about
            to_pool |= issued & ~about           # u's verdicts about others survive
        if q.rows is not None and len(q.rows):
            erase[np.asarray(q.rows, dtype=np.int64)] = True
            to_pool[np.asarray(q.rows, dtype=np.int64)] = True
    # a verdict about a deleted account has no subject left to keep it for
    if accounts:
        to_pool &= ~np.isin(Tr.edges[:, 1], accounts)
    return erase, to_pool, sorted(set(accounts))


def apply_vpe(D: "C.Dataset", G: "C.Graph", tt: int, requests, cfg: VPEConfig,
              rng: np.random.Generator) -> VPEResult:
    """Transform the platform's data by VPE; return the graphs to retrain on."""
    Tr = C.train_split(G, tt)
    erase, to_pool, accounts = _erased_sets(Tr, requests)
    g = max(1, int(cfg.granularity))
    n_windows = (tt + g - 1) // g
    anon = np.arange(D.n, D.n + n_windows, dtype=np.int64)
    sign = np.where(S.neg_mask(Tr.labels), -1, 1)

    # pools: distinct erased raters per window
    pools = {}
    counts = {}
    for r in np.where(to_pool)[0]:
        w = int(Tr.snap[r]) // g
        u, v, s = int(Tr.edges[r, 0]), int(Tr.edges[r, 1]), int(sign[r])
        pools.setdefault(w, set()).add(u)
        counts.setdefault((w, v, s), []).append(u)

    new_e, new_lab, new_snap, new_time, pooled = [], [], [], [], []
    for (w, v, s), raters in sorted(counts.items()):
        c = len(raters)
        small = len(pools[w]) < cfg.m_min
        if small and np.isfinite(cfg.eps):
            c_rel = max(0, int(round(c + rng.laplace(0.0, 1.0 / cfg.eps))))
        else:
            c_rel = c
        snap_w = min((w + 1) * g - 1, tt - 1)
        t_w = int(G.time[G.snap == snap_w].max()) if (G.snap == snap_w).any() else 0
        for _ in range(c_rel):
            new_e.append((int(anon[w]), v))
            new_lab.append(s)
            new_snap.append(snap_w)
            new_time.append(t_w)
        pooled.append(dict(window=w, target=v, sign=s, raters=sorted(raters), count=c,
                           released=c_rel, pool_size=len(pools[w]), noised=bool(small and np.isfinite(cfg.eps))))

    keep_full = ~G.mask_of(Tr.rid[erase])
    Gk = G.subset(keep_full)
    if new_e:
        e = np.array(new_e, dtype=np.int64)
        lab = S.one_hot_from_sign(np.array(new_lab))
        rid = np.arange(len(D.edges), len(D.edges) + len(e), dtype=np.int64)
        Gv = C.Graph(np.concatenate([Gk.edges, e]), np.concatenate([Gk.labels, lab]),
                     np.concatenate([Gk.time, np.array(new_time, dtype=np.int64)]),
                     np.concatenate([Gk.snap, np.array(new_snap, dtype=np.int64)]),
                     D.n + n_windows, np.concatenate([Gk.rid, rid]))
    else:
        Gv = C.Graph(Gk.edges, Gk.labels, Gk.time, Gk.snap, D.n + n_windows, Gk.rid)
    Dv = C.Dataset(D.name, Gv.edges, Gv.labels, Gv.time, Gv.n, D.T, D.cfg,
                   meta=dict(D.meta, vpe=True))
    info = dict(granularity=g, eps=cfg.eps, m_min=cfg.m_min, n_windows=n_windows,
                n_erased=int(erase.sum()), n_pooled_verdicts=int(to_pool.sum()),
                n_released_edges=len(new_e), n_accounts=len(accounts),
                pool_sizes={int(w): len(p) for w, p in pools.items()})
    return VPEResult(Dv, Gv, C.train_split(Gv, tt), anon, pooled,
                     {int(w): sorted(p) for w, p in pools.items()}, accounts, info)


def plain_erasure(D, G, tt, requests):
    """The standard erasure the platform would do without VPE (for comparison)."""
    Tr = C.train_split(G, tt)
    erase, _, accounts = _erased_sets(Tr, requests)
    return Tr.rid[erase], accounts


def retrain_on(res: VPEResult, seed: int, tt: int, epochs: int, self_term: bool = False):
    """Exact retraining on the VPE-transformed data (no identity left in weights). self_term: the G4 flag of
    the model it replaces (keeper.laundering.core.self_term_of), passed to make_model."""
    return C.train(C.make_model(res.D, res.Tr, seed, tt, self_term=self_term), res.Tr, tt, epochs=epochs)
