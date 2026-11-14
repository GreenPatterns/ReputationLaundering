"""Regime-2 baseline retention methods of the CPPR driver keeper.cppr.cp_vpe_gpu (Task 4.3, experiment E5; protocol
docs/preregistration.md Section 2.7). --methods author,singleton,withdrawal selects them.

Each baseline keeps, in another form, the verdicts that CPPR pools: the erased training ratings that
keeper.cppr.vpe._erased_sets marks to_pool (every rating of an edge-level request, and every rating a deleted account
issued about an account that stays). Ratings about a deleted account are erased with nothing kept, as under CPPR.
  author      retain with author: nothing is erased, so each rating stays with its author inside the model. On the
              direct arm this is the base model itself (gain 0 by construction, no training; the driver reports it
              so); on an injected arm the model trains on the injected graph, the fakes with their own identities and
              the tiers the run's tier function gave them.
  singleton   per-rating pseudonym: the erased rows leave and each kept rating is re-emitted at its own snapshot and
              time from its own fresh node, appended after the arm's nodes. The fresh node takes its rater's tier, so
              its gate weight is the rater's tier weight. Its linkage candidates are the members of the CPPR pool that
              would have held the rating (same window and tier; CPPR's merged pool, apply_cp_vpe), so the verdict dump
              writes one row per rating and candidate in the schema of keeper.cppr.dumps, method singleton.
  withdrawal  withdrawal counts: Plain Erasure's graph (the direct arm passes the driver's, with its snapshots held
              under --hold-test-snapshot; an injected arm's keeps its snapshots, as Plain VPE's and CPPR's do), and two
              input columns appended to every account before the model is built: log1p of the positive and of the
              negative kept ratings withdrawn from it. The tier-prototype overwrite of the protected
              attach_scem_and_credibility_gate averages every column, so the count columns are restored to their
              per-node values right after it; the original columns get exactly the prototypes they get without counts.
None of them takes a random generator, so they cannot move a draw of the driver's streams; nor do they change the
releases or tiers the other methods train on. CPPR's functions are used unchanged (protected; Task 4.3 brief).
"""
from functools import partial
from typing import NamedTuple

import numpy as np
import torch

from keeper.cppr import vpe as V
from keeper.cppr.protocol import BASELINES
from keeper.laundering import core as C
from keeper.laundering import signed as S

AUTHOR_DIRECT_NOTE = ("retain with author on the direct arm is the base model (nothing erased): its gains are 0 by "
                      "construction and its test metrics are the base model's; it is not trained")
_NO_NODES = np.zeros(0, dtype=np.int64)


class Release(NamedTuple):
    """What one method trains on: res (a keeper.cppr.vpe.VPEResult with D, G, Tr; singleton's also holds its verdicts,
    pools and fresh nodes), the tiers cred, and the trainer (None: the driver's train_gated_model)."""
    res: object
    cred: np.ndarray
    train: object = None


def _kept_rows(Tr, requests) -> tuple:
    """(erase, rows): the erased rows of Tr and the positions of the kept (to_pool) ratings among them."""
    erase, to_pool, _ = V._erased_sets(Tr, requests)
    return erase, np.flatnonzero(to_pool)


def author_release(D, G, tt: int, cred) -> Release:
    """Retain with author: the arm's own graph and tiers, nothing erased."""
    return Release(V.VPEResult(D, G, C.train_split(G, tt), _NO_NODES), np.asarray(cred))


def singleton_release(D, G, tt: int, requests, cred, res_cp) -> Release:
    """Per-rating pseudonym: G without the erased rows, plus one fresh node per kept rating (ids D.n, D.n + 1, ...) that
    rates the same target with the same sign at the rating's own snapshot and time. ``res_cp`` is CPPR's release of
    the same arm, graph and requests; its pools are the linkage candidates. Tiers: the arm's, then each fresh node's
    rater's."""
    cred = np.asarray(cred)
    if len(cred) != D.n:
        raise ValueError(f"{len(cred)} tiers for {D.n} accounts")
    Tr = C.train_split(G, tt)
    erase, rows = _kept_rows(Tr, requests)
    raters, targets = Tr.edges[rows, 0], Tr.edges[rows, 1]
    sign = np.where(S.neg_mask(Tr.labels[rows]), -1, 1)
    nodes = np.arange(D.n, D.n + len(rows), dtype=np.int64)
    Gk = G.subset(~G.mask_of(Tr.rid[erase]))
    Gs = C.Graph(np.concatenate([Gk.edges, np.stack([nodes, targets], axis=1)]),
                 np.concatenate([Gk.labels, S.one_hot_from_sign(sign)]),
                 np.concatenate([Gk.time, Tr.time[rows]]), np.concatenate([Gk.snap, Tr.snap[rows]]),
                 D.n + len(rows), np.concatenate([Gk.rid, np.arange(len(D.edges), len(D.edges) + len(rows))]))
    Ds = C.Dataset(D.name, Gs.edges, Gs.labels, Gs.time, Gs.n, D.T, D.cfg, meta=dict(D.meta, singleton=True))
    g, n_levels = int(res_cp.info["granularity"]), int(res_cp.info["n_levels"])
    pooled, windows = [], (Tr.snap[rows] // g).tolist()
    for a, u, v, s, w in zip(nodes.tolist(), raters.tolist(), targets.tolist(), sign.tolist(), windows):
        c_u = int(cred[u])
        key = w * n_levels + c_u                         # CPPR's cell key: the pool that would have held the rating
        pooled.append(dict(window=key, raw_window=w, cred_level=c_u, target=v, sign=s, raters=[u], count=1,
                           released=1, pool_size=len(res_cp.pools[key]), noised=False, node=a))
    info = dict(granularity=g, n_levels=n_levels, n_erased=int(erase.sum()), n_pooled_verdicts=len(rows),
                n_released_edges=len(rows), n_accounts=len(res_cp.removed_accounts))
    res = V.VPEResult(Ds, Gs, C.train_split(Gs, tt), nodes, pooled, res_cp.pools, res_cp.removed_accounts, info)
    return Release(res, np.concatenate([cred, cred[raters]]))


def withdrawal_counts(Tr, requests, n_nodes: int) -> np.ndarray:
    """[n_nodes, 2] float32: log1p of the positive and of the negative kept ratings withdrawn from each account."""
    _, rows = _kept_rows(Tr, requests)
    v, neg = Tr.edges[rows, 1], S.neg_mask(Tr.labels[rows])
    counts = np.stack([np.bincount(v[~neg], minlength=n_nodes), np.bincount(v[neg], minlength=n_nodes)], axis=1)
    return np.log1p(counts).astype(np.float32)


def withdrawal_release(D, G, tt: int, requests, cred, erased=None) -> Release:
    """Withdrawal counts: Plain Erasure's graph ``erased`` = (D_p, G_p) when given (the direct arm), else G without the
    erased rows, snapshots kept; trained by train_withdrawal_model with the counts of withdrawal_counts."""
    Tr = C.train_split(G, tt)
    if erased is None:
        erased = (D, G.subset(~G.mask_of(Tr.rid[_kept_rows(Tr, requests)[0]])))
    Dk, Gk = erased
    counts = withdrawal_counts(Tr, requests, int(Gk.n))
    return Release(V.VPEResult(Dk, Gk, C.train_split(Gk, tt), _NO_NODES), np.asarray(cred),
                   partial(train_withdrawal_model, counts=counts))


def restore_counts(model, counts) -> None:
    """Write ``counts`` back into the last columns of the model's input features X_raw, in place."""
    X = model.structural_layer.X_raw
    with torch.no_grad():
        X[:, X.shape[1] - counts.shape[1]:] = torch.as_tensor(counts, dtype=X.dtype, device=X.device)


def train_withdrawal_model(D, Tr, seed: int, tt: int, cred_levels, epochs: int = 150, gate_norm: str = "snapshot",
                           *, counts):
    """train_gated_model with the count columns: the model is built through core.make_model with ``counts`` appended to
    its input features, gated by the protected attach_scem_and_credibility_gate, its count columns restored after the
    tier-prototype overwrite, then trained."""
    from keeper.cppr import cp_vpe_gpu as CV          # the driver imports this module: a top-level import is circular
    m = C.make_model(D, Tr, seed, tt, extra_features=counts)
    CV.attach_scem_and_credibility_gate(m, cred_levels, CV.CRED_WEIGHTS, gate_norm=gate_norm)
    restore_counts(m, counts)
    return C.train(m, Tr, tt, epochs=epochs)


def releases(D, G, tt: int, requests, cred, res_cp, methods, erased=None) -> dict:
    """{name: Release} of the baselines among ``methods`` (RunProtocol.method_keys), in BASELINES order.
    ``res_cp`` is CPPR's release of the arm (singleton's candidates), ``erased`` Plain Erasure's (D_p, G_p) if any."""
    build = dict(author=lambda: author_release(D, G, tt, cred),
                 singleton=lambda: singleton_release(D, G, tt, requests, cred, res_cp),
                 withdrawal=lambda: withdrawal_release(D, G, tt, requests, cred, erased))
    return {name: build[name]() for name in BASELINES if name in methods}


def direct_fields(s: dict) -> dict:
    """The direct_laundering fields of the baselines among the per-method means ``s`` (cp_vpe_gpu._direct_arm), after
    the stored fields: gains (edge and account targets), test MCC, singleton's linkage, and the note on author."""
    out = {}
    for name in (b for b in BASELINES if b in s):
        out.update({f"gain_{name}_edge": s[name]["edge"], f"gain_{name}_account": s[name]["account"],
                    f"t1_mcc_{name}": s[name]["mcc"]})
        if s[name]["linkage"] is not None:
            out[f"linkage_{name}"] = s[name]["linkage"]
    if "author" in s:
        out["note_author"] = AUTHOR_DIRECT_NOTE
    return out
