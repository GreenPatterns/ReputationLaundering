"""Per-target and per-verdict dumps of the CPPR driver (Task 3.2): --dump-targets and --dump-verdicts of
keeper.cppr.cp_vpe_gpu.

Each dump is one JSON file (dump_table) that the driver writes with keeper.laundering.core.dump and its --force
flag, so it carries the provenance block and obeys the overwrite guard of ruling R27. It holds ``columns``,
``rows`` (one list per row, in column order), ``n_rows``, the name of the result file it belongs to (``result``)
and that run's ``config`` block.

Target rows: one per target, arm and method.
  arm, method     direct with methods plain, vpe and cppr; whitewash, badmouth and control with vpe and cppr; in every
                  arm also the Regime-2 baselines author, singleton and withdrawal that --methods selects (Task 4.3,
                  keeper.cppr.baselines; author's direct rows are the base model's, gain 0)
  kind            the target's cohort: edge (edge-level laundering targets, also used by whitewash and control),
                  account (account-level laundering targets) or trusted (bad-mouth victims)
  target, r0, r_after, gain
                  reputation before any deletion, after the arm's retraining, and r_after - r0
  n_neg_before, n_pos_before
                  the target's negative and positive training in-edges in the arm's graph before erasure (the
                  injected graph for whitewash and bad-mouth)
  n_neg_after, n_pos_after, n_out_after
                  its in-edges by sign and its out-edges in the training graph the model was trained on; a pooled
                  verdict counts as an in-edge from its pool node
  edgeless_after  no training edge left at all
Verdict rows: one per pooled verdict and candidate rater of a Plain VPE or CPPR release, and one per re-emitted
rating and candidate of a per-rating pseudonym release (method singleton, Task 4.3): its candidates are the members of
the CPPR pool that would have held the rating, its pool_id that pool's key, and the scores those of keeper.cppr.linkage
on the pseudonym release and its model (the fresh nodes count as pooled nodes), so that E6 scores it as it scores CPPR.
  arm, method, verdict    the release and the verdict's index in it
  pool_id, window, pool_tier, pool_size
                  the pool key (CPPR: window * 4 + tier), the time window, the pool's tier (CPPR only, else None)
                  and the number of candidates
  target, sign    the pooled verdict
  candidate, candidate_tier, label
                  a pool member, its tier in the arm, and 1 if it issued the verdict
  structure, timing, model_prob, degree, activity, activity_window
                  keeper.cppr.linkage.verdict_scores for the candidate; None for a single-member pool
"""
import os

import numpy as np

from keeper.cppr import linkage as L
from keeper.laundering import core as C
from keeper.laundering import signed as S

TARGET_COLUMNS = ("arm", "method", "kind", "target", "r0", "r_after", "gain", "n_neg_before", "n_pos_before",
                  "n_neg_after", "n_pos_after", "n_out_after", "edgeless_after")
VERDICT_COLUMNS = ("arm", "method", "verdict", "pool_id", "window", "pool_tier", "pool_size", "target", "sign",
                   "candidate", "candidate_tier", "label", "structure", "timing", "model_prob", "degree", "activity",
                   "activity_window")
_FEATURE_KEYS = ("structure", "timing", "model", "degree", "activity", "activity_window")   # verdict_scores keys


def _edge_counts(Tr) -> tuple:
    """(negative in-edges, positive in-edges, out-edges) per account of a graph with edges, labels and n."""
    n = int(Tr.n)
    edges = np.asarray(Tr.edges, dtype=np.int64).reshape(-1, 2)
    neg = S.neg_mask(np.asarray(Tr.labels).reshape(-1, 2))
    return (np.bincount(edges[neg, 1], minlength=n), np.bincount(edges[~neg, 1], minlength=n),
            np.bincount(edges[:, 0], minlength=n))


def target_rows(arm: str, method: str, kind: str, targets, r0, r_after, before, after) -> list:
    """Rows of TARGET_COLUMNS for ``targets``: reputations r0 and r_after (same order), and the edge counts in the
    training graphs ``before`` (before erasure) and ``after`` (the one the model was trained on)."""
    neg_b, pos_b, _ = _edge_counts(before)
    neg_a, pos_a, out_a = _edge_counts(after)
    rows = []
    for v, r_0, r_1 in zip(targets, r0, r_after):
        v = int(v)
        n_left = int(neg_a[v] + pos_a[v] + out_a[v])
        rows.append([arm, method, kind, v, float(r_0), float(r_1), float(r_1) - float(r_0), int(neg_b[v]),
                     int(pos_b[v]), int(neg_a[v]), int(pos_a[v]), int(out_a[v]), n_left == 0])
    return rows


def verdict_rows(arm: str, method: str, res, model, tt: int, cred) -> list:
    """Rows of VERDICT_COLUMNS for every pooled verdict of the release ``res`` and every member of its pool.
    ``model`` is the model retrained on the release and ``cred`` the tiers it was trained with."""
    rows = []
    for idx, p, scores in L.verdict_scores(res, model, tt):
        pool = res.pools[p["window"]]
        raters = {int(u) for u in p["raters"]}
        tier = p.get("cred_level")
        head = [arm, method, int(idx), int(p["window"]), int(p.get("raw_window", p["window"])),
                None if tier is None else int(tier), len(pool), int(p["target"]), int(p["sign"])]
        for u in pool:
            u = int(u)
            features = [None] * len(_FEATURE_KEYS) if scores is None else [scores[k][u] for k in _FEATURE_KEYS]
            rows.append(head + [u, int(cred[u]), int(u in raters)] + features)
    return rows


def dump_paths(result_file: str, folders: dict, force: bool = False) -> dict:
    """{kind: absolute path} of the requested dumps, <folder>/<result file stem>_<kind>.json for each kind whose
    folder is set. Each path that exists is refused unless force (ruling R27), so a run stops before training."""
    stem = os.path.splitext(os.path.basename(result_file))[0]
    return {kind: C.result_path(os.path.join(os.path.abspath(folder), f"{stem}_{kind}.json"), force)
            for kind, folder in folders.items() if folder}


def dump_table(columns, rows: list, meta: dict) -> dict:
    """The JSON object of one dump; the driver writes it with keeper.laundering.core.dump and its --force flag."""
    return dict(meta, columns=list(columns), n_rows=len(rows), rows=rows)
