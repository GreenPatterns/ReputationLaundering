"""Rater-linkage attacks on VPE's pooled verdicts (proposal §5b, §6, K3).

Threat: an observer of the POST-erasure platform (the released graph and model)
tries to tell which member of a pool issued a pooled verdict (A_w -> v, s). The
anonymity set is the pool P_w (distinct raters erased in window w). Scores per
candidate u in P_w for the verdict on v:

  structure  Adamic-Adar between u and v in the released training graph (u's
             remaining edges; pooled nodes excluded). Link prediction is the
             classic way to re-identify an anonymised edge.
  timing     u's remaining activity inside window w's snapshots, relative to u's
             overall activity; a rater active in w is more likely its source.
  model      the released model's probability that u -> v carries sign s, i.e.
             the model-based (fingerprint-style) signal: does the model still
             "know" that u judged v this way?
  combined   the mean of the three per-pool rank percentiles.

Metric: for each pooled verdict and each of its true raters, the AUC of the true
rater's score against the other pool members (ties count 1/2); averaged over
verdicts. 0.5 is chance. Top-1 accuracy is also reported. Verdicts whose pool
has a single member are perfectly linkable and are counted separately (their
AUC is undefined, not 0.5).

The attacker may exploit a signal in either direction (e.g. erasing u's rating
LOWERS u's own in-window activity, so low timing activity points at u). Each
signal's direction is therefore cross-fitted: learned on one half of the
verdicts, applied to the other, and vice versa. Ranks use average ties.

verdict_scores returns the per-candidate scores, with each candidate's degree
and activity, for every pooled verdict; linkage_attack scores them, and the
per-verdict dumps of keeper.cppr.dumps store them (Task 3.2).
"""
import numpy as np
import scipy.sparse as sp
import torch

from keeper.laundering import core as C


def _undirected(Tr: "C.Graph", exclude: np.ndarray, n: int):
    keep = ~np.isin(Tr.edges[:, 0], exclude) & ~np.isin(Tr.edges[:, 1], exclude)
    e = Tr.edges[keep]
    a = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()
    a.data[:] = 1.0
    return a


def _auc_true_vs_rest(scores: dict, true_u: int):
    others = [s for u, s in scores.items() if u != true_u]
    if not others:
        return None
    t = scores[true_u]
    return float(np.mean([1.0 if t > o else 0.5 if t == o else 0.0 for o in others]))


def _rank_pct(scores: dict) -> dict:
    """Rank percentile in [0, 1] with AVERAGE ranks for ties (no id-order bias)."""
    from scipy.stats import rankdata
    us = list(scores)
    r = rankdata([scores[u] for u in us], method="average") - 1.0
    return {u: float(x) / max(len(us) - 1, 1) for u, x in zip(us, r)}


@torch.no_grad()
def verdict_scores(res, model, tt: int) -> list:
    """The attacker's scores for every pooled verdict of a VPE release, in res.pooled order.

    One (index, verdict, scores) per verdict. scores is None when the verdict's pool has fewer than two members
    (perfectly linkable, so not scored). Otherwise it maps each signal (structure, timing, model) and three
    context features to {candidate: value} over the pool's members: degree (undirected degree in the released
    training graph without the pooled nodes), activity (the candidate's remaining edges not sent by a pooled
    node) and activity_window (those inside the verdict's window).
    """
    Tr = res.Tr
    n = Tr.n
    A = _undirected(Tr, res.anon_nodes, n)
    deg = np.asarray(A.sum(1)).ravel()
    inv_log = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
    snaps_of_window = {}
    act_total = np.bincount(Tr.edges[~np.isin(Tr.edges[:, 0], res.anon_nodes)].ravel(),
                            minlength=n).astype(float)
    z = C.embed(model, Tr, tt)
    W = model.regression_weights.to(z.device)

    records = []
    for idx, p in enumerate(res.pooled):
        pool = res.pools[p["window"]]
        if len(pool) < 2:
            records.append((idx, p, None))
            continue
        w = p.get("raw_window", p["window"])   # time window; for CP-VPE p["window"] is the (window, tier) cell key
        g = res.info["granularity"]
        if w not in snaps_of_window:
            snaps = list(range(w * g, min((w + 1) * g, tt)))
            m = np.isin(Tr.snap, snaps) & ~np.isin(Tr.edges[:, 0], res.anon_nodes)
            snaps_of_window[w] = np.bincount(Tr.edges[m].ravel(), minlength=n).astype(float)
        act_w = snaps_of_window[w]
        v, s = p["target"], p["sign"]
        nv = A[v].indices
        cand = np.array(pool, dtype=np.int64)
        struct = {int(u): float(inv_log[np.intersect1d(A[u].indices, nv)].sum()) for u in cand}
        timing = {int(u): float(act_w[u] / act_total[u]) if act_total[u] > 0 else 0.0
                  for u in cand}
        f = torch.cat((z[torch.as_tensor(cand, device=z.device)],
                       z[torch.full((len(cand),), v, device=z.device)]), 1) @ W
        p_trust = torch.softmax(f, 1)[:, 0].cpu().numpy()
        modl = {int(u): float(p_trust[i] if s > 0 else 1 - p_trust[i])
                for i, u in enumerate(cand)}
        records.append((idx, p, dict(structure=struct, timing=timing, model=modl,
                                     degree={int(u): float(deg[u]) for u in cand},
                                     activity={int(u): float(act_total[u]) for u in cand},
                                     activity_window={int(u): float(act_w[u]) for u in cand})))
    return records


@torch.no_grad()
def linkage_attack(res, model, tt: int) -> dict:
    """Linkage AUC per signal over all pooled verdicts of a VPE release."""
    signals = ("structure", "timing", "model")
    items = []                       # (verdict index, true rater, {signal: scores})
    singletons = 0
    for idx, p, scores in verdict_scores(res, model, tt):
        if scores is None:
            singletons += len(p["raters"])
            continue
        for true_u in p["raters"]:
            items.append((idx, int(true_u), {k: scores[k] for k in signals}))

    # cross-fitted orientation: direction of each signal learned on the other half
    def raw_auc(sig, subset):
        a = [_auc_true_vs_rest(sc[sig], u) for _, u, sc in subset]
        a = [x for x in a if x is not None]
        return float(np.mean(a)) if a else 0.5
    halves = ([it for it in items if it[0] % 2 == 0], [it for it in items if it[0] % 2 == 1])
    per_signal = {k: [] for k in signals + ("combined",)}
    top1 = {k: [] for k in per_signal}
    orient_log = []
    for h in (0, 1):
        fit, ev = halves[1 - h], halves[h]
        sgn = {k: (1.0 if raw_auc(k, fit) >= 0.5 else -1.0) for k in signals}
        orient_log.append(sgn)
        for _, true_u, sc in ev:
            oriented = {k: {u: sgn[k] * x for u, x in sc[k].items()} for k in signals}
            ranks = {k: _rank_pct(oriented[k]) for k in signals}
            comb = {u: float(np.mean([ranks[k][u] for k in signals])) for u in ranks["model"]}
            for k, scr in list(oriented.items()) + [("combined", comb)]:
                auc = _auc_true_vs_rest(scr, true_u)
                if auc is not None:
                    per_signal[k].append(auc)
                    top1[k].append(float(scr[true_u] == max(scr.values())))
    sizes = [len(res.pools[p["window"]]) for p in res.pooled]
    return dict(
        auc={k: (float(np.mean(v)) if v else None) for k, v in per_signal.items()},
        top1={k: (float(np.mean(v)) if v else None) for k, v in top1.items()},
        n_verdicts_scored=len(per_signal["combined"]), n_singleton_verdicts=int(singletons),
        orientation_per_half=orient_log,
        pool_size=dict(min=int(min(sizes)) if sizes else None,
                       median=float(np.median(sizes)) if sizes else None,
                       max=int(max(sizes)) if sizes else None),
        chance_top1=float(np.mean([1 / len(res.pools[p["window"]]) for p in res.pooled
                                   if len(res.pools[p["window"]]) > 1])) if sizes else None)
