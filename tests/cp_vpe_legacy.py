"""Frozen copies of the CPPR driver at commit ee67679 (before Task 3.2): the oracles of tests/test_cp_vpe_inject.py.

Not collected (no test_ prefix). Each function below is copied verbatim from the base commit, so a test can run
the old and the new code on the same inputs and require identical outputs:
* CPVPEConfig, apply_cp_vpe, build_inject_then_erase_graph, run_single_seed from keeper/cppr/cp_vpe_gpu.py;
* _undirected, _auc_true_vs_rest, _rank_pct, linkage_attack from keeper/cppr/linkage.py.
The copied run_single_seed calls L.linkage_attack, which here is the frozen linkage_attack below, and the
module-level train_gated_model, which the tests replace by a stub (no model is trained on the CPU).
"""
import collections
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

from keeper.cppr import vpe as V
from keeper.cppr.cp_vpe_gpu import train_gated_model  # noqa: F401  (replaced by a stub in the tests)
from keeper.credibility.tiers import TIER_WEIGHTS as CRED_WEIGHTS
from keeper.credibility.tiers import legacy_levels as compute_credibility_levels
from keeper.laundering import core as C
from keeper.laundering import metrics as M
from keeper.laundering import signed as S

BASE_COMMIT = "ee676797b228d75f6886ac11102cba23355d59d7"


# ---------------------------------------------------------------- keeper/cppr/cp_vpe_gpu.py at ee67679
@dataclass
class CPVPEConfig:
    granularity: int = 1
    eps: float = float("inf")
    m_min: int = 10
    cred_weights: tuple = CRED_WEIGHTS


def apply_cp_vpe(D: "C.Dataset", G: "C.Graph", tt: int, requests,
                 cred_levels: np.ndarray, cfg: CPVPEConfig,
                 rng: np.random.Generator) -> tuple:
    """Transform the graph via Credibility-Preserving VPE (CP-VPE)."""
    Tr = C.train_split(G, tt)
    erase, to_pool, accounts = V._erased_sets(Tr, requests)
    g = max(1, int(cfg.granularity))
    n_windows = (tt + g - 1) // g
    n_levels = len(cfg.cred_weights)
    n_anon = n_windows * n_levels
    anon_nodes = np.arange(D.n, D.n + n_anon, dtype=np.int64)
    sign = np.where(S.neg_mask(Tr.labels), -1, 1)

    def anon_id(w: int, c_u: int) -> int:
        return int(D.n + w * n_levels + c_u)

    cell_pools = collections.defaultdict(set)
    counts = collections.defaultdict(list)
    for r in np.where(to_pool)[0]:
        w = int(Tr.snap[r]) // g
        u, v, s = int(Tr.edges[r, 0]), int(Tr.edges[r, 1]), int(sign[r])
        c_u = int(cred_levels[u])
        cell_pools[(w, c_u)].add(u)
        counts[(w, c_u, v, s)].append(u)

    merged_pools = {}
    for (w, c_u), members in cell_pools.items():
        if len(members) >= cfg.m_min:
            merged_pools[(w, c_u)] = sorted(members)
        else:
            union_c = set()
            for (w2, c2), m2 in cell_pools.items():
                if c2 == c_u:
                    union_c |= m2
            merged_pools[(w, c_u)] = sorted(union_c)

    new_e, new_lab, new_snap, new_time, pooled = [], [], [], [], []
    pools_by_key = {}
    for (w, c_u, v, s), raters in sorted(counts.items()):
        c = len(raters)
        cell_key = w * n_levels + c_u
        eff_pool = merged_pools[(w, c_u)]
        pools_by_key[cell_key] = eff_pool
        small = len(eff_pool) < cfg.m_min
        if small and np.isfinite(cfg.eps):
            c_rel = max(0, int(round(c + rng.laplace(0.0, 1.0 / cfg.eps))))
        else:
            c_rel = c
        snap_w = min((w + 1) * g - 1, tt - 1)
        t_w = int(G.time[G.snap == snap_w].max()) if (G.snap == snap_w).any() else 0
        a_node = anon_id(w, c_u)
        for _ in range(c_rel):
            new_e.append((a_node, v))
            new_lab.append(s)
            new_snap.append(snap_w)
            new_time.append(t_w)
        pooled.append(dict(
            window=cell_key, raw_window=w, cred_level=c_u,
            target=v, sign=s, raters=sorted(raters), count=c,
            released=c_rel, pool_size=len(eff_pool),
            noised=bool(small and np.isfinite(cfg.eps))
        ))

    keep_full = ~G.mask_of(Tr.rid[erase])
    Gk = G.subset(keep_full)
    if new_e:
        e = np.array(new_e, dtype=np.int64)
        lab = S.one_hot_from_sign(np.array(new_lab))
        rid = np.arange(len(D.edges), len(D.edges) + len(e), dtype=np.int64)
        Gv = C.Graph(
            np.concatenate([Gk.edges, e]),
            np.concatenate([Gk.labels, lab]),
            np.concatenate([Gk.time, np.array(new_time, dtype=np.int64)]),
            np.concatenate([Gk.snap, np.array(new_snap, dtype=np.int64)]),
            D.n + n_anon,
            np.concatenate([Gk.rid, rid]),
        )
    else:
        Gv = C.Graph(Gk.edges, Gk.labels, Gk.time, Gk.snap, D.n + n_anon, Gk.rid)

    Dv = C.Dataset(D.name, Gv.edges, Gv.labels, Gv.time, Gv.n, D.T, D.cfg,
                   meta=dict(D.meta, cp_vpe=True))
    ext_cred = np.zeros(D.n + n_anon, dtype=np.int64)
    ext_cred[:len(cred_levels)] = cred_levels
    for w in range(n_windows):
        for c_u in range(n_levels):
            ext_cred[anon_id(w, c_u)] = c_u

    info = dict(
        granularity=g, eps=cfg.eps, m_min=cfg.m_min,
        n_windows=n_windows, n_levels=n_levels,
        n_erased=int(erase.sum()), n_pooled_verdicts=int(to_pool.sum()),
        n_released_edges=len(new_e), n_accounts=len(accounts),
        pool_sizes={int(k): len(p) for k, p in pools_by_key.items()},
    )
    res = V.VPEResult(Dv, Gv, C.train_split(Gv, tt), anon_nodes, pooled,
                      pools_by_key, accounts, info)
    return res, ext_cred


def build_inject_then_erase_graph(D: "C.Dataset", G: "C.Graph", tt: int,
                                  cred_levels: np.ndarray, targets: list,
                                  sign_val: int, k_fake: int,
                                  honest_reqs: list, rng: np.random.Generator,
                                  sybil_ring: bool = True, seeded: bool = True):
    n_fake = k_fake * len(targets)
    fake_ids = np.arange(D.n, D.n + n_fake, dtype=np.int64)
    new_e, new_lab, new_snap, new_time = [], [], [], []
    fake_reqs = list(honest_reqs)
    ptr = 0
    for v in targets:
        grp = [int(fake_ids[ptr + j]) for j in range(k_fake)]
        snap_f = int(rng.integers(5, tt))
        t_f = int(G.time[G.snap == snap_f].max()) if (G.snap == snap_f).any() else 0
        for j in range(k_fake):
            f_u = grp[j]
            new_e.append((f_u, int(v)))
            new_lab.append(sign_val)
            new_snap.append(snap_f)
            new_time.append(t_f)
            if sybil_ring and sign_val > 0:
                nxt = grp[(j + 1) % k_fake]
                new_e.append((f_u, nxt))
                new_lab.append(+1)
                new_snap.append(snap_f)
                new_time.append(t_f)
            fake_reqs.append(V.Request(account=f_u, tag="inject_then_erase", target=int(v)))
        ptr += k_fake

    e_arr = np.array(new_e, dtype=np.int64)
    lab_arr = S.one_hot_from_sign(np.array(new_lab))
    rid_arr = np.arange(len(D.edges), len(D.edges) + len(e_arr), dtype=np.int64)
    G_inj = C.Graph(
        np.concatenate([G.edges, e_arr]),
        np.concatenate([G.labels, lab_arr]),
        np.concatenate([G.time, np.array(new_time, dtype=np.int64)]),
        np.concatenate([G.snap, np.array(new_snap, dtype=np.int64)]),
        D.n + n_fake,
        np.concatenate([G.rid, rid_arr]),
    )
    order = np.argsort(G_inj.snap, kind="mergesort")
    G_inj = G_inj.subset(order)
    D_inj = C.Dataset(D.name, G_inj.edges, G_inj.labels, G_inj.time, G_inj.n, D.T, D.cfg,
                      meta=dict(D.meta, injected_fakes=n_fake))
    Tr_inj = C.train_split(G_inj, tt)
    cred_inj = compute_credibility_levels(Tr_inj, D.n + n_fake, seeded=seeded)
    # Edge-level requests hold row positions of the original training list, and
    # inserting the fakes shifts every later position. Re-point each request to
    # the same edges (same row ids) in the injected training list.
    Tr0 = C.train_split(G, tt)
    pos_of_rid = {int(r): i for i, r in enumerate(Tr_inj.rid)}
    fake_reqs = [
        replace(q, rows=np.array([pos_of_rid[int(r)] for r in Tr0.rid[np.asarray(q.rows)]],
                                 dtype=np.int64))
        if q.rows is not None and len(q.rows) else q
        for q in fake_reqs
    ]
    return D_inj, G_inj, cred_inj, fake_reqs


def run_single_seed(ds: str, seed: int, epochs: int = 150, rho: float = 0.02,
                    g: int = 1, k_fake: int = 3, n_edge: int = 15, n_acc: int = 5,
                    out_name: str | None = None, gate_norm: str = "snapshot", force: bool = False):
    T0 = time.time()
    result_name = out_name or (f"cp_vpe_{ds}_s{seed}.json" if gate_norm == "snapshot"
                               else f"cpgate_{gate_norm}_{ds}_s{seed}.json")
    C.result_path(result_name, force)         # refuse an existing result before the training, not after it
    C.set_deterministic(True)
    C.require_gpu()
    D = C.load(ds)
    mem = C.apply_memory_profile(D)
    G = C.full_graph(D)
    tt = D.tt
    Tr = C.train_split(G, tt)
    panel = C.Panel(Tr, n_probe=256, seed=1234)
    cred_levels = compute_credibility_levels(Tr, D.n, seeded=True)

    cohort, neg, pos = C.pick_targets(Tr, max_neg=3, strict=True)
    trusted_cohort = np.array(sorted([
        v for v, p_rows in pos.items() if len(p_rows) >= 3 and len(neg.get(v, [])) == 0
    ]), dtype=np.int64)

    m0 = train_gated_model(D, Tr, seed, tt, cred_levels, epochs=epochs, gate_norm=gate_norm)
    z0 = C.embed(m0, Tr, tt)
    r0_cohort = dict(zip(cohort.tolist(), panel.reputation(m0, z0, cohort).tolist()))
    r0_trusted = dict(zip(trusted_cohort.tolist(), panel.reputation(m0, z0, trusted_cohort).tolist()))
    base_tm = C.test_metrics(m0, z0, D, G)

    rng = np.random.default_rng(3000 + seed)
    honest_rng = np.random.default_rng(int(1e6 * rho) + 17 * seed)
    reqs, edge_t, acc_t = V.workload(Tr, cohort, neg, n_edge, n_acc, rho,
                                     3000 + seed, honest_rng) if hasattr(V, "workload") else (None, None, None)
    if reqs is None:
        from keeper.cppr.vpe_frontier import workload
        reqs, edge_t, acc_t = workload(Tr, cohort, neg, n_edge, n_acc, rho, 3000 + seed, honest_rng)

    honest_only_reqs = [q for q in reqs if q.tag == "honest"]

    erase_rids, _ = V.plain_erasure(D, G, tt, reqs)
    Dp, ridp = C.erased_dataset(D, erase_rids)
    Gp = C.full_graph(Dp, rid=ridp)
    Trp = C.train_split(Gp, tt)
    mp = train_gated_model(Dp, Trp, seed, tt, cred_levels, epochs=epochs, gate_norm=gate_norm)
    zp = C.embed(mp, Trp, tt)
    tm_plain = C.test_metrics(mp, zp, Dp, Gp)
    gain_plain_edge = (panel.reputation(mp, zp, np.array(edge_t)) - np.array([r0_cohort[v] for v in edge_t])).tolist()
    gain_plain_acc = (panel.reputation(mp, zp, np.array(acc_t)) - np.array([r0_cohort[v] for v in acc_t])).tolist()

    res_vpe = V.apply_vpe(D, G, tt, reqs, V.VPEConfig(g, float("inf"), 10), rng)
    w_arr = np.asarray(CRED_WEIGHTS, dtype=np.float64)
    pool_w_vpe = {
        int(w_idx): float(np.mean([w_arr[cred_levels[u]] for u in p_members]))
        for w_idx, p_members in res_vpe.pools.items() if len(p_members) > 0
    }
    cred_vpe = np.concatenate([cred_levels, np.zeros(len(res_vpe.anon_nodes), dtype=np.int64)])
    mv_plain = train_gated_model(res_vpe.D, res_vpe.Tr, seed, tt, cred_vpe, epochs=epochs, gate_norm=gate_norm,
                                 plain_vpe_anon_nodes=res_vpe.anon_nodes,
                                 plain_vpe_pool_weights=pool_w_vpe)
    zv_plain = C.embed(mv_plain, res_vpe.Tr, tt)
    tm_vpe = C.test_metrics(mv_plain, zv_plain, res_vpe.D, res_vpe.G)
    gain_vpe_edge = (panel.reputation(mv_plain, zv_plain, np.array(edge_t)) - np.array([r0_cohort[v] for v in edge_t])).tolist()
    gain_vpe_acc = (panel.reputation(mv_plain, zv_plain, np.array(acc_t)) - np.array([r0_cohort[v] for v in acc_t])).tolist()
    link_vpe = L.linkage_attack(res_vpe, mv_plain, tt)

    res_cp, cred_cp = apply_cp_vpe(D, G, tt, reqs, cred_levels, CPVPEConfig(g, float("inf"), 10), rng)
    mv_cp = train_gated_model(res_cp.D, res_cp.Tr, seed, tt, cred_cp, epochs=epochs, gate_norm=gate_norm)
    zv_cp = C.embed(mv_cp, res_cp.Tr, tt)
    tm_cp = C.test_metrics(mv_cp, zv_cp, res_cp.D, res_cp.G)
    gain_cp_edge = (panel.reputation(mv_cp, zv_cp, np.array(edge_t)) - np.array([r0_cohort[v] for v in edge_t])).tolist()
    gain_cp_acc = (panel.reputation(mv_cp, zv_cp, np.array(acc_t)) - np.array([r0_cohort[v] for v in acc_t])).tolist()
    link_cp = L.linkage_attack(res_cp, mv_cp, tt)

    D_ww, G_ww, cred_ww, reqs_ww = build_inject_then_erase_graph(
        D, G, tt, cred_levels, edge_t, +1, k_fake, honest_only_reqs, rng, sybil_ring=True, seeded=True
    )
    res_ww_vpe = V.apply_vpe(D_ww, G_ww, tt, reqs_ww, V.VPEConfig(g, float("inf"), 10), rng)
    pool_w_ww = {
        int(w_idx): float(np.mean([w_arr[cred_ww[u]] for u in p_members]))
        for w_idx, p_members in res_ww_vpe.pools.items() if len(p_members) > 0
    }
    cred_ww_vpe = np.concatenate([cred_ww, np.zeros(len(res_ww_vpe.anon_nodes), dtype=np.int64)])
    m_ww_vpe = train_gated_model(res_ww_vpe.D, res_ww_vpe.Tr, seed, tt, cred_ww_vpe, epochs=epochs, gate_norm=gate_norm,
                                 plain_vpe_anon_nodes=res_ww_vpe.anon_nodes,
                                 plain_vpe_pool_weights=pool_w_ww)
    z_ww_vpe = C.embed(m_ww_vpe, res_ww_vpe.Tr, tt)
    r_ww_vpe = panel.reputation(m_ww_vpe, z_ww_vpe, np.array(edge_t))
    gain_ww_vpe = float(np.mean(r_ww_vpe - np.array([r0_cohort[v] for v in edge_t])))

    res_ww_cp, cred_ww_cp = apply_cp_vpe(D_ww, G_ww, tt, reqs_ww, cred_ww, CPVPEConfig(g, float("inf"), 10), rng)
    m_ww_cp = train_gated_model(res_ww_cp.D, res_ww_cp.Tr, seed, tt, cred_ww_cp, epochs=epochs, gate_norm=gate_norm)
    z_ww_cp = C.embed(m_ww_cp, res_ww_cp.Tr, tt)
    r_ww_cp = panel.reputation(m_ww_cp, z_ww_cp, np.array(edge_t))
    gain_ww_cp = float(np.mean(r_ww_cp - np.array([r0_cohort[v] for v in edge_t])))

    bm_targets = rng.choice(trusted_cohort, size=min(len(edge_t), len(trusted_cohort)), replace=False).tolist()
    D_bm, G_bm, cred_bm, reqs_bm = build_inject_then_erase_graph(
        D, G, tt, cred_levels, bm_targets, -1, k_fake, honest_only_reqs, rng, sybil_ring=False, seeded=True
    )
    res_bm_vpe = V.apply_vpe(D_bm, G_bm, tt, reqs_bm, V.VPEConfig(g, float("inf"), 10), rng)
    pool_w_bm = {
        int(w_idx): float(np.mean([w_arr[cred_bm[u]] for u in p_members]))
        for w_idx, p_members in res_bm_vpe.pools.items() if len(p_members) > 0
    }
    cred_bm_vpe = np.concatenate([cred_bm, np.zeros(len(res_bm_vpe.anon_nodes), dtype=np.int64)])
    m_bm_vpe = train_gated_model(res_bm_vpe.D, res_bm_vpe.Tr, seed, tt, cred_bm_vpe, epochs=epochs, gate_norm=gate_norm,
                                 plain_vpe_anon_nodes=res_bm_vpe.anon_nodes,
                                 plain_vpe_pool_weights=pool_w_bm)
    z_bm_vpe = C.embed(m_bm_vpe, res_bm_vpe.Tr, tt)
    r_bm_vpe = panel.reputation(m_bm_vpe, z_bm_vpe, np.array(bm_targets))
    drop_bm_vpe = float(np.mean(r_bm_vpe - np.array([r0_trusted[v] for v in bm_targets])))

    res_bm_cp, cred_bm_cp = apply_cp_vpe(D_bm, G_bm, tt, reqs_bm, cred_bm, CPVPEConfig(g, float("inf"), 10), rng)
    m_bm_cp = train_gated_model(res_bm_cp.D, res_bm_cp.Tr, seed, tt, cred_bm_cp, epochs=epochs, gate_norm=gate_norm)
    z_bm_cp = C.embed(m_bm_cp, res_bm_cp.Tr, tt)
    r_bm_cp = panel.reputation(m_bm_cp, z_bm_cp, np.array(bm_targets))
    drop_bm_cp = float(np.mean(r_bm_cp - np.array([r0_trusted[v] for v in bm_targets])))

    mp_e = M.mean_or_none(gain_plain_edge)
    mv_e = M.mean_or_none(gain_vpe_edge)
    mcp_e = M.mean_or_none(gain_cp_edge)

    out = dict(
        ds=ds, seed=seed, epochs=epochs, rho=rho, granularity=g, k_fake=k_fake,
        gate_norm=gate_norm,
        temporal_encoder="SCEM",
        memory_profile=mem, base_t1=base_tm["t1"],
        direct_laundering=dict(
            gain_plain_edge=mp_e,
            gain_vpe_edge=mv_e,
            gain_cp_vpe_edge=mcp_e,
            reduction_plain_vpe=(None if not mp_e else 1.0 - mv_e / mp_e),
            reduction_cp_vpe=(None if not mp_e else 1.0 - mcp_e / mp_e),
            gain_plain_account=M.mean_or_none(gain_plain_acc),
            gain_vpe_account=M.mean_or_none(gain_vpe_acc),
            gain_cp_vpe_account=M.mean_or_none(gain_cp_acc),
            t1_mcc_plain=tm_plain["t1"]["mcc"],
            t1_mcc_vpe=tm_vpe["t1"]["mcc"],
            t1_mcc_cp_vpe=tm_cp["t1"]["mcc"],
            d_t1_mcc_cp_vs_plain=(tm_cp["t1"]["mcc"] - tm_plain["t1"]["mcc"]),
            linkage_plain_vpe=link_vpe,
            linkage_cp_vpe=link_cp,
        ),
        inject_then_erase_whitewash=dict(
            gain_plain_vpe=gain_ww_vpe,
            gain_cp_vpe=gain_ww_cp,
            suppression_ratio=1.0 - (gain_ww_cp / max(gain_ww_vpe, 1e-9)),
        ),
        inject_then_erase_badmouth=dict(
            drop_plain_vpe=drop_bm_vpe,
            drop_cp_vpe=drop_bm_cp,
            suppression_ratio=1.0 - (abs(drop_bm_cp) / max(abs(drop_bm_vpe), 1e-9)),
        ),
        runtime_s=round(time.time() - T0, 1),
    )
    path = C.dump(out, result_name, D, force=force)
    print(f"[{ds} s={seed} SCEM] Direct laundering: plain={mp_e:.4f} -> VPE={mv_e:.4f} -> CP-VPE={mcp_e:.4f} | "
          f"Inject-Erase WW: VPE={gain_ww_vpe:.4f} -> CP-VPE={gain_ww_cp:.4f} | "
          f"Inject-Erase BM: VPE={drop_bm_vpe:.4f} -> CP-VPE={drop_bm_cp:.4f} | "
          f"CP-VPE AUC={link_cp['auc']['combined']:.3f} ({out['runtime_s']}s)")
    print(f"wrote {path}")
    return out


# ---------------------------------------------------------------- keeper/cppr/linkage.py at ee67679
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
def linkage_attack(res, model, tt: int) -> dict:
    """Linkage AUC per signal over all pooled verdicts of a VPE release."""
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

    signals = ("structure", "timing", "model")
    items = []                       # (verdict index, true rater, {signal: scores})
    singletons = 0
    for idx, p in enumerate(res.pooled):
        pool = res.pools[p["window"]]
        if len(pool) < 2:
            singletons += len(p["raters"])
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
        for true_u in p["raters"]:
            items.append((idx, int(true_u), dict(structure=struct, timing=timing, model=modl)))

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


L = SimpleNamespace(linkage_attack=linkage_attack)   # the copied run_single_seed calls L.linkage_attack
