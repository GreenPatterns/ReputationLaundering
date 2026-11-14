"""CPPR workload facts quoted in the paper (read-only, no training).

Replays the data transformations of keeper/cppr/cp_vpe_gpu.py run_single_seed
for every dataset and seed, in the same random-draw order, and reports:
the Plain VPE pool weights and Tier-0 shares, the share of scored retained
ratings that come from deleted accounts, the background requests of the
inject-then-erase arms that still point at their intended edges (the requests
store row positions, and inserting the fakes shifts later positions), the
bad-mouth targets whose tier drops because the fakes' -1 ratings count in
their negative in-degree, and the Plain Erasure test-set sizes after the
snapshots are re-cut. It trains no model and changes no file.

Run from the repository root:
    CUDA_VISIBLE_DEVICES="" ~/.conda/envs/ps/bin/python -m keeper.paperkit.cppr_workload_facts
"""
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO)

import numpy as np  # noqa: E402

from keeper.laundering import core as C  # noqa: E402
from keeper.cppr import vpe as V  # noqa: E402
from keeper.laundering import signed as S  # noqa: E402
from keeper.cppr.vpe_frontier import workload  # noqa: E402
from keeper.cppr.cp_vpe_gpu import (compute_credibility_levels, apply_cp_vpe,  # noqa: E402
                                   build_inject_then_erase_graph, CPVPEConfig,
                                   CRED_WEIGHTS)


W = np.asarray(CRED_WEIGHTS)


def items_of(res):
    """(group idx, true rater, pool) as linkage.py builds them."""
    out = []
    for idx, p in enumerate(res.pooled):
        pool = res.pools[p["window"]]
        if len(pool) < 2:
            continue
        for u in p["raters"]:
            out.append((idx, int(u), pool, p))
    return out


def run(ds, seed):
    D = C.load(ds)
    G = C.full_graph(D)
    tt = D.tt
    Tr = C.train_split(G, tt)
    cred = compute_credibility_levels(Tr, D.n, seeded=True)
    cohort, neg, pos = C.pick_targets(Tr, max_neg=3, strict=True)
    trusted = np.array(sorted([v for v, p in pos.items()
                               if len(p) >= 3 and len(neg.get(v, [])) == 0]), dtype=np.int64)
    rho = 0.02
    honest_rng = np.random.default_rng(int(1e6 * rho) + 17 * seed)
    reqs, edge_t, acc_t = workload(Tr, cohort, neg, 15, 5, rho, 3000 + seed, honest_rng)
    honest = [q for q in reqs if q.tag == "honest"]
    rng = np.random.default_rng(3000 + seed)
    out = dict(ds=ds, seed=seed, n_edge_t=len(edge_t), n_acc_t=len(acc_t),
               n_honest=len(honest), n_train=len(Tr.edges))

    # deleted accounts
    accs = sorted({q.account for q in reqs if q.account is not None})
    out["n_deleted_accounts"] = len(accs)

    # ---------------- direct arm pooling
    res_v = V.apply_vpe(D, G, tt, reqs, V.VPEConfig(1, float("inf"), 10), rng)
    res_c, cred_c = apply_cp_vpe(D, G, tt, reqs, cred, CPVPEConfig(1, float("inf"), 10), rng)
    out["n_erased"] = res_v.info["n_erased"]
    out["n_pooled_verdicts"] = res_v.info["n_pooled_verdicts"]
    # plain VPE pool weights
    wbar = [float(np.mean([W[cred[u]] for u in m])) for m in res_v.pools.values() if m]
    out["wbar_min_med_max"] = (min(wbar), float(np.median(wbar)), max(wbar))
    tier0_share = [float(np.mean([cred[u] == 0 for u in m])) for m in res_v.pools.values() if m]
    out["pool_tier0_share_med"] = float(np.median(tier0_share))
    for name, res in (("vpe", res_v), ("cp", res_c)):
        its = items_of(res)
        n_items = len(its)
        multi = sum(1 for p in res.pooled if len(p["raters"]) > 1)
        dup = sum(1 for p in res.pooled if len(set(p["raters"])) < len(p["raters"]))
        acc_items = sum(1 for _, u, _, _ in its if u in accs)
        # fraction of items whose pool contains another deleted account
        out[name] = dict(n_items=n_items, n_groups=len(res.pooled),
                         groups_multi_rater=multi, groups_dup_rater=dup,
                         items_true_is_deleted_account=acc_items,
                         share_items_deleted_account=acc_items / max(n_items, 1))
        # merged pools (cp only): items whose candidate pool spans other windows
        if name == "cp":
            merged = 0
            for _, _, pool, p in its:
                cell_members = {int(r) for q in res.pooled
                                if q["raw_window"] == p["raw_window"]
                                and q["cred_level"] == p["cred_level"] for r in q["raters"]}
                if len(set(pool)) > len(cell_members):
                    merged += 1
            out[name]["items_with_merged_pool"] = merged

    # timing signal of deleted accounts vs others (released graph res_c)
    Trc = res_c.Tr
    an = res_c.anon_nodes
    act_total = np.bincount(Trc.edges[~np.isin(Trc.edges[:, 0], an)].ravel(),
                            minlength=Trc.n)
    out["deleted_accounts_with_zero_remaining_activity"] = int(sum(act_total[a] == 0 for a in accs))

    # ---------------- Plain Erasure recut
    erase_rids, _ = V.plain_erasure(D, G, tt, reqs)
    Dp, ridp = C.erased_dataset(D, erase_rids)
    Gp = C.full_graph(Dp, rid=ridp)
    test_orig = set(G.rid[G.snap == tt].tolist())
    test_p = set(Gp.rid[Gp.snap == tt].tolist())
    train_p = set(Gp.rid[Gp.snap < tt].tolist())
    out["plain_erasure"] = dict(n_erased=len(erase_rids),
                                orig_test=len(test_orig), plain_test=len(test_p),
                                orig_test_edges_in_plain_training=len(test_orig & train_p))
    rows, t1, _ = C.task_masks(Gp, tt)
    rows_c, t1_c, _ = C.task_masks(res_c.G, tt)
    out["t1_n_plain"] = int(t1.sum())
    out["t1_n_cppr"] = int(t1_c.sum())

    # ---------------- inject arms
    D_ww, G_ww, cred_ww, reqs_ww = build_inject_then_erase_graph(
        D, G, tt, cred, edge_t, +1, 3, honest, rng, sybil_ring=True, seeded=True)
    bm_targets = rng.choice(trusted, size=min(len(edge_t), len(trusted)), replace=False).tolist()
    D_bm, G_bm, cred_bm, reqs_bm = build_inject_then_erase_graph(
        D, G, tt, cred, bm_targets, -1, 3, honest, rng, sybil_ring=False, seeded=True)
    for name, Gi, targets, reqs_i in (("ww", G_ww, edge_t, reqs_ww), ("bm", G_bm, bm_targets, reqs_bm)):
        Tri = C.train_split(Gi, tt)
        rows_o = np.concatenate([q.rows for q in honest])
        rows_h = np.concatenate([q.rows for q in reqs_i if q.tag == "honest"])
        same = Tri.rid[rows_h] == Tr.rid[rows_o]
        fake_hit = int(np.sum(Tri.edges[rows_h].max(1) >= D.n))
        inc_t = int(np.sum(np.isin(Tri.edges[rows_h], targets).any(1)))
        inc_t_orig = int(np.sum(np.isin(Tr.edges[rows_o], targets).any(1)))
        out[name] = dict(background_rows_same_edge=int(same.sum()),
                         background_rows_total=len(rows_h),
                         background_rows_hit_fake_edges=fake_hit,
                         background_incident_to_attack_targets=inc_t,
                         intended_background_incident_to_attack_targets=inc_t_orig)
    out["ww"]["target_tier_before"] = [int(cred[v]) for v in edge_t]
    out["ww"]["target_tier_after"] = [int(cred_ww[v]) for v in edge_t]
    out["ww"]["fake_tiers"] = sorted(set(cred_ww[D.n:].tolist()))
    tb = [int(cred[v]) for v in bm_targets]
    ta = [int(cred_bm[v]) for v in bm_targets]
    out["bm"]["target_tier_before"] = tb
    out["bm"]["target_tier_after"] = ta
    out["bm"]["n_targets_demoted"] = int(sum(a < b for a, b in zip(ta, tb)))
    out["bm"]["n_targets_to_tier0"] = int(sum(a == 0 and b > 0 for a, b in zip(ta, tb)))
    out["bm"]["n_tier_changes_elsewhere"] = int(np.sum(cred_bm[:D.n] != cred))
    out["ww"]["n_tier_changes_elsewhere"] = int(np.sum(cred_ww[:D.n] != cred))
    return out


def summarize(rows):
    rng = lambda xs: f"{min(xs):.3g} to {max(xs):.3g}"
    ww = [r["ww"] for r in rows]
    bm = [r["bm"] for r in rows]
    return dict(
        deleted_accounts=rng([r["n_deleted_accounts"] for r in rows]),
        wbar_min_to_max=f"{min(r['wbar_min_med_max'][0] for r in rows):.2f} to "
                        f"{max(r['wbar_min_med_max'][2] for r in rows):.2f}",
        pool_tier0_share_median=rng([r["pool_tier0_share_med"] for r in rows]),
        deleted_account_share_vpe=rng([r["vpe"]["share_items_deleted_account"] for r in rows]),
        deleted_account_share_cppr=rng([r["cp"]["share_items_deleted_account"] for r in rows]),
        background_same_edge_share_ww=rng([w["background_rows_same_edge"] / w["background_rows_total"] for w in ww]),
        background_same_edge_share_bm=rng([b["background_rows_same_edge"] / b["background_rows_total"] for b in bm]),
        background_hit_fake_ww=rng([w["background_rows_hit_fake_edges"] for w in ww]),
        background_touch_ww_targets=rng([w["background_incident_to_attack_targets"] for w in ww]),
        background_touch_bm_targets=rng([b["background_incident_to_attack_targets"] for b in bm]),
        bm_targets_demoted=rng([b["n_targets_demoted"] for b in bm]),
        bm_targets_to_tier0=rng([b["n_targets_to_tier0"] for b in bm]),
        ww_tier_changes=rng([w["n_tier_changes_elsewhere"] for w in ww]),
        plain_erasure_test_edges_in_training=rng([r["plain_erasure"]["orig_test_edges_in_plain_training"] for r in rows]),
        t1_plain=rng([r["t1_n_plain"] for r in rows]),
        t1_cppr=rng([r["t1_n_cppr"] for r in rows]),
    )


if __name__ == "__main__":
    datasets = sys.argv[1:] or ["bitcoinalpha", "bitcoinotc", "wiki-rfa"]
    for ds in datasets:
        rows = [run(ds, s) for s in range(5)]
        print(ds, json.dumps(summarize(rows), indent=1))
