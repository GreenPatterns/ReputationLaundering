"""VPE evaluation: laundering resistance vs rater-linkage frontier (H5, K3).

Workload (per background erasure rate rho): honest users erase a random rho
fraction of training ratings, one request per rating; laundering targets have
their distrust in-edges erased (edge-level, C1/C2), and a second set of targets
has the accounts of their distrust raters deleted (adaptive attack: whole
endorser/rater accounts, §6). All requests in a pool window form one pool.

For each rho the platform serves either
  plain  standard erasure, exact retraining (the laundering surface), or
  VPE    verdict-preserving erasure with (granularity g, eps), exact retraining.
Measured: laundering gain per target (H5: VPE cuts it >= 80% vs plain), T1/T2
MCC (H5: within 0.01 of plain erasure), and linkage AUC per signal (H5: <= 0.6
at some practical pool size; K3: > 0.75 at every practical pool size kills VPE).

Adaptive small-pool attack: rho = 0, so the laundering raters are alone in their
windows; noise is then applied to every pool smaller than m_min.

Run: CUDA_VISIBLE_DEVICES=0 <ps python> -m laundering.vpe_frontier --ds bitcoinalpha --seed 0
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402

from keeper.laundering import core as C  # noqa: E402
from keeper.cppr import linkage as L  # noqa: E402
from keeper.laundering import metrics as M  # noqa: E402
from keeper.laundering import unlearn as U  # noqa: E402
from keeper.cppr import vpe as V  # noqa: E402


def workload(Tr, cohort, neg, n_edge, n_acc, rho, target_seed, honest_rng):
    """Honest + laundering requests; laundering targets/raters kept disjoint.

    The laundering targets depend only on ``target_seed``, so they are the SAME
    in every setting; only the honest background varies with rho.
    """
    pick = np.random.default_rng(target_seed).permutation(cohort)
    edge_t, acc_t, used = [], [], set()
    for v in pick.tolist():
        raters = set(Tr.edges[neg[v], 0].tolist())
        if v in used or raters & used:
            continue
        if len(edge_t) < n_edge:
            edge_t.append(v)
        elif len(acc_t) < n_acc:
            acc_t.append(v)
        else:
            break
        used |= raters | {v}
    reqs = [V.Request(rows=np.asarray(neg[v]), tag="launder_edge", target=v) for v in edge_t]
    for v in acc_t:
        for a in sorted(set(Tr.edges[neg[v], 0].tolist())):
            reqs.append(V.Request(account=a, tag="launder_account", target=v))
    protect = np.isin(Tr.edges[:, 1], edge_t + acc_t) | np.isin(Tr.edges[:, 0], edge_t + acc_t)
    pool = np.where(~protect)[0]
    k = int(round(rho * len(Tr.edges)))
    for r in (honest_rng.choice(pool, k, replace=False) if k else []):
        reqs.append(V.Request(rows=np.array([r]), tag="honest"))
    return reqs, edge_t, acc_t


def gains(model, Tr, tt, panel, targets, r0):
    if not targets:
        return []
    z = C.embed(model, Tr, tt)
    return (panel.reputation(model, z, np.array(targets)) - np.array([r0[v] for v in targets])).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="bitcoinalpha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n-edge", type=int, default=15, help="edge-level laundering targets")
    ap.add_argument("--n-acc", type=int, default=5, help="account-level laundering targets")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    ap.add_argument("--granularity", type=int, nargs="+", default=[1, 3, 7])
    ap.add_argument("--eps", type=float, nargs="+", default=[float("inf"), 1.0, 0.5])
    ap.add_argument("--small-pool-m-min", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing result file")
    a = ap.parse_args()
    out_name = a.out or f"vpe_{a.ds}_s{a.seed}.json"
    C.result_path(out_name, a.force)          # refuse an existing result before the work, not after it
    T0 = time.time()
    C.set_deterministic(True)
    C.require_gpu()
    D = C.load(a.ds)
    mem = C.apply_memory_profile(D)
    G = C.full_graph(D)
    tt = D.tt
    Tr = C.train_split(G, tt)
    panel = C.Panel(Tr, n_probe=256, seed=1234)
    cohort, neg, _ = C.pick_targets(Tr, max_neg=3, strict=True)
    m0 = C.train(C.make_model(D, Tr, a.seed, tt), Tr, tt, epochs=a.epochs)
    z0 = C.embed(m0, Tr, tt)
    r0 = dict(zip(cohort.tolist(), panel.reputation(m0, z0, cohort).tolist()))
    base_tm = C.test_metrics(m0, z0, D, G)
    rng = np.random.default_rng(3_000 + a.seed)

    runs = []
    settings = [(rho, g, e, 10 ** 9) for rho in a.rates for g in a.granularity for e in a.eps]
    settings += [(0.0, 1, e, a.small_pool_m_min) for e in a.eps]      # small-pool attack
    plain_cache = {}
    for rho, g, eps, m_min in settings:
        t1 = time.time()
        honest_rng = np.random.default_rng(int(1e6 * rho) + 17 * a.seed)
        reqs, edge_t, acc_t = workload(Tr, cohort, neg, a.n_edge, a.n_acc, rho,
                                       3_000 + a.seed, honest_rng)
        if rho not in plain_cache:
            erase_rids, _ = V.plain_erasure(D, G, tt, reqs)
            up = U.exact_retrain(D, G, tt, erase_rids, a.seed, a.epochs, self_term=C.self_term_of(m0))
            zp = up.embed(tt)
            plain_cache[rho] = dict(
                edge=gains(up.model, up.Tr, tt, panel, edge_t, r0),
                acc=gains(up.model, up.Tr, tt, panel, acc_t, r0),
                tm=C.test_metrics(up.model, zp, D, up.G))
        pl = plain_cache[rho]
        res = V.apply_vpe(D, G, tt, reqs, V.VPEConfig(g, eps, m_min), rng)
        mv = V.retrain_on(res, a.seed, tt, a.epochs, self_term=C.self_term_of(m0))
        zv = C.embed(mv, res.Tr, tt)
        tm = C.test_metrics(mv, zv, res.D, res.G)
        ge = gains(mv, res.Tr, tt, panel, edge_t, r0)
        ga = gains(mv, res.Tr, tt, panel, acc_t, r0)
        link = L.linkage_attack(res, mv, tt)

        def reduction(v_, p_):
            mp, mv_ = M.mean_or_none(p_), M.mean_or_none(v_)
            return None if not mp or mv_ is None else 1.0 - mv_ / mp
        rec = dict(rho=rho, granularity=g, eps=eps, m_min=m_min,
                   small_pool_attack=bool(rho == 0.0),
                   n_requests=len(reqs), vpe_info=res.info,
                   gain_plain_edge=M.mean_or_none(pl["edge"]), gain_vpe_edge=M.mean_or_none(ge),
                   gain_plain_account=M.mean_or_none(pl["acc"]), gain_vpe_account=M.mean_or_none(ga),
                   reduction_edge=reduction(ge, pl["edge"]), reduction_account=reduction(ga, pl["acc"]),
                   t1_mcc_vpe=tm["t1"]["mcc"], t1_mcc_plain=pl["tm"]["t1"]["mcc"],
                   t2_mcc_vpe=tm["t2"]["mcc"], t2_mcc_plain=pl["tm"]["t2"]["mcc"],
                   d_t1_mcc_vs_plain=(None if tm["t1"]["mcc"] is None or pl["tm"]["t1"]["mcc"] is None
                                      else tm["t1"]["mcc"] - pl["tm"]["t1"]["mcc"]),
                   linkage=link, gains_vpe_edge=ge, gains_plain_edge=pl["edge"],
                   seconds=round(time.time() - t1, 1))
        runs.append(rec)
        print(f"  rho={rho} g={g} eps={eps} m_min={m_min}: gain plain {rec['gain_plain_edge']} "
              f"-> vpe {rec['gain_vpe_edge']} (red {rec['reduction_edge']}) | acc red "
              f"{rec['reduction_account']} | dMCC {rec['d_t1_mcc_vs_plain']} | linkage AUC "
              f"{link['auc']} pool median {link['pool_size']['median']} ({rec['seconds']}s)",
              flush=True)

    def h5_ok(r):
        lk = r["linkage"]["auc"]["combined"]
        return (r["reduction_edge"] is not None and r["reduction_edge"] >= 0.8
                and r["d_t1_mcc_vs_plain"] is not None and abs(r["d_t1_mcc_vs_plain"]) <= 0.01
                and lk is not None and lk <= 0.6)
    practical = [r for r in runs if not r["small_pool_attack"]]
    out = dict(ds=a.ds, seed=a.seed, epochs=a.epochs, memory_profile=mem,
               base_t1=base_tm["t1"], runs=runs,
               h5=dict(any_setting_meets_all=any(h5_ok(r) for r in practical),
                       settings_meeting=[(r["rho"], r["granularity"], r["eps"])
                                         for r in practical if h5_ok(r)]),
               k3=dict(min_linkage_auc_combined=min(
                   (r["linkage"]["auc"]["combined"] for r in practical
                    if r["linkage"]["auc"]["combined"] is not None), default=None),
                       kill=all((r["linkage"]["auc"]["combined"] or 0) > 0.75 for r in practical)),
               runtime_s=round(time.time() - T0, 1))
    path = C.dump(out, out_name, D, force=a.force)
    print(f"H5 any setting meets all: {out['h5']['any_setting_meets_all']} | "
          f"K3 kill: {out['k3']['kill']} (min combined AUC {out['k3']['min_linkage_auc_combined']})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
