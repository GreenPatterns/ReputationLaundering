"""Laundering on the second learned model (SGCN) and a non-learned baseline
(EigenTrust), using the same target draw as probe4 (proposal §6 "Models").

The question SGCN answers: is LocalGDTE's near-zero indirect share (C3) a
property of trust learning in general, or of LocalGDTE's snapshot-local message
passing? SGCN propagates over the whole static signed graph.

Arms per target (SGCN): direct full (unlink, exact_retrain), C3 white-box greedy
(unlink, exact_retrain), control (matched trust erasure), placebo (far edges).
EigenTrust: direct and control, reported as a change in percentile among active
nodes (its scores are not probabilities).

Run: CUDA_VISIBLE_DEVICES=0 <ps python> -m laundering.probe_baselines --ds bitcoinalpha --seed 0
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402

from keeper.laundering import attack as A  # noqa: E402
from keeper.laundering import baselines as B  # noqa: E402
from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import metrics as M  # noqa: E402
from keeper.laundering.probe4 import stratified_draw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="bitcoinalpha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-targets", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing result file")
    a = ap.parse_args()
    out_name = a.out or f"baselines_{a.ds}_s{a.seed}.json"
    C.result_path(out_name, a.force)          # refuse an existing result before the work, not after it
    T0 = time.time()
    C.set_deterministic(True)
    C.require_gpu()
    D = C.load(a.ds)
    G = C.full_graph(D)
    tt = D.tt
    Tr = C.train_split(G, tt)
    panel = C.Panel(Tr, n_probe=256, seed=1234)
    active = np.unique(Tr.edges.ravel())
    cohort, neg, pos = C.pick_targets(Tr, max_neg=3, strict=True)
    n_pos_of = {int(v): len(pos.get(int(v), [])) for v in cohort}
    tg, sizes = stratified_draw(cohort, n_pos_of, a.n_targets)

    def fit(G_):
        return B.SGCN(D.n, a.seed, epochs=a.epochs).fit(G_)

    m0 = fit(Tr)
    z0 = m0.embed(Tr)
    r0 = dict(zip(tg.tolist(), m0.reputation(z0, panel, tg).tolist()))
    t_et = B.eigentrust(Tr, D.n)
    pct = lambda t, v: float((t[active] < t[v]).mean())    # noqa: E731
    print(f"[{a.ds} s{a.seed}] SGCN fitted | sample {len(tg)}", flush=True)

    def sgcn_arm(v, mask):
        if not mask.any():
            return None
        cut = Tr.subset(~mask)
        zu = m0.embed(cut, recompute_features=True)
        mr = fit(cut)
        zr = mr.embed(cut)
        return dict(unlink=float(m0.reputation(zu, panel, [v])[0] - r0[v]),
                    exact_retrain=float(mr.reputation(zr, panel, [v])[0] - r0[v]))

    def et_arm(v, mask):
        if not mask.any():
            return None
        t1 = B.eigentrust(Tr.subset(~mask), D.n)
        return dict(pct_before=pct(t_et, v), pct_after=pct(t1, v),
                    d_pct=pct(t1, v) - pct(t_et, v))

    recs = []
    for i, v in enumerate(tg.tolist()):
        t1 = time.time()
        rng = np.random.default_rng(10_000 * a.seed + v)
        k = len(neg[v])
        npos = n_pos_of[v]
        md = np.zeros(len(Tr.edges), bool)
        md[neg[v]] = True
        sc = B.SGCNScorer(m0, Tr, panel, v)
        p3 = A.plan_indirect(sc, Tr, v, k)
        tpos = np.asarray(pos.get(v, []), dtype=np.int64)
        mc = np.zeros(len(Tr.edges), bool)
        if len(tpos):
            mc[np.sort(rng.choice(tpos, min(k, len(tpos)), replace=False))] = True
        mp = np.zeros(len(Tr.edges), bool)
        rp = A.placebo_rows(Tr, v, k, rng)
        mp[rp] = True
        rec = dict(target=v, n_neg=k, n_pos=npos, stratum=M.stratum_of(npos), r0=r0[v],
                   count_gain=dict(laplace=float(M.counting_gain(npos, k, k, "laplace")),
                                   raw=float(M.counting_gain(npos, k, k, "raw"))),
                   sgcn=dict(direct=sgcn_arm(v, md),
                             c3_white=(sgcn_arm(v, p3.mask(Tr)) if p3.mask(Tr).any()
                                       else dict(unlink=0.0, exact_retrain=0.0)),
                             c3_trace=p3.trace, c3_candidates=p3.n_candidates,
                             control=sgcn_arm(v, mc), placebo=sgcn_arm(v, mp)),
                   eigentrust=dict(direct=et_arm(v, md), control=et_arm(v, mc)))
        recs.append(rec)
        print(f"  {i + 1}/{len(tg)} v={v} {rec['stratum']} SGCN direct "
              f"{rec['sgcn']['direct']} C3 {rec['sgcn']['c3_white']} "
              f"({time.time() - t1:.0f}s)", flush=True)

    strata = [r["stratum"] for r in recs]
    summ = {}
    for op in ("unlink", "exact_retrain"):
        dg = [r["sgcn"]["direct"][op] for r in recs]
        ig = [r["sgcn"]["c3_white"][op] for r in recs]
        cg = [r["count_gain"]["laplace"] for r in recs]
        summ[op] = dict(direct=M.by_stratum(dg, strata), c3=M.by_stratum(ig, strata),
                        control=M.by_stratum([None if r["sgcn"]["control"] is None else
                                              r["sgcn"]["control"][op] for r in recs], strata),
                        placebo=M.by_stratum([None if r["sgcn"]["placebo"] is None else
                                              r["sgcn"]["placebo"][op] for r in recs], strata),
                        amplification_laplace=M.amplification(dg, cg),
                        indirect_share=M.indirect_share(ig, dg),
                        direct_post_stratified=M.post_stratified_mean(dg, strata, sizes))
    summ["eigentrust"] = dict(
        direct_d_pct=M.by_stratum([r["eigentrust"]["direct"]["d_pct"] for r in recs], strata),
        control_d_pct=M.by_stratum([None if r["eigentrust"]["control"] is None else
                                    r["eigentrust"]["control"]["d_pct"] for r in recs], strata))
    out = dict(ds=a.ds, seed=a.seed, epochs=a.epochs, model="SGCN (PyG SignedGCN)",
               strata_sizes=sizes, sample=tg.tolist(), summary=summ, targets=recs,
               runtime_s=round(time.time() - T0, 1))
    path = C.dump(out, out_name, D, force=a.force)
    print(f"SGCN exact: amp {summ['exact_retrain']['amplification_laplace']} share "
          f"{summ['exact_retrain']['indirect_share']}\nwrote {path}")


if __name__ == "__main__":
    main()
