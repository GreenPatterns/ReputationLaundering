"""Incidental laundering from honest erasures (proposal §6), with no adversary.

Scenarios: accounts leave the platform and every edge incident to them is
erased (GDPR account deletion).
  prolific   the top-R raters by distrust issued leave (R in --top-r).
  random     a uniform random fraction rho of active accounts leaves (rho in
             --rates; "realistic" deletion rates are not public, so a range is
             swept and stated).
For each scenario the platform applies unlink (parameters unchanged) and exact
retraining. The effect is measured on every distrust-majority target (at most
``budget`` distrust in-edges) that did not itself leave: gain, crossings of
p=0.5 (literal and conditional), and how many targets lost at least one
distrust in-edge. Everything is stratified by positive history (A3).

Run: CUDA_VISIBLE_DEVICES=0 <ps python> -m laundering.incidental --ds bitcoinalpha --seed 0
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402

from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import metrics as M  # noqa: E402
from keeper.laundering import signed as S  # noqa: E402
from keeper.laundering import unlearn as U  # noqa: E402


def departing_rows(Tr, accounts) -> np.ndarray:
    acc = np.asarray(sorted(accounts), dtype=np.int64)
    return np.where(np.isin(Tr.edges[:, 0], acc) | np.isin(Tr.edges[:, 1], acc))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="bitcoinalpha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--top-r", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--rates", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    ap.add_argument("--draws", type=int, default=3, help="random-deletion draws per rate")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing result file")
    a = ap.parse_args()
    out_name = a.out or f"incidental_{a.ds}_s{a.seed}.json"
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
    cohort, neg, pos = C.pick_targets(Tr, max_neg=a.budget, strict=True)
    n_pos_of = {int(v): len(pos.get(int(v), [])) for v in cohort}
    active = np.unique(Tr.edges.ravel())
    is_neg = S.neg_mask(Tr.labels)
    issued = np.bincount(Tr.edges[is_neg, 0], minlength=Tr.n)

    m0 = C.train(C.make_model(D, Tr, a.seed, tt), Tr, tt, epochs=a.epochs)
    r0 = panel.reputation(m0, C.embed(m0, Tr, tt), cohort)
    print(f"[{a.ds} s{a.seed}] cohort {len(cohort)} | top rater issues "
          f"{int(issued.max())} distrust edges", flush=True)

    scenarios = [("prolific", dict(top_r=r), set(np.argsort(-issued, kind="stable")[:r].tolist()))
                 for r in a.top_r]
    rng = np.random.default_rng(1_000 + a.seed)
    for rho in a.rates:
        for d in range(a.draws):
            k = max(1, int(round(rho * len(active))))
            scenarios.append(("random", dict(rate=rho, draw=d),
                              set(rng.choice(active, k, replace=False).tolist())))

    results = []
    for kind, params, accounts in scenarios:
        t1 = time.time()
        rows = departing_rows(Tr, accounts)
        mask = np.zeros(len(Tr.edges), bool)
        mask[rows] = True
        stay = np.array([int(v) not in accounts for v in cohort])
        tg = cohort[stay]
        touched = np.array([bool(mask[neg[int(v)]].any()) for v in tg])
        strata = [M.stratum_of(n_pos_of[int(v)]) for v in tg]
        rec = dict(kind=kind, params=params, n_accounts=len(accounts),
                   n_edges_erased=int(mask.sum()),
                   n_distrust_erased=int((mask & is_neg).sum()),
                   n_targets=int(len(tg)), n_targets_lost_distrust=int(touched.sum()),
                   ops={})
        for op in ("unlink", "exact_retrain"):
            u = (U.unlink(m0, G, Tr, mask, tt) if op == "unlink" else
                 U.exact_retrain(D, G, tt, Tr.rid[mask], a.seed, a.epochs, self_term=C.self_term_of(m0)))
            z = u.embed(tt)
            rn = panel.reputation(u.model, z, tg)
            gain = rn - r0[stay]
            tm = C.test_metrics(u.model, z, D, u.G)
            rec["ops"][op] = dict(
                gain=M.by_stratum(list(gain), strata),
                gain_touched=M.by_stratum(list(gain[touched]),
                                          [s for s, t in zip(strata, touched) if t]),
                flip={h: M.flip_rates(r0[stay][[h == "all" or s == h for s in strata]],
                                      rn[[h == "all" or s == h for s in strata]])
                      for h in M.STRATA},
                t1_mcc=tm["t1"]["mcc"], t2_mcc=tm["t2"]["mcc"])
        results.append(rec)
        f = rec["ops"]["exact_retrain"]["flip"]["all"]
        print(f"  {kind:8s} {params} accounts={len(accounts)} edges={rec['n_edges_erased']} "
              f"targets touched {rec['n_targets_lost_distrust']}/{len(tg)} | "
              f"retrain crossings {f['n_crossed']} (literal {f['literal']}) "
              f"({time.time() - t1:.0f}s)", flush=True)

    out = dict(ds=a.ds, seed=a.seed, epochs=a.epochs, budget=a.budget,
               memory_profile=mem, cohort_size=int(len(cohort)),
               top20_raters_distrust_share=float(np.sort(issued)[::-1][:20].sum()
                                                 / max(int(is_neg.sum()), 1)),
               scenarios=results, runtime_s=round(time.time() - T0, 1),
               note="random-deletion rates are an assumption swept over a range; "
                    "no public platform deletion rate was available")
    path = C.dump(out, out_name, D, force=a.force)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
