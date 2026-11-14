"""Pre-registered hyperparameter selection for the approximate unlearning operators.

The proposal names NegGrad+ and an influence-based operator but fixes no
hyperparameters. They are chosen here, BEFORE any probe run, on calibration
targets drawn disjointly from the probe's evaluation draw. The selection rule is
fixed in advance and follows the unlearning literature's yardstick, closeness to
exact retraining (Kurmanji et al. 2023):

    minimise the mean |forget CE(op) - forget CE(exact retrain)|, both measured
    on the SERVED state (model + erased inference graph),
    subject to mean Task-1 MCC drop <= MAX_MCC_DROP (utility preserved).

Forgetting is measured on the served graph because in LocalGDTE an edge's own
label is a message-passing input: on the pre-erasure graph every model, even an
exactly retrained one, reads F's signs back (measured: delta CE ~0.000), so that
yardstick cannot see forgetting. Exact retraining and unlink have no
hyperparameters and are run as references.
Writes ``results/laundering/calibration_<ds>_s<seed>.json`` (an existing file only with --force).
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402

from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import unlearn as U  # noqa: E402

MAX_MCC_DROP = 0.02
PROBE_DRAW_SEED = 7          # probe4's evaluation draw; calibration must avoid it
CAL_DRAW_SEED = 99

NEGGRAD_GRID = [dict(beta=b, lr=lr, epochs=10)
                for b in (0.95, 0.99) for lr in (1e-4, 5e-4, 1e-3, 5e-3)]
INFLUENCE_GRID = [dict(damping=d) for d in (0.1, 0.3, 1.0, 3.0)]
HEAD_GRID = [dict(damping=d) for d in (1e-3, 1e-2, 1e-1, 1.0)]


def calibration_targets(cohort: np.ndarray, n_probe: int, n_cal: int) -> np.ndarray:
    probe = set(np.random.default_rng(PROBE_DRAW_SEED).choice(
        cohort, min(n_probe, len(cohort)), replace=False).tolist())
    rest = np.array(sorted(set(cohort.tolist()) - probe), dtype=np.int64)
    if len(rest) == 0:
        raise SystemExit("cohort fully used by the probe draw; no calibration targets")
    rng = np.random.default_rng(CAL_DRAW_SEED)
    return np.sort(rng.choice(rest, min(n_cal, len(rest)), replace=False))


def evaluate(u, D, Tr0, panel, v, r0, forget, tt, mcc0):
    z = u.embed(tt)
    tm = C.test_metrics(u.model, z, D, u.G)
    return dict(d_rep=float(panel.reputation(u.model, z, [v])[0] - r0),
                forget_ce_served=U.forget_ce_served(u, u.Tr, Tr0, forget, tt),
                forget_ce_fullgraph=U.forget_ce(u, Tr0, forget, tt),
                mcc_t1=tm["t1"]["mcc"], d_mcc_t1=(None if tm["t1"]["mcc"] is None
                                                  else tm["t1"]["mcc"] - mcc0),
                mcc_t2=tm["t2"]["mcc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="bitcoinalpha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n-cal", type=int, default=5)
    ap.add_argument("--n-probe", type=int, default=50)
    ap.add_argument("--force", action="store_true", help="overwrite an existing result file")
    a = ap.parse_args()
    out_name = f"calibration_{a.ds}_s{a.seed}.json"
    C.result_path(out_name, a.force)          # refuse an existing result before the work, not after it
    C.set_deterministic(True)
    C.require_gpu()
    D = C.load(a.ds)
    C.apply_memory_profile(D)
    G = C.full_graph(D)
    tt = D.tt
    Tr = C.train_split(G, tt)
    panel = C.Panel(Tr, n_probe=256, seed=1234)
    cohort, neg, _ = C.pick_targets(Tr, max_neg=3, strict=True)
    tg = calibration_targets(cohort, a.n_probe, a.n_cal)
    t0 = time.time()
    m0 = C.train(C.make_model(D, Tr, a.seed, tt), Tr, tt, epochs=a.epochs)
    z0 = C.embed(m0, Tr, tt)
    mcc0 = C.test_metrics(m0, z0, D, G)["t1"]["mcc"]
    r0 = dict(zip(tg.tolist(), panel.reputation(m0, z0, tg).tolist()))
    print(f"[{a.ds} s{a.seed}] base T1 MCC {mcc0:.4f} | {len(tg)} calibration targets "
          f"| base train {time.time() - t0:.0f}s", flush=True)

    rows = []
    for v in tg.tolist():
        forget = np.zeros(len(Tr.edges), bool)
        forget[neg[v]] = True
        cases = [("exact_retrain",                      # first: it is the reference
                  lambda: U.exact_retrain(D, G, tt, Tr.rid[forget], a.seed, a.epochs,
                                          self_term=C.self_term_of(m0)), {}),
                 ("unlink", lambda: U.unlink(m0, G, Tr, forget, tt), {}),
                 ("unlink_frozen_x",
                  lambda: U.unlink(m0, G, Tr, forget, tt, recompute_features=False), {})]
        cases += [("neggrad_plus", (lambda h=h: U.neggrad_plus(m0, G, Tr, forget, tt, **h)), h)
                  for h in NEGGRAD_GRID]
        if U.full_hvp_feasible(Tr):
            cases += [("influence_newton",
                       (lambda h=h: U.influence_unlearn(m0, G, Tr, forget, tt, **h)), h)
                      for h in INFLUENCE_GRID]
        cases += [("head_newton", (lambda h=h: U.head_newton(m0, G, Tr, forget, tt, **h)), h)
                  for h in HEAD_GRID]
        for name, fn, hp in cases:
            t1 = time.time()
            u = fn()
            res = evaluate(u, D, Tr, panel, v, r0[v], forget, tt, mcc0)
            if name == "exact_retrain":
                ref = res["forget_ce_served"]
            res["gap_to_exact"] = abs(res["forget_ce_served"] - ref)
            rows.append(dict(target=v, operator=name, hp=hp, seconds=time.time() - t1,
                             info={k: w for k, w in u.info.items()
                                   if isinstance(w, (int, float, str, bool))}, **res))
            print(f"  v={v} {name:16s} {hp} dRep {res['d_rep']:+.4f} "
                  f"CE_F(served) {res['forget_ce_served']:.3f} gap {res['gap_to_exact']:.3f} "
                  f"dMCC1 {res['d_mcc_t1']} ({time.time() - t1:.1f}s)", flush=True)

    def pick(op, grid):
        best, table = None, []
        for hp in grid:
            sel = [r for r in rows if r["operator"] == op and r["hp"] == hp]
            gap = float(np.mean([r["gap_to_exact"] for r in sel]))
            dm = [r["d_mcc_t1"] for r in sel if r["d_mcc_t1"] is not None]
            drop = float(-np.mean(dm)) if dm else None
            ok = drop is not None and drop <= MAX_MCC_DROP
            table.append(dict(hp=hp, mean_gap_to_exact=gap, mean_mcc_drop=drop,
                              mean_forget_ce_served=float(np.mean(
                                  [r["forget_ce_served"] for r in sel])),
                              feasible=ok))
            if ok and (best is None or gap < best["mean_gap_to_exact"]):
                best = table[-1]
        return best, table

    ng_best, ng_table = pick("neggrad_plus", NEGGRAD_GRID)
    feasible = U.full_hvp_feasible(Tr)
    in_best, in_table = pick("influence_newton", INFLUENCE_GRID) if feasible else (None, [])
    hd_best, hd_table = pick("head_newton", HEAD_GRID)
    out = dict(ds=a.ds, seed=a.seed, epochs=a.epochs, base_mcc_t1=mcc0,
               rule=f"min mean |CE_F(op) - CE_F(exact)| on the served state, s.t. mean T1 MCC drop <= {MAX_MCC_DROP}",
               calibration_targets=tg.tolist(), probe_draw_seed=PROBE_DRAW_SEED,
               neggrad=dict(selected=ng_best, table=ng_table),
               influence=dict(selected=in_best, table=in_table,
                              feasible=feasible,
                              infeasible_reason=None if feasible else
                              "full-parameter exact HVP exceeds 48 GB (measured OOM)"),
               head_newton=dict(selected=hd_best, table=hd_table), rows=rows,
               runtime_s=round(time.time() - t0, 1))
    path = C.dump(out, out_name, D, force=a.force)
    print(f"NegGrad+ selected: {ng_best}\ninfluence selected: {in_best}\n"
          f"head_newton selected: {hd_best}\nwrote {path}")


if __name__ == "__main__":
    main()
