"""CPU checks for keeper/laundering/calibration_guard.py.

Run: <ps python> -m tests.test_calibration_guard (or the file path); pytest runs the same checks through
test_calibration_guard_checks.
"""
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

from keeper.laundering import calibration_guard as CG  # noqa: E402
from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import probe4 as P4  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def synthetic_probe4_run():
    rng = np.random.default_rng(0)

    def op():
        return {"gain": float(rng.normal(0.05, 0.02)), "d_t1_mcc": float(rng.normal(0, 0.01)),
                "pct_after": float(rng.uniform())}

    def rec(t, stratum):
        return {"target": t, "stratum": stratum, "r0": float(rng.uniform(0.2, 0.45)),
                "count_gain": {"laplace": 0.1, "raw": 0.2},
                "direct": {"ops": {o: op() for o in P4.DIRECT_OPS_ALL}},
                "account": {"ops": {o: op() for o in P4.CHEAP_OPS}},
                "c3_white": {"ops": {o: op() for o in P4.CHEAP_OPS}},
                "c3_grey": {"ops": {"unlink": None, "exact_retrain": None}},
                "control": None, "placebo": {"ops": {o: op() for o in P4.CHEAP_OPS}}}

    recs = [rec(t, "pos" if i % 2 else "nopos") for i, t in enumerate(range(11, 19))]
    cx = SimpleNamespace(strata_sizes={"pos": 20, "nopos": 30}, direct_ops=P4.DIRECT_OPS_ALL)
    co = {"n_batches": 2, "flip": {}, "flip_tau": {}, "gain": {},
          "r_new": {r["target"]: r["r0"] + 0.1 for r in recs}}
    run = {"ds": "synthetic", "seed": 0, "strata_sizes": cx.strata_sizes, "base": {"tau": 0.5},
           "calibration": {"influence_feasible": True}, "targets": recs, "full_cohort": co,
           "summary": P4.summarise(cx, recs, co, {"tau": 0.5})}
    return json.loads(json.dumps(run, default=float))         # as stored on disk


def main():
    print("G1  probe4 summaries without calibration targets")
    run = synthetic_probe4_run()
    out, hit = CG.clean_probe4(run, {12, 99})
    check("stored summary reproduced, target 12 removed", hit == [12]
          and len(out["targets"]) == 7
          and out["summary"]["arms"]["direct"]["exact_retrain"]["gain"]["all"]["n"] == 7)
    same_run, none = CG.clean_probe4(run, {99})
    check("no overlap leaves the run unchanged", none == [] and same_run is run)
    bad = json.loads(json.dumps(run))
    bad["summary"]["arms"]["direct"]["exact_retrain"]["gain"]["all"]["mean"] += 1.0
    try:
        CG.clean_probe4(bad, set())
        check("tampered summary is rejected", False)
    except RuntimeError:
        check("tampered summary is rejected", True)

    print("G2  federated and verification records")
    fed = sorted(glob.glob(os.path.join(C.RES, "fed_laundering_*_s*.json")))
    for p in fed[:2]:
        r = json.load(open(p))
        out, hit = CG.clean_fed(r, CG.calibration_targets(r["ds"]))
        check(f"{os.path.basename(p)} reproduces stored h6", True, f"excluded {hit}")
    if not fed:
        print("  SKIP  no fed_laundering results yet")
    ver = {"ds": "synthetic", "seed": 0, "h4": {"pass_rate": 0.5, "n": 4},
           "requests": [{"kind": "launder_C1", "target": 1, "passes_all": True},
                        {"kind": "launder_C2_edge", "target": 1, "passes_all": True},
                        {"kind": "launder_C1", "target": 2, "passes_all": False},
                        {"kind": "launder_C2_edge", "target": 2, "passes_all": False},
                        {"kind": "benign_random", "target": None, "passes_all": True}]}
    out, hit = CG.clean_verification(ver, {1})
    check("verification rate recomputed without target 1",
          hit == [1] and out["h4"]["pass_rate"] == 0.0 and out["h4"]["n"] == 2)

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


def test_calibration_guard_checks():
    """pytest entry point: the checks of main(), which ends with sys.exit(1 if FAILS else 0)."""
    try:
        main()
    except SystemExit as stop:
        assert stop.code == 0, FAILS


if __name__ == "__main__":
    main()
