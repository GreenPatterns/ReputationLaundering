"""Keep calibration targets out of every evaluated statistic.

Defect fixed here (found by the 2026-10-09 paper audit): ``calibrate_operators``
excludes an *unstratified* seed-7 draw from its calibration candidates, but
``probe4`` evaluates a *stratified* seed-7 draw, so the two sets can overlap.
Measured: 2 of 5 calibration targets are also evaluated on bitcoinalpha
(3684, 3724) and on wiki-rfa (6328, 9537); none on bitcoinotc.

Calibration tunes only U3 to U5 and the weight-channel damping, so results
under U1 (unlink) and U2 (exact retraining), including the full-cohort U2 flip
rate, do not depend on it. Every per-target statistic of the experiments that
use the calibrated operators (probe4, fed_laundering, verification) is
recomputed here from the stored per-target records, without the calibration
targets. Each recomputation first reproduces the stored statistic on the
unfiltered records, so a mismatch between this module and the experiment code
raises instead of silently changing a result.

The stored calibration files are not re-run: re-running calibration would
change the tuned settings of the running sweep.
"""
import json
import math
import os
from types import SimpleNamespace

from keeper.laundering import core as C
from keeper.laundering import metrics as M

REL_TOL = 1e-9


def calibration_targets(ds: str, path: str | None = None) -> set:
    """Target ids that the dataset's calibration was tuned on (empty if no file)."""
    path = path or os.path.join(C.RES, f"calibration_{ds}_s0.json")
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {int(t) for t in json.load(fh).get("calibration_targets", [])}


def same(a, b, tol: float = REL_TOL) -> bool:
    """Equality of JSON-like values: NaN equals NaN, floats compared relatively."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) \
            and not isinstance(b, bool):
        a, b = float(a), float(b)
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def _json(x):
    return json.loads(json.dumps(x, default=float))


# ---------------------------------------------------------------- probe4
def probe4_summary(run: dict, exclude: set) -> dict:
    """probe4's ``summarise`` on the stored records, without ``exclude``."""
    from keeper.laundering import probe4 as P4
    feasible = (run.get("calibration") or {}).get("influence_feasible", True)
    cx = SimpleNamespace(strata_sizes=run["strata_sizes"],
                         direct_ops=tuple(op for op in P4.DIRECT_OPS_ALL
                                          if op != "influence_newton" or feasible))
    co = run.get("full_cohort")
    if co is not None:                     # JSON turned the int target keys into strings
        co = dict(co, r_new={int(k): v for k, v in co["r_new"].items()})
    recs = [r for r in run["targets"] if int(r["target"]) not in exclude]
    return _json(P4.summarise(cx, recs, co, {"tau": run["base"]["tau"]}))


def clean_probe4(run: dict, exclude: set) -> tuple:
    """(run with calibration targets removed from its summary, excluded ids)."""
    if not same(probe4_summary(run, set()), run["summary"]):
        raise RuntimeError(f"probe4 {run['ds']} s{run['seed']}: re-summarising the stored "
                           "records does not reproduce the stored summary")
    hit = sorted(exclude & {int(r["target"]) for r in run["targets"]})
    if not hit:
        return run, []
    out = dict(run, summary=probe4_summary(run, set(hit)))
    out["targets"] = [r for r in run["targets"] if int(r["target"]) not in set(hit)]
    return out, hit


# ---------------------------------------------------------------- fed_laundering
def fed_retained(run: dict, exclude: set) -> dict:
    """Mean retained projection per aggregator, as fed_laundering's ``h6``."""
    recs = [r for r in run["targets"] if int(r["target"]) not in exclude]
    out = {}
    for agg in ("geomedian", "krum"):
        vals = [r["aggs"][agg]["retained_projection"] for r in recs]
        out[agg] = M.by_stratum(vals, [r["stratum"] for r in recs])["all"]["mean"]
    return out


def clean_fed(run: dict, exclude: set) -> tuple:
    if not same(fed_retained(run, set()), run["h6"]["retained_projection"]):
        raise RuntimeError(f"fed_laundering {run['ds']} s{run['seed']}: recomputed h6 "
                           "does not reproduce the stored value")
    hit = sorted(exclude & {int(r["target"]) for r in run["targets"]})
    if not hit:
        return run, []
    h6 = dict(run["h6"], retained_projection=fed_retained(run, set(hit)))
    return dict(run, h6=h6), hit


# ---------------------------------------------------------------- verification
def verification_rate(run: dict, exclude: set) -> tuple:
    """(H4 pass rate over laundering requests, their number), without ``exclude``."""
    laund = [r["passes_all"] for r in run["requests"]
             if r["kind"].startswith("launder") and (r["target"] is None
                                                      or int(r["target"]) not in exclude)]
    return (float(sum(laund) / len(laund)) if laund else None), len(laund)


def clean_verification(run: dict, exclude: set) -> tuple:
    rate, n = verification_rate(run, set())
    if not (same(rate, run["h4"]["pass_rate"]) and n == run["h4"]["n"]):
        raise RuntimeError(f"verification {run['ds']} s{run['seed']}: recomputed H4 rate "
                           "does not reproduce the stored value")
    hit = sorted(exclude & {int(r["target"]) for r in run["requests"]
                            if r["target"] is not None})
    if not hit:
        return run, []
    rate, n = verification_rate(run, set(hit))
    return dict(run, h4=dict(run["h4"], pass_rate=rate, n=n)), hit
