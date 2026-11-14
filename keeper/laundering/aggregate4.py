"""Multi-seed verdicts for H1-H3, K1-K2 from probe4 runs (proposal §6).

Decision rule, applied to every claim: the statistic is the mean over seeds,
``sd`` the standard deviation across seeds, and ``f = 0.02`` the proposal's
floor. A threshold claim "x >= c" is
    SUPPORTED      if mean - sd - f >= c
    NOT SUPPORTED  if mean + sd + f <  c
    INCONCLUSIVE   otherwise.
With fewer than MIN_SEEDS seeds no verdict is issued (the old aggregator issued
verdicts from n=1 with sd=0; audit M11). The proposal asks for 5 seeds, and a
run with fewer is flagged even when a verdict is issued.

H1 uses the literal §6 flip rate (denominator: all distrust-majority targets):
the full-cohort batched count where it exists, otherwise the post-stratified
sample estimate. H2 uses the Laplace counting baseline (amendment A2) and the
indirect share with exact retraining in both numerator and denominator.

Calibration targets are removed from every per-target statistic that depends on
the calibrated operators (probe4, fed_laundering, verification) before any
verdict; see laundering/calibration_guard.py for the defect and the checks.

Usage: <ps python> -m keeper.laundering.aggregate4 [--glob 'probe4_*_s*.json'] [--force]
(writes results/laundering/probe4_aggregate.json; an existing file is replaced only with --force)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import calibration_guard as CG  # noqa: E402

FLOOR = 0.02
MIN_SEEDS = 2
SPEC_SEEDS = 5


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else None), len(xs)


def threshold_verdict(xs, c, strict=False):
    m, sd, n = ms(xs)
    if m is None:
        return dict(verdict="NO DATA", mean=None, sd=None, n_seeds=n)
    if n < MIN_SEEDS or sd is None:
        return dict(verdict=f"NO VERDICT (n_seeds={n} < {MIN_SEEDS})", mean=m, sd=sd, n_seeds=n)
    lo, hi = m - sd - FLOOR, m + sd + FLOOR
    ok = lo > c if strict else lo >= c
    bad = hi <= c if strict else hi < c
    v = "SUPPORTED" if ok else "NOT SUPPORTED" if bad else "INCONCLUSIVE"
    if n < SPEC_SEEDS:
        v += f" (n_seeds={n} < {SPEC_SEEDS} spec)"
    return dict(verdict=v, mean=m, sd=sd, n_seeds=n, band=[lo, hi], threshold=c)


def get(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p) if isinstance(d, dict) else None
    return d


def h1_rate(run):
    fc = run.get("full_cohort")
    if fc:
        return fc["flip"]["all"]["literal"], "full_cohort_batched"
    ps = get(run, "summary", "h1_sample_post_stratified", "p0.5", "estimate")
    return ps, "sample_post_stratified"


def verdicts(runs):
    out = {}
    h1 = [h1_rate(r)[0] for r in runs]
    out["H1_cheapness_flip>=0.30"] = dict(threshold_verdict(h1, 0.30),
                                         source=sorted({h1_rate(r)[1] for r in runs}))
    amp = [get(r, "summary", "amplification", "laplace", "exact_retrain", "all") for r in runs]
    share = [get(r, "summary", "indirect_share", "c3_white", "exact_retrain", "all")
             for r in runs]
    va = threshold_verdict(amp, 1.5, strict=True)
    vs = threshold_verdict(share, 0.3)
    both = ("SUPPORTED" if va["verdict"].startswith("SUPPORTED")
            and vs["verdict"].startswith("SUPPORTED") else
            "NOT SUPPORTED" if (va["verdict"].startswith("NOT")
                                or vs["verdict"].startswith("NOT")) else
            "INCONCLUSIVE" if va["mean"] is not None else va["verdict"])
    out["H2_amplification>1.5_and_share>=0.3"] = dict(
        verdict=both, amplification=va, indirect_share=vs,
        amplification_raw=threshold_verdict(
            [get(r, "summary", "amplification", "raw", "exact_retrain", "all")
             for r in runs], 1.5, strict=True),
        indirect_share_greybox=threshold_verdict(
            [get(r, "summary", "indirect_share", "c3_grey", "exact_retrain", "all")
             for r in runs], 0.3))
    d = [get(r, "summary", "h3_exact_minus_neggrad", "all", "mean") for r in runs]
    m, sd, n = ms(d)
    if m is None or sd is None or n < MIN_SEEDS:
        v3 = f"NO VERDICT (n_seeds={n})" if m is not None else "NO DATA"
    else:
        v3 = "SUPPORTED (non-inferior)" if m >= -(sd + FLOOR) else "NOT SUPPORTED"
    out["H3_exact>=neggrad"] = dict(verdict=v3, mean_exact_minus_neggrad=m, sd=sd,
                                    n_seeds=n, wilcoxon_p_per_seed=[
                                        get(r, "summary", "h3_wilcoxon_p") for r in runs])
    g = [get(r, "summary", "arms", "direct", "exact_retrain", "gain", "all", "mean")
         for r in runs]
    mg, sdg, ng = ms(g)
    mh1, _, _ = ms(h1)
    if mg is None or sdg is None or ng < MIN_SEEDS:
        k1 = f"NO VERDICT (n_seeds={ng})"
    else:
        flat = abs(mg) <= sdg + FLOOR and (mh1 is not None and mh1 < 0.10)
        k1 = "KILL (models do not respond)" if flat else "NOT TRIGGERED"
    out["K1_flat"] = dict(verdict=k1, mean_gain=mg, sd=sdg, flip_literal=mh1)
    ma, _, _ = ms(amp)
    msh, _, _ = ms(share)
    if ma is None or msh is None or len([x for x in amp if x is not None]) < MIN_SEEDS:
        k2 = "NO VERDICT"
    else:
        k2 = "TRIVIAL (reframing is the user's decision)" if (ma <= 1 and abs(msh) < 0.1) \
            else "NOT TRIGGERED"
    out["K2_trivial"] = dict(verdict=k2, amplification=ma, indirect_share=msh)
    pl = [get(r, "summary", "placebo_floor", "sd") for r in runs]
    out["diag_placebo_floor_sd"] = ms(pl)[:2]
    out["diag_direct_gain_by_stratum"] = {
        h: ms([get(r, "summary", "arms", "direct", "exact_retrain", "gain", h, "mean")
               for r in runs])[:2] for h in ("all", "pos", "nopos")}
    out["diag_control_by_stratum"] = {
        h: ms([get(r, "summary", "arms", "control", "exact_retrain", "gain", h, "mean")
               for r in runs])[:2] for h in ("all", "pos", "nopos")}
    return out


def below_verdict(xs, c):
    """Claim "x < c": SUPPORTED if mean + sd + f < c, NOT if mean - sd - f >= c."""
    m, sd, n = ms(xs)
    if m is None:
        return dict(verdict="NO DATA", mean=None, sd=None, n_seeds=n)
    if n < MIN_SEEDS or sd is None:
        return dict(verdict=f"NO VERDICT (n_seeds={n} < {MIN_SEEDS})", mean=m, sd=sd, n_seeds=n)
    v = ("SUPPORTED" if m + sd + FLOOR < c else
         "NOT SUPPORTED" if m - sd - FLOOR >= c else "INCONCLUSIVE")
    if n < SPEC_SEEDS:
        v += f" (n_seeds={n} < {SPEC_SEEDS} spec)"
    return dict(verdict=v, mean=m, sd=sd, n_seeds=n, threshold=c)


def load_family(pattern):
    out = {}
    for p in sorted(glob.glob(os.path.join(C.RES, pattern))):
        r = json.load(open(p))
        out.setdefault(r["ds"], []).append(r)
    return out


EXCLUDED = {}                     # (family, ds) -> {seed: excluded calibration targets}


def _cal_source(run):
    src = get(run, "calibration", "source")
    return src if src and os.path.exists(src) else None


def without_calibration(family, run, clean):
    """Apply a calibration_guard cleaner and record what it removed."""
    run, hit = clean(run, CG.calibration_targets(run["ds"], _cal_source(run)))
    EXCLUDED.setdefault((family, run["ds"]), {})[run["seed"]] = hit
    return run


def other_hypotheses():
    """H4 (verification), H5/K3 (VPE), H6 (federated), and SGCN as a second model."""
    res = {}
    for ds, rs in load_family("verification_*_s*.json").items():
        rs = [without_calibration("verification", r, CG.clean_verification) for r in rs]
        laund = sum(r["h4"]["n"] for r in rs)
        passed = sum(r["h4"]["pass_rate"] * r["h4"]["n"] for r in rs)
        rate = passed / laund if laund else None
        res.setdefault(ds, {})["H4_verification_pass>=0.99"] = dict(
            verdict=("NO DATA" if rate is None else
                     "SUPPORTED" if rate >= 0.99 else "NOT SUPPORTED"),
            pooled_pass_rate=rate, n_requests=laund, seeds=[r["seed"] for r in rs],
            note="pooled rate vs 0.99; the +-0.02 gain floor is not applied to a "
                 "rate bounded by 1")
    for ds, rs in load_family("vpe_*_s*.json").items():
        keys = sorted({(x["rho"], x["granularity"], x["eps"]) for r in rs for x in r["runs"]
                       if not x["small_pool_attack"]}, key=str)
        table = []
        for k in keys:
            pts = [x for r in rs for x in r["runs"]
                   if (x["rho"], x["granularity"], x["eps"]) == k and not x["small_pool_attack"]]
            red, sred, _ = ms([x["reduction_edge"] for x in pts])
            dm, sdm, _ = ms([abs(x["d_t1_mcc_vs_plain"]) if x["d_t1_mcc_vs_plain"] is not None
                             else None for x in pts])
            auc, sauc, n = ms([x["linkage"]["auc"]["combined"] for x in pts])
            sd0 = lambda s: s if s is not None else 0.0      # noqa: E731
            meets = (n >= MIN_SEEDS and red is not None and red - sd0(sred) >= 0.8
                     and dm is not None and dm + sd0(sdm) <= 0.01
                     and auc is not None and auc + sd0(sauc) <= 0.6)
            table.append(dict(rho=k[0], granularity=k[1], eps=k[2], n_seeds=n,
                              reduction=red, reduction_sd=sred, abs_dmcc=dm,
                              linkage_auc=auc, linkage_auc_sd=sauc, meets_h5=meets,
                              median_pool=ms([x["linkage"]["pool_size"]["median"]
                                              for x in pts])[0]))
        practical = [t for t in table if t["linkage_auc"] is not None]
        res.setdefault(ds, {})["H5_VPE"] = dict(
            verdict=("SUPPORTED" if any(t["meets_h5"] for t in table) else
                     "NOT SUPPORTED" if table and all(t["n_seeds"] >= MIN_SEEDS for t in table)
                     else "NO VERDICT"),
            frontier=table)
        res[ds]["K3_VPE_linkage"] = dict(
            verdict=("NO DATA" if not practical else
                     "KILL (linkage > 0.75 everywhere)" if all(
                         t["linkage_auc"] - (t["linkage_auc_sd"] or 0) > 0.75 for t in practical)
                     else "NOT TRIGGERED"),
            min_auc=min((t["linkage_auc"] for t in practical), default=None))
    for ds, rs in load_family("fed_laundering_*_s*.json").items():
        rs = [without_calibration("fed_laundering", r, CG.clean_fed) for r in rs]
        res.setdefault(ds, {})["H6_federated"] = dict(
            geomedian=below_verdict([r["h6"]["retained_projection"]["geomedian"] for r in rs], 0.5),
            krum=below_verdict([r["h6"]["retained_projection"]["krum"] for r in rs], 0.5),
            note="retained projection of the unlearning update relative to FedAvg")
    for ds, rs in load_family("baselines_*_s*.json").items():
        res.setdefault(ds, {})["SGCN_second_model"] = dict(
            amplification_laplace_exact=threshold_verdict(
                [get(r, "summary", "exact_retrain", "amplification_laplace") for r in rs],
                1.5, strict=True),
            indirect_share_exact=threshold_verdict(
                [get(r, "summary", "exact_retrain", "indirect_share") for r in rs], 0.3))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="probe4_*_s*.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing probe4_aggregate.json")
    a = ap.parse_args()
    out_name = "probe4_aggregate.json"
    C.result_path(out_name, a.force)          # a derived aggregate is regenerated only with the flag
    extra = other_hypotheses()
    for ds, d in extra.items():
        print(f"\n[{ds}] other hypotheses")
        for k, v in d.items():
            print(f"  {k:30s} {v.get('verdict', '')}")
    runs = []
    for p in sorted(glob.glob(os.path.join(C.RES, a.glob))):
        if p.endswith(".partial.jsonl"):
            continue
        r = json.load(open(p))
        if "summary" in r:
            runs.append(without_calibration("probe4", r, CG.clean_probe4))
    if not runs and not extra:
        sys.exit("no probe4 or other experiment runs found")
    by_ds = {}
    for r in runs:
        by_ds.setdefault(r["ds"], []).append(r)
    hosts = {get(r, "provenance", "gpu_name") for r in runs}
    if len(hosts) > 1:
        print(f"WARNING: runs come from different GPUs {hosts}; results do not transfer "
              "across hardware, so they are not pooled", file=sys.stderr)
    summary = {}
    for ds, rs in sorted(by_ds.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        summary[ds] = dict(seeds=[r["seed"] for r in rs],
                           code=sorted({get(r, "provenance", "code_sha256")[:12] for r in rs}),
                           verdicts=verdicts(rs))
        print(f"\n[{ds}] seeds {summary[ds]['seeds']}")
        for k, v in summary[ds]["verdicts"].items():
            if isinstance(v, dict) and "verdict" in v:
                print(f"  {k:40s} {v['verdict']}  mean={v.get('mean', v.get('mean_gain'))}")
    excluded = {f"{fam}/{ds}": {str(k): v for k, v in seeds.items()}
                for (fam, ds), seeds in sorted(EXCLUDED.items())}
    for key, seeds in excluded.items():
        hits = sorted({t for v in seeds.values() for t in v})
        print(f"calibration targets excluded from {key}: {hits if hits else 'none'}")
    path = C.dump(dict(per_dataset=summary, other_hypotheses=extra,
                       excluded_calibration_targets=excluded), out_name, force=a.force)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
