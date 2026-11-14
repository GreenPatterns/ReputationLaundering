"""H8 (E9a, the self-term ablation) and E8 (the calibrated cold start, descriptive), from the E9a runs (Task 4.8).

  python -m keeper.analysis.e9a_e8 [--results-root ABS_REPO_ROOT] [--prereg PATH] [--n-boot 10000] [--force]

Inputs: the runs of keeper.cluster.e9a_tasks under --results-root (results/laundering/e9a/<id>.json) and their dumps
(results/dumps/e9a/<id>.npz). A missing run is listed, never imputed. Writes results/analysis/e9a_e8.json (core.dump,
atomic, R27) and a generated e9a_e8.md, then the dump folder's SHA-256 manifest (keeper.analysis.dump_manifest) when it
is absent or stale. CPU only.

Integrity, per run (each failure is listed; the endpoints are still computed from the runs that could be read, status
is "integrity_failure" and the exit status 2): the protocol's sha256; git_dirty False and git_untracked 0; one commit
over all runs; ds, seed, self_term, epochs (150) and no --smoke as its task; the target lists, stratum sizes, active
ids, incoming counts and test-snapshot rows (cold_start.held_out_snapshot) equal those recomputed from the network's
data (reference); the dump's recorded sha256; and every per-target r0, r, gain, n_after and pct_after of the JSON
recomputed exactly from the dump (r is the target's own entry of its population row).

H8 (protocol Sections 1 and 4), per network, on the 50 drawn targets, seeds and targets crossed:
  (a) mean U1 gain with self_term on; rule: 95% interval > 0; p = max(one-sided one-sample seed-t p of the per-seed
      means, (1 + #{crossed replicates <= 0}) / (B + 1));
  (b) R = ratio of the seed-averaged mean U1 gains, on over off (pairs: same seed and target, both runs present);
      rule: lower bound >= 0.5, read three ways (met, inconclusive, partial); contradicted when the whole interval lies
      below 0 (ruling R62); p = max(linearized one-sided seed-t p of R - 0.5, (1 + #{replicates < 0.5}) / (B + 1)).
  Headline interval: hull of the crossed bootstrap percentile interval and the seed-t interval (linearized for R), not
  clipped (signed). B counts the defined replicates. Fewer than 80% of the seeds: not evaluable. Network p: max over
  (a), (b); hypothesis p: max over networks (intersection-union), for the decisions step's Holm correction. The
  nested bootstrap interval is a sensitivity.
Secondary (intervals, no rule): gain with self_term off; probe4's post-stratified mean; the >= 4 stratum's gains;
percentiles before and after U1; edgeless shares (a property of the graph); base Task 1 test MCC.

E8 (Section 2.10, ruling R60), on the self-term-off runs: cold_start.evaluate for m in (1, 2, 4) with each target's
post-U1 population; target_proxy[i] = population[i, position of target i] (the same array as row i); every report
must say laundering_population == "per-target post-U1", its uncalibrated percentiles must equal the run's pct_after,
and the target ids are stored with the rows. Summaries over seeds: laundering percentiles (drawn and >= 4 targets),
the newcomer rank cost and the low-degree test MCC.
"""
import argparse
import datetime
import json
import math
import os
import platform
import socket
import sys
import zipfile
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
from scipy.stats import t as student_t

from keeper.analysis import cold_start as CS
from keeper.analysis import dump_manifest as DM
from keeper.analysis.bootstrap import (column_means_stat, combine_mean, combine_ratio, crossed_replicates, mean_stat,
                                       nested_bootstrap, per_seed_values, percentile_interval)
from keeper.analysis.intervals import BOOT_SEED, PROPORTION, Boot, clip_to_support, headline_interval, seed_t_for
from keeper.analysis.provenance import git_state
from keeper.cluster import e9a_tasks as T
from keeper.cluster import protocol_settings as PS
from keeper.cppr.protocol import sha256_file
from keeper.laundering import core as C
from keeper.laundering import e9a_runner as R
from keeper.laundering import metrics as M

SCHEMA = "e9a_e8/1"
THRESHOLD = 0.5                  # H8(b)
MIN_SEED_SHARE = Fraction(4, 5)
NOT_EVALUABLE = "not evaluable"
ANALYSIS = os.path.join(C.ROOT, "results", "analysis")
MCC_SUPPORT = (-1.0, 1.0)
EXIT_OK, EXIT_INTEGRITY = 0, 2


# ------------------------------------------------------------------------ inputs
def reference(ds: str) -> SimpleNamespace:
    """What every run of ds must record, recomputed from the data (CPU, no model)."""
    D = C.load(ds)
    G = C.full_graph(D)
    Tr = C.train_split(G, D.tt)
    drawn, stratum, sizes = R.target_lists(Tr)
    rows, dst, y = CS.held_out_snapshot(G, D.tt)
    return SimpleNamespace(drawn=drawn, stratum=stratum, sizes=sizes, active=np.unique(Tr.edges.ravel()),
                           n_in=CS.incoming_counts(Tr.edges[:, 1], D.n), test_rows=rows, test_dst=dst, test_y=y)


def expected_runs(results_root: str, networks) -> list:
    """(task row, ds, seed, self_term) of every registered run of the networks, from keeper.cluster.e9a_tasks."""
    grid = {T.task_id(ds, s, st): (ds, s, st) for ds in networks for s in PS.SEEDS_5 for st in T.SELF_TERMS}
    return [(row, *grid[row.id]) for row in T.e9a_tasks(results_root) if row.id in grid]


def _run_failures(res: dict, ds: str, seed: int, st: str, prereg_sha: str | None, ref) -> list:
    cfg, prov = res.get("config", {}), res.get("provenance", {})
    checks = [
        ((res.get("preregistration") or {}).get("sha256") == prereg_sha, "protocol sha256 differs from the protocol's"),
        (prov.get("git_dirty") is False, f"git_dirty is {prov.get('git_dirty')!r}, not False"),
        (prov.get("git_untracked") == 0, f"git_untracked is {prov.get('git_untracked')!r}, not 0"),
        ((res.get("ds"), res.get("seed"), res.get("self_term"), res.get("epochs")) == (ds, seed, st == "on", PS.EPOCHS),
         f"ds/seed/self_term/epochs {(res.get('ds'), res.get('seed'), res.get('self_term'), res.get('epochs'))} are "
         f"not the task's {(ds, seed, st == 'on', PS.EPOCHS)}"),
        (cfg.get("smoke") is None, "a --smoke run"),
        (cfg.get("targets_drawn") == ref.drawn.tolist(), "drawn targets differ from probe4's draw"),
        (cfg.get("targets_stratum") == ref.stratum.tolist(), "stratum differs from the >= 4 distrust-majority accounts"),
        (cfg.get("strata_sizes") == ref.sizes, "strata sizes differ from probe4's cohort"),
        ([r.get("target") for r in res.get("targets", [])] == ref.drawn.tolist() + ref.stratum.tolist(),
         "per-target records are not drawn + stratum in order")]
    return [message for ok, message in checks if not ok]


def _dump_failures(res: dict, z: dict, ref) -> list:
    targets = np.concatenate([ref.drawn, ref.stratum])
    same = [(np.array_equal(z["active_ids"], ref.active), "active ids"), (np.array_equal(z["n_in"], ref.n_in), "n_in"),
            (np.array_equal(z["test_rows"], ref.test_rows), "test rows"),
            (np.array_equal(z["test_dst"], ref.test_dst), "test destinations"),
            (np.array_equal(z["test_y"], ref.test_y), "test labels"), (np.array_equal(z["target_ids"], targets), "targets"),
            (z["population"].shape == (len(targets), len(ref.active)), "population shape"),
            (len(z["test_p"]) == len(ref.test_rows), "test probabilities")]
    failures = [f"dump {name} differ from the data's" for ok, name in same if not ok]
    if failures:
        return failures
    pos = np.searchsorted(z["active_ids"], targets)
    own = z["population"][np.arange(len(targets)), pos].astype(np.float64)
    base = z["base_proxy"][pos].astype(np.float64)
    recs = res["targets"]
    pct = CS.percentiles_after_deletion(z["active_ids"], z["base_proxy"], targets, own, z["population"])
    for name, got, want in (("r", own, [r["r"] for r in recs]), ("r0", base, [r["r0"] for r in recs]),
                            ("gain", own - base, [r["gain"] for r in recs]),
                            ("n_after", z["target_n"], [r["n_after"] for r in recs]),
                            ("pct_after", pct, [r["pct_after"] for r in recs])):
        bad = np.flatnonzero(np.asarray(got, dtype=np.float64) != np.asarray(want, dtype=np.float64))
        if len(bad):
            failures.append(f"{name} of target {int(targets[bad[0]])} ({len(bad)} targets) differs between the JSON "
                            "and the dump")
    return failures


def load_runs(results_root: str, networks, prereg_sha: str | None, refs: dict) -> tuple:
    """(runs {(ds, seed, self_term): SimpleNamespace(id, res, z)}, missing, failures, inputs {relpath: sha256},
    the sorted git commits of the runs)."""
    runs, missing, failures, inputs = {}, [], [], {}
    dump_dir = os.path.join(results_root, "results", "dumps", "e9a")
    for row, ds, seed, st in expected_runs(results_root, networks):
        if not os.path.isfile(row.done_path):
            missing.append(dict(run=row.id, reason="result missing"))
            continue
        inputs[os.path.relpath(row.done_path, results_root)] = sha256_file(row.done_path)
        with open(row.done_path) as f:
            res = json.load(f)
        problems = _run_failures(res, ds, seed, st, prereg_sha, refs[ds])
        dump = os.path.join(dump_dir, os.path.basename((res.get("dump") or {}).get("path") or f"{row.id}.npz"))
        if not os.path.isfile(dump):
            missing.append(dict(run=row.id, reason="dump missing"))
            failures += [dict(run=row.id, detail=p) for p in problems]
            continue
        sha = inputs[os.path.relpath(dump, results_root)] = sha256_file(dump)
        if sha != (res.get("dump") or {}).get("sha256"):
            problems.append("dump sha256 differs from the recorded one")
        try:
            with np.load(dump) as npz:
                z = {k: npz[k] for k in npz.files}
        except (OSError, ValueError, zipfile.BadZipFile) as err:
            failures += [dict(run=row.id, detail=p) for p in problems + [f"dump unreadable: {err}"]]
            continue
        if not problems:
            problems = _dump_failures(res, z, refs[ds])
        failures += [dict(run=row.id, detail=p) for p in problems]
        runs[(ds, seed, st)] = SimpleNamespace(id=row.id, res=res, z=z)
    commits = sorted({r.res.get("provenance", {}).get("git_commit") or "none" for r in runs.values()})
    if len(commits) > 1:
        failures.append(dict(run=None, detail=f"the runs come from {len(commits)} commits: {commits}"))
    return runs, missing, failures, dict(sorted(inputs.items())), commits


# ------------------------------------------------------------------------ statistics
def p_seed_t(values, null: float = 0.0):
    """One-sided seed-level t p-value of mean(values) > null (df = n - 1); None with fewer than two values."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return None
    excess, sd = v.mean() - null, v.std(ddof=1)
    if sd == 0:
        return 0.0 if excess > 0 else 1.0
    return float(student_t.sf(excess / (sd / math.sqrt(len(v))), len(v) - 1))


def p_ratio_seed_t(num, den, threshold: float):
    """One-sided linearized seed-t p-value of R = mean(num) / mean(den) > threshold, with the residuals of
    bootstrap.ratio_seed_t_interval (df = n - 1); None with fewer than two seeds or mean(den) = 0."""
    a, b = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    if len(a) < 2 or b.mean() == 0:
        return None
    ratio = a.mean() / b.mean()
    se = ((a - ratio * b) / b.mean()).std(ddof=1) / math.sqrt(len(a))
    if se == 0:
        return 0.0 if ratio > threshold else 1.0
    return float(student_t.sf((ratio - threshold) / se, len(a) - 1))


def p_bootstrap(replicates, threshold: float = 0.0, strict: bool = False) -> float:
    """(1 + #{replicates <= threshold}) / (B + 1), with < when strict; B counts the defined replicates."""
    r = np.asarray(replicates, dtype=float)
    r = r[np.isfinite(r)]
    return (1 + int(np.sum(r < threshold) if strict else np.sum(r <= threshold))) / (len(r) + 1)


def summary(cells: dict, name: str, boot: Boot, *, ratio: bool = False, support=None, unit=None) -> tuple:
    """(fields, crossed replicates or None) of a mean (or, with ratio, the ratio of seed-averaged column means) over
    cells seed -> {target: [values]}: estimate, per-seed values and the headline interval (hull of the crossed
    bootstrap and the seed-t interval; Clopper-Pearson for a 0/1 field at 0 or 1), clipped to support."""
    stat, combine = (column_means_stat, combine_ratio) if ratio else (mean_stat, combine_mean)
    seeds = sorted(cells)
    per_seed = per_seed_values(cells, stat) if cells else []
    out = dict(n_seeds=len(seeds), seeds=seeds, n_pairs=sum(len(c) for c in cells.values()),
               n_targets=len(set().union(*(set(c) for c in cells.values()))) if cells else 0)
    if ratio:
        num, den = [p[0] for p in per_seed], [p[1] for p in per_seed]
        out.update(estimate=float(np.mean(num) / np.mean(den)) if per_seed and np.mean(den) != 0 else None,
                   numerator_mean=float(np.mean(num)) if per_seed else None,
                   denominator_mean=float(np.mean(den)) if per_seed else None,
                   by_seed={s: n / d if d else None for s, n, d in zip(seeds, num, den)},
                   by_seed_numerator=dict(zip(seeds, num)), by_seed_denominator=dict(zip(seeds, den)))
    else:
        out.update(estimate=float(np.mean(per_seed)) if per_seed else None, by_seed=dict(zip(seeds, per_seed)))
    if len(seeds) < 2:
        return dict(out, ci95=None, interval_note="fewer than two seeds"), None
    try:
        point, reps, _ = crossed_replicates(cells, stat, combine, boot.n_boot, boot.seed_for(name, "crossed"))
        boot_ci = percentile_interval(reps)
        nested = list(nested_bootstrap(cells, stat, combine, boot.n_boot, boot.seed_for(name, "nested"))[1:])
    except FloatingPointError as err:
        return dict(out, ci95=None, interval_note=f"bootstrap refused: {err}"), None
    seed_t = seed_t_for(cells, stat, combine)
    ci, source = headline_interval(point, boot_ci, seed_t, cells, unit)
    clipped, changed = clip_to_support(ci, support)
    out.update(ci95=clipped, ci95_bootstrap=boot_ci, ci95_seed_t=seed_t, ci95_nested=nested,
               ci95_source=source + (f", clipped to {list(support)}" if changed else ""),
               bootstrap_streams={d: boot.seed_for(name, d) for d in ("crossed", "nested")})
    return out, reps


def difference_decision(ci, evaluable: bool) -> dict:
    """Rule "interval > 0": met above 0, contradicted below 0."""
    if not evaluable or ci is None:
        return dict(decision=NOT_EVALUABLE, met=False, contradicted=False)
    met, contradicted = ci[0] > 0, ci[1] < 0
    return dict(decision="met" if met else "contradicted" if contradicted else "not met", met=bool(met),
                contradicted=bool(contradicted))


def threshold_decision(ci, threshold: float, evaluable: bool) -> dict:
    """Rule "lower bound >= threshold", read three ways (Section 1); contradicted only below 0 (ruling R62)."""
    if not evaluable or ci is None:
        return dict(decision=NOT_EVALUABLE, reading=None, met=False, contradicted=False)
    reading = "met" if ci[0] >= threshold else "inconclusive" if ci[1] >= threshold else "partial"
    return dict(decision=reading, reading=reading, met=reading == "met", contradicted=bool(ci[1] < 0))


def component(cells: dict, name: str, boot: Boot, *, threshold=None, dropped: int = 0) -> dict:
    """A confirmatory component: H8(a) (threshold None: a mean tested against 0) or H8(b) (a ratio against
    threshold), with its intervals, p-values and decision."""
    ratio = threshold is not None
    out, reps = summary(cells, name, boot, ratio=ratio)
    expected = len(PS.SEEDS_5)
    evaluable = reps is not None and Fraction(out["n_seeds"], expected) >= MIN_SEED_SHARE
    out.update(kind="ratio" if ratio else "difference", n_seeds_expected=expected, n_pairs_dropped=int(dropped),
               rule=f"95% interval lower bound >= {threshold:g}" if ratio else "95% interval > 0")
    if reps is None:
        return dict(out, evaluable=False, p_value=None, p_seed_t=None, p_bootstrap=None,
                    **(threshold_decision(None, threshold, False) if ratio else difference_decision(None, False)))
    if ratio:
        seeds = out["seeds"]
        p_t = p_ratio_seed_t([out["by_seed_numerator"][s] for s in seeds],
                             [out["by_seed_denominator"][s] for s in seeds], threshold)
        p_b, decision = p_bootstrap(reps, threshold, strict=True), threshold_decision(out["ci95"], threshold, evaluable)
    else:
        p_t = p_seed_t(list(out["by_seed"].values()))
        p_b, decision = p_bootstrap(reps, 0.0), difference_decision(out["ci95"], evaluable)
    return dict(out, evaluable=evaluable, p_value=None if p_t is None else max(p_t, p_b), p_seed_t=p_t,
                p_bootstrap=p_b, **decision)


def network_summary(components: dict) -> dict:
    """One network: evaluable when every component is; rule met when every component meets it; p = the largest."""
    evaluable = all(c["evaluable"] and c["p_value"] is not None for c in components.values())
    met = evaluable and all(c["met"] for c in components.values())
    return dict(evaluable=evaluable, interval_rule_met=met,
                p_value=max(c["p_value"] for c in components.values()) if evaluable else None,
                contradicted=sorted(n for n, c in components.items() if c["contradicted"]),
                status=NOT_EVALUABLE if not evaluable else "interval rule met" if met else "interval rule not met")


def hypothesis_summary(networks: dict) -> dict:
    """H8 over the registered networks (its rule holds "on each network"); p = the largest network p-value."""
    missing = [ds for ds in PS.NETWORKS if ds not in networks]
    evaluable = not missing and all(n["evaluable"] for n in networks.values())
    met = evaluable and all(n["interval_rule_met"] for n in networks.values())
    return dict(evaluable=evaluable, interval_rule_met=met, networks_missing=missing,
                p_value=max(n["p_value"] for n in networks.values()) if evaluable else None,
                contradicted={ds: n["contradicted"] for ds, n in networks.items() if n["contradicted"]},
                status_before_holm=(NOT_EVALUABLE if not evaluable else
                                    "interval rule met on every network" if met else "interval rule not met"))


# ------------------------------------------------------------------------ cells
def cells_of(runs: dict, ds: str, st: str, group: str, field: str) -> dict:
    """seed -> {target: [record[field]]} over the runs of (ds, self_term st), targets of one group."""
    out = {}
    for (d, seed, s), run in sorted(runs.items()):
        cells = {r["target"]: [float(r[field])] for r in run.res["targets"] if r["group"] == group}
        if d == ds and s == st and cells:
            out[seed] = cells
    return out


def paired(on: dict, off: dict) -> tuple:
    """(seed -> {target: [on, off]} over the (seed, target) pairs in both, the number of unpaired cells dropped)."""
    cells, dropped = {}, 0
    for seed in sorted(set(on) | set(off)):
        a, b = on.get(seed, {}), off.get(seed, {})
        dropped += len(set(a) ^ set(b))
        if set(a) & set(b):
            cells[seed] = {t: [a[t][0], b[t][0]] for t in sorted(set(a) & set(b))}
    return cells, dropped


def h8_for(runs: dict, ds: str, boot: Boot) -> dict:
    on, off = cells_of(runs, ds, "on", "drawn", "gain"), cells_of(runs, ds, "off", "drawn", "gain")
    b_cells, dropped = paired(on, off)
    components = dict(a=component(on, "h8/a", boot), b=component(b_cells, "h8/b", boot, threshold=THRESHOLD,
                                                                  dropped=dropped))
    return dict(components=components, **network_summary(components))


def secondary_for(runs: dict, ds: str, ref, boot: Boot) -> dict:
    out = {}
    for st in T.SELF_TERMS:
        for group in R.GROUPS:
            for field, support in (("gain", None), ("pct_before", PROPORTION), ("pct_after", PROPORTION)):
                out[f"{field}_{group}_{st}"] = summary(cells_of(runs, ds, st, group, field),
                                                       f"secondary/{field}_{group}_{st}", boot, support=support)[0]
        post = {}
        for (d, seed, s), run in sorted(runs.items()):
            if d == ds and s == st:
                drawn = [r for r in run.res["targets"] if r["group"] == "drawn"]
                post[seed] = M.post_stratified_mean([r["gain"] for r in drawn],
                                                    [M.stratum_of(r["n_pos"]) for r in drawn], ref.sizes)
        values = [v for v in post.values() if v is not None]
        out[f"gain_drawn_post_stratified_{st}"] = dict(estimate=float(np.mean(values)) if values else None,
                                                       by_seed=post, strata_sizes=ref.sizes)
        mcc = {seed: [run.res["base"]["test_metrics"]["t1"]["mcc"]] for (d, seed, s), run in sorted(runs.items())
               if d == ds and s == st and run.res["base"]["test_metrics"]["t1"]["mcc"] is not None}
        out[f"base_t1_mcc_{st}"] = summary(mcc, f"secondary/base_t1_mcc_{st}", boot, support=MCC_SUPPORT)[0]
    any_run = next((run for (d, _, _), run in sorted(runs.items()) if d == ds), None)
    if any_run is not None:                         # edgeless after U1 is a property of the graph: one census
        for group in R.GROUPS:
            flags = [bool(r["edgeless"]) for r in any_run.res["targets"] if r["group"] == group]
            out[f"edgeless_{group}"] = dict(n=len(flags), n_edgeless=int(sum(flags)),
                                            share=sum(flags) / len(flags) if flags else None, source="graph census")
    return out


# ------------------------------------------------------------------------ E8
def e8_run(run) -> tuple:
    """(record, failures) of one self-term-off run: cold_start.evaluate with the per-target post-U1 populations."""
    z, recs = run.z, run.res["targets"]
    pos = np.searchsorted(z["active_ids"], z["target_ids"])
    population = z["population"].astype(np.float64)
    target_proxy = population[np.arange(len(pos)), pos]           # the same array as each population row
    report = CS.evaluate(z["active_ids"], z["base_proxy"].astype(np.float64), z["n_in"], z["target_ids"],
                         target_proxy, z["target_n"], z["test_dst"], z["test_y"], z["test_p"].astype(np.float64),
                         CS.MS, target_population=population)
    if report["laundering_population"] != "per-target post-U1":
        raise AssertionError(f"{run.id}: E8 ranked on a {report['laundering_population']!r} population")
    failures = []
    if report["uncalibrated"]["laundering_pct"] != [r["pct_after"] for r in recs]:
        failures.append("E8's uncalibrated percentiles differ from the run's pct_after")
    return dict(target_ids=z["target_ids"].tolist(), groups=[r["group"] for r in recs], report=report), failures


def e8_for(runs: dict, ds: str, boot: Boot) -> tuple:
    per_seed, failures = {}, []
    for (d, seed, st), run in sorted(runs.items()):
        if d == ds and st == "off":
            per_seed[seed], problems = e8_run(run)
            failures += [dict(run=run.id, detail=p) for p in problems]
    out = dict(by_seed=per_seed, summary={})

    def pct_cells(group, pick):
        return {s: {t: [v] for t, g, v in zip(rec["target_ids"], rec["groups"], pick(rec["report"])) if g == group}
                for s, rec in per_seed.items()}

    def scalar_cells(pick):
        return {s: [pick(rec["report"])] for s, rec in per_seed.items() if pick(rec["report"]) is not None}

    for group in R.GROUPS:
        out["summary"][f"laundering_pct_{group}_uncalibrated"] = summary(
            pct_cells(group, lambda r: r["uncalibrated"]["laundering_pct"]), f"e8/pct_{group}_uncal", boot,
            support=PROPORTION)[0]
    for m in map(str, CS.MS):
        for group in R.GROUPS:
            out["summary"][f"laundering_pct_{group}_m{m}"] = summary(
                pct_cells(group, lambda r: r["by_m"][m]["laundering_pct"]), f"e8/pct_{group}_m{m}", boot,
                support=PROPORTION)[0]
        out["summary"][f"newcomer_change_m{m}"] = summary(
            scalar_cells(lambda r: r["by_m"][m]["newcomer"]["mean_change"]), f"e8/newcomer_m{m}", boot)[0]
        for key in ("base", "calibrated"):
            out["summary"][f"low_degree_mcc_{key}_m{m}"] = summary(
                scalar_cells(lambda r: r["by_m"][m]["low_degree"][key]["mcc"]), f"e8/low_degree_{key}_m{m}", boot,
                support=MCC_SUPPORT)[0]
        out["summary"][f"low_degree_delta_mcc_m{m}"] = summary(
            scalar_cells(lambda r: r["by_m"][m]["low_degree"]["delta_mcc"]), f"e8/low_degree_delta_m{m}", boot)[0]
    return out, failures


# ------------------------------------------------------------------------ the analysis
def analyse(results_root: str, prereg: str, networks=PS.NETWORKS, n_boot: int = 10_000) -> dict:
    """The whole analysis (no provenance block)."""
    results_root, networks = os.path.abspath(results_root), tuple(networks)
    prereg_sha = sha256_file(prereg) if os.path.isfile(prereg) else None
    failures = [] if prereg_sha else [dict(run=None, detail=f"protocol file {prereg} not found")]
    refs = {ds: reference(ds) for ds in networks}
    runs, missing, run_failures, inputs, commits = load_runs(results_root, networks, prereg_sha, refs)
    failures += run_failures
    h8, secondary, e8 = {}, {}, {}
    for ds in networks:
        boot = Boot(ds, n_boot)
        h8[ds] = h8_for(runs, ds, boot)
        secondary[ds] = secondary_for(runs, ds, refs[ds], boot)
        e8[ds], e8_failures = e8_for(runs, ds, boot)
        failures += e8_failures
    return dict(
        schema=SCHEMA, status="integrity_failure" if failures else "ok",
        protocol=dict(path=prereg, sha256=prereg_sha),
        settings=dict(networks=list(networks), networks_registered=list(PS.NETWORKS), seeds=list(PS.SEEDS_5),
                      self_terms=list(T.SELF_TERMS), epochs=PS.EPOCHS, n_boot=int(n_boot), boot_seed=BOOT_SEED,
                      threshold=THRESHOLD, min_seed_share=str(MIN_SEED_SHARE), ms=list(CS.MS),
                      groups=list(R.GROUPS)),
        integrity=dict(ok=not failures, complete=not missing, failures=failures, missing_runs=missing,
                       n_expected=len(expected_runs(results_root, networks)), n_present=len(runs),
                       git_commits=commits, inputs_sha256=inputs),
        hypotheses=dict(H8=dict(claim="The laundering gain is not an artifact of the missing self term",
                                networks=h8, **hypothesis_summary(h8))),
        secondary=secondary, e8=e8)


def render_markdown(result: dict) -> str:
    """The generated summary (every number from result)."""
    def f(x, spec=".4f"):
        return "n/a" if x is None else format(x, spec)

    def ci(c):
        return "n/a" if c.get("ci95") is None else f"[{f(c['ci95'][0])}, {f(c['ci95'][1])}]"

    h8 = result["hypotheses"]["H8"]
    lines = ["# E9a (H8) and E8: generated summary", "",
             f"Status: {result['status']}. Runs: {result['integrity']['n_present']} of "
             f"{result['integrity']['n_expected']}. Protocol sha256: {result['protocol']['sha256']}.", "",
             f"H8 hypothesis p-value (before Holm): {f(h8['p_value'], '.3g')}; {h8['status_before_holm']}.", "",
             "| network | (a) mean gain, on | 95% CI | p | decision | (b) ratio on/off | 95% CI | p | reading |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for ds, net in h8["networks"].items():
        a, b = net["components"]["a"], net["components"]["b"]
        lines.append(f"| {ds} | {f(a['estimate'])} | {ci(a)} | {f(a['p_value'], '.3g')} | {a['decision']} | "
                     f"{f(b['estimate'])} | {ci(b)} | {f(b['p_value'], '.3g')} | {b['decision']} |")
    lines += ["", "## Secondary (E9a)", "", "| network | quantity | estimate | 95% CI |", "| --- | --- | --- | --- |"]
    for ds, sec in result["secondary"].items():
        for key, c in sec.items():
            value = c.get("estimate", c.get("share"))
            lines.append(f"| {ds} | {key} | {f(value)} | {ci(c) if 'ci95' in c else c.get('source', '')} |")
    lines += ["", "## E8 (descriptive, self term off)", "", "| network | quantity | estimate | 95% CI |",
              "| --- | --- | --- | --- |"]
    for ds, e8 in result["e8"].items():
        for key, c in e8["summary"].items():
            lines.append(f"| {ds} | {key} | {f(c['estimate'])} | {ci(c)} |")
    integrity = result["integrity"]
    lines += ["", "## Integrity", ""] + [f"- missing: {m['run']} ({m['reason']})" for m in integrity["missing_runs"]]
    lines += [f"- failure: {x['run']}: {x['detail']}" for x in integrity["failures"]] or ["- no integrity failure"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ CLI
def provenance(result: dict, args) -> dict:
    import scipy
    return dict(**git_state(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                host=socket.gethostname(), argv=list(sys.argv), python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, results_root=os.path.abspath(args.results_root),
                code_sha256=C.CODE_SHA256_AT_IMPORT, inputs_sha256=result["integrity"]["inputs_sha256"])


def update_manifest(results_root: str) -> str:
    """Write the dump folder's manifest when it is absent or differs from the folder; returns what was done."""
    folder = os.path.join(os.path.abspath(results_root), "results", "dumps", "e9a")
    if not os.path.isdir(folder):
        return "no dump folder"
    try:
        diff, n = DM.check_manifest(folder)
        if not any(diff.values()):
            return f"manifest unchanged ({n} files)"
    except DM.ManifestError:
        pass
    path, n = DM.write_manifest(folder, force=True)
    return f"wrote {path} ({n} files)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="H8 (E9a) and E8 analyses of the E9a runs (CPU only).")
    ap.add_argument("--results-root", default=C.ROOT, help="the main repository, whose results/ holds the runs")
    ap.add_argument("--prereg", default=os.path.join(C.ROOT, PS.PREREG), help="the protocol file")
    ap.add_argument("--networks", nargs="+", default=list(PS.NETWORKS), choices=PS.NETWORKS,
                    help="networks to analyse (a subset leaves H8 not evaluable)")
    ap.add_argument("--n-boot", type=int, default=10_000, help="bootstrap replicates (the protocol's 10,000)")
    ap.add_argument("--out", default=os.path.join(ANALYSIS, "e9a_e8.json"))
    ap.add_argument("--md", default=os.path.join(ANALYSIS, "e9a_e8.md"))
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    a = ap.parse_args(argv)
    out_path = C.result_path(os.path.abspath(a.out), a.force)        # refuse before the work, not after it
    md_path = C.result_path(os.path.abspath(a.md), a.force)
    result = analyse(a.results_root, a.prereg, a.networks, a.n_boot)
    result["provenance"] = provenance(result, a)
    for path in (out_path, md_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    C.dump(result, out_path, force=a.force)
    text = render_markdown(json.loads(json.dumps(C._finite(result), default=C._json_default)))
    C.write_atomic(md_path, lambda f: f.write(text), replace=C.overwrite_allowed(a.force))
    print(update_manifest(a.results_root))
    for failure in result["integrity"]["failures"]:
        print(f"integrity failure: {failure['run']}: {failure['detail']}", file=sys.stderr)
    code = EXIT_INTEGRITY if result["integrity"]["failures"] else EXIT_OK
    print(f"wrote {out_path} and {md_path} (status {result['status']}, H8 p = "
          f"{result['hypotheses']['H8']['p_value']}, exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
