"""E2 analysis (Task 4.5): H6a and H6b from the E2 runs, the secondary outcomes and the integrity checks.

  python -m keeper.analysis.e2_analysis --results-root ABS_REPO_ROOT [--prereg docs/preregistration.md]
         [--n-boot 10000] [--force]

Reads results/scem/e2/e2_*.json under the results root (one file per network, seed, stream and model; written by
keeper.scem.e2_driver) and writes results/analysis/e2.json (keeper.laundering.core.dump: atomic, with provenance) and
results/analysis/e2.md. An existing output is replaced only with --force.

Endpoint (docs/preregistration.md, Sections 1, 2.4, 3 and 4). A target's attack gain under a model is (proxy on the
attack stream - proxy on the clean twin) x direction (+1 whitewash, -1 bad-mouth), from the two runs of the same
network, seed and model. A contrast pairs two models by seed and target:
  H6a  per network and attack kind, lag 1: gain(scem_nohaz) - gain(scem);
  H6b  per network and attack kind, lag 1: gain(s3r) - gain(scem).
A missing run drops its seed from the contrast; it is listed, never imputed.

Statistics. Seed and target are crossed, with missing cells (targets recur across seeds). Estimate: the mean of the
per-seed means. Headline 95% interval: the hull of the crossed bootstrap percentile interval (seeds and target ids
resampled independently; keeper.analysis.bootstrap.crossed_replicates, on the random stream of
keeper.analysis.intervals.Boot for the network and component) and the seed-t interval (df = seeds - 1). Component p:
max(one-sided seed-t p, (1 + #{replicates <= 0}) / (B + 1)), B the defined replicates. A component with results for
fewer than 80% of its registered seeds is not evaluable. A hypothesis is evaluable when all six components (three
networks, two kinds) are; its p is the largest component p (Holm across the eight hypotheses is the decisions step's,
Task 4.9); its interval rule is met when every component's interval lies above 0. A component interval wholly below 0
is reported as contradicted. The nested bootstrap interval is reported as a sensitivity.

Secondary (intervals, no test): per network, lag (1 and 3), kind and model, the mean gain, the mean percentile shift
and the contrast with scem; per network, model and stream, the learned phi (SCEM models) and the test MCC.

Integrity (exit 2, nothing written): every run's protocol sha256 equals --prereg's; its git state is clean; its eps,
edge cap and guard are the registered ones (keeper.cluster.protocol_settings); its SCEM settings are those of
keeper.scem.e2_driver.SCEM_MODELS; one frozen calibration and one clean-level digest per network; one stream plan per
network and seed; within a network, seed and model (the clean twin and its attack runs) the same model seed, epochs,
settings, targets, directions, panel and commit; no duplicate run. Runs outside the registered set are listed and
left out.
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from fractions import Fraction

import numpy as np
from scipy.stats import t as student_t

from keeper.analysis.bootstrap import (N_BOOT, crossed_replicates, hull, mean_stat, nested_bootstrap,
                                       per_seed_values, percentile_interval, seed_t_interval)
from keeper.analysis.intervals import BOOT_SEED, Boot
from keeper.analysis.provenance import git_state
from keeper.cluster import protocol_settings as ps
from keeper.cluster.e2_tasks import FOLDER, e2_runs, task_id
from keeper.cppr.protocol import preregistration
from keeper.laundering import core as C
from keeper.scem.e2_driver import STREAM_LAGS, check_settings
from keeper.scem.e2_streams import KINDS, N_TARGETS

MIN_SEED_SHARE = Fraction(4, 5)
REFERENCE = "scem"
HYPOTHESES = {"H6a": ("scem_nohaz", REFERENCE), "H6b": ("s3r", REFERENCE)}
PRIMARY_STREAM = "lag1"
ATTACK_STREAMS = ("lag1", "lag3")
NOT_EVALUABLE = "not evaluable"
DIRECTION = {"whitewash": 1, "badmouth": -1}


# ------------------------------------------------------------------------ runs
def load_records(folder: str) -> list:
    """Every e2_*.json in folder, with its file name under "_file"."""
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "e2_*.json"))):
        with open(path) as f:
            out.append(dict(json.load(f), _file=os.path.basename(path)))
    return out


def key_of(r: dict) -> tuple:
    return r["ds"], int(r["seed"]), r["stream"], r["model"]


def run_set(records: list) -> tuple:
    """(index of the registered runs by key, missing task ids, unexpected task ids, duplicate task ids)."""
    expected = set(e2_runs())
    seen = defaultdict(list)
    for r in records:
        seen[key_of(r)].append(r)
    index = {k: rs[0] for k, rs in seen.items() if k in expected}
    return (index, [task_id(*k) for k in e2_runs() if k not in seen],
            sorted(task_id(*k) for k in seen if k not in expected),
            sorted(task_id(*k) for k, rs in seen.items() if len(rs) > 1))


# ------------------------------------------------------------------------ integrity
def _record_failures(r: dict, prereg_sha: str) -> list:
    ds, model, name = r["ds"], r["model"], r["_file"]
    fails = []
    prov, plan, mb = r.get("provenance") or {}, r["plan"], r.get("mb")
    if (r.get("prereg") or {}).get("sha256") != prereg_sha:
        fails.append(f"{name}: protocol sha256 {(r.get('prereg') or {}).get('sha256')} is not {prereg_sha}")
    if prov.get("git_dirty") is not False or not prov.get("git_commit"):
        fails.append(f"{name}: git state not clean (commit {prov.get('git_commit')}, dirty {prov.get('git_dirty')})")
    if r["epochs"] != ps.EPOCHS or r["model_seed"] != r["seed"]:
        fails.append(f"{name}: epochs {r['epochs']} or model seed {r['model_seed']} is not registered")
    registered = dict(eps=ps.EPS_CAPPED[ds], edge_cap_units=ps.EDGE_CAP_UNITS, neg_from=ps.NEG_FROM)
    for block, label in ((plan, "plan"), (mb, "M-B")):
        if block is not None and any(block.get(k) != v for k, v in registered.items()):
            fails.append(f"{name}: {label} has {[block.get(k) for k in registered]}, registered {list(registered.values())}")
    if (model == "s3r") != (mb is None) or (model == "s3r") != (r.get("settings") is None):
        fails.append(f"{name}: M-B and SCEM settings must be present exactly for the SCEM models")
    elif model != "s3r":
        try:
            check_settings(model, r["settings"])
        except RuntimeError as err:
            fails.append(f"{name}: {err}")
    direction = [DIRECTION.get(k) for k in r["kind"]]
    if len(r["targets"]) != 2 * N_TARGETS or direction != r["direction"] or len(r["proxy"]) != len(r["targets"]):
        fails.append(f"{name}: targets, kinds, directions or proxies are inconsistent")
    return fails


def _same(groups: dict, fn, label: str) -> list:
    """A failure per group whose records disagree on fn(record)."""
    out = []
    for key, rs in sorted(groups.items()):
        values = {json.dumps(fn(r), sort_keys=True) for r in rs}
        if len(values) > 1:
            out.append(f"{'/'.join(map(str, key))}: {label} differs across {sorted(r['_file'] for r in rs)}")
    return out


def integrity(records: list, prereg_sha: str, duplicates: list) -> list:
    """The failures of the integrity checks (an empty list when every check passes)."""
    fails = [f"duplicate run {d}" for d in duplicates]
    for r in records:
        try:
            fails += _record_failures(r, prereg_sha)
        except (KeyError, TypeError) as err:
            fails.append(f"{r.get('_file')}: malformed record ({err!r})")
    if fails:
        return fails
    by_ds, by_seed, by_model = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        by_ds[(r["ds"],)].append(r)
        by_seed[(r["ds"], r["seed"])].append(r)
        by_model[(r["ds"], r["seed"], r["model"])].append(r)
    scem_by_ds = {k: [r for r in rs if r["mb"] is not None] for k, rs in by_ds.items()}
    fails += _same(scem_by_ds, lambda r: r["mb"]["calibration"], "the frozen M-B calibration")
    fails += _same(by_ds, lambda r: r["plan"]["clean_levels_sha256"], "the clean stream's levels")
    fails += _same(by_seed, lambda r: r["plan"]["sha256"], "the stream plan")
    twin = ("model_seed", "epochs", "settings", "targets", "direction", "kind")
    fails += _same(by_model, lambda r: [r[k] for k in twin], "the clean-twin settings (" + ", ".join(twin) + ")")
    fails += _same(by_model, lambda r: r["panel"]["sources_sha256"], "the panel")
    fails += _same(by_model, lambda r: r["provenance"].get("git_commit"), "the code commit")
    for r in records:
        if r["stream"] == "clean" and r["mb"] is not None and r["mb"]["levels_sha256"] != r["plan"]["clean_levels_sha256"]:
            fails.append(f"{r['_file']}: the clean twin's M-B levels are not the clean stream's")
    return fails


# ------------------------------------------------------------------------ cells and statistics
def run_gains(index: dict, ds: str, seed: int, stream: str, model: str, kind: str, field: str = "proxy"):
    """{target: (field under attack - field on the clean twin) x direction} of one kind, or None if a run is missing."""
    attack, clean = index.get((ds, seed, stream, model)), index.get((ds, seed, "clean", model))
    if attack is None or clean is None:
        return None
    rows = zip(attack["targets"], attack["kind"], attack["direction"], attack[field], clean[field])
    return {int(t): float(d * (a - c)) for t, k, d, a, c in rows if k == kind}


def cells(index: dict, ds: str, stream: str, kind: str, model: str, seeds, reference: str | None = None,
          field: str = "proxy") -> dict:
    """seed -> {target: gain(model)} or, with reference, {target: gain(model) - gain(reference)} (pairs by seed and
    target); a seed missing a run is left out."""
    out = {}
    for seed in seeds:
        g = run_gains(index, ds, seed, stream, model, kind, field)
        r = None if reference is None else run_gains(index, ds, seed, stream, reference, kind, field)
        if g is None or (reference is not None and r is None):
            continue
        g = g if r is None else {t: g[t] - r[t] for t in g if t in r}
        if g:
            out[seed] = g
    return out


def p_seed_t(values, null: float = 0.0):
    """One-sided seed-level t p-value of mean(values) > null (df = n - 1); None with fewer than two values."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return None
    excess, sd = v.mean() - null, v.std(ddof=1)
    if sd == 0:
        return 0.0 if excess > 0 else 1.0
    return float(student_t.sf(excess / (sd / math.sqrt(len(v))), len(v) - 1))


def p_bootstrap(replicates, threshold: float = 0.0) -> float:
    """(1 + #{replicates <= threshold}) / (B + 1); B counts the defined replicates."""
    r = np.asarray(replicates, dtype=float)
    r = r[np.isfinite(r)]
    return (1 + int(np.sum(r <= threshold))) / (len(r) + 1)


def decision(ci, evaluable: bool) -> dict:
    """Rule "interval > 0": met when the lower bound is above 0, contradicted when the upper bound is below 0."""
    if not evaluable or ci is None:
        return dict(decision=NOT_EVALUABLE, met=False, contradicted=False)
    met, contradicted = ci[0] > 0, ci[1] < 0
    return dict(decision="met" if met else "contradicted" if contradicted else "not met", met=bool(met),
                contradicted=bool(contradicted))


def _untested(out: dict, confirmatory: bool, reason: str) -> dict:
    out.update(evaluable=False, not_evaluable_reason=reason, ci95=None)
    if confirmatory:
        out.update(p_value=None, p_seed_t=None, p_bootstrap=None, **decision(None, False))
    return out


def component(cell_map: dict, name: str, boot: Boot, expected: int, confirmatory: bool = True) -> dict:
    """Estimate, intervals and (confirmatory) p-values and decision of a mean over targets; expected is the number of
    registered seeds and name keys boot's random streams."""
    seeds = sorted(cell_map)
    per_seed = per_seed_values(cell_map, mean_stat) if cell_map else []
    out = dict(n_seeds=len(seeds), n_seeds_expected=int(expected), seeds=seeds,
               n_targets=len(set().union(*(set(c) for c in cell_map.values()))) if cell_map else 0,
               n_pairs=sum(len(c) for c in cell_map.values()),
               evaluable=bool(expected) and Fraction(len(seeds), int(expected)) >= MIN_SEED_SHARE,
               estimate=float(np.mean(per_seed)) if per_seed else None, by_seed=dict(zip(seeds, per_seed)))
    if len(per_seed) < 2:
        return _untested(out, confirmatory, "fewer than two seeds")
    seed_t = seed_t_interval(per_seed)
    try:
        _, replicates, _ = crossed_replicates(cell_map, n_boot=boot.n_boot, seed=boot.seed_for(name, "crossed"))
        boot_ci = percentile_interval(replicates)
        _, nested_lo, nested_hi = nested_bootstrap(cell_map, n_boot=boot.n_boot, seed=boot.seed_for(name, "nested"))
    except FloatingPointError as err:
        return _untested(out, confirmatory, f"bootstrap refused: {err}")
    ci = hull(boot_ci, seed_t)
    out.update(ci95=ci, ci95_bootstrap=boot_ci, ci95_seed_t=seed_t, ci95_nested=[nested_lo, nested_hi],
               ci95_source="hull of the crossed bootstrap and seed-t intervals",
               bootstrap_streams={d: boot.seed_for(name, d) for d in ("crossed", "nested")})
    if confirmatory:
        p_t, p_b = p_seed_t(per_seed), p_bootstrap(replicates)
        out.update(p_seed_t=p_t, p_bootstrap=p_b, p_value=max(p_t, p_b), **decision(ci, out["evaluable"]))
    return out


def summary(components: dict) -> dict:
    """A hypothesis over its components (network -> kind -> component): evaluable when all are; p the largest."""
    flat = {f"{ds}/{kind}": c for ds, by_kind in components.items() for kind, c in by_kind.items()}
    evaluable = all(c["evaluable"] and c.get("p_value") is not None for c in flat.values())
    met = evaluable and all(c["met"] for c in flat.values())
    return dict(evaluable=evaluable, interval_rule_met=met,
                p_value=max(c["p_value"] for c in flat.values()) if evaluable else None,
                contradicted=sorted(k for k, c in flat.items() if c.get("contradicted")),
                not_evaluable=sorted(k for k, c in flat.items() if not c["evaluable"]),
                status_before_holm=(NOT_EVALUABLE if not evaluable else
                                    "interval rule met on every network and kind" if met else "interval rule not met"))


# ------------------------------------------------------------------------ the analysis
def registered_seeds(ds: str, stream: str, *models) -> list:
    """The seeds with a registered run of every given model on the stream (and so its clean twin)."""
    runs = set(e2_runs())
    return sorted(s for s in ps.SEEDS_E2_H6 if all((ds, s, stream, m) in runs for m in models))


def hypotheses(index: dict, n_boot: int) -> dict:
    out = {}
    for h, (model, reference) in HYPOTHESES.items():
        comps = {}
        for ds in ps.NETWORKS:
            boot, seeds = Boot(ds, n_boot), registered_seeds(ds, PRIMARY_STREAM, model, reference)
            comps[ds] = {kind: component(cells(index, ds, PRIMARY_STREAM, kind, model, seeds, reference),
                                         f"{h}/{kind}", boot, len(seeds)) for kind in KINDS}
        out[h] = dict(contrast=f"gain({model}) - gain({reference})", stream=PRIMARY_STREAM, rule="95% interval > 0",
                      **summary(comps), components=comps)
    return out


def secondary(index: dict, n_boot: int) -> dict:
    """network -> stream -> kind -> model -> {gain, percentile_shift[, minus_scem]} (intervals, no test). The lag-1
    contrasts of the H6 models with scem are the hypotheses' components and are not repeated here."""
    primary = {(PRIMARY_STREAM, model) for model, _ in HYPOTHESES.values()}
    out = {}
    for ds in ps.NETWORKS:
        boot = Boot(ds, n_boot)
        out[ds] = {}
        for stream in ATTACK_STREAMS:
            out[ds][stream] = {kind: {} for kind in KINDS}
            for kind in KINDS:
                for model in ps.E2_MODELS:
                    seeds = registered_seeds(ds, stream, model)
                    if not seeds:
                        continue
                    name = f"{stream}/{kind}/{model}"
                    block = dict(
                        gain=component(cells(index, ds, stream, kind, model, seeds), f"{name}/gain", boot,
                                       len(seeds), confirmatory=False),
                        percentile_shift=component(cells(index, ds, stream, kind, model, seeds, field="percentile"),
                                                   f"{name}/percentile", boot, len(seeds), confirmatory=False))
                    both = registered_seeds(ds, stream, model, REFERENCE)
                    if model != REFERENCE and both and (stream, model) not in primary:
                        block[f"minus_{REFERENCE}"] = component(
                            cells(index, ds, stream, kind, model, both, REFERENCE), f"{name}/minus_{REFERENCE}", boot,
                            len(both), confirmatory=False)
                    out[ds][stream][kind][model] = block
    return out


def seed_summary(values: dict) -> dict:
    """Per-seed values (None dropped), their mean, range and seed-t interval."""
    v = {s: float(x) for s, x in sorted(values.items()) if x is not None}
    xs = list(v.values())
    return dict(by_seed=v, n_seeds=len(xs), mean=float(np.mean(xs)) if xs else None,
                min=min(xs) if xs else None, max=max(xs) if xs else None, ci95_seed_t=seed_t_interval(xs))


def per_run(index: dict, field: str, models) -> dict:
    """network -> model -> stream -> seed_summary of a per-run field (the learned phi, the test MCC)."""
    out = {}
    for ds in ps.NETWORKS:
        out[ds] = {m: {st: seed_summary({s: index[(ds, s, st, m)][field] for s in ps.SEEDS_E2_H6
                                          if (ds, s, st, m) in index})
                       for st in ps.E2_STREAMS} for m in models}
    return out


def analyse(records: list, prereg: dict, n_boot: int = N_BOOT) -> dict:
    """The analysis of the runs (after integrity passed)."""
    index, missing, unexpected, _ = run_set(records)
    scem_models = [m for m in ps.E2_MODELS if m != "s3r"]
    return dict(
        experiment="E2", hypotheses_tested=list(HYPOTHESES), prereg=prereg, n_boot=int(n_boot), boot_seed=BOOT_SEED,
        runs=dict(expected=len(e2_runs()), used=len(index), missing=missing, unexpected=unexpected,
                  commits=sorted({r["provenance"]["git_commit"] for r in index.values()}),
                  honest_levels_changed=sum(r["mb"].get("honest_levels_changed", 0) for r in index.values()
                                            if r["mb"] is not None)),
        hypotheses=hypotheses(index, n_boot), secondary=secondary(index, n_boot),
        learned_phi=per_run(index, "learned_phi", scem_models), test_mcc=per_run(index, "test_mcc", ps.E2_MODELS),
        stream_lags=STREAM_LAGS, analysis_git=git_state())


# ------------------------------------------------------------------------ report
def _f(x, spec: str = ".4f") -> str:
    return "n/a" if x is None else format(x, spec)


def _ci(ci) -> str:
    return "n/a" if ci is None else f"[{_f(ci[0])}, {_f(ci[1])}]"


def render(result: dict) -> str:
    """e2.md, generated from the result."""
    runs = result["runs"]
    lines = ["# E2: SCEM end to end (H6a, H6b)", "",
             f"Generated by keeper.analysis.e2_analysis from {runs['used']} runs ({len(runs['missing'])} of "
             f"{runs['expected']} registered runs missing). Protocol sha256 `{result['prereg']['sha256']}`; "
             f"{result['n_boot']} bootstrap replicates; code commits {', '.join(runs['commits']) or 'none'}.", ""]
    for h, block in result["hypotheses"].items():
        lines += [f"## {h}: {block['contrast']}, {block['stream']}", "",
                  f"{block['status_before_holm']}; hypothesis p (largest component p, before Holm) "
                  f"{_f(block['p_value'], '.3g')}.", "",
                  "| network | kind | seeds | pairs | estimate | 95% CI (hull) | nested bootstrap | p | decision |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for ds, by_kind in block["components"].items():
            for kind, c in by_kind.items():
                lines.append(f"| {ds} | {kind} | {c['n_seeds']}/{c['n_seeds_expected']} | {c['n_pairs']} | "
                             f"{_f(c['estimate'])} | {_ci(c['ci95'])} | {_ci(c.get('ci95_nested'))} | "
                             f"{_f(c.get('p_value'), '.3g')} | {c['decision']} |")
        lines.append("")
    lines += ["## Secondary: mean attack gain and its contrast with scem", "",
              "The lag-1 contrasts of s3r and scem_nohaz with scem are H6b and H6a (above).", "",
              "| network | stream | kind | model | seeds | gain | 95% CI | percentile shift | minus scem | 95% CI |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for ds, by_stream in result["secondary"].items():
        for stream, by_kind in by_stream.items():
            for kind, by_model in by_kind.items():
                for model, b in by_model.items():
                    g, p, d = b["gain"], b["percentile_shift"], b.get(f"minus_{REFERENCE}") or {}
                    lines.append(f"| {ds} | {stream} | {kind} | {model} | {g['n_seeds']}/{g['n_seeds_expected']} | "
                                 f"{_f(g['estimate'])} | {_ci(g['ci95'])} | {_f(p['estimate'])} | "
                                 f"{_f(d.get('estimate'))} | {_ci(d.get('ci95'))} |")
    for field, title in (("learned_phi", "Learned phi"), ("test_mcc", "Test MCC (Task 1)")):
        lines += ["", f"## {title}", "", "| network | model | stream | seeds | mean | min | max | seed-t 95% CI |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for ds, by_model in result[field].items():
            for model, by_stream in by_model.items():
                for stream, s in by_stream.items():
                    if s["n_seeds"]:
                        lines.append(f"| {ds} | {model} | {stream} | {s['n_seeds']} | {_f(s['mean'])} | "
                                     f"{_f(s['min'])} | {_f(s['max'])} | {_ci(s['ci95_seed_t'])} |")
    lines += ["", "## Runs", "", f"Missing ({len(runs['missing'])}): {', '.join(runs['missing']) or 'none'}.", "",
              f"Unexpected, left out ({len(runs['unexpected'])}): {', '.join(runs['unexpected']) or 'none'}.", "",
              f"Honest account-snapshot tiers that differ from the clean stream's, summed over the SCEM attack runs: "
              f"{runs['honest_levels_changed']}.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keeper.analysis.e2_analysis", description="E2 analysis (H6a, H6b).")
    ap.add_argument("--results-root", default=C.ROOT, help="the repository whose results/ holds the runs")
    ap.add_argument("--prereg", default=ps.PREREG, help="the protocol file whose sha256 every run must carry")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--force", action="store_true", help="replace existing outputs")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.results_root)
    out_dir = os.path.join(root, "results", "analysis")
    try:
        json_path = C.result_path(os.path.join(out_dir, "e2.json"), a.force)
        md_path = C.result_path(os.path.join(out_dir, "e2.md"), a.force)
        prereg = preregistration(a.prereg)
    except (FileExistsError, FileNotFoundError) as err:
        ap.error(str(err))
    records = load_records(os.path.join(root, *FOLDER))
    index, _, _, duplicates = run_set(records)
    failures = integrity(list(index.values()), prereg["sha256"], duplicates)
    if failures:
        print("e2_analysis: integrity checks failed; nothing written:", *failures, sep="\n  ", file=sys.stderr)
        return 2
    result = analyse(records, prereg, a.n_boot)
    os.makedirs(out_dir, exist_ok=True)
    C.dump(result, json_path, force=a.force)
    C.write_atomic(md_path, lambda f: f.write(render(result)), replace=C.overwrite_allowed(a.force))
    h = result["hypotheses"]
    print(f"wrote {json_path} and {md_path}: " + "; ".join(f"{k} {v['status_before_holm']} (p {_f(v['p_value'], '.3g')})"
                                                          for k, v in h.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
