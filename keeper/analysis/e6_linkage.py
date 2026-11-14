"""E6: linkage of CPPR's pooled verdicts against hand-crafted, single-signal and learned insiders (Task 4.6; H7; protocol
docs/preregistration.md Sections 1, 2.8, 3 and 4). CPU only:

  CUDA_VISIBLE_DEVICES="" python -m keeper.analysis.e6_linkage --results-root ABS_REPO_ROOT [--out E6.json --md E6.md]
      [--force]

Inputs, read-only: the per-verdict dumps (keeper.cppr.dumps.VERDICT_COLUMNS) of the E3 runs that write them (capped BTP
and BFS, seeds 5-14: results/dumps/e3/<stem>_verdicts.json) and of the E5 g = 0 runs (seeds 5-9:
results/dumps/e5/<id>_verdicts.json), each with its result file. Families: cppr_btp (method cppr of the capped-BTP E3
runs; H7), cppr_bfs (cppr of the BFS E3 runs) and pseudonym (method singleton of E5, the per-rating pseudonym), always
on the direct arm. The attackers and the G3 metric are keeper.analysis.linkage_attackers'.

Integrity. An E3 run must pass keeper.analysis.cppr_runs' checks (registered settings, protocol sha256, clean commit,
target dump); an E5 run the same registered settings with E5's arms and methods (its target dump, unused here, is not
read). Its verdict dump must belong to it
(result name, config, row count, recorded path), have VERDICT_COLUMNS, consistent pool rows, and the sha256 of the
bytes parsed must equal its entry in the dump folder's manifest (keeper.analysis.dump_manifest, MANIFEST_SHA256.txt). A
run that fails is excluded with every failure listed; any failure, or a missing protocol file, exits 2. A missing run
(result or dump) is listed, never imputed; a family with results for fewer than 80% of its registered seeds is not
evaluable.

H7, per network: per seed the maximum over ATTACKERS of the seed's G3 on cppr_btp (defined when every attacker is);
estimate the mean over seeds; one-sided p of mean < 0.60 and the headline 95% interval from the Nadeau-Bengio corrected t
(factor 1 + J n_test / n_train with n_test / n_train = 1 / (J - 1) for leave one seed out, J the seeds); supported on a
network when the upper bound is below 0.60, contradicted when the lower bound is above it. Hypothesis p = max over
networks, for the decisions step's Holm correction. Sensitivities: single-member pools scored as AUC 1; the nested
bootstrap of the endpoint (seeds, then pooled verdicts within seed). Secondary, per family and attacker: G3 with the
corrected-t interval, tie-split top-1 against chance (mean 1 / pool size), the learned attacker's permutation importance
and the direction of the timing signal.

Outputs: e6.json (keeper.laundering.core.dump: provenance, R27 overwrite guard, atomic) and the generated e6.md; an
existing output is refused without --force before any work.
"""
import argparse
import datetime
import json
import os
import platform
import resource
import socket
import sys
import time
import zlib
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from keeper.analysis import cppr_runs as R
from keeper.analysis import dump_manifest as DM
from keeper.analysis import linkage_attackers as LA
from keeper.analysis.bootstrap import N_BOOT, percentile_interval
from keeper.analysis.intervals import BOOT_SEED
from keeper.analysis.provenance import git_state
from keeper.cluster import protocol_settings as PS
from keeper.cppr import dumps as DP
from keeper.laundering import core as C

SCHEMA = "keeper.analysis.e6_linkage/1"
NULL_AUC = 0.60
MIN_SEED_SHARE = Fraction(4, 5)
NOT_EVALUABLE = "not evaluable"
EXIT_OK, EXIT_INTEGRITY = 0, 2
ANALYSIS = os.path.join(C.ROOT, "results", "analysis")
ARM = "direct"
MODULE_FILES = ("keeper/analysis/e6_linkage.py", "keeper/analysis/linkage_attackers.py",
                "keeper/analysis/cppr_runs.py", "keeper/analysis/dump_manifest.py", "keeper/analysis/bootstrap.py",
                "keeper/cluster/protocol_settings.py", "keeper/cppr/dumps.py", "keeper/cppr/linkage.py")


@dataclass(frozen=True)
class Family:
    """A set of runs whose verdicts are scored: source experiment, tier setting, the dump's method and the result's
    recorded linkage block (keeper.cppr.linkage.linkage_attack)."""
    name: str
    source: str
    tier: str
    method: str
    recorded: str
    seeds: tuple


FAMILIES = (Family("cppr_btp", "e3", PS.CAPPED, "cppr", "linkage_cp_vpe", tuple(PS.SEEDS_E3)),
            Family("cppr_bfs", "e3", PS.BFS, "cppr", "linkage_cp_vpe", tuple(PS.SEEDS_E3)),
            Family("pseudonym", "e5", PS.CAPPED, "singleton", "linkage_singleton", tuple(PS.SEEDS_5)))
H7_FAMILY = "cppr_btp"


# ------------------------------------------------------------------------ loading and integrity
def e5_stem(ds: str, seed: int) -> str:
    return f"e5_{PS.CAPPED}_g0_{ds}_s{seed}"


def e5_settings(ds: str, seed: int, prereg_sha256) -> dict:
    """An E5 g = 0 run's registered settings: E3's capped-BTP run of the seed with E5's arms and methods."""
    want = R.registered_settings(R.RunKey("e3", PS.CAPPED, ds, seed), prereg_sha256)
    want.update({"config.arms": PS.E5_ARMS_G0.split(","), "config.methods": PS.E5_METHODS.split(",")})
    return want


def _settings_failures(result: dict, want: dict) -> list:
    out = [dict(field=path, detail=f"{path} is {R.lookup(result, path)!r}, not the registered {value!r}")
           for path, value in want.items() if not R._same(R.lookup(result, path), value)]
    commit = R.lookup(result, "provenance.git_commit")
    if not (isinstance(commit, str) and commit):
        out.append(dict(field="provenance.git_commit", detail=f"provenance.git_commit is {commit!r}, not a commit"))
    return out


class DumpManifests:
    """The dump folders' manifests (path -> sha256), read once each; a missing or malformed one is a failure."""

    def __init__(self, paths: dict):
        self.paths, self.entries, self.errors = dict(paths), {}, {}
        for source, path in self.paths.items():
            try:
                self.entries[source] = DM.read_manifest(path)
            except DM.ManifestError as error:
                self.errors[source] = str(error)

    def failure(self, source: str, name: str, digest: str) -> dict | None:
        if source in self.errors:
            return dict(field="verdicts.manifest", detail=self.errors[source])
        want = self.entries[source].get(name)
        if want is None:
            return dict(field="verdicts.manifest", detail=f"{name} is not listed in {self.paths[source]}")
        if want != digest:
            return dict(field="verdicts.sha256", detail=f"{name} sha256 {digest} differs from the manifest's {want}")
        return None


def run_source(family: Family, ds: str, seed: int, root: str) -> SimpleNamespace:
    """Stem, result file and verdict dump of one expected run of ``family``."""
    if family.source == "e3":
        key = R.RunKey("e3", family.tier, ds, seed)
        return SimpleNamespace(key=key, stem=key.stem, result_file=key.result_file(root),
                               verdicts=os.path.join(root, "results", "dumps", "e3", f"{key.stem}_verdicts.json"))
    stem = e5_stem(ds, seed)
    return SimpleNamespace(key=None, stem=stem, result_file=os.path.join(root, "results", "cppr", "e5", f"{stem}.json"),
                           verdicts=os.path.join(root, "results", "dumps", "e5", f"{stem}_verdicts.json"))


def load_one(family: Family, ds: str, seed: int, root: str, manifest: R.Manifest, dumps: DumpManifests,
             prereg_sha256) -> tuple:
    """(SimpleNamespace(table, recorded, seconds) or None, missing reason or None, failures). The verdict dump is
    parsed only when the run passes its checks; a run with a failure must not be used."""
    src = run_source(family, ds, seed, root)
    if not os.path.exists(src.result_file):
        return None, "result missing", []
    if not os.path.exists(src.verdicts):
        return None, "verdict dump missing", []
    t0 = time.time()
    if src.key is not None:
        run, reason, failures = R.load_run(src.key, root, manifest, prereg_sha256)
        if reason is not None:
            return None, reason, []
        result = None if run is None else run.result
    else:
        try:
            result = manifest.json(src.result_file)
        except (OSError, ValueError) as error:
            return None, None, [dict(field="file", detail=f"cannot read the result: {error}")]
        failures = _settings_failures(result, e5_settings(ds, seed, prereg_sha256)) if isinstance(result, dict) \
            else [dict(field="file", detail="the result is not a JSON object")]
    if failures:
        return None, None, failures
    data = manifest.read(src.verdicts)
    bad = dumps.failure(family.source, os.path.basename(src.verdicts), manifest.sha256[manifest.key(src.verdicts)])
    if bad is not None:
        return None, None, [bad]
    try:
        dump = json.loads(data)
    except ValueError as error:
        return None, None, [dict(field="verdicts.file", detail=f"cannot parse the verdict dump: {error}")]
    del data
    failures = [dict(f, field="verdicts:" + f["field"])
                for f in R._dump_failures(SimpleNamespace(stem=src.stem), result, dump, kind="verdicts")]
    if dump.get("columns") != list(DP.VERDICT_COLUMNS):
        failures.append(dict(field="verdicts.columns", detail=f"columns {dump.get('columns')} are not VERDICT_COLUMNS"))
    if failures:
        return None, None, failures
    table, problems = LA.table_from_rows(dump["columns"], dump["rows"], ARM, family.method)
    failures = [dict(field="verdicts.rows", detail=p) for p in problems]
    if not len(table.verdict) and not table.n_single_verdicts:
        failures.append(dict(field="verdicts.rows", detail=f"no {ARM}/{family.method} verdict rows"))
    recorded = ((result.get("direct_laundering") or {}).get(family.recorded))
    return SimpleNamespace(table=table, recorded=recorded, seconds=time.time() - t0), None, failures


def load_family(family: Family, ds: str, root: str, manifest: R.Manifest, dumps: DumpManifests,
                prereg_sha256) -> tuple:
    """(seed -> loaded run, missing, failures, seconds per run) of one family on one network."""
    used, missing, failures, seconds = {}, [], [], {}
    for seed in family.seeds:
        stem = run_source(family, ds, seed, root).stem
        got, reason, problems = load_one(family, ds, seed, root, manifest, dumps, prereg_sha256)
        if reason is not None:
            missing.append(dict(run=stem, family=family.name, reason=reason))
        failures += [dict(run=stem, **p) for p in problems]
        if got is not None and not problems:
            used[seed] = got
            seconds[stem] = round(got.seconds, 2)
    return used, missing, failures, seconds


# ------------------------------------------------------------------------ statistics of one family
def _mean(x) -> float | None:
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else None


def _decision(ci, evaluable: bool) -> str:
    if not evaluable or ci is None:
        return NOT_EVALUABLE
    return "supported" if ci[1] < NULL_AUC else "contradicted" if ci[0] > NULL_AUC else "not supported"


def _evaluable(n_seeds: int, family: Family) -> bool:
    return Fraction(n_seeds, len(family.seeds)) >= MIN_SEED_SHARE


def endpoint_block(per_seed: dict, family: Family, extra: dict | None = None) -> dict:
    """The corrected-t reading of seed -> value against NULL_AUC."""
    seeds = sorted(s for s, v in per_seed.items() if v is not None)
    test = LA.corrected_t([per_seed[s] for s in seeds], NULL_AUC)
    evaluable = _evaluable(len(seeds), family)
    return dict(by_seed={str(s): per_seed[s] for s in sorted(per_seed)}, seeds=seeds, n_seeds=len(seeds),
                n_seeds_expected=len(family.seeds), evaluable=evaluable, estimate=test["mean"], ci95=test["ci95"],
                p_value=test["p_below"] if evaluable else None, corrected_t=test,
                decision=_decision(test["ci95"], evaluable), **(extra or {}))


def analyse_family(family: Family, runs: dict, n_boot: int, ds: str) -> dict:
    """Every attacker's per-seed G3, top-1 and chance, the max over attackers (with its sensitivities), the learned
    attacker's importances and the direction of each signal, for one family on one network."""
    tables = {s: r.table for s, r in sorted(runs.items())}
    scores, details = LA.attacker_scores(tables)
    items = {s: {a: LA.item_metrics(tables[s], sc) for a, sc in by.items() if sc is not None}
             for s, by in scores.items()}
    attackers = {}
    for a in LA.ATTACKERS:
        g = {s: (float(items[s][a]["auc"].mean()) if a in items[s] and len(items[s][a]["auc"]) else None)
             for s in tables}
        top1 = {s: (float(items[s][a]["top1"].mean()) if a in items[s] and len(items[s][a]["top1"]) else None)
                for s in tables}
        attackers[a] = endpoint_block(g, family, dict(top1_by_seed={str(s): v for s, v in top1.items()},
                                                      top1=_mean(top1.values())))
    complete = {s for s in tables if all(a in items[s] and len(items[s][a]["auc"]) for a in LA.ATTACKERS)}
    best = {s: (max(attackers[a]["by_seed"][str(s)] for a in LA.ATTACKERS) if s in complete else None) for s in tables}
    argmax = {str(s): max(LA.ATTACKERS, key=lambda a: attackers[a]["by_seed"][str(s)]) for s in sorted(complete)}
    singles_as_one = {}
    for s in tables:
        t, n = tables[s], int(tables[s].label.sum())
        singles_as_one[s] = (None if s not in complete else
                             max((attackers[a]["by_seed"][str(s)] * n + t.n_single_items) / (n + t.n_single_items)
                                 for a in LA.ATTACKERS))
    sums = {s: LA.verdict_sums(items[s], LA.ATTACKERS) for s in sorted(complete)}
    boot = None
    if len(sums) >= 2:
        reps = LA.nested_bootstrap_max(sums, n_boot, [BOOT_SEED, zlib.crc32(f"{ds}/{family.name}".encode())])
        boot = dict(ci95=percentile_interval(reps), n_boot=int(n_boot),
                    p_value=float((1 + np.sum(reps >= NULL_AUC)) / (len(reps) + 1)))
    max_block = endpoint_block(best, family, dict(argmax_by_seed=argmax))
    max_block["sensitivity_single_member_pools_as_auc_1"] = endpoint_block(singles_as_one, family)
    max_block["sensitivity_nested_bootstrap"] = boot
    n_items = {s: int(t.label.sum()) for s, t in tables.items()}
    n_single = {s: int(t.n_single_items) for s, t in tables.items()}
    chance = {s: (float(np.mean(1.0 / np.bincount(t.verdict)[t.verdict[t.label]])) if n_items[s] else None)
              for s, t in tables.items()}
    return dict(
        seeds=sorted(tables), n_seeds_expected=len(family.seeds), evaluable=_evaluable(len(tables), family),
        n_pooled_verdicts={str(s): t.n_verdicts + t.n_single_verdicts for s, t in tables.items()},
        n_items={str(s): n_items[s] for s in tables}, n_single_member_items={str(s): n_single[s] for s in tables},
        single_member_item_share=(sum(n_single.values()) / max(1, sum(n_single.values()) + sum(n_items.values()))),
        n_candidate_rows={str(s): int(len(t.verdict)) for s, t in tables.items()},
        chance_top1_by_seed={str(s): v for s, v in chance.items()}, chance_top1=_mean(chance.values()),
        attackers=attackers, max_over_attackers=max_block,
        learned=learned_block(details["learned"]), signals=signal_block(tables, details),
        handcrafted_check=handcrafted_check(runs, attackers["handcrafted"], details, n_items))


def learned_block(learned: dict) -> dict:
    """Mean permutation importance over folds (drop of the held-out G3), and the timing partial dependence: a LOWER
    timing share marks the true rater when the mean predicted probability falls from the 10% to the 90% quantile."""
    folds = {s: d for s, d in learned.items() if d is not None}
    if not folds:
        return dict(n_folds=0)
    imp = {f: float(np.mean([d["permutation_importance"][f] for d in folds.values()])) for f in LA.FEATURES}
    pd_by_fold = {str(s): d["timing_pd"] for s, d in folds.items()}
    lo, hi = str(LA.PD_QUANTILES[0]), str(LA.PD_QUANTILES[-1])
    lower = sum(d["timing_pd"][lo] > d["timing_pd"][hi] for d in folds.values())
    return dict(n_folds=len(folds), permutation_importance_mean=imp,
                permutation_importance_by_fold={str(s): d["permutation_importance"] for s, d in folds.items()},
                importance_ranking=sorted(imp, key=lambda f: -imp[f]), timing_pd_by_fold=pd_by_fold,
                timing_direction="lower" if 2 * lower > len(folds) else "higher" if 2 * lower < len(folds) else "mixed",
                timing_lower_marks_true_rater_folds=int(lower),
                n_train_rows_by_fold={str(s): d["n_train_rows"] for s, d in folds.items()})


def signal_block(tables: dict, details: dict) -> dict:
    """Per signal: the raw (unoriented) G3 over every item of the family (below 0.5: a LOWER value marks the true
    rater), the leave-one-seed-out signs, and the hand-crafted attacker's half orientations."""
    out = {}
    for k in LA.SIGNALS:
        j = LA.FEATURES.index(k)
        raw = np.concatenate([LA.item_metrics(t, t.X[:, j])["auc"] for t in tables.values()])
        signs = details["signal_sign"][k]
        out[k] = dict(raw_g3_pooled=float(raw.mean()) if len(raw) else None,
                      lower_marks_true_rater=bool(raw.mean() < 0.5) if len(raw) else None,
                      loso_sign_by_seed={str(s): v for s, v in signs.items()})
    out["handcrafted_orientation_by_seed"] = {str(s): o for s, o in details["handcrafted_orientation"].items()}
    return out


def handcrafted_check(runs: dict, block: dict, details: dict, n_items: dict) -> dict:
    """The recomputed hand-crafted G3 against the driver's recorded linkage block (auc.combined) per seed. The driver
    counts a rater listed twice for one verdict twice; the dump keeps one row per candidate, so a small difference in
    items, and hence in AUC, can remain."""
    out = {}
    for s, r in sorted(runs.items()):
        rec = r.recorded or {}
        auc = (rec.get("auc") or {}).get("combined")
        mine = block["by_seed"][str(s)]
        out[str(s)] = dict(recorded_auc=auc, recomputed_auc=mine, recorded_items=rec.get("n_verdicts_scored"),
                           dump_items=n_items[s], difference=None if auc is None or mine is None else mine - auc,
                           recorded_orientation=rec.get("orientation_per_half"),
                           recomputed_orientation=details["handcrafted_orientation"][s])
    diffs = [abs(v["difference"]) for v in out.values() if v["difference"] is not None]
    return dict(by_seed=out, max_abs_difference=max(diffs) if diffs else None)


# ------------------------------------------------------------------------ the whole analysis
def h7_block(per_network: dict, networks) -> dict:
    """H7: per network the cppr_btp max-over-attackers reading; hypothesis p = max over networks when every registered
    network is evaluable."""
    nets = {ds: fams[H7_FAMILY]["max_over_attackers"] for ds, fams in per_network.items() if H7_FAMILY in fams}
    missing = [ds for ds in PS.NETWORKS if ds not in nets]
    evaluable = not missing and all(n["evaluable"] for n in nets.values())
    met = evaluable and all(n["decision"] == "supported" for n in nets.values())
    return dict(claim="An insider cannot link CPPR's pooled verdicts (criterion G3)",
                rule="upper bound of the corrected-t 95% interval of the per-seed max-over-attackers G3 AUC < 0.60, on "
                     "each network",
                networks={ds: dict(estimate=n["estimate"], ci95=n["ci95"], p_value=n["p_value"],
                                   decision=n["decision"], n_seeds=n["n_seeds"], evaluable=n["evaluable"])
                          for ds, n in nets.items()},
                networks_missing=missing, evaluable=evaluable, interval_rule_met=met,
                p_value=max(n["p_value"] for n in nets.values()) if evaluable else None,
                status_before_holm=(NOT_EVALUABLE if not evaluable else "interval rule met on every network" if met
                                    else "interval rule not met: the paper states that criterion G3 is unmet"),
                contradicted=sorted(ds for ds, n in nets.items() if n["decision"] == "contradicted"))


def analyse(results_root: str, prereg: str, networks=PS.NETWORKS, families=tuple(f.name for f in FAMILIES),
            n_boot: int = N_BOOT, dump_manifests: dict | None = None, manifest: R.Manifest | None = None) -> tuple:
    """(result without provenance, timings). ``dump_manifests`` maps e3 / e5 to manifest files (default: each dump
    folder's MANIFEST_SHA256.txt)."""
    root = os.path.abspath(results_root)
    manifest = R.Manifest(root) if manifest is None else manifest
    prereg_sha256 = manifest.add(prereg) if os.path.isfile(prereg) else None
    failures = [] if prereg_sha256 else [dict(run=None, field="protocol", detail=f"protocol file {prereg} not found")]
    R.check_run_set(R.expected_runs(tuple(networks)), root, tuple(networks))
    paths = {src: os.path.join(root, "results", "dumps", src, DM.MANIFEST) for src in ("e3", "e5")}
    paths.update(dump_manifests or {})
    dumps = DumpManifests(paths)
    chosen = [f for f in FAMILIES if f.name in families]
    per_network, missing, seconds, used = {}, [], dict(load_s={}, analysis_s={}), {}
    for ds in networks:
        per_network[ds] = {}
        for fam in chosen:
            runs, miss, fails, secs = load_family(fam, ds, root, manifest, dumps, prereg_sha256)
            missing, failures = missing + miss, failures + fails
            seconds["load_s"].update(secs)
            used[f"{ds}/{fam.name}"] = sorted(runs)
            if not runs:
                per_network[ds][fam.name] = dict(seeds=[], n_seeds_expected=len(fam.seeds), evaluable=False,
                                                 max_over_attackers=endpoint_block({}, fam))
                continue
            t0 = time.time()
            per_network[ds][fam.name] = analyse_family(fam, runs, n_boot, ds)
            seconds["analysis_s"][f"{ds}/{fam.name}"] = round(time.time() - t0, 2)
            del runs
    excluded = sorted({f["run"] for f in failures if f["run"] is not None})
    import sklearn
    result = dict(
        schema=SCHEMA, status="integrity_failure" if failures else "ok",
        protocol=dict(path=manifest.key(prereg), sha256=prereg_sha256),
        settings=dict(networks=list(networks), families=[f.name for f in chosen], arm=ARM, null_auc=NULL_AUC,
                      attackers=list(LA.ATTACKERS), features=list(LA.FEATURES), n_boot=int(n_boot),
                      n_boot_registered=N_BOOT, boot_seed=BOOT_SEED, min_seed_share=str(MIN_SEED_SHARE),
                      sklearn=sklearn.__version__,
                      learner="HistGradientBoostingClassifier(random_state=0, early_stopping=False), default "
                              "hyperparameters otherwise",
                      dump_manifests={src: manifest.key(p) for src, p in paths.items()}),
        integrity=dict(ok=not failures, failures=failures, missing_runs=missing, excluded_runs=excluded,
                       used_seeds=used, dump_manifest_errors=dumps.errors),
        h7=h7_block(per_network, networks), networks=per_network, method=METHOD)
    return result, seconds


METHOD = dict(
    g3="Per seed and attacker, the mean over (pooled verdict, true rater) items of the within-pool AUC of the true "
       "rater against the other pool members (ties 1/2), on the direct arm's verdicts. An item is a distinct true "
       "rater of a verdict (a dump row with label 1). Single-member pools are excluded and counted; the sensitivity "
       "scores each of their items as AUC 1.",
    attackers="handcrafted: keeper.cppr.linkage.linkage_attack's combined score, recomputed from the dump's "
              "structure, timing and model_prob with its within-run half orientation (verdict index parity). "
              "Single signals: structure, timing, model_prob, degree, activity, activity_window, each oriented by the "
              "sign of its mean item AUC on the other seeds. learned: HistGradientBoostingClassifier(random_state=0, "
              "early_stopping=False) on structure, timing, model_prob, degree, activity, activity_window, "
              "candidate_tier, pool_size, pool_tier, fitted on the other seeds' candidate rows of the same family "
              "(label = true rater), scoring the held-out seed.",
    h7="Per seed, the maximum over the attackers of G3 on the capped-BTP E3 runs (cppr_btp); defined when every "
       "attacker is. Test: one-sided corrected t of mean < 0.60, variance factor 1 + J n_test / n_train with "
       "n_test / n_train = 1 / (J - 1) (leave one seed out; J = seeds); headline interval: the two-sided 95% "
       "corrected-t interval, clipped to [0, 1]. Supported on a network when its upper bound is < 0.60; "
       "contradicted when its lower bound is > 0.60; fewer than 80% of the seeds: not evaluable.",
    secondary="Top-1: per item 1/k when the true rater ties k members at the pool's maximum score, else 0 (the "
              "driver's block counts any tie at the maximum as a hit); chance: the mean of 1 / pool size over items. "
              "Learned attacker: permutation importance = drop of the held-out seed's G3 when one feature is "
              "permuted over its rows (one permutation, seeded); timing partial dependence at the 10/50/90% "
              "quantiles of the training rows' timing. Bootstrap sensitivity: nested, seeds then pooled verdicts "
              "within seed, the attackers' scores fixed.")


# ------------------------------------------------------------------------ report
def _f(x, fmt="{:.3f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def _ci(ci) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def render_markdown(r: dict) -> str:
    h7 = r["h7"]
    lines = ["# E6: linkage of CPPR's pooled verdicts (generated by keeper.analysis.e6_linkage)", "",
             f"Status: {r['status']}. H7 p (max over networks): {_f(h7['p_value'], '{:.4g}')}; "
             f"{h7['status_before_holm']}.", "", "## H7 (cppr_btp, max over attackers, corrected t against 0.60)", "",
             "| network | seeds | estimate | 95% CI | p | decision |", "| --- | --- | --- | --- | --- | --- |"]
    for ds, n in h7["networks"].items():
        lines.append(f"| {ds} | {n['n_seeds']} | {_f(n['estimate'])} | {_ci(n['ci95'])} | "
                     f"{_f(n['p_value'], '{:.4g}')} | {n['decision']} |")
    for ds, fams in r["networks"].items():
        for name, fam in fams.items():
            lines += ["", f"## {ds} / {name}", ""]
            if not fam.get("seeds"):
                lines.append("No run used.")
                continue
            mx = fam["max_over_attackers"]
            lines += [f"Seeds {fam['seeds']} of {fam['n_seeds_expected']}; single-member item share "
                      f"{fam['single_member_item_share']:.4f}; chance top-1 {_f(fam['chance_top1'], '{:.4f}')}. "
                      f"Max over attackers: {_f(mx['estimate'])} {_ci(mx['ci95'])} ({mx['decision']}); singles as 1: "
                      f"{_f(mx['sensitivity_single_member_pools_as_auc_1']['estimate'])}; nested bootstrap "
                      f"{_ci((mx['sensitivity_nested_bootstrap'] or {}).get('ci95'))}.", "",
                      "| attacker | G3 | 95% CI (corrected t) | top-1 |", "| --- | --- | --- | --- |"]
            for a, b in fam["attackers"].items():
                lines.append(f"| {a} | {_f(b['estimate'])} | {_ci(b['ci95'])} | {_f(b['top1'], '{:.4f}')} |")
            learned = fam["learned"]
            if learned.get("n_folds"):
                imp = learned["permutation_importance_mean"]
                lines += ["", "Learned attacker, permutation importance (G3 drop): " +
                          ", ".join(f"{f} {imp[f]:+.4f}" for f in learned["importance_ranking"]) +
                          f". Timing direction: {learned['timing_direction']} share marks the true rater "
                          f"({learned['timing_lower_marks_true_rater_folds']} of {learned['n_folds']} folds lower)."]
            sig = fam["signals"]
            lines += ["", "Raw signal G3 (below 0.5: lower value marks the true rater): " +
                      ", ".join(f"{k} {_f(sig[k]['raw_g3_pooled'])}" for k in LA.SIGNALS) +
                      f". Hand-crafted vs driver block: max |diff| "
                      f"{_f(fam['handcrafted_check']['max_abs_difference'], '{:.2e}')}."]
    integ = r["integrity"]
    lines += ["", "## Integrity", "", f"Failures: {len(integ['failures'])}; missing runs: {len(integ['missing_runs'])}."]
    lines += [f"- failure {f['run']}: {f['detail']}" for f in integ["failures"]]
    lines += [f"- missing {m['run']}: {m['reason']}" for m in integ["missing_runs"]]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ provenance and CLI
def provenance(manifest: R.Manifest, args, seconds: dict) -> dict:
    import scipy
    import sklearn
    modules = R.Manifest(C.ROOT)
    return dict(**git_state(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                host=socket.gethostname(), argv=list(sys.argv), python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS"), results_root=os.path.abspath(args.results_root),
                code_sha256=C.CODE_SHA256_AT_IMPORT,
                module_sha256={rel: modules.add(os.path.join(C.ROOT, rel)) for rel in MODULE_FILES},
                inputs_sha256=dict(sorted(manifest.sha256.items())), timings=seconds,
                peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))


def parse(argv=None):
    ap = argparse.ArgumentParser(description="E6 linkage analysis (H7; CPU only).")
    ap.add_argument("--results-root", default=C.ROOT, help="the main repository, whose results/ holds the runs")
    ap.add_argument("--prereg", default=os.path.join(C.ROOT, PS.PREREG), help="the protocol file")
    ap.add_argument("--networks", nargs="+", default=list(PS.NETWORKS), choices=PS.NETWORKS)
    ap.add_argument("--families", nargs="+", default=[f.name for f in FAMILIES], choices=[f.name for f in FAMILIES])
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="nested bootstrap replicates (sensitivity)")
    ap.add_argument("--e3-dump-manifest", default=None, help="default: results/dumps/e3/MANIFEST_SHA256.txt")
    ap.add_argument("--e5-dump-manifest", default=None, help="default: results/dumps/e5/MANIFEST_SHA256.txt")
    ap.add_argument("--out", default=os.path.join(ANALYSIS, "e6.json"))
    ap.add_argument("--md", default=os.path.join(ANALYSIS, "e6.md"))
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    out_path = C.result_path(os.path.abspath(a.out), a.force)      # refuse before the work, not after it
    md_path = C.result_path(os.path.abspath(a.md), a.force)
    manifest = R.Manifest(a.results_root)
    overrides = {src: os.path.abspath(p) for src, p in (("e3", a.e3_dump_manifest), ("e5", a.e5_dump_manifest)) if p}
    result, seconds = analyse(a.results_root, a.prereg, a.networks, a.families, a.n_boot, overrides, manifest)
    result["provenance"] = provenance(manifest, a, seconds)
    for path in (out_path, md_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    C.dump(result, out_path, force=a.force)
    text = render_markdown(json.loads(json.dumps(C._finite(result), default=C._json_default)))
    C.write_atomic(md_path, lambda f: f.write(text), replace=C.overwrite_allowed(a.force))
    code = EXIT_INTEGRITY if result["integrity"]["failures"] else EXIT_OK
    for failure in result["integrity"]["failures"]:
        print(f"integrity failure: {failure['run']}: {failure['detail']}", file=sys.stderr)
    print(f"wrote {out_path} and {md_path} (status {result['status']}, exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
