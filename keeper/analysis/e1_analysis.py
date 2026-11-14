"""E1 analysis (Task 4.4): census statistics of the SCEM engine study, the H4 and H5 rules, and integrity checks.

  python -m keeper.analysis.e1_analysis [--in DIR] [--out-dir DIR] [--prereg PATH] [--force]

Reads every JSON that keeper.scem.e1_runner wrote into DIR (default results/scem/e1), merges each study's shards per
network and checks that every shard is present and that the units cover each study's unit list exactly once. Writes
OUT_DIR/e1.json (keeper.laundering.core.dump, atomic) and e1.md (default OUT_DIR = results/analysis).

The unit is the victim record (protocol Section 3), and E1 is a census: every statistic is over all victim records of
a cell, with no sampling. census() gives n, mean, the 95th percentile (numpy's linear interpolation) and the maximum.

Integrity (assert_integrity): every output carries the protocol's sha256, equal to that of --prereg; one git commit
across the outputs of the single-record studies (e1c_single, e1d_h5_single) and one across all the other outputs (the
single-record studies were added after the first E1 run, without a change to the engine); the engine's defaults as
the engine has them now, with L_f = 1; each policy's keyword arguments
and closure age as keeper.scem.e1_runner defines them; the stream's K and t_seed; and every BTP calibration with the
eps, edge_cap_units and neg_from of keeper.cluster.protocol_settings, K(t) = ceil(log2 n_ref(t)), and, on the training
stream, K and n_ref of the whole stream equal to K_EXPECTED and N_REF_EXPECTED.

Rules (Section 1):
  H4  With M-B, at g = 0, lag 1, k in {5, 50}, ring off and on: the census mean of attack - clean <= 0.01 on each
      network (p95 and max reported). Deterministic; not part of the Holm family.
  H5  From the single-record studies, one attacked record per stream: on the 21-snapshot stream (e1c_single) and on
      the 88-snapshot stream (e1d_h5_single), per network: for every committing policy P, |attack - clean| <= 1e-12
      for every (lag, record) pair eligible under P (lag + L_f >= A_P and t_k + lag + L_f <= K - 1), and no_commit's
      mean over the same pairs >= 0.01; with no eligible pair, P is untested; a network and stream whose
      single-record output is missing is pending. A pair above 1e-12 is a code defect: the outputs are written with
      defect = true and the exit status is 1.
  The batched studies (e1c, e1d_h5) attack several records in one stream, so the other records' attacks reach a
  record while it is in escrow; their eligible differences are reported as descriptive (h5_batched) and set no
  verdict.

Exit status: 0; 1 for an H5 defect; 2 for a usage or integrity error (nothing is written).
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

from keeper.analysis.dump_manifest import sha256_file
from keeper.analysis.provenance import git_state
from keeper.cluster import protocol_settings as ps
from keeper.credibility import tiers
from keeper.laundering import core as C
from keeper.scem import e1_runner as R

IN_DIR = os.path.join(C.ROOT, "results", "scem", "e1")
OUT_DIR = os.path.join(C.ROOT, "results", "analysis")
H4_MAX_MEAN = 0.01
H4_CELLS = tuple(dict(k=k, g=0, lag=1, ring=ring, farming=False) for k in (5, 50) for ring in (False, True))
H5_TOL = 1e-12
H5_CONTRAST = 0.01
BOUND_TOL = 1e-12                         # rounding allowance of the property1_bound comparison (as Task 3.3)
TRAINING = ("e1a", "e1b", "e1c", "e1c_single")          # studies on the training stream (snapshots 0-20)
STREAM_LABEL = {"e1c": "21", "e1d_h5": str(R.H5_CUT)}   # the batched window studies -> their stream
SINGLE_LABEL = {"e1c_single": "21", "e1d_h5_single": str(R.H5_CUT)}   # H5's single-record studies
eligible = R.eligible


class IntegrityError(ValueError):
    pass


# ------------------------------------------------------------------ loading and merging


def load_outputs(folder: str) -> list:
    """(file name, document) of every runner output in folder, sorted by name."""
    out = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json"):
            with open(os.path.join(folder, name)) as f:
                doc = json.load(f)
            if doc.get("study") in R.STUDIES:
                out.append((name, doc))
    return out


def _expected_units(doc) -> list:
    return [(c["index"], b) for c in doc["cells"] for b in range(c["n_batches"])]


def merge_shards(docs: list) -> dict:
    """(study, ds) -> one document whose units are the union of its shards'. IntegrityError when a shard is missing
    or repeated, when the shards disagree on their cells, or when the units do not cover the unit list exactly once."""
    groups = collections.defaultdict(list)
    for doc in docs:
        groups[doc["study"], doc["ds"]].append(doc)
    merged = {}
    for key, group in sorted(groups.items()):
        counts = {d["shard"]["count"] for d in group}
        indices = sorted(d["shard"]["index"] for d in group)
        if len(counts) != 1 or indices != list(range(counts.pop())):
            raise IntegrityError(f"{key}: shards {indices} do not form 0..N-1 of one N")
        first = dict(group[0])
        if "units" in first:
            if any(d["cells"] != first["cells"] or d["n_units"] != first["n_units"] for d in group):
                raise IntegrityError(f"{key}: the shards disagree on their cells")
            units = sorted((u for d in group for u in d["units"]), key=lambda u: (u["cell"], u["batch"]))
            if [(u["cell"], u["batch"]) for u in units] != _expected_units(first):
                raise IntegrityError(f"{key}: the units do not cover the unit list exactly once")
            first["units"] = units
        merged[key] = first
    return merged


# ------------------------------------------------------------------ integrity


def _calibration_problems(cal: dict, ds: str, t_seed: int, K: int, training: bool) -> list:
    variant = cal["variant"]
    want = dict(eps=(ps.EPS_CAPPED if variant == "capped" else ps.EPS_UNCAPPED)[ds],
                edge_cap_units=ps.EDGE_CAP_UNITS if variant == "capped" else None, neg_from=ps.NEG_FROM, t_seed=t_seed)
    problems = [f"{variant} {k} = {cal[k]!r}, registered {v!r}" for k, v in want.items() if cal[k] != v]
    if len(cal["K_t"]) != K or any(k != tiers.default_K(n) for k, n in zip(cal["K_t"], cal["n_ref_t"])):
        problems.append(f"{variant} K(t) is not ceil(log2 n_ref(t)) over {K} rows")
    if training and (cal["K_full"], cal["n_ref_full"]) != (ps.K_EXPECTED[ds], ps.N_REF_EXPECTED[ds]):
        problems.append(f"{variant} whole-stream K, n_ref = {cal['K_full']}, {cal['n_ref_full']}; registered "
                        f"{ps.K_EXPECTED[ds]}, {ps.N_REF_EXPECTED[ds]}")
    return problems


def _stream_problems(stream: dict, ds: str, K: int, t_seed: int, training: bool) -> list:
    problems = [] if (stream["K"], stream["t_seed"]) == (K, t_seed) else [
        f"stream K, t_seed = {stream['K']}, {stream['t_seed']}; expected {K}, {t_seed}"]
    if training and stream["n_ref"] != ps.N_REF_EXPECTED[ds]:
        problems.append(f"stream n_ref = {stream['n_ref']}, registered {ps.N_REF_EXPECTED[ds]}")
    return problems


def _policy_problems(policies: dict) -> list:
    problems = []
    for name, row in policies.items():
        kw = {k: v for k, v in row.items() if k not in ("A", "horizon_days")}
        A = R.horizon_A(name)
        if name not in R.POLICIES or kw != R.POLICIES[name] or row["A"] != (A if np.isfinite(A) else None):
            problems.append(f"policy {name} = {row}, the runner defines {R.POLICIES.get(name)} with A = {A}")
    return problems


def doc_problems(doc: dict, protocol_sha: str) -> list:
    """Every way one output departs from the registered settings (module docstring)."""
    study, ds = doc["study"], doc["ds"]
    training = study in TRAINING
    problems = []
    if (doc.get("preregistration") or {}).get("sha256") != protocol_sha:
        problems.append("the protocol sha256 differs from --prereg's")
    if doc["engine_defaults"] != json.loads(json.dumps(R.engine_defaults())) or doc["engine_defaults"]["L_f"] != 1:
        problems.append("the engine defaults differ from the engine's, or L_f != 1")
    if not doc["provenance"].get("git_commit"):
        problems.append("no git commit in the provenance")
    if study == "e1d":
        for cut, block in doc["cuts"].items():
            K, t_seed = int(cut), R.CUT_T_SEED[int(cut)]
            problems += _stream_problems(block["stream"], ds, K, t_seed, False) + _policy_problems(block["policies"])
            problems += [p for cal in block["calibration"].values() for p in _calibration_problems(cal, ds, t_seed, K,
                                                                                                  False)]
        return problems
    K, t_seed = (R.H5_CUT, R.CUT_T_SEED[R.H5_CUT]) if study.startswith("e1d_h5") else (21, tiers.T_SEED)
    problems += _stream_problems(doc["stream"], ds, K, t_seed, training) + _policy_problems(doc["policies"])
    for cal in doc.get("calibration", {}).values():
        problems += _calibration_problems(cal, ds, t_seed, K, training)
    return problems


def assert_integrity(merged: dict, protocol_sha: str) -> dict:
    """IntegrityError listing every problem; else the inputs' code state (one commit for the single-record studies
    and one for the others, and whether any input was dirty)."""
    problems = [f"{study} {ds}: {p}" for (study, ds), doc in merged.items() for p in doc_problems(doc, protocol_sha)]
    commits = {single: {doc["provenance"].get("git_commit") for (study, _), doc in merged.items()
                        if (study in SINGLE_LABEL) == single} for single in (False, True)}
    for single, found in commits.items():
        if len(found) > 1:
            problems.append(f"the {'single-record' if single else 'other'} outputs come from {len(found)} commits: "
                            f"{sorted(map(str, found))}")
    if problems:
        raise IntegrityError("; ".join(problems))
    return dict(git_commit=commits[False].pop() if commits[False] else None,
                git_commit_single=commits[True].pop() if commits[True] else None,
                any_dirty=any(doc["provenance"].get("git_dirty") for doc in merged.values()))


# ------------------------------------------------------------------ statistics and rules


def census(values) -> dict:
    """n, mean, 95th percentile and maximum over every unit (None when there is none)."""
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return dict(n=0, mean=None, p95=None, max=None)
    return dict(n=len(x), mean=float(x.mean()), p95=float(np.percentile(x, 95)), max=float(x.max()))


def _cat(units, mb: str, key: str) -> np.ndarray:
    return np.concatenate([np.asarray(u[mb][key], dtype=np.float64) for u in units]) if units else np.zeros(0)


def e1b_rows(doc: dict) -> list:
    """Per cell and M-B setting: the census of attack - clean (total) and attack - comparator (direct), the share of
    victim records whose direct difference lies within property1_bound, honest tier changes and skipped records."""
    by_cell = collections.defaultdict(list)
    for u in doc["units"]:
        by_cell[u["cell"]].append(u)
    rows = []
    for cell in doc["cells"]:
        units = by_cell[cell["index"]]
        for mb in ("off", "on"):
            attack, clean = _cat(units, mb, "attack"), _cat(units, mb, "clean")
            direct = attack - _cat(units, mb, "comparator")
            within = direct <= _cat(units, mb, "bound") + BOUND_TOL
            rows.append(dict({k: cell[k] for k in ("index", "k", "g", "lag", "ring", "farming")}, mb=mb,
                             n_batches=len(units), skipped_seed_window=len(cell["skipped_seed_window"]),
                             skipped_beyond_stream=len(cell["skipped_beyond_stream"]),
                             total=census(attack - clean), direct=census(direct),
                             share_within_bound=float(within.mean()) if len(within) else None,
                             mean_attack=census(attack)["mean"], mean_clean=census(clean)["mean"],
                             honest_tier_changed_sum=sum(u["honest_tier_changed"] for u in units),
                             honest_tier_changed_max=max((u["honest_tier_changed"] for u in units), default=0),
                             max_conservation_err=max((u[mb]["max_conservation_err"] for u in units), default=0.0),
                             monotone_verified=all(u[mb]["monotone_verified"] for u in units)))
    return rows


def h4_rule(tables: dict) -> dict:
    """tables: ds -> e1b_rows. Per network, the H4 cells' census of attack - clean under M-B; the verdict."""
    per_ds = {}
    for ds in ps.NETWORKS:
        rows = tables.get(ds)
        cells = []
        for spec in H4_CELLS:
            match = [r for r in rows or () if r["mb"] == "on" and all(r[k] == v for k, v in spec.items())]
            stats = match[0]["total"] if match else census([])
            cells.append(dict(spec, **stats, holds=None if stats["mean"] is None else stats["mean"] <= H4_MAX_MEAN))
        holds = [c["holds"] for c in cells]
        per_ds[ds] = dict(cells=cells, status="not_evaluable" if None in holds else "holds" if all(holds) else "fails")
    statuses = [v["status"] for v in per_ds.values()]
    verdict = ("not_supported" if "fails" in statuses else "not_evaluable" if "not_evaluable" in statuses
               else "supported")
    return dict(threshold=H4_MAX_MEAN, networks=per_ds, verdict=verdict)


def eligible_diffs(doc: dict, policy: str) -> tuple:
    """(attack - clean of policy, attack - clean of no_commit) over the units' records eligible under policy."""
    K, L_f, A = doc["stream"]["K"], doc["L_f"], doc["policies"][policy]["A"]
    diff, contrast = [np.zeros(0)], [np.zeros(0)]
    for u in doc["units"]:
        mask = eligible(u["t_k"], u["lag"], A, L_f, K)
        diff.append((np.asarray(u["attack"][policy]) - np.asarray(u["clean"][policy]))[mask])
        contrast.append((np.asarray(u["attack"]["no_commit"]) - np.asarray(u["clean"]["no_commit"]))[mask])
    return np.concatenate(diff), np.concatenate(contrast)


def h5_policies(doc: dict) -> dict:
    """Per committing policy: the eligible records' attack - clean, no_commit's on the same records, and the status:
    untested, defect (a record with |attack - clean| above 1e-12), contrast_below_threshold (no_commit's mean < 0.01)
    or holds."""
    out = {}
    for p in R.COMMITTING:
        diff, contrast = eligible_diffs(doc, p)
        if not len(diff):
            status = "untested"
        elif np.abs(diff).max() > H5_TOL:
            status = "defect"
        else:
            status = "holds" if contrast.mean() >= H5_CONTRAST else "contrast_below_threshold"
        out[p] = dict(A=doc["policies"][p]["A"], n_eligible=len(diff), max=census(diff)["max"],
                      max_abs=float(np.abs(diff).max()) if len(diff) else None,
                      no_commit=census(contrast), status=status)
    return out


def h5_rule(blocks: dict) -> dict:
    """blocks: (stream label, ds) -> h5_policies of a single-record study. The verdict: defect if any policy has a
    defect; not_supported if a contrast is below its threshold; pending while a network of either stream is missing;
    else supported when some policy holds. Untested policies and pending blocks are listed."""
    statuses = [row["status"] for block in blocks.values() for row in block.values()]
    pending = [f"{label}/{ds}" for label in ("21", str(R.H5_CUT)) for ds in ps.NETWORKS if (label, ds) not in blocks]
    if "defect" in statuses:
        verdict = "defect"
    elif "contrast_below_threshold" in statuses:
        verdict = "not_supported"
    elif pending:
        verdict = "pending"
    else:
        verdict = "supported" if "holds" in statuses else "not_evaluable"
    return dict(tolerance=H5_TOL, contrast=H5_CONTRAST, defect=verdict == "defect", verdict=verdict, pending=pending,
                untested=sorted(f"{label}/{ds}/{p}" for (label, ds), block in blocks.items()
                                for p, row in block.items() if row["status"] == "untested"),
                blocks={f"{label}/{ds}": block for (label, ds), block in sorted(blocks.items())})


def batched_rows(doc: dict) -> dict:
    """Descriptive, from a batched study (several records attacked in one stream): per committing policy, the records
    eligible under it, the share whose |attack - clean| exceeds 1e-12, and the mean and maximum of |attack - clean|."""
    out = {}
    for p in R.COMMITTING:
        diff = np.abs(eligible_diffs(doc, p)[0])
        out[p] = dict(A=doc["policies"][p]["A"], n_eligible=len(diff),
                      share_nonzero=float((diff > H5_TOL).mean()) if len(diff) else None,
                      mean_abs=census(diff)["mean"], max_abs=census(diff)["max"])
    return out


def window_by_lag(doc: dict) -> dict:
    """Descriptive: per policy and lag, the census of attack - clean over every attacked record."""
    out = {}
    for p in doc["policies"]:
        out[p] = {str(lag): census(np.concatenate([np.asarray(u["attack"][p]) - np.asarray(u["clean"][p])
                                                   for u in doc["units"] if u["lag"] == lag] or [np.zeros(0)]))
                  for lag in doc["lags"]}
    return out


# ------------------------------------------------------------------ the report


def analyse(merged: dict) -> dict:
    e1b = {ds: e1b_rows(doc) for (study, ds), doc in merged.items() if study == "e1b"}
    windows = {(STREAM_LABEL[study], ds): doc for (study, ds), doc in merged.items() if study in STREAM_LABEL}
    singles = {(SINGLE_LABEL[study], ds): doc for (study, ds), doc in merged.items() if study in SINGLE_LABEL}
    return dict(
        e1a={ds: dict(policies=doc["policies"], metrics=doc["metrics"]) for (study, ds), doc in merged.items()
             if study == "e1a"},
        e1b=e1b, h4=h4_rule(e1b),
        e1c={f"{label}/{ds}": window_by_lag(doc) for (label, ds), doc in sorted(windows.items())},
        h5=h5_rule({key: h5_policies(doc) for key, doc in singles.items()}),
        h5_batched={f"{label}/{ds}": batched_rows(doc) for (label, ds), doc in sorted(windows.items())},
        e1d={ds: {cut: {k: block[k] for k in ("stream", "days_per_snapshot", "policies", "metrics")}
                  for cut, block in doc["cuts"].items()} for (study, ds), doc in merged.items() if study == "e1d"},
        missing=sorted(f"{study}/{ds}" for study in R.STUDIES for ds in ps.NETWORKS if (study, ds) not in merged))


def _f(x, digits=4) -> str:
    return "n/a" if x is None else f"{x:.{digits}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{digits}f}"


def render_md(result: dict) -> str:
    code = result["code"]
    lines = ["# E1: SCEM engine (H4, H5)", "",
             f"Inputs from git {str(code['git_commit'])[:7]} (single-record studies: "
             f"{str(code.get('git_commit_single'))[:7]})"
             f"{' (an input had uncommitted changes)' if code['any_dirty'] else ''}; protocol sha256 "
             f"{result['protocol_sha256'][:12]}. Missing: {', '.join(result['missing']) or 'none'}.", "",
             f"**H4: {result['h4']['verdict']}** (M-B, g = 0, lag 1: census mean of attack - clean <= "
             f"{H4_MAX_MEAN}).", "", "| network | k | ring | n | mean | p95 | max | holds |", "|" + "---|" * 8]
    for ds, block in result["h4"]["networks"].items():
        lines += [f"| {ds} | {c['k']} | {'on' if c['ring'] else 'off'} | {c['n']} | {_f(c['mean'])} | {_f(c['p95'])} | "
                  f"{_f(c['max'])} | {c['holds']} |" for c in block["cells"]]
    h5 = result["h5"]
    lines += ["", f"**H5: {h5['verdict']}** (one attacked record per stream: every eligible (lag, record) pair has "
              f"|attack - clean| <= {H5_TOL}; no_commit's mean >= {H5_CONTRAST}). Untested: "
              f"{', '.join(h5['untested']) or 'none'}. Pending: {', '.join(h5['pending']) or 'none'}.", "",
              "| stream/network | policy | A | eligible | max abs | no_commit mean | status |", "|" + "---|" * 7]
    for key, block in h5["blocks"].items():
        lines += [f"| {key} | {p} | {_f(r['A'], 3)} | {r['n_eligible']} | {_f(r['max_abs'])} | "
                  f"{_f(r['no_commit']['mean'])} | {r['status']} |" for p, r in block.items()]
    lines += [f"| {key} | all | | | | | pending |" for key in h5["pending"]]
    lines += ["", "## Batched attacks (descriptive): several records attacked in one stream", "",
              "The batched studies (e1c, e1d_h5) attack several records (at most one per warner) in one stream, so "
              "the other records' attacks reach a record while it is in escrow (reliability kappa, self-surprise). "
              "These numbers set no verdict.", "",
              f"| stream/network | policy | A | eligible | share > {H5_TOL} | mean abs | max abs |", "|" + "---|" * 7]
    for key, block in result["h5_batched"].items():
        lines += [f"| {key} | {p} | {_f(r['A'], 3)} | {r['n_eligible']} | {_f(r['share_nonzero'], 3)} | "
                  f"{_f(r['mean_abs'])} | {_f(r['max_abs'])} |" for p, r in block.items()]
    lines += ["", "## E1a: release policies without attack", "",
              "| network | policy | M-B | silent-sender rev. | future-only | honest mass rev. | all retained | "
              "latency 1 | latency 4 |", "|---|---|---|---|---|---|---|---|---|"]
    for ds, block in result["e1a"].items():
        for p, row in block["metrics"].items():
            lines += [f"| {ds} | {p} | {mb} | {_f(m['silent_sender_revocation'])} | "
                      f"{_f(m['silent_sender_revocation_future_only'])} | {_f(m['honest_mass_revoked'])} | "
                      f"{_f(m['all_retained'])} | {_f(m['latency_age1'])} | {_f(m['latency_age4'])} |"
                      for mb, m in row.items()]
    lines += ["", "## E1b: adversarial witnesses (Erlang N = 2)", "",
              "| network | k | g | lag | ring | farm | M-B | n | total mean | p95 | max | direct mean | max | "
              "within bound | tier chg | skipped |", "|" + "---|" * 16]
    for ds, rows in result["e1b"].items():
        lines += [f"| {ds} | {r['k']} | {r['g']} | {r['lag']} | {int(r['ring'])} | {int(r['farming'])} | {r['mb']} | "
                  f"{r['total']['n']} | {_f(r['total']['mean'])} | {_f(r['total']['p95'])} | {_f(r['total']['max'])} | "
                  f"{_f(r['direct']['mean'])} | {_f(r['direct']['max'])} | {_f(r['share_within_bound'], 3)} | "
                  f"{r['honest_tier_changed_sum']} | {r['skipped_seed_window'] + r['skipped_beyond_stream']} |"
                  for r in rows]
    lines += ["", "## E1d: granularity", "", "| network | snapshots | days/snapshot | policy | M-B | A (days) | "
              "silent-sender rev. | honest mass rev. | latency 1 | latency 4 |", "|" + "---|" * 10]
    for ds, cuts in result["e1d"].items():
        for cut, block in cuts.items():
            for p, row in block["metrics"].items():
                lines += [f"| {ds} | {cut} | {_f(block['days_per_snapshot'], 3)} | {p} | {mb} | "
                          f"{_f(block['policies'][p]['horizon_days'], 3)} | {_f(m['silent_sender_revocation'])} | "
                          f"{_f(m['honest_mass_revoked'])} | {_f(m['latency_age1'])} | {_f(m['latency_age4'])} |"
                          for mb, m in row.items()]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keeper.analysis.e1_analysis", description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="folder", default=IN_DIR, help="the runner outputs (default %(default)s)")
    ap.add_argument("--out-dir", default=OUT_DIR, help="where e1.json and e1.md go (default %(default)s)")
    ap.add_argument("--prereg", default=os.path.join(C.ROOT, ps.PREREG), help="the protocol file")
    ap.add_argument("--force", action="store_true", help="replace existing outputs")
    a = ap.parse_args(argv)
    json_path, md_path = (os.path.join(os.path.abspath(a.out_dir), f"e1.{ext}") for ext in ("json", "md"))
    try:
        for path in (json_path, md_path):
            C.result_path(path, a.force)
        loaded = load_outputs(a.folder)
        merged = merge_shards([doc for _, doc in loaded])
        protocol_sha = sha256_file(a.prereg)
        code = assert_integrity(merged, protocol_sha)
    except (IntegrityError, FileExistsError, FileNotFoundError) as error:
        print(f"e1_analysis: {error}", file=sys.stderr)
        return 2
    result = dict(protocol_sha256=protocol_sha, code=code,
                  inputs={name: sha256_file(os.path.join(a.folder, name)) for name, _ in loaded}, **analyse(merged))
    result["defect"] = result["h5"]["defect"]
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    C.dump(dict(result, provenance=dict(C.provenance(), **git_state())), json_path, force=a.force)
    C.write_atomic(md_path, lambda f: f.write(render_md(result)), replace=C.overwrite_allowed(a.force))
    print(f"wrote {json_path} and {md_path}; H4 {result['h4']['verdict']}, H5 {result['h5']['verdict']}")
    if result["defect"]:
        print("e1_analysis: DEFECT: H5's exact property is violated (see h5.blocks)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
