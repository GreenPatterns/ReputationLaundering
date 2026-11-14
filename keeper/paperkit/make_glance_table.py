"""Generate results/paper/tables/tab_laundering_glance.tex and results/paper/glance_values.json
from stored result files only.  `--check` regenerates in memory and fails on any difference."""
import argparse
import glob
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = "results/paper/tables/tab_laundering_glance.tex"
VALUES = "results/paper/glance_values.json"
DATASETS = ("bitcoinalpha", "bitcoinotc", "wiki-rfa", "epinions-dated")
HEADERS = ("Alpha", "OTC", "Wiki-RfA", "Epinions")
EXTRA = "results/analysis/laundering_extra.json"
LEGACY = "results/legacy/probe4_aggregate.json"
NOTE = (r"$\mathcal{U}_1$ keeps the trained parameters and only removes the deleted ratings from the graph that "
        r"the model reads. Platform percentile: share of active accounts with a lower reputation. The targets are "
        r"a stratified sample, about half of them with at least one endorsement; cohort-weighted rows weight the "
        r"two groups by their shares of the cohort, and the other rows are sample means.")


def fnum(x, dec, signed=False):
    s = f"{abs(x):.{dec}f}"
    neg = x < 0 and float(s) != 0
    if neg:
        return f"${'-'}{s}$"
    return ("+" if signed else "") + s


def pct(x, dec):
    return f"{100 * x:.{dec}f}"


def _get(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def _cohort_calib(extra, ds):
    return set(extra[ds].get("calibration_excluded") or [])


def _c3_all(root, ds, excluded):
    """Per-seed mean of the C3 (white) U2 gain over every evaluated target, 0 when no C3 candidate."""
    files = sorted(glob.glob(str(root / f"results/laundering/probe4_{ds}_s*.json")))
    seeds = []
    for f in files:
        gains = []
        for t in json.load(open(f))["targets"]:
            if t["target"] in excluded:
                continue
            op = t["c3_white"]["ops"]["exact_retrain"]
            gains.append(0.0 if op is None else op["gain"])
        seeds.append(sum(gains) / len(gains))
    return statistics.mean(seeds), [Path(f).relative_to(root).as_posix() for f in files]


def _escape_u2(root, ds, excluded, level=0.10):
    """Share of evaluations (seed, target) that start below ``level`` of the platform and are at or above it
    after full retraining (U2), with the two counts."""
    files = sorted(glob.glob(str(root / f"results/laundering/probe4_{ds}_s*.json")))
    below = escaped = 0
    for f in files:
        for t in json.load(open(f))["targets"]:
            if t["target"] in excluded or t["pct_before"] >= level:
                continue
            below += 1
            escaped += t["direct"]["ops"]["exact_retrain"]["pct_after"] >= level
    return escaped / below, (escaped, below), [Path(f).relative_to(root).as_posix() for f in files]


def _eigentrust(root, ds):
    files = sorted(glob.glob(str(root / f"results/laundering/baselines_{ds}_s*.json")))
    if not files:
        return None, []
    vals = [json.load(open(f))["summary"]["eigentrust"]["direct_d_pct"]["all"]["mean"] for f in files]
    return statistics.mean(vals), [Path(f).relative_to(root).as_posix() for f in files]


def _cell(value, fmt, files, key_path):
    return {"value": value, "text": fmt, "sources": [{"file": f, "key_path": key_path} for f in files]}


def compute_cells(root):
    extra = json.load(open(root / EXTRA))
    legacy = json.load(open(root / LEGACY))
    rows = {}

    def put(name, ds, cell):
        rows.setdefault(name, {})[ds] = cell

    for ds in DATASETS:
        e = extra[ds]

        def ex(path, fmt, dec):
            v = _get(e, path)
            return _cell(v, fmt(v, dec), [EXTRA], f"{ds}." + ".".join(path))
        put("cohort", ds, _cell(e["cohort"], f"{e['cohort']:,}".replace(",", "{,}"), [EXTRA], f"{ds}.cohort"))
        cw = e["cohort_weighted"]
        put("share_pos", ds, _cell(cw["cohort_share_pos"], pct(cw["cohort_share_pos"], 0), [EXTRA],
                                   f"{ds}.cohort_weighted.cohort_share_pos"))
        put("gain_u2_post", ds, ex(("cohort_weighted", "gain_u2"), lambda v, d: fnum(v, d, True), 3))
        for name, suffix in (("percentile", ""), ("percentile_nopos", "_nopos"), ("percentile_pos", "_pos")):
            b, a = cw["pct_before" + suffix], cw["pct_after_u2" + suffix]
            put(name, ds, _cell([b, a], f"{pct(b, 1)} $\\to$ {pct(a, 1)}", [EXTRA],
                                f"{ds}.cohort_weighted.pct_before{suffix}, {ds}.cohort_weighted.pct_after_u2{suffix}"))
        put("edgeless", ds, ex(("edgeless", "share"), lambda v, d: pct(v, d), 0))
        put("r0", ds, ex(("r0", "cohort_mean"), lambda v, d: fnum(v, d), 3))
        put("gain_u2", ds, ex(("amplification", "mean_gain_u2"), lambda v, d: fnum(v, d, True), 3))
        put("u1_share", ds, ex(("shares", "u1_over_u2"), lambda v, d: pct(v, d), 1))
        v, counts, files = _escape_u2(root, ds, _cohort_calib(extra, ds))
        put("escape", ds, _cell(v, pct(v, 1), files,
                                f"targets[*]: pct_before < 0.10 and direct.ops.exact_retrain.pct_after >= 0.10 "
                                f"(calibration targets excluded): {counts[0]} of {counts[1]} evaluations"))
        v, files = _c3_all(root, ds, _cohort_calib(extra, ds))
        put("c3", ds, _cell(v, fnum(v, 3, True), files,
                            "targets[*].c3_white.ops.exact_retrain.gain (0 if null; calibration targets excluded; mean per seed, then over seeds)"))
        sg = legacy["other_hypotheses"].get(ds, {}).get("SGCN_second_model")
        if sg is None:
            put("sgcn", ds, _cell(None, "n/a", [], ""))
        else:
            v = sg["direct_gain_exact"]["mean"]
            put("sgcn", ds, _cell(v, fnum(v, 3, True), [LEGACY],
                                  f"other_hypotheses.{ds}.SGCN_second_model.direct_gain_exact.mean"))
        v, files = _eigentrust(root, ds)
        if v is None:
            put("eigentrust", ds, _cell(None, "n/a", [], ""))
        else:
            put("eigentrust", ds, _cell(v, fnum(v, 3, True), files,
                                        "summary.eigentrust.direct_d_pct.all.mean (mean over seeds)"))
    return rows


ROW_LABELS = (
    ("cohort", "Targets in cohort"),
    ("share_pos", r"With an endorsement (\% of cohort)"),
    ("edgeless", r"Left with no rating after deletion (\% of cohort)"),
    ("r0", "Reputation before, mean"),
    ("gain_u2", r"Gain, full retraining ($\mathcal{U}_2$)"),
    ("gain_u2_post", r"\quad cohort-weighted"),
    ("u1_share", r"Gain, inference-only unlinking ($\mathcal{U}_1$), share of $\mathcal{U}_2$ (\%)"),
    ("percentile", r"Platform percentile (\%), before $\to$ after $\mathcal{U}_2$, cohort-weighted"),
    ("percentile_nopos", r"\quad targets without an endorsement"),
    ("percentile_pos", r"\quad targets with an endorsement"),
    ("escape", r"Targets in the lowest 10\% that leave it after $\mathcal{U}_2$ (\%)"),
    ("c3", r"Gain, indirect deletion"),
    ("sgcn", "Gain in a second GNN (SGCN)"),
    ("eigentrust", "Shift in EigenTrust (percentile)"),
)


def render_tex(rows):
    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Clean-data reputation laundering at a glance (direct deletion of at most three "
        r"warnings; mean over 5 seeds).}",
        r"\label{tab:laundering_glance}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{1.0in}rrrr@{}}", r"\toprule",
        " & " + " & ".join(rf"\textbf{{{h}}}" for h in HEADERS) + r" \\", r"\midrule",
    ]
    for key, label in ROW_LABELS:
        lines.append(label + " & " + " & ".join(rows[key][ds]["text"] for ds in DATASETS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\parbox{\columnwidth}{\scriptsize " + NOTE + "}", r"\end{table}", ""]
    return "\n".join(lines)


def build(root=ROOT):
    root = Path(root)
    rows = compute_cells(root)
    files = sorted({s["file"] for r in rows.values() for c in r.values() for s in c["sources"]})
    sha = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}
    values = {"generator": "keeper/paperkit/make_glance_table.py", "cells": rows, "sha256": sha}
    return render_tex(rows), json.dumps(values, indent=1, sort_keys=True) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    root = Path(args.root)
    tex, js = build(root)
    out = {TEX: tex, VALUES: js}
    if args.check:
        bad = [p for p, t in out.items() if not (root / p).exists() or (root / p).read_text() != t]
        if bad:
            print("STALE:", ", ".join(bad), file=sys.stderr)
            return 1
        print("glance table up to date")
        return 0
    for p, t in out.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
