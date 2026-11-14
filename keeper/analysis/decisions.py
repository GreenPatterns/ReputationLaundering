"""Decisions over H1-H8 (Task 4.9; protocol docs/preregistration.md Sections 1 and 4). CPU only, no statistics of its own
beyond Holm's step-down:

  python -m keeper.analysis.decisions [--results-root ROOT] [--out PATH] [--md PATH] [--force]

Inputs, read-only, under ROOT/results/analysis (ROOT defaults to this repository): cppr_confirm.json (H1, H2w, H2b,
H3a, H3b; keeper.analysis.cppr_confirm), e2.json (H6a, H6b; e2_analysis), e6.json (H7; e6_linkage), e9a_e8.json (H8;
e9a_e8) and e1.json (H4, H5; e1_analysis). A missing input file makes its hypotheses "pending" (listed), never an error,
so the paper can be drafted while runs finish. Outputs: ROOT/results/analysis/decisions.json (keeper.laundering.core
.dump: atomic, overwrite guard of R27, refused without --force) and the generated decisions.md beside it. Reruns on the
same inputs are bit-identical (the provenance block holds the inputs' sha256 only).

Rules (Section 1):
  - Components. Each producing module gives a component's estimate, two-sided 95% headline interval ci95, p-value and
    evaluable flag (fewer than 80% of its seeds: not evaluable). The reading is recomputed here from ci95 and the
    registered rule: "interval > c" is met above c, contradicted below c, else not met; "interval < c" (H7, c = 0.60)
    is the mirror image; "lower bound >= c" (the removal thresholds H1(b), H2w(b), H2b(b), H8(b)) is read three ways:
    met (lower bound >= c), inconclusive (only the upper bound reaches c), partial (upper bound < c), and is
    contradicted only when its interval lies wholly below 0 (ruling R62, as in the modules). A reading that differs
    from the module's own decision is listed under notes.
  - Hypothesis p-value: the largest component p-value over its components and the three networks (intersection-union),
    defined when every component is evaluable. Holm's step-down over the eight (FAMILY) at family-wise ALPHA; a
    pending or not-evaluable hypothesis enters with p = 1. Holm's adjusted p-values never decrease when a raw p-value
    grows, so these provisional values are upper bounds of the final ones: a "supported" stays supported when the
    missing p-values arrive (holm.complete says whether all eight were available).
  - Decision, first match: pending (input missing); contradicted (a component contradicted); not evaluable (a
    component not evaluable or missing; reported as not supported); supported (every component met and Holm-adjusted
    p < ALPHA); not supported.
  - H3a and H5 are exact properties: a violation is a code defect, the outputs are written and the exit status is 3.
    H4 is a deterministic census rule outside the Holm family: e1.json's verdict.
"""
import argparse
import json
import os
import re
import sys

from keeper.analysis.dump_manifest import sha256_file
from keeper.cluster import protocol_settings as PS
from keeper.laundering import core as C

SCHEMA = "keeper.analysis.decisions/1"
ALPHA = 0.05
FAMILY = ("H1", "H2w", "H2b", "H3b", "H6a", "H6b", "H7", "H8")
EXIT_OK, EXIT_CODE_DEFECT = 0, 3
SOURCES = dict(cppr_confirm="results/analysis/cppr_confirm.json", e2="results/analysis/e2.json",
               e6="results/analysis/e6.json", e9a_e8="results/analysis/e9a_e8.json", e1="results/analysis/e1.json")
# hypothesis -> (source, {component: (operator, constant)}, regex of the run ids it depends on)
RULES = {
    "H1": ("cppr_confirm", {"a_difference": (">", 0.0), "b_cut": (">=", 0.8)}, r"^e3_btpc_"),
    "H2w": ("cppr_confirm", {"a_difference": (">", 0.0), "b_suppression": (">=", 0.8)}, r"^e3_btpc_"),
    "H2b": ("cppr_confirm", {"a_difference": (">", 0.0), "b_suppression": (">=", 0.8)}, r"^e3_btpc_"),
    "H3b": ("cppr_confirm", {"difference": (">", 0.0)}, r"^e[34]_(btpc|bfs)_"),
    "H6a": ("e2", {"whitewash": (">", 0.0), "badmouth": (">", 0.0)}, r"^e2_(scem_nohaz|scem)_(clean|lag1)_"),
    "H6b": ("e2", {"whitewash": (">", 0.0), "badmouth": (">", 0.0)}, r"^e2_(s3r|scem)_(clean|lag1)_"),
    "H7": ("e6", {"max_auc": ("<", 0.60)}, r"^e3_btpc_"),
    "H8": ("e9a_e8", {"a": (">", 0.0), "b": (">=", 0.5)}, r"."),
}
OTHER_RUNS = dict(H3a=r"_btpc_", H4=r"^e1b/", H5=r"^e1(c|d_h5)/")
CLAIMS = dict(H1="CPPR removes direct laundering", H2w="CPPR confines whitewash fakes without attack edges",
              H2b="CPPR confines bad-mouth fakes without attack edges",
              H3a="BTP admits a bounded number of fakes per attack edge",
              H3b="Under one attack edge, the attacker's best whitewash gain is smaller against capped BTP than BFS",
              H4="M-B makes witness laundering cost attack edges", H5="Commitment bounds the window of vulnerability",
              H6a="SCEM's revocation (not its delay) reduces a silent sender's effect",
              H6b="SCEM reduces a silent sender's effect relative to the S3R backbone",
              H7="An insider cannot link CPPR's pooled verdicts (criterion G3)",
              H8="The laundering gain is not an artifact of the missing self term")
THREE_WAY = "three-way reading of the removal threshold per network (see readings)"
NULL_READINGS = dict(
    H1=THREE_WAY, H2w=THREE_WAY, H2b=THREE_WAY, H8=THREE_WAY,
    H3b="BTP bounds Tier >= 1 admission per attack edge (H3a); under the count gate, the attacker's best gain per "
        "attack edge is not reduced.",
    H6a="reported as not supported in the section that makes the claim",
    H6b="reported as not supported in the section that makes the claim",
    H7="criterion G3 is unmet",
    H4="M-B reduces, not removes, witness laundering, with the measured residual")
READING_TEXT = {"met": "removal met", "inconclusive": "removal not established",
                "partial": "the mechanism reduces the effect (with the measured share)"}
MODULE_WORDS = {"supported": "met", "not supported": "not met"}       # e6's words for a component's decision


# ------------------------------------------------------------------------ Holm
def holm(pvalues: dict) -> dict:
    """Holm-adjusted p-values of {name: p}; a None p enters as 1 (and keeps its rank among the largest). Ties keep
    the same adjusted value (cumulative maximum); names break ties in the ordering, deterministically."""
    m = len(pvalues)
    order = sorted(pvalues, key=lambda n: (1.0 if pvalues[n] is None else pvalues[n], n))
    out, running = {}, 0.0
    for rank, name in enumerate(order):
        p = 1.0 if pvalues[name] is None else pvalues[name]
        running = max(running, min(1.0, (m - rank) * p))
        out[name] = running
    return out


# ------------------------------------------------------------------------ components and hypotheses
def read_component(raw: dict | None, rule: tuple) -> dict:
    """The reading of one component under rule (operator, constant) from its interval (see the module docstring)."""
    op, c = rule
    raw = raw or {}
    ci, p = raw.get("ci95"), raw.get("p_value")
    out = dict(rule=f"{'lower bound >=' if op == '>=' else 'interval ' + op} {c:g}", estimate=raw.get("estimate"),
               ci95=ci, p_value=p, n_seeds=raw.get("n_seeds"), n_seeds_expected=raw.get("n_seeds_expected"),
               module_decision=raw.get("decision"))
    if not raw:
        return dict(out, evaluable=False, reading="missing", met=False, contradicted=False)
    if not raw.get("evaluable") or ci is None or None in ci or p is None:
        return dict(out, evaluable=False, reading="not evaluable", met=False, contradicted=False)
    lo, hi = ci
    if op == ">=":
        reading = "met" if lo >= c else "inconclusive" if hi >= c else "partial"
        contradicted = hi < 0.0
    else:
        met, contradicted = (lo > c, hi < c) if op == ">" else (hi < c, lo > c)
        reading = "met" if met else "contradicted" if contradicted else "not met"
    return dict(out, evaluable=True, reading=reading, met=reading == "met", contradicted=bool(contradicted))


def raw_components(h: str, doc: dict) -> dict:
    """network -> component -> the module's component block, for each producing module's layout."""
    source = RULES[h][0]
    if source == "e2":
        return (doc.get("hypotheses") or {}).get(h, {}).get("components") or {}
    if source == "e6":
        return {ds: {"max_auc": n} for ds, n in ((doc.get("h7") or {}).get("networks") or {}).items()}
    nets = (doc.get("hypotheses") or {}).get(h, {}).get("networks") or {}
    return {ds: n.get("components") or {} for ds, n in nets.items()}


def module_p(h: str, doc: dict):
    block = doc.get("h7") if RULES[h][0] == "e6" else (doc.get("hypotheses") or {}).get(h)
    return (block or {}).get("p_value")


def missing_runs(source: str, doc: dict | None) -> list:
    """Every run the source lists as missing or excluded, as 'id (reason)' strings."""
    if doc is None:
        return []
    integrity = doc.get("integrity") or {}
    rows = {"e2": (doc.get("runs") or {}).get("missing"), "e1": doc.get("missing")}.get(
        source, integrity.get("missing_runs"))
    out = [f"{r['run']} ({r.get('reason', 'missing')})" if isinstance(r, dict) else str(r) for r in rows or []]
    excluded = integrity.get("excluded_runs") or {}
    out += [f"{run} (excluded)" for run in (excluded if isinstance(excluded, (list, dict)) else [])]
    return out


def hypothesis(h: str, doc: dict | None) -> dict:
    source, rules, runs = RULES[h]
    base = dict(claim=CLAIMS[h], source=SOURCES[source], null_reading=NULL_READINGS[h])
    if doc is None:
        return dict(base, decision="pending", evaluable=False, p_value=None, rule_met=False, contradicted=[],
                    components={}, readings={}, missing_runs=[], notes=[])
    raw = raw_components(h, doc)
    comps = {ds: {name: read_component((raw.get(ds) or {}).get(name), rule) for name, rule in rules.items()}
             for ds in PS.NETWORKS}
    flat = {f"{ds}/{name}": c for ds, by in comps.items() for name, c in by.items()}
    evaluable = all(c["evaluable"] for c in flat.values())
    p = max(c["p_value"] for c in flat.values()) if evaluable else None
    notes = [f"{k}: reading {c['reading']!r} differs from the module's {c['module_decision']!r}"
             for k, c in flat.items() if c["evaluable"] and c["module_decision"] is not None
             and MODULE_WORDS.get(c["module_decision"], c["module_decision"]) != c["reading"]]
    if evaluable and module_p(h, doc) != p:
        notes.append(f"intersection-union p {p} differs from the module's {module_p(h, doc)}")
    readings = {ds: {name: READING_TEXT.get(c["reading"], c["reading"]) if rules[name][0] == ">=" else c["reading"]
                     for name, c in by.items()} for ds, by in comps.items()}
    return dict(base, evaluable=evaluable, p_value=p, rule_met=evaluable and all(c["met"] for c in flat.values()),
                contradicted=sorted(k for k, c in flat.items() if c["contradicted"]), components=comps,
                readings=readings, missing_runs=[r for r in missing_runs(source, doc) if re.search(runs, r)],
                notes=notes, input_status=doc.get("status"))


def decide(block: dict, p_holm) -> str:
    if block.get("decision") == "pending":
        return "pending"
    if block["contradicted"]:
        return "contradicted"
    if not block["evaluable"]:
        return "not evaluable"
    return "supported" if block["rule_met"] and p_holm < ALPHA else "not supported"


# ------------------------------------------------------------------------ exact and deterministic rules
def h3a_block(doc: dict | None) -> dict:
    if doc is None:
        return dict(claim=CLAIMS["H3a"], source=SOURCES["cppr_confirm"], decision="pending", code_defect=False)
    h3a = doc.get("h3a") or {}
    violations = h3a.get("violations") or []
    decision = "defect" if violations else "untested" if not h3a.get("n_runs") else "holds"
    return dict(claim=CLAIMS["H3a"], source=SOURCES["cppr_confirm"], decision=decision, code_defect=bool(violations),
                n_runs=h3a.get("n_runs"), n_groups=h3a.get("n_groups"), violations=violations,
                missing_runs=[r for r in missing_runs("cppr_confirm", doc) if re.search(OTHER_RUNS["H3a"], r)])


def e1_blocks(doc: dict | None) -> tuple:
    """(H4, H5) from e1.json; the verdict words are e1_analysis' (supported, not_supported, not_evaluable, defect)."""
    if doc is None:
        return tuple(dict(claim=CLAIMS[h], source=SOURCES["e1"], decision="pending", code_defect=False)
                     for h in ("H4", "H5"))
    h4, h5 = doc.get("h4") or {}, doc.get("h5") or {}
    words = lambda v: (v or "missing").replace("_", " ")        # noqa: E731
    runs = missing_runs("e1", doc)
    return (dict(claim=CLAIMS["H4"], source=SOURCES["e1"], decision=words(h4.get("verdict")), code_defect=False,
                 null_reading=NULL_READINGS["H4"], threshold=h4.get("threshold"),
                 networks={ds: dict(status=words(n.get("status")), cells=n.get("cells"))
                           for ds, n in (h4.get("networks") or {}).items()},
                 missing_runs=[r for r in runs if re.search(OTHER_RUNS["H4"], r)]),
            dict(claim=CLAIMS["H5"], source=SOURCES["e1"], decision=words(h5.get("verdict")),
                 code_defect=bool(h5.get("defect")), untested=h5.get("untested") or [],
                 blocks=h5.get("blocks") or {}, missing_runs=[r for r in runs if re.search(OTHER_RUNS["H5"], r)]))


# ------------------------------------------------------------------------ the whole step
def load_inputs(results_root: str) -> tuple:
    """({source: parsed JSON or None}, {relative path: sha256} of the files present)."""
    docs, shas = {}, {}
    for name, rel in SOURCES.items():
        path = os.path.join(results_root, rel)
        if not os.path.isfile(path):
            docs[name] = None
            continue
        shas[rel] = sha256_file(path)
        with open(path) as f:
            docs[name] = json.load(f)
    return docs, shas


def analyse(results_root: str) -> dict:
    docs, shas = load_inputs(os.path.abspath(results_root))
    hyps = {h: hypothesis(h, docs[RULES[h][0]]) for h in FAMILY}
    adjusted = holm({h: b["p_value"] for h, b in hyps.items()})
    complete = all(b["p_value"] is not None for b in hyps.values())
    for h, b in hyps.items():
        b["p_holm"] = adjusted[h] if b["p_value"] is not None else None
        b["decision"] = decide(b, adjusted[h])
        b["null_reading_applies"] = b["decision"] not in ("supported", "pending")
        # only a Holm-limited "not supported" can still change, when the missing p-values arrive
        b["provisional"] = b["decision"] == "pending" or (b["decision"] == "not supported" and b["rule_met"]
                                                           and not complete)
    h3a = h3a_block(docs["cppr_confirm"])
    h4, h5 = e1_blocks(docs["e1"])
    code_defect = h3a["code_defect"] or h5["code_defect"]
    pending = [h for h in FAMILY + ("H3a", "H4", "H5")
               if (hyps.get(h) or dict(H3a=h3a, H4=h4, H5=h5)[h])["decision"] == "pending"]
    return dict(
        schema=SCHEMA, alpha=ALPHA, family=list(FAMILY), networks=list(PS.NETWORKS),
        holm=dict(complete=complete,
                  note="pending or not-evaluable hypotheses enter Holm with p = 1; the adjusted p-values are upper "
                       "bounds of the final ones"),
        inputs={name: dict(path=rel, present=docs[name] is not None, sha256=shas.get(rel),
                           status=(docs[name] or {}).get("status")) for name, rel in SOURCES.items()},
        pending=pending, code_defect=code_defect, hypotheses=hyps, exact=dict(H3a=h3a, H5=h5),
        deterministic=dict(H4=h4),
        provenance=dict(generator="keeper.analysis.decisions", inputs_sha256=dict(sorted(shas.items()))))


# ------------------------------------------------------------------------ report
def _f(x, spec=".4f") -> str:
    return "n/a" if x is None else format(x, spec)


def _ci(ci) -> str:
    return "n/a" if not ci or None in ci else f"[{_f(ci[0])}, {_f(ci[1])}]"


def render_markdown(r: dict) -> str:
    lines = ["# Decisions over H1-H8: generated summary", "",
             f"Holm over {len(r['family'])} hypotheses at family-wise {r['alpha']}; complete: {r['holm']['complete']} "
             f"({r['holm']['note']}). Code defect: {r['code_defect']}.", "",
             "Pending (input missing): " + (", ".join(r["pending"]) or "none") + ".", "",
             "| hypothesis | raw p (IU) | Holm-adjusted p | decision | contradicted | null reading (if not supported) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for h, b in r["hypotheses"].items():
        decision = b["decision"] + (" (provisional)" if b["provisional"] and b["decision"] != "pending" else "")
        lines.append(f"| {h} | {_f(b['p_value'], '.3g')} | {_f(b['p_holm'], '.3g')} | {decision} | "
                     f"{', '.join(b['contradicted']) or '-'} | {b['null_reading'] if b['null_reading_applies'] else '-'} |")
    lines += ["", "## Components", "",
              "| hypothesis | network | component | rule | estimate | 95% interval | p | reading | Holm-adjusted p | "
              "decision | missing runs |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for h, b in r["hypotheses"].items():
        for ds, by in b["components"].items():
            n_missing = sum(ds in run for run in b["missing_runs"])
            for name, c in by.items():
                lines.append(f"| {h} | {ds} | {name} | {c['rule']} | {_f(c['estimate'])} | {_ci(c['ci95'])} | "
                             f"{_f(c['p_value'], '.3g')} | {b['readings'][ds][name]} | {_f(b['p_holm'], '.3g')} | "
                             f"{b['decision']} | {n_missing} |")
    h3a, h5, h4 = r["exact"]["H3a"], r["exact"]["H5"], r["deterministic"]["H4"]
    lines += ["", "## Exact properties and the deterministic rule", "",
              f"- H3a: {h3a['decision']}" + ("" if h3a["decision"] == "pending" else
                                             f" ({h3a['n_runs']} capped-BTP runs, {h3a['n_groups']} groups, "
                                             f"{len(h3a['violations'])} violations)") + ".",
              f"- H5: {h5['decision']}" + ("" if h5["decision"] == "pending" else
                                           f"; untested: {', '.join(h5['untested']) or 'none'}") + ".",
              f"- H4: {h4['decision']}" + ("" if h4["decision"] == "pending" else
                                           f" (census mean of attack - clean <= {h4['threshold']}; null reading: "
                                           f"{NULL_READINGS['H4']})") + "."]
    for ds, n in (h4.get("networks") or {}).items():
        cells = "; ".join(f"k={c['k']} ring={c['ring']}: mean {_f(c['mean'])}, p95 {_f(c['p95'])}, max {_f(c['max'])}"
                          for c in n["cells"] or [])
        lines.append(f"  - {ds}: {n['status']} ({cells})")
    lines += ["", "## Inputs", ""] + [f"- {i['path']}: {'present' if i['present'] else 'missing (pending)'}"
                                       f"{'' if not i['status'] else ', status ' + i['status']}"
                                       for i in r["inputs"].values()]
    lines += ["", "## Missing runs", ""]
    for h, b in list(r["hypotheses"].items()) + [("H3a", h3a), ("H4", h4), ("H5", h5)]:
        runs = b.get("missing_runs") or []
        lines.append(f"- {h}: {len(runs)}" + (": " + ", ".join(runs) if runs else ""))
    notes = [f"- {h}: {n}" for h, b in r["hypotheses"].items() for n in b["notes"]]
    return "\n".join(lines + ["", "## Notes", ""] + (notes or ["- none"])) + "\n"


# ------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Holm decisions over H1-H8 (CPU only).")
    ap.add_argument("--results-root", default=C.ROOT, help="the repository whose results/analysis holds the inputs")
    ap.add_argument("--out", help="default: RESULTS_ROOT/results/analysis/decisions.json")
    ap.add_argument("--md", help="default: RESULTS_ROOT/results/analysis/decisions.md")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    a = ap.parse_args(argv)
    folder = os.path.join(os.path.abspath(a.results_root), "results", "analysis")
    out = C.result_path(os.path.abspath(a.out or os.path.join(folder, "decisions.json")), a.force)
    md = C.result_path(os.path.abspath(a.md or os.path.join(folder, "decisions.md")), a.force)
    result = analyse(a.results_root)
    for path in (out, md):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    C.dump(result, out, force=a.force)
    text = render_markdown(json.loads(json.dumps(C._finite(result), default=C._json_default)))
    C.write_atomic(md, lambda f: f.write(text), replace=C.overwrite_allowed(a.force))
    print(f"wrote {out} and {md}; pending: {', '.join(result['pending']) or 'none'}")
    for h, b in result["hypotheses"].items():
        print(f"  {h}: {b['decision']} (p {b['p_value']}, Holm {b['p_holm']})")
    if result["code_defect"]:
        print("decisions: CODE DEFECT: an exact property (H3a or H5) is violated", file=sys.stderr)
        return EXIT_CODE_DEFECT
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
