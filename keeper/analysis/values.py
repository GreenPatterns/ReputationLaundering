"""The paper's values map and LaTeX value macros (Task 4.9, part 2), so that the paper never types a number by hand.

  python -m keeper.analysis.values [--results-root ROOT] [--out-root ROOT] [--check]

Reads result files under RESULTS_ROOT (default: this repository), read-only, and writes under OUT_ROOT (default: this
repository):
  results/paper/keeper_values.json  {"generator", "inputs_sha256", "values": {key: entry}} (results/paper/values.json,
                                    which older generators use, is an input here and never written)
  results/paper/keeper_values.tex   one macro per key, read by \\val{<key>}
  docs/keeper_values_keys.md        every key with its text, format and source
--check regenerates all three in memory and exits 1 when a file on disk differs (0 when all are current).

Entry: value (the number or string), text (formatted for the paper), format, source (file relative to RESULTS_ROOT, a
glob for several files, or "derived"), path and kind. kind "pointer": path is the RFC 6901 JSON pointer of the value
in source; "computed": path describes the computation; "derived": computed from other keys (DERIVED).

Keys: lower-case, dot-separated segments of [a-z0-9_]; networks are alpha, otc, wiki and epinions. An input file that
does not exist yet contributes no key, so its \\val{} prints a bold red ??key?? in the PDF until it appears; a value
that is null in its source is written as "n/a".

Texts. Numbers are math-mode text WITHOUT dollar signs: the paper writes $\\val{h8.alpha.gain_on.est}$, which turns
the plain '-' into a math minus. Formats: signed3 (+0.268), fixed3 (0.726), pct1 and pct0 (100 x, 1 or 0 decimals;
pct0 only where glance_values.json shows a count share as an integer percent), spct1 (signed percentage points), p2
(2 significant digits), g3 (3 significant digits, for the generic flattening), g (shortest), int (1{,}234). Below
1e-4 or from 1e6, p2 and g3 write a\\times10^{b}. String values (format str: decisions, readings, verdicts) are text
mode: \\val{dec.h8.decision}.

Coverage, in order: (a) e9a_e8.json: H8 per network and component, the secondary outcomes, E8 (prefixes h8, e9a,
e8); (b) decisions.json (dec); (c) glance_values.json (glance), laundering_extra.json (laundering), tiers_report.json
(tiers); (d) every numeric leaf of cppr_confirm.json, e1.json, e2.json, e5.json, e6.json and e7.json under the prefixes
cppr, e1, e2, e5, e6 and e7 (GENERIC), skipping the subtrees in SKIP and lists longer than MAX_LIST items, plus the
string leaves named in STRING_LEAVES; (e) results/paper/values.json: operator gains, the U1/U2 shares and the incidental,
verification and fed blocks (stored); (f) the sign control from the probe4 records (stored.<net>.sign_control); (g) the
ranges of DERIVED. Adding a derived range is one line in DERIVED.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

from keeper.analysis.decisions import FAMILY
from keeper.laundering import core as C

GENERATOR = "keeper/analysis/values.py"
OUT_JSON, OUT_TEX, OUT_MD = "results/paper/keeper_values.json", "results/paper/keeper_values.tex", "docs/keeper_values_keys.md"
NET = {"bitcoinalpha": "alpha", "bitcoinotc": "otc", "wiki-rfa": "wiki", "epinions-dated": "epinions"}
NETS3, NETS4 = tuple(NET)[:3], tuple(NET)
E9A, DEC = "results/analysis/e9a_e8.json", "results/analysis/decisions.json"
GLANCE, LX = "results/paper/glance_values.json", "results/analysis/laundering_extra.json"
TIERS, STORED = "results/analysis/tiers_report.json", "results/paper/values.json"
GENERIC = (("cppr", "results/analysis/cppr_confirm.json"), ("e1", "results/analysis/e1.json"),
           ("e2", "results/analysis/e2.json"), ("e5", "results/analysis/e5.json"), ("e6", "results/analysis/e6.json"),
           ("e7", "results/analysis/e7.json"))
SKIP = {"provenance", "bootstrap_streams", "inputs", "inputs_sha256", "module_sha256", "method", "sha256", "git_commits",
        "commits", "calibration_source"}
MAX_LIST = 10
STRING_LEAVES = {"decision", "reading", "status", "status_before_holm", "verdict"}
P_LEAVES = {"p", "p_value", "p_seed_t", "p_bootstrap", "p_below", "p_holm"}
KEY = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")
GLANCE_FORMATS = dict(c3="signed3", cohort="int", edgeless="pct0", eigentrust="signed3", escape="pct1",
                      gain_u2="signed3", percentile="pct1", r0="fixed3", sgcn="signed3", u1_share="pct1")
# (key under laundering.<net>, block path, field, format); interval ends from the block's ci95 when present
LX_FIELDS = (("edgeless_share", ("edgeless",), "share", "pct1"),
             ("cold_start_pct", ("edgeless",), "cold_start_pct_mean", "pct1"),
             ("pct_before", ("percentile",), "before_mean", "pct1"),
             ("pct_after_u1", ("percentile",), "after_mean_u1", "pct1"),
             ("pct_after_u2", ("percentile",), "after_mean_u2", "pct1"),
             ("gain_u2", ("amplification",), "mean_gain_u2", "signed3"),
             *((f"amp.{y}", ("amplification",), y, "fixed3")
               for y in ("laplace", "raw", "base_rate", "logreg", "logreg_target_loo")),
             *((f"amp_split.{s}", ("amp_split",), s, "fixed3") for s in ("edgeless", "edge_keeping")),
             *((f"partial.{k}", ("partial_deletion_u1", "share_of_full"), k, "pct1")
               for k in ("k1_of_2", "k1_of_3", "k2_of_3")),
             ("u1_share", ("shares",), "u1_over_u2", "pct1"), ("sgcn_u1_share", ("shares",), "sgcn_u1_over_u2", "pct1"))
TIER_VARIANTS = ("capped", "uncapped")
CALIBRATION = (("eps", "eps", "g"), ("k", "K", "int"), ("n_ref", "n_ref", "int"), ("guarantee", "guarantee_per_edge", "int"),
               ("best_star", "best_star", "int"), ("best_star_lower_bound", "best_star_lower_bound", "str"))
OPS = ("unlink", "unlink_frozen_x", "exact_retrain", "neggrad_plus", "influence_newton", "head_newton")
SECONDARY_FORMATS = ((r"newcomer_change", "spct1"), (r"delta", "signed3"), (r"^gain", "signed3"),
                     (r"pct", "pct1"), (r"mcc", "fixed3"))


def both(name: str, pattern: str) -> tuple:
    """The range of pattern over the three fully dated networks and over all four."""
    return (f"range.{name}.dated3", pattern, NETS3), (f"range.{name}.all4", pattern, NETS4)


# (key prefix, member key pattern with {net}, networks): writes <prefix>.min and <prefix>.max
DERIVED = (
    ("range.h8.gain_on", "h8.{net}.gain_on.est", NETS3),
    ("range.h8.ratio", "h8.{net}.ratio.est", NETS3),
    *both("gain_u2", "laundering.{net}.gain_u2.est"),
    *both("u1_share", "laundering.{net}.u1_share.est"),
    *both("pct_before", "laundering.{net}.pct_before.est"),
    *both("pct_after_u2", "laundering.{net}.pct_after_u2.est"),
    *both("edgeless_share", "laundering.{net}.edgeless_share.est"),
    *both("cold_start_pct", "laundering.{net}.cold_start_pct.est"),
    *both("partial.k1_of_3", "laundering.{net}.partial.k1_of_3.est"),
    *both("partial.k2_of_3", "laundering.{net}.partial.k2_of_3.est"),
    *both("partial.k1_of_2", "laundering.{net}.partial.k1_of_2.est"),
    *both("sign_control", "stored.{net}.sign_control.mean"),
    *both("unlink_share", "stored.{net}.unlink_share"),
    *both("unlink_frozen_x_share", "stored.{net}.unlink_frozen_x_share"),
)


# ------------------------------------------------------------------------ formatting
def _fixed(x: float, decimals: int) -> str:
    s = f"{x:.{decimals}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def _signed(x: float, decimals: int) -> str:
    s = _fixed(x, decimals)
    return s if s.startswith("-") else "+" + s


def _significant(x: float, digits: int) -> str:
    """digits significant digits; decimal notation from 1e-4 up to 1e6, else a\\times10^{b}."""
    if x == 0:
        return "0"
    rounded = float(f"{x:.{digits - 1}e}")
    exponent = math.floor(math.log10(abs(rounded)))
    if -4 <= exponent < 6:
        return _fixed(rounded, max(0, digits - 1 - exponent))
    mantissa, power = f"{x:.{digits - 1}e}".split("e")
    return f"{mantissa}\\times10^{{{int(power)}}}"


def latex_escape(s: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", s.replace("\\", r"\textbackslash{}"))


def text_of(value, fmt: str) -> str:
    if value is None:
        return "n/a"
    if fmt == "str":
        return latex_escape(str(value).lower() if isinstance(value, bool) else str(value))
    x = float(value)
    return {"signed3": lambda: _signed(x, 3), "fixed3": lambda: _fixed(x, 3), "pct1": lambda: _fixed(100 * x, 1),
            "pct0": lambda: _fixed(100 * x, 0), "spct1": lambda: _signed(100 * x, 1),
            "p2": lambda: _significant(x, 2), "g3": lambda: _significant(x, 3), "g": lambda: f"{x:g}",
            "int": lambda: f"{round(x):,}".replace(",", "{,}")}[fmt]()


# ------------------------------------------------------------------------ JSON pointers
def pointer(parts) -> str:
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def resolve(doc, ptr: str):
    for part in ptr.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        doc = doc[int(part)] if isinstance(doc, list) else doc[part]
    return doc


def segment(name) -> str:
    """A JSON key as a key segment: network names shortened, lower case, other characters as '_'."""
    s = str(name)
    for full, short in NET.items():
        s = s.replace(full, short)
    return re.sub(r"[^a-z0-9_]", "_", s.lower())


# ------------------------------------------------------------------------ the collector
class Values:
    """The values map under construction, with the parsed inputs (cached) and their sha256."""

    def __init__(self, results_root: str):
        self.root, self.entries, self.docs, self.sha = results_root, {}, {}, {}

    def doc(self, rel: str):
        """The parsed JSON file rel (relative to the results root), None when it does not exist."""
        if rel not in self.docs:
            path = os.path.join(self.root, rel)
            self.docs[rel] = None
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    raw = f.read()
                self.sha[rel] = hashlib.sha256(raw).hexdigest()
                self.docs[rel] = json.loads(raw)
        return self.docs[rel]

    def put(self, key: str, value, fmt: str, source: str, path: str, kind: str = "pointer") -> None:
        if not KEY.match(key):
            raise ValueError(f"invalid key {key!r}")
        if key in self.entries:
            raise ValueError(f"duplicate key {key!r} ({path} in {source})")
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        self.entries[key] = dict(value=value, text=text_of(value, fmt), format=fmt, source=source, path=path,
                                 kind=kind)

    def at(self, key: str, rel: str, parts, fmt: str) -> None:
        """The value at parts (a list of JSON keys and indexes) of rel; a KeyError names a schema change."""
        ptr = pointer(parts)
        self.put(key, resolve(self.doc(rel), ptr), fmt, rel, ptr)

    def interval(self, key: str, rel: str, parts, fmt: str, field=None, ci_key: str = "ci95") -> None:
        """key.est and, when the interval exists, key.lo and key.hi: from parts' estimate and ci95, or (field given)
        from parts[field] and parts.ci95[field] (the keeper.analysis.intervals.Block layout)."""
        parts = list(parts)
        est, ci = (parts + ["estimate"], parts + [ci_key]) if field is None else (parts + [field], parts + [ci_key, field])
        self.at(f"{key}.est", rel, est, fmt)
        node = self.doc(rel)
        for p in ci:
            node = node.get(p) if isinstance(node, dict) else None
        if isinstance(node, list) and len(node) == 2:
            self.at(f"{key}.lo", rel, ci + [0], fmt)
            self.at(f"{key}.hi", rel, ci + [1], fmt)

    def flatten(self, prefix: str, rel: str, node, parts: list, strings: bool = True) -> None:
        """Every numeric leaf (and, with strings, every string leaf named in STRING_LEAVES) of node under prefix (the
        generic flattening)."""
        last = parts[-1] if parts else ""
        if isinstance(node, dict):
            for k, v in node.items():
                if k not in SKIP:
                    self.flatten(f"{prefix}.{segment(k)}", rel, v, parts + [k], strings)
        elif isinstance(node, list):
            if len(node) <= MAX_LIST:
                names = ("lo", "hi") if len(node) == 2 and str(last).startswith("ci95") else range(len(node))
                for name, (i, v) in zip(names, enumerate(node)):
                    self.flatten(f"{prefix}.{name}", rel, v, parts + [i], strings)
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, (int, float)):
            fmt = "p2" if last in P_LEAVES else "int" if isinstance(node, int) else "g3"
            self.put(prefix, node, fmt, rel, pointer(parts))
        elif strings and isinstance(node, str) and last in STRING_LEAVES:
            self.put(prefix, node, "str", rel, pointer(parts))


# ------------------------------------------------------------------------ (a) e9a_e8.json
def secondary_format(name: str) -> str:
    return next(fmt for pattern, fmt in SECONDARY_FORMATS + ((r".", "g3"),) if re.search(pattern, name))


def add_e9a(v: Values) -> None:
    d = v.doc(E9A)
    if d is None:
        return
    h8 = ["hypotheses", "H8"]
    v.at("h8.p", E9A, h8 + ["p_value"], "p2")
    v.at("h8.status", E9A, h8 + ["status_before_holm"], "str")
    for ds in NETS3:
        net, base = NET[ds], h8 + ["networks", ds]
        v.at(f"h8.{net}.p", E9A, base + ["p_value"], "p2")
        for comp, name, fmt in (("a", "gain_on", "signed3"), ("b", "ratio", "fixed3")):
            cp, key = base + ["components", comp], f"h8.{net}.{name}"
            v.interval(key, E9A, cp, fmt)
            v.at(f"{key}.p", E9A, cp + ["p_value"], "p2")
            v.at(f"{key}.reading", E9A, cp + ["decision"], "str")
            v.at(f"{key}.n_seeds", E9A, cp + ["n_seeds"], "int")
            for i, end in enumerate(("lo", "hi")):
                if d["hypotheses"]["H8"]["networks"][ds]["components"][comp].get("ci95_nested"):
                    v.at(f"{key}.nested.{end}", E9A, cp + ["ci95_nested", i], fmt)
        for field, block in d["secondary"].get(ds, {}).items():
            parts, key = ["secondary", ds, field], f"e9a.{net}.{field}"
            if field.startswith("edgeless_"):
                v.at(f"{key}.share", E9A, parts + ["share"], "pct1")
                v.at(f"{key}.n", E9A, parts + ["n"], "int")
                v.at(f"{key}.n_edgeless", E9A, parts + ["n_edgeless"], "int")
            else:
                v.interval(key, E9A, parts, secondary_format(field))
        for field in d["e8"].get(ds, {}).get("summary", {}):
            v.interval(f"e8.{net}.{field}", E9A, ["e8", ds, "summary", field], secondary_format(field))


# ------------------------------------------------------------------------ (b) decisions.json
def add_decisions(v: Values) -> None:
    d = v.doc(DEC)
    if d is None:
        return
    for h in FAMILY:
        parts, key = ["hypotheses", h], f"dec.{h.lower()}"
        v.at(f"{key}.p_iu", DEC, parts + ["p_value"], "p2")
        v.at(f"{key}.p_holm", DEC, parts + ["p_holm"], "p2")
        v.at(f"{key}.decision", DEC, parts + ["decision"], "str")
        for ds, by in d["hypotheses"][h].get("readings", {}).items():
            for comp in by:
                v.at(f"{key}.{NET[ds]}.{segment(comp)}.reading", DEC, parts + ["readings", ds, comp], "str")
    for group, h in (("exact", "H3a"), ("exact", "H5"), ("deterministic", "H4")):
        v.at(f"dec.{h.lower()}.decision", DEC, [group, h, "decision"], "str")


# ------------------------------------------------------------------------ (c) stored results the paper already uses
def add_glance(v: Values) -> None:
    d = v.doc(GLANCE)
    if d is None:
        return
    for row, cells in d["cells"].items():
        fmt = GLANCE_FORMATS.get(row, "g3")
        for ds, cell in cells.items():
            key, parts = f"glance.{segment(row)}.{NET[ds]}", ["cells", row, ds, "value"]
            if isinstance(cell["value"], list):
                for i, end in enumerate(("before", "after")):
                    v.at(f"{key}.{end}", GLANCE, parts + [i], fmt)
            else:
                v.at(key, GLANCE, parts, fmt)


def add_laundering(v: Values) -> None:
    if v.doc(LX) is None:
        return
    for ds in NETS4:
        for name, block, field, fmt in LX_FIELDS:
            v.interval(f"laundering.{NET[ds]}.{name}", LX, [ds, *block], fmt, field=field)


def add_tiers(v: Values) -> None:
    d = v.doc(TIERS)
    if d is None:
        return
    rows = {(r["network"], r["variant"]): i for i, r in enumerate(d["calibration_table"])}
    for ds in NETS3:
        net = NET[ds]
        for variant in TIER_VARIANTS:
            i = rows[(ds, variant)]
            for name, field, fmt in CALIBRATION:
                if d["calibration_table"][i][field] is not None:
                    v.at(f"tiers.{net}.{variant}.{name}", TIERS, ["calibration_table", i, field], fmt)
            for t in range(4):
                v.at(f"tiers.{net}.{variant}.tier{t}_share", TIERS,
                     ["networks", ds, "variants", variant, "btp", "tier_shares", t], "pct1")
        for t in range(4):
            v.at(f"tiers.{net}.legacy.tier{t}_share", TIERS, ["networks", ds, "legacy", "tier_shares", t], "pct1")
        for method, block in d["networks"][ds]["demotion_census"]["own"]["methods"].items():
            for j, row in enumerate(block["rows"]):
                parts = ["networks", ds, "demotion_census", "own", "methods", method, "rows", j]
                v.at(f"tiers.{net}.census.{segment(method)}.k{row['k']}.dropped", TIERS, parts + ["share_dropped"], "pct1")
                v.at(f"tiers.{net}.census.{segment(method)}.k{row['k']}.to_tier0", TIERS, parts + ["share_to_tier0"],
                     "pct1")


# ------------------------------------------------------------------------ (d) generic, (e) values.json, (f) sign control
def add_generic(v: Values) -> None:
    for prefix, rel in GENERIC:
        if v.doc(rel) is not None:
            v.flatten(prefix, rel, v.doc(rel), [])


def add_stored(v: Values) -> None:
    d = v.doc(STORED)
    if d is None:
        return
    for ds in NETS4:
        net, ops = NET[ds], d["probe4"][ds]["ops"]
        for op in OPS:
            parts = ["probe4", ds, "ops", op]
            if ops.get(op) is None:
                v.at(f"stored.{net}.{op}.gain.mean", STORED, parts, "signed3")
                continue
            v.at(f"stored.{net}.{op}.gain.mean", STORED, parts + ["gain", "mean"], "signed3")
            v.at(f"stored.{net}.{op}.gain.sd", STORED, parts + ["gain", "sd"], "fixed3")
        v.at(f"stored.{net}.exact_retrain.d_t1_mcc.mean", STORED, ["probe4", ds, "ops", "exact_retrain", "d_t1_mcc", "mean"],
             "signed3")
        exact = ["probe4", ds, "ops", "exact_retrain", "gain", "mean"]
        for op in ("unlink", "unlink_frozen_x"):
            num = ["probe4", ds, "ops", op, "gain", "mean"]
            v.put(f"stored.{net}.{op}_share", resolve(d, pointer(num)) / resolve(d, pointer(exact)), "pct1", STORED,
                  f"{pointer(num)} / {pointer(exact)}", kind="computed")
    for block in ("incidental", "verification", "fed"):
        v.flatten(f"stored.{block}", STORED, d[block], [block], strings=False)


def add_sign_control(v: Values) -> None:
    """Per seed, the mean exact-retrain gain of the control arm over the targets with a control entry, without the
    calibration targets of laundering_extra.json; then the mean over seeds."""
    if v.doc(LX) is None:
        return
    for ds in NETS4:
        pattern = f"results/laundering/probe4_{ds}_s*.json"
        files = sorted((f for f in glob.glob(os.path.join(v.root, pattern))
                        if re.search(r"_s(\d+)\.json$", f)), key=lambda f: int(re.search(r"_s(\d+)\.json$", f)[1]))
        if not files:
            continue
        excluded = set(v.doc(LX)[ds].get("calibration_excluded") or [])
        means = {}
        for path in files:
            rel = os.path.relpath(path, v.root)
            seed = int(re.search(r"_s(\d+)\.json$", path)[1])
            gains = [t["control"]["ops"]["exact_retrain"]["gain"] for t in v.doc(rel)["targets"]
                     if t.get("control") is not None and t["target"] not in excluded]
            means[seed] = float(np.mean(gains))
            how = "/targets/*/control/ops/exact_retrain/gain, control not null, calibration_excluded of " + LX
            v.put(f"stored.{NET[ds]}.sign_control.s{seed}", means[seed], "signed3", rel, f"mean of {how}", "computed")
            v.put(f"stored.{NET[ds]}.sign_control.n.s{seed}", len(gains), "int", rel, f"count of {how}", "computed")
        v.put(f"stored.{NET[ds]}.sign_control.mean", float(np.mean(list(means.values()))), "signed3", pattern,
              "mean over seeds of stored.<net>.sign_control.s<seed>", "computed")


def add_derived(v: Values) -> None:
    for prefix, pattern, nets in DERIVED:
        members = [pattern.format(net=NET[ds]) for ds in nets]
        if not all(m in v.entries and v.entries[m]["value"] is not None for m in members):
            continue
        fmt, values = v.entries[members[0]]["format"], [v.entries[m]["value"] for m in members]
        for end, pick in (("min", min), ("max", max)):
            v.put(f"{prefix}.{end}", pick(values), fmt, "derived", f"{end} of {', '.join(members)}", "derived")


# ------------------------------------------------------------------------ outputs
def build(results_root: str) -> tuple:
    """(values, inputs_sha256) from the files under results_root."""
    v = Values(os.path.abspath(results_root))
    for step in (add_e9a, add_decisions, add_glance, add_laundering, add_tiers, add_generic, add_stored,
                 add_sign_control, add_derived):
        step(v)
    return dict(sorted(v.entries.items())), dict(sorted(v.sha.items()))


def render_json(values: dict, sha: dict) -> str:
    return json.dumps(dict(generator=GENERATOR, inputs_sha256=sha, values=values), indent=1, sort_keys=True,
                      allow_nan=False) + "\n"


def render_tex(values: dict) -> str:
    lines = [f"% Generated by {GENERATOR} ({OUT_JSON}); do not edit. Keys and sources: {OUT_MD}.",
             "% \\input{keeper_values} in the preamble (it loads xcolor). Numbers are math-mode text without dollar signs:",
             "% write $\\val{h8.alpha.gain_on.est}$. String values (decisions, readings) are text: \\val{dec.h8.decision}.",
             "% An undefined key prints a bold red ??key?? in the PDF.",
             "\\RequirePackage{xcolor}",
             "\\providecommand{\\keepervalmissing}[1]{\\textcolor{red}{\\textbf{??\\texttt{\\detokenize{#1}}??}}}",
             "\\newcommand{\\val}[1]{\\ifcsname keeperval@#1\\endcsname\\csname keeperval@#1\\endcsname"
             "\\else\\keepervalmissing{#1}\\fi}"]
    lines += [f"\\expandafter\\def\\csname keeperval@{k}\\endcsname{{{e['text']}}}" for k, e in values.items()]
    return "\n".join(lines) + "\n"


def render_md(values: dict) -> str:
    lines = ["# KEEPER paper values: the keys (generated)", "",
             f"Generated by `{GENERATOR}`; do not edit. Each key prints its text through `\\val{{key}}` "
             f"(`{OUT_TEX}`); numbers are math-mode text (`$\\val{{key}}$`), format `str` is text mode. Sources are "
             "relative to the results root; a path starting with `/` is a JSON pointer.", "",
             f"{len(values)} keys.", "", "| key | text | format | source | path |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| `{k}` | `{e['text']}` | {e['format']} | {e['source']} | `{e['path']}` |"
              for k, e in values.items()]
    return "\n".join(lines) + "\n"


def outputs(results_root: str) -> tuple:
    """({output path relative to the out root: text}, number of keys)."""
    values, sha = build(results_root)
    return {OUT_JSON: render_json(values, sha), OUT_TEX: render_tex(values), OUT_MD: render_md(values)}, len(values)


def read_text(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The paper's values map and LaTeX value macros (CPU only).")
    ap.add_argument("--results-root", default=C.ROOT, help="the repository whose results/ holds the inputs")
    ap.add_argument("--out-root", default=C.ROOT, help="the repository the three outputs are written into")
    ap.add_argument("--check", action="store_true", help="exit 1 if an output on disk differs from a regeneration")
    a = ap.parse_args(argv)
    files, n = outputs(a.results_root)
    if a.check:
        stale = [rel for rel, text in files.items() if read_text(os.path.join(a.out_root, rel)) != text]
        print(f"stale: {', '.join(stale)}" if stale else "keeper values up to date", file=sys.stderr if stale else None)
        return 1 if stale else 0
    for rel, text in files.items():
        path = os.path.join(os.path.abspath(a.out_root), rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        C.write_atomic(path, lambda f, t=text: f.write(t), replace=True)
    print(f"wrote {', '.join(files)} ({n} keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
