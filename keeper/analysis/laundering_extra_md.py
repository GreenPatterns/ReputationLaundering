"""Markdown summary of results/analysis/laundering_extra.json (Task 2.1).

Each number of docs/reviews/2026-10-26/review_evidence.md sections 1 to 3, and each untabulated number
of the Task 2.1 context, stands next to the measured value with "match" or "mismatch". A value matches
when the measured value rounds to it at the precision it is stated in (within half a unit of its last
digit). A note explains a mismatch where the cause was traced; no tolerance is widened to make a value
match. The new numbers of the re-analysis follow, with their 95% intervals.
"""
from dataclasses import dataclass

DS_NAMES = {"bitcoinalpha": "Bitcoin-Alpha", "bitcoinotc": "Bitcoin-OTC", "wiki-rfa": "Wiki-RfA",
            "epinions-dated": "Epinions-Dated"}
PRIMARY = ("bitcoinalpha", "bitcoinotc", "wiki-rfa")
ALL4 = PRIMARY + ("epinions-dated",)
EPS = 1e-12
RATER_TERM_BAR = 0.003            # the paper's bound on the rater term (as in laundering_extra)


def _half_unit(decimals: int) -> float:
    return 0.5 * 10.0 ** -decimals + EPS


def _show(m, scale: float, decimals: int, unit: str) -> str:
    """A measured value with two more decimals than the expected one; an integer count as it is."""
    if isinstance(m, int) and scale == 1.0:
        return f"{m}{unit}"
    return f"{m * scale:.{decimals + 2}f}{unit}"


@dataclass(frozen=True)
class Value:
    """A number stated to ``decimals`` places, after multiplying the measured value by ``scale``."""
    x: float
    decimals: int
    scale: float = 1.0
    unit: str = ""

    def check(self, m) -> bool:
        return m is not None and abs(m * self.scale - self.x) <= _half_unit(self.decimals)

    def expected(self) -> str:
        return f"{self.x:.{self.decimals}f}{self.unit}"

    def measured(self, m) -> str:
        return "n/a" if m is None else _show(m, self.scale, self.decimals, self.unit)


@dataclass(frozen=True)
class Range:
    """A range "lo to hi" stated to ``decimals`` places; the measured value is (min, max)."""
    lo: float
    hi: float
    decimals: int
    scale: float = 1.0
    unit: str = ""

    def check(self, m) -> bool:
        return m is not None and all(abs(a * self.scale - b) <= _half_unit(self.decimals)
                                     for a, b in zip(m, (self.lo, self.hi)))

    def expected(self) -> str:
        return f"{self.lo:.{self.decimals}f}{self.unit} to {self.hi:.{self.decimals}f}{self.unit}"

    def measured(self, m) -> str:
        if m is None:
            return "n/a"
        return " to ".join(_show(x, self.scale, self.decimals, self.unit) for x in m)


@dataclass(frozen=True)
class Count:
    """k of n, optionally with the share stated as a whole percent; the measured value is (k, n)."""
    k: int
    n: int
    pct: float | None = None

    def check(self, m) -> bool:
        if m is None or tuple(m) != (self.k, self.n):
            return False
        return self.pct is None or abs(100.0 * m[0] / m[1] - self.pct) <= _half_unit(0)

    def expected(self) -> str:
        return f"{self.k}/{self.n}" + ("" if self.pct is None else f" ({self.pct:.0f}%)")

    def measured(self, m) -> str:
        return "n/a" if m is None else f"{m[0]}/{m[1]} ({100.0 * m[0] / m[1]:.1f}%)"


@dataclass(frozen=True)
class PlusMinus:
    """x +- e: the measured mean (optionally (mean, sd)) lies within e of x."""
    x: float
    e: float

    def check(self, m) -> bool:
        mean = m[0] if isinstance(m, (tuple, list)) else m
        return mean is not None and abs(mean - self.x) <= self.e + EPS

    def expected(self) -> str:
        return f"{self.x} +- {self.e}"

    def measured(self, m) -> str:
        if isinstance(m, (tuple, list)):
            return f"{m[0]:.5f} +- {m[1]:.5f} (mean +- sd)"
        return f"{m:.5f}"


@dataclass(frozen=True)
class Qualitative:
    """A worded claim checked as lo <= measured <= hi."""
    text: str
    lo: float
    hi: float

    def check(self, m) -> bool:
        return m is not None and self.lo <= m <= self.hi

    def expected(self) -> str:
        return f"{self.text} ({self.lo:g} to {self.hi:g})"

    def measured(self, m) -> str:
        return "n/a" if m is None else f"{m:.3f}"


def status(expect, measured) -> str:
    return "match" if expect.check(measured) else "mismatch"


# ------------------------------------------------------------------------ getters
def at(ds: str, *path):
    def get(res):
        x = res.get(ds)
        for p in path:
            x = None if x is None else x.get(p) if isinstance(x, dict) else x[p]
        return x
    return get


def span(ds: str, *path):
    def get(res):
        vals = [v for v in at(ds, *path)(res) if v is not None]
        return (min(vals), max(vals)) if vals else None
    return get


def pair(ds: str, path_k, path_n):
    return lambda res: (at(ds, *path_k)(res), at(ds, *path_n)(res))


def seed0_spread(ds: str, *keys):
    def get(res):
        s0 = at(ds, "edgeless", "spread")(res)[0]
        return tuple(s0[k] for k in keys)
    return get


def max_over(dss, *path):
    return lambda res: max(at(ds, *path)(res) for ds in dss if ds in res)


# ------------------------------------------------------------------------ expected values
PCT = dict(scale=100.0, unit="%")
EV1, EV2, EV3, CTX = ("evidence S1", "evidence S2", "evidence S3", "context")


def _rows_s1() -> list:
    edgeless = {"bitcoinalpha": (81, 135, 60), "bitcoinotc": (258, 398, 65), "wiki-rfa": (107, 155, 69)}
    proxy = {"bitcoinalpha": (0.935, 0.973), "bitcoinotc": (0.870, 0.936), "wiki-rfa": (0.559, 0.793)}
    pct = {"bitcoinalpha": (11, 16), "bitcoinotc": (15, 18), "wiki-rfa": (13, 28)}
    rows = []
    for ds in PRIMARY:
        rows += [(f"edgeless/{ds}", EV1, "cohort targets edgeless after C2", ds, Count(*edgeless[ds]),
                  pair(ds, ("edgeless", "n"), ("edgeless", "cohort"))),
                 (f"cold_proxy/{ds}", EV1, "edgeless targets' U1 proxy, range of seed means", ds,
                  Range(*proxy[ds], 3), span(ds, "edgeless", "cold_start_proxy")),
                 (f"cold_pct/{ds}", EV1, "edgeless targets' platform percentile, range of seed means", ds,
                  Range(*pct[ds], 0, **PCT), span(ds, "edgeless", "cold_start_pct"))]
    a = "bitcoinalpha"
    rows += [("s0_n", EV1, "seed 0: sampled edgeless targets", a, Value(21, 0),
              lambda res: at(a, "edgeless", "spread")(res)[0]["n"]),
             ("s0_proxy", EV1, "seed 0: their U1 proxy", a, PlusMinus(0.9733, 0.0001),
              seed0_spread(a, "mean", "sd")),
             ("s0_r0", EV1, "seed 0: their starting proxies r0", a, Range(0.34, 0.87, 2),
              seed0_spread(a, "r0_min", "r0_max"))]
    return rows


def _rows_s2() -> list:
    yard = {"bitcoinalpha": (1.38, 0.41, 0.86), "bitcoinotc": (1.29, 0.38, 0.84),
            "wiki-rfa": (2.02, 0.67, 1.44), "epinions-dated": (1.43, 0.41, 0.96)}
    split = {"bitcoinalpha": (1.79, 1.17, 1.09, 1.81), "bitcoinotc": (1.68, 1.11, 0.89, 1.91),
             "wiki-rfa": (2.19, 1.89, 1.88, 2.16)}
    rows = []
    for ds in ALL4:
        for name, x in zip(("laplace", "raw", "base_rate"), yard[ds]):
            rows.append((f"amp_{name}/{ds}", EV2, f"amplification, {name} yardstick", ds, Value(x, 2),
                         at(ds, "amplification", name)))
    for ds in PRIMARY:
        for name, x in zip(("edgeless", "edge_keeping", "pos", "nopos"), split[ds]):
            rows.append((f"split_{name}/{ds}", EV2, f"Laplace amplification, {name} targets", ds,
                         Value(x, 2), at(ds, "amp_split", name)))
    for ds in ALL4:
        rows.append((f"lap_cross/{ds}", EV2, "Laplace yardstick conditional flip rate (cohort)", ds,
                     Value(100, 0, **PCT), at(ds, "amplification", "laplace_crossing", "conditional_flip_rate")))
    return rows


def _rows_s3() -> list:
    tau = {"bitcoinalpha": (0.062, 0.097), "bitcoinotc": (0.025, 0.256), "wiki-rfa": (0.281, 0.526),
           "epinions-dated": (0.150, 0.312)}
    before = {"bitcoinalpha": 3.2, "bitcoinotc": 5.9, "wiki-rfa": 5.6, "epinions-dated": 9.3}
    after = {"bitcoinalpha": 37, "bitcoinotc": 39, "wiki-rfa": 56, "epinions-dated": 52}
    med = {"bitcoinalpha": (15, 32), "bitcoinotc": (20, 42), "wiki-rfa": (18, 42), "epinions-dated": (34, 48)}
    r0c = {"bitcoinalpha": 0.622, "bitcoinotc": 0.588, "wiki-rfa": 0.148, "epinions-dated": 0.454}
    r0s = {"bitcoinalpha": 0.629, "bitcoinotc": 0.638, "wiki-rfa": 0.230, "epinions-dated": 0.522}
    rows = []
    for ds in ALL4:
        rows += [(f"tau/{ds}", EV3, "MCC-optimal threshold, range over seeds", ds, Range(*tau[ds], 3),
                  at(ds, "tau_mcc", "range")),
                 (f"pct_before/{ds}", EV3, "mean platform percentile before", ds, Value(before[ds], 1, **PCT),
                  at(ds, "percentile", "before_mean")),
                 (f"pct_after/{ds}", EV3, "mean platform percentile after, U1", ds, Value(after[ds], 0, **PCT),
                  at(ds, "percentile", "after_mean_u1")),
                 (f"pct_median/{ds}", EV3, "median percentile after U1, range over seeds", ds,
                  Range(*med[ds], 0, **PCT), at(ds, "percentile", "after_median_u1_range"))]
        for field in ("escape_bottom5", "escape_bottom10"):
            rows.append((f"{field}/{ds}", EV3, f"{field.replace('_', ' ')} under U1", ds,
                         Value(100, 0, **PCT), at(ds, "percentile", field)))
        rows += [(f"r0_cohort/{ds}", EV3, "mean r0, full cohort", ds, Value(r0c[ds], 3),
                  at(ds, "r0", "cohort_mean")),
                 (f"r0_sample/{ds}", EV3, "mean r0, sampled targets", ds, Value(r0s[ds], 3),
                  at(ds, "r0", "sample_mean"))]
    for ds, hi, n in (("bitcoinalpha", 5, 135), ("bitcoinotc", 38, 398)):
        rows.append((f"below_tau/{ds}", EV3, f"cohort targets below tau before deletion (of {n})", ds,
                     Range(0, hi, 0), span(ds, "tau_mcc", "cohort_below_tau")))
    for ds in ("bitcoinalpha", "bitcoinotc"):
        rows.append((f"quartile/{ds}", EV3, "share of targets left in the bottom quartile after U1", ds,
                     Qualitative("about half", 0.4, 0.6), at(ds, "percentile", "frac_after_below25")))
    rows.append(("u1u2", EV3, "U1/U2 as a ratio of five-seed means", "bitcoinalpha", Value(0.99245, 5),
                 at("bitcoinalpha", "shares", "u1_over_u2")))
    return rows


def _rows_ctx() -> list:
    rows = []
    for ds, x in zip(PRIMARY, (97.8, 99.0, 99.4)):
        rows.append((f"sgcn/{ds}", CTX, "SGCN U1 share of the U2 gain", ds, Value(x, 1, **PCT),
                     at(ds, "shares", "sgcn_u1_over_u2")))
    for ds, tgt, mean, mabs, k, n in (("bitcoinalpha", 11.94, -0.0022, 0.0075, 135, 240),
                                      ("bitcoinotc", 9.43, -0.0006, 0.0071, 128, 250)):
        rows += [(f"tgt/{ds}", CTX, "target-term logit shift under U1", ds, Value(tgt, 2),
                  at(ds, "logit", "target_term_shift")),
                 (f"src/{ds}", CTX, "rater-term signed mean shift", ds, Value(mean, 4),
                  at(ds, "logit", "rater_term_mean")),
                 (f"src_abs/{ds}", CTX, "rater-term mean absolute shift", ds, Value(mabs, 4),
                  at(ds, "logit", "rater_term_mean_abs")),
                 (f"src_over/{ds}", CTX, "evaluations with rater-term magnitude above 0.003", ds, Count(k, n),
                  pair(ds, ("logit", "n_over_0.003"), ("logit", "n_eval")))]
    rows += [("dmcc_c2", CTX, "max single-retrain dMCC magnitude, C2 under U2", "all four", Value(0.073, 3),
              max_over(ALL4, "max_dmcc", "c2_u2")),
             ("dmcc_placebo", CTX, "max single-retrain dMCC magnitude, placebo under U2", "all four", Value(0.108, 3),
              max_over(ALL4, "max_dmcc", "placebo_u2"))]
    return rows


EXPECTED = _rows_s1() + _rows_s2() + _rows_s3() + _rows_ctx()

NOTES = {
    "s0_r0": "review_isolated2.py prints only the first 12 of the 21 rows; over those 12 rows r0 runs from "
             "0.3385 to 0.8727, which is the 0.34 to 0.87 of the review. Over all 21 edgeless targets it "
             "runs from 0.218 to 0.873.",
    "pct_before/wiki-rfa": "review_probe4.py prints the seed average to four decimals (0.0555), and 5.6% "
                           "rounds that printed value a second time. The unrounded 5.549% is 5.5%.",
    "pct_median/epinions-dated": "review_probe4.py prints the seed-2 median to three decimals (0.475), and "
                                 "48% rounds that printed value a second time. The unrounded 47.49% is 47%.",
    "tau/epinions-dated": "The stored seed-1 threshold is 0.3113 (review_probe4.py prints 0.3113), so the "
                          "upper end is 0.311. No stored value rounds to 0.312; the review's 0.312 looks "
                          "like a transcription slip.",
}


# ------------------------------------------------------------------------ rendering
def reproduction_rows(result: dict) -> list:
    out = []
    for key, source, quantity, ds, expect, get in EXPECTED:
        if ds in DS_NAMES and ds not in result:
            continue
        try:
            m = get(result)
        except (KeyError, TypeError, IndexError, ValueError):
            m = None
        out.append(dict(key=key, source=source, quantity=quantity, dataset=DS_NAMES.get(ds, ds),
                        expected=expect.expected(), measured=expect.measured(m), status=status(expect, m)))
    return out


def _fmt(v, ci, digits: int = 3, scale: float = 1.0) -> str:
    if v is None:
        return "n/a"
    s = f"{v * scale:.{digits}f}"
    return s if ci is None else f"{s} [{ci[0] * scale:.{digits}f}, {ci[1] * scale:.{digits}f}]"


def _ci(block: dict, field: str, digits: int = 3, scale: float = 1.0) -> str:
    return _fmt(block.get(field), (block.get("ci95") or {}).get(field), digits, scale)


def _aucs(d: dict) -> str:
    amp = d["amplification"]
    return f"{amp['logreg_model']['auc_heldout']:.3f} / {amp['logreg_target_loo_model']['auc_heldout']:.3f}"


def _components(block: dict, field: str, digits: int = 2) -> str:
    """The three intervals behind a headline: crossed bootstrap / seed-t / nested."""
    return " / ".join(_fmt(block[field], block[key].get(field), digits).split(" ", 1)[-1]
                      for key in ("ci95_bootstrap", "ci95_seed_t", "ci95_nested"))


def _frac_over(g: dict) -> str:
    ci, source, t = g["ci95"]["frac_over_0.003"], g["ci95_source"]["frac_over_0.003"], g["targets_over_0.003"]
    where = f" over {t['n_targets']} targets" if source.startswith("Clopper") else ""
    return (f"{100 * g['frac_over_0.003']:.1f}% ({g['n_over_0.003']}/{g['n_eval']} evaluations; "
            f"{t['n_targets_any_over']}/{t['n_targets']} targets with any); "
            f"CI [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]%{where}")


def _escape(block: dict, field: str) -> str:
    c, ci, source = block["escape_counts"][field], block["ci95"][field], block["ci95_source"][field]
    where = f" over {c['n_targets_below']} targets" if source.startswith("Clopper") else ""
    return (f"{100 * block[field]:.1f}% ({c['n_escaped']}/{c['n_below']}); "
            f"CI [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]%{where}")


NEW_ROWS = (
    ("edgeless after C2 (cohort census)",
     lambda d: f"{d['edgeless']['n']}/{d['edgeless']['cohort']} ({100 * d['edgeless']['share']:.1f}%)"),
    ("cold-start proxy, U1 (mean of seeds)", lambda d: _ci(d["edgeless"], "cold_start_proxy_mean")),
    ("cold-start platform percentile, U1", lambda d: _ci(d["edgeless"], "cold_start_pct_mean")),
    ("escape from the bottom 5%, U1", lambda d: _escape(d["percentile"], "escape_bottom5")),
    ("escape from the bottom 10%, U1", lambda d: _escape(d["percentile"], "escape_bottom10")),
    ("left in the bottom quartile, U1", lambda d: _ci(d["percentile"], "frac_after_below25")),
    ("amplification, Laplace", lambda d: _ci(d["amplification"], "laplace", 2)),
    ("Laplace: crossed bootstrap / seed-t / nested intervals",
     lambda d: _components(d["amplification"], "laplace")),
    ("amplification, raw fraction", lambda d: _ci(d["amplification"], "raw", 2)),
    ("amplification, base-rate prior", lambda d: _ci(d["amplification"], "base_rate", 2)),
    ("amplification, degree logreg (R28: label left out at both ends)",
     lambda d: _ci(d["amplification"], "logreg", 2)),
    ("amplification, degree logreg, target-only LOO (R20)",
     lambda d: _ci(d["amplification"], "logreg_target_loo", 2)),
    ("degree logreg held-out AUC, R28 / R20", _aucs),
    ("Laplace amplification, edgeless / edge-keeping",
     lambda d: f"{_ci(d['amp_split'], 'edgeless', 2)} / {_ci(d['amp_split'], 'edge_keeping', 2)}"),
    ("partial deletion gain, 1 of 2 (share of full)",
     lambda d: f"{_ci(d['partial_deletion_u1'], 'k1_of_2')} "
               f"({_ci(d['partial_deletion_u1']['share_of_full'], 'k1_of_2', 2)})"),
    ("partial deletion gain, 1 of 3 (share of full)",
     lambda d: f"{_ci(d['partial_deletion_u1'], 'k1_of_3')} "
               f"({_ci(d['partial_deletion_u1']['share_of_full'], 'k1_of_3', 2)})"),
    ("partial deletion gain, 2 of 3 (share of full)",
     lambda d: f"{_ci(d['partial_deletion_u1'], 'k2_of_3')} "
               f"({_ci(d['partial_deletion_u1']['share_of_full'], 'k2_of_3', 2)})"),
    ("U1/U2 share, LocalGDTE", lambda d: _ci(d["shares"], "u1_over_u2", 4)),
    ("U1/U2 share, SGCN", lambda d: _ci(d["shares"], "sgcn_u1_over_u2", 4)),
    ("target-term logit shift, U1", lambda d: _ci(d["logit"], "target_term_shift", 2)),
    ("rater-term mean absolute shift, U1", lambda d: _ci(d["logit"], "rater_term_mean_abs", 5)),
    ("rater term above 0.003: evaluations, targets", lambda d: _frac_over(d["logit"])),
    ("max single-retrain dMCC magnitude, C2 / placebo (U2)",
     lambda d: f"{d['max_dmcc']['c2_u2']:.4f} / {d['max_dmcc']['placebo_u2']:.4f}"),
    ("mean dMCC, C2 under U2", lambda d: _ci(d["max_dmcc"], "mean_c2_u2", 4)),
    ("MCC-optimal tau, mean of seeds", lambda d: _ci(d["tau_mcc"], "mean", 3)),
)


def _new_numbers(result: dict) -> list:
    dss = [ds for ds in ALL4 if ds in result]
    lines = ["| Quantity | " + " | ".join(DS_NAMES[ds] for ds in dss) + " |",
             "|---" * (len(dss) + 1) + "|"]
    for label, get in NEW_ROWS:
        cells = []
        for ds in dss:
            try:
                cells.append(get(result[ds]))
            except (KeyError, TypeError):
                cells.append("n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def _design_note(result: dict) -> str:
    nested = [DS_NAMES[ds] for ds in ALL4 if ds in result and result[ds].get("ci_design") == "nested"]
    if not nested:
        return "Every dataset drew the same targets in every seed, so all four use the crossed bootstrap. "
    return f"{', '.join(nested)} drew different targets across seeds and fall(s) back to the nested bootstrap. "


def _undefined_note(result: dict) -> str:
    items = [f"{DS_NAMES[ds]} {field} ({', '.join(f'{d} {100 * v:.2f}%' for d, v in shares.items())})"
             for ds in ALL4 if ds in result
             for field, shares in (result[ds].get("ci_undefined_replicates") or {}).items()]
    return ("Undefined bootstrap replicates (dropped, at most 1% allowed): "
            + ("none in any field." if not items else "; ".join(items) + "."))


# ------------------------------------------------------------------------ negative findings (generated)
DMCC_BOUND = 0.007                # the paper's bound on the mean test-MCC change under U2
AMP_YARDSTICKS = (("laplace", "Laplace"), ("raw", "raw fraction"), ("base_rate", "base-rate prior"),
                  ("logreg", "degree logreg (R28)"), ("logreg_target_loo", "degree logreg, target-only LOO (R20)"))
AMP_SPLITS = (("edgeless", "edgeless targets"), ("edge_keeping", "edge-keeping targets"),
              ("pos", "targets with a trust in-edge"), ("nopos", "targets without one"))


def _amplification_items(result: dict):
    """(dataset, label, point, headline interval) for every amplification in the result."""
    for ds in ALL4:
        if ds not in result:
            continue
        amp, split = result[ds]["amplification"], result[ds]["amp_split"]
        for key, label in AMP_YARDSTICKS:
            yield ds, f"{label} yardstick, all targets", amp.get(key), amp["ci95"].get(key)
        for key, label in AMP_SPLITS:
            yield ds, f"Laplace yardstick, {label}", split.get(key), split["ci95"].get(key)


def amplification_findings(result: dict) -> tuple:
    """(intervals that include 1, intervals entirely below 1), each a list of readable items."""
    includes, below = [], []
    for ds, label, v, ci in _amplification_items(result):
        if v is None or ci is None:
            continue
        item = f"{DS_NAMES[ds]}, {label}: {v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
        if ci[0] <= 1.0 <= ci[1]:
            includes.append(item)
        elif ci[1] < 1.0:
            below.append(item)
    return includes, below


def _interval(v, ci, digits: int = 4, sign: bool = True) -> str:
    f = f"+.{digits}f" if sign else f".{digits}f"
    return f"{v:{f}} [{ci[0]:{f}}, {ci[1]:{f}}]"


def _claim_laplace(d: dict) -> tuple:
    v, ci = d["amplification"]["laplace"], d["amplification"]["ci95"]["laplace"]
    return ci[0] > 1.0, f"point estimate {v:.2f}, interval [{ci[0]:.2f}, {ci[1]:.2f}]"


def _claim_rater(d: dict) -> tuple:
    g = d["logit"]
    k, n = g["n_over_0.003"], g["n_eval"]
    detail = (f"signed mean {g['rater_term_mean']:+.4f}, mean magnitude "
              f"{_interval(g['rater_term_mean_abs'], g['ci95']['rater_term_mean_abs'], sign=False)}, above "
              f"{RATER_TERM_BAR} in {k}/{n} ({100 * k / n:.0f}%) of evaluations")
    return k == 0, detail


def _claim_dmcc_bound(d: dict) -> tuple:
    m = d["max_dmcc"]
    v, ci = m["mean_c2_u2"], m["ci95"]["mean_c2_u2"]
    where = "inside" if abs(v) <= DMCC_BOUND else "outside"
    return (-DMCC_BOUND <= ci[0] and ci[1] <= DMCC_BOUND,
            f"point estimate {v:+.4f} ({where} +-{DMCC_BOUND}), interval [{ci[0]:+.4f}, {ci[1]:+.4f}]")


def _claim_placebo(d: dict) -> tuple:
    m = d["max_dmcc"]
    return m["placebo_u2"] >= m["c2_u2"], f"placebo {m['placebo_u2']:.4f}, C2 {m['c2_u2']:.4f}"


def _claim_dmcc_zero(d: dict) -> tuple:
    m = d["max_dmcc"]
    v, ci = m["mean_c2_u2"], m["ci95"]["mean_c2_u2"]
    return ci[0] <= 0.0 <= ci[1], f"point estimate {v:+.4f}, interval [{ci[0]:+.4f}, {ci[1]:+.4f}]"


CLAIMS = (    # (id, claim as the current paper makes it, criterion, datasets it is made for, check)
    ("P1", "C2 gains under U2 exceed the Laplace counting shift (amplification above 1)",
     "supported only if the lower bound of the 95% headline interval of the Laplace amplification exceeds 1",
     ALL4, _claim_laplace),
    ("P2", f"the rater term a.z_u moves by less than {RATER_TERM_BAR} under U1",
     f"supported only if no evaluation's rater-term shift exceeds {RATER_TERM_BAR} in magnitude",
     ("bitcoinalpha", "bitcoinotc"), _claim_rater),
    ("P3", f"the mean test-MCC change under U2 is at most {DMCC_BOUND} in magnitude",
     f"supported only if the 95% headline interval lies within +-{DMCC_BOUND}; not supported does not "
     f"mean the point estimate exceeds {DMCC_BOUND}", ALL4, _claim_dmcc_bound),
    ("P4", "the random-edge placebo changes test MCC by as much as a C2 retrain (largest single change)",
     "supported only if the placebo's largest single change is at least C2's (maxima, no interval)",
     ALL4, _claim_placebo),
    ("P5", "utility monitoring does not reveal the attack: the mean test-MCC change under C2 and U2 is "
           "indistinguishable from zero", "supported only if the 95% headline interval includes 0",
     ALL4, _claim_dmcc_zero),
)


def claim_checks(result: dict) -> tuple:
    """(unsupported, supported, not_checked) over every claim and every dataset it is made for.

    unsupported and supported hold (claim id, dataset, detail); not_checked holds (claim id, dataset,
    reason) for a dataset that was not analysed or lacks a field the check reads.
    """
    unsupported, supported, not_checked = [], [], []
    for cid, _, _, scope, check in CLAIMS:
        for ds in scope:
            if ds not in result:
                not_checked.append((cid, ds, "dataset not analysed"))
                continue
            try:
                ok, detail = check(result[ds])
            except (KeyError, TypeError) as err:
                not_checked.append((cid, ds, f"missing field ({type(err).__name__}: {err})"))
                continue
            (supported if ok else unsupported).append((cid, ds, detail))
    return unsupported, supported, not_checked


def _partial_note(result: dict) -> str:
    short = {ds: (result[ds]["partial_deletion_u1"]["n_curve_points_short"],
                  result[ds]["partial_deletion_u1"]["n_curve_points"]) for ds in ALL4 if ds in result}
    text = ("Partial deletion. Its gains come from the stored greedy curves, scored by attack.Scorer with the "
            "trained input features held fixed (the unlink_frozen_x convention, not U1 with recomputed "
            "features); the share of full deletion divides by the same targets' unlink_frozen_x gain. ")
    stopped = [f"{DS_NAMES[ds]} {k} of {n}" for ds, (k, n) in short.items() if k]
    if not stopped:
        return text + "No greedy run stopped early."
    return (text + "Curve points from greedy runs that stopped early (no positive marginal gain), and so "
            "deleted fewer than k' warnings: " + ", ".join(stopped) + "; they are kept as stored.")


def paper_notes(result: dict) -> list:
    """Markdown lines of the "Notes for the paper" section, every finding computed from ``result``."""
    includes, below = amplification_findings(result)
    unsupported, supported, not_checked = claim_checks(result)
    claims = {cid: (text, criterion) for cid, text, criterion, _, _ in CLAIMS}
    passed = {}
    for cid, ds, _ in supported:
        passed.setdefault(cid, []).append(DS_NAMES[ds])
    lines = ["Every item below is computed from the headline intervals (ci95, rulings R30 and R31) when this "
             "summary is written; Phase 5 reports each negative finding where its claim is made.", "",
             "Amplification intervals that include 1 (the gain is not shown to differ from the yardstick's "
             "shift):", ""]
    lines += [f"- {x}" for x in includes] or ["- none"]
    lines += ["", "Amplification intervals entirely below 1 (the yardstick moves more than the model):", ""]
    lines += [f"- {x}" for x in below] or ["- none"]
    lines += ["", "Claims of the current paper, each with the criterion it is checked by:", ""]
    lines += [f"- {cid}: {text}. Criterion: {criterion}." for cid, (text, criterion) in claims.items()]
    lines += ["", "Claims the data or intervals do not support:", ""]
    lines += [f"- {cid}, {DS_NAMES[ds]}: {detail}." for cid, ds, detail in unsupported] or ["- none"]
    lines += ["", "Claims checked and supported: " + ("; ".join(f"{cid} ({', '.join(v)})" for cid, v in
                                                           passed.items()) or "none") + "."]
    lines += ["", "Claims not checked: " + ("; ".join(f"{cid}, {DS_NAMES.get(ds, ds)} ({reason})"
                                                    for cid, ds, reason in not_checked) or "none") + "."]
    lines += ["", f"- {_partial_note(result)}"]
    return lines


def render_markdown(result: dict) -> str:
    prov = result.get("provenance", {})
    rows = reproduction_rows(result)
    bad = [r for r in rows if r["status"] != "match"]
    lines = ["# Laundering re-analysis (Task 2.1)", "",
             f"Generated by `python -m keeper.analysis.laundering_extra` from the stored results in "
             f"`results/laundering/` (git {str(prov.get('git_commit'))[:7]}"
             f"{', uncommitted changes' if prov.get('git_dirty') else ''}, {prov.get('utc')}). "
             f"Each interval is the 95% headline interval of ruling R30: the hull of the two-way crossed "
             f"bootstrap interval (ruling R29; {prov.get('n_boot')} replicates, each resampling the seeds and, "
             f"independently, the target set once) and a t-interval over the per-seed statistics with "
             f"n_seeds - 1 degrees of freedom. A 0/1 field at exactly 0 or 1 uses the Clopper-Pearson "
             f"interval over its distinct targets instead. Each headline interval is clipped to its field's "
             f"support (ruling R31; the unclipped one is in `ci95_unclipped`). The component intervals are in the JSON "
             f"(`ci95_bootstrap`, `ci95_seed_t`, `ci95_nested`, with `ci95_source`). {_design_note(result)}"
             f"{_undefined_note(result)} Full values: `results/analysis/laundering_extra.json`.", "",
             "## Reproduction of the review's numbers", "",
             f"{len(rows) - len(bad)} of {len(rows)} match at the precision the review states them in; "
             f"{len(bad)} do not.", "",
             "| Source | Quantity | Dataset | Expected | Measured | Status |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['source']} | {r['quantity']} | {r['dataset']} | {r['expected']} | {r['measured']} | "
              f"{r['status']} |" for r in rows]
    if bad:
        lines += ["", "### Mismatches", ""]
        lines += [f"- `{r['key']}`: expected {r['expected']}, measured {r['measured']}. "
                  f"{NOTES.get(r['key'], 'Cause not traced.')}" for r in bad]
    lines += ["", "## New numbers", "",
              "Means over seeds with their 95% headline intervals in brackets. Partial deletion uses the "
              "greedy curves' unlink gains (trained input features kept); its share divides by the same "
              "targets' full deletion gain.", ""] + _new_numbers(result)
    lines += ["", "## Notes for the paper", ""] + paper_notes(result)
    return "\n".join(lines) + "\n"
