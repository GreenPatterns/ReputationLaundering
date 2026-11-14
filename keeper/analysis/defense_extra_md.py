"""Markdown summary of results/analysis/defense_extra.json (Task 2.2).

Each number of docs/reviews/2026-10-26/review_evidence.md sections 4 to 6 stands next to the measured
value with "match" or "mismatch", and so do the paper's R(0/1/3) at the ceiling hazard and the
supplement's single-warning hazards. The matching rule is Task 2.1's: a value matches when the measured
value lies within half a unit of its last stated digit. No tolerance is widened to make a value match.
The new numbers follow: the legacy tier coverage, the single-warning table under both closed forms, the
engine's hazard map and the CPPR seed statistics (pre-revision runs, superseded by E3). The notes for the
paper come last, and each is computed from the values.
"""
from dataclasses import dataclass

from keeper.analysis.laundering_extra_md import EPS, Range, Value, status

DS_NAMES = {"bitcoinalpha": "Bitcoin-Alpha", "bitcoinotc": "Bitcoin-OTC", "wiki-rfa": "Wiki-RfA"}
NETWORKS = tuple(DS_NAMES)
PCT = dict(scale=100.0, unit="%")
EV4, EV5, EV6 = "evidence S4", "evidence S5", "evidence S6"
PAPER, SUPP = "paper (R(s) at lambda 2)", "supplement S2"
CEILING = 2.0
FULL, FLOOR = 1.0, 0.20                     # one fully reliable witness; the floor after snapshot 4
PRE_REVISION_TITLE = "pre-revision runs, superseded by E3"


def _half(decimals: int) -> float:
    return 0.5 * 10.0 ** -decimals + EPS


@dataclass(frozen=True)
class MeanSd:
    """"mean +- sd" stated to ``decimals`` places; the measured value is (mean, sd)."""
    mean: float
    sd: float
    decimals: int

    def check(self, m) -> bool:
        return (m is not None and abs(m[0] - self.mean) <= _half(self.decimals)
                and abs(m[1] - self.sd) <= _half(self.decimals))

    def expected(self) -> str:
        return f"{self.mean:+.{self.decimals}f} +- {self.sd:.{self.decimals}f}"

    def measured(self, m) -> str:
        d = self.decimals + 2
        return "n/a" if m is None else f"{m[0]:+.{d}f} +- {m[1]:.{d}f}"


@dataclass(frozen=True)
class Seq:
    """Per-seed values stated to ``decimals`` places, in seed order."""
    values: tuple
    decimals: int

    def check(self, m) -> bool:
        return (m is not None and len(m) == len(self.values)
                and all(abs(a - b) <= _half(self.decimals) for a, b in zip(m, self.values)))

    def expected(self) -> str:
        return ", ".join(f"{x:.{self.decimals}f}" for x in self.values)

    def measured(self, m) -> str:
        return "n/a" if m is None else ", ".join(f"{x:.{self.decimals + 2}f}" for x in m)


# ------------------------------------------------------------------------ getters
def at(*path):
    def get(res):
        x = res
        for p in path:
            x = None if x is None else x.get(p) if isinstance(x, dict) else x[p]
        return x
    return get


def both(get_a, get_b):
    """(a, b) from two getters, or None if either value is missing."""
    def get(res):
        a, b = get_a(res), get_b(res)
        return None if a is None or b is None else (a, b)
    return get


def grid(lam: float, lag: int, key: str):
    def get(res):
        rows = [r for r in res["single_warning"]["grid"] if abs(r["lam"] - lam) < 1e-12 and r["lag"] == lag]
        return rows[0][key] if rows else None
    return get


def hazard(kappa: float, channel: int = 0):
    def get(res):
        rows = [r for r in res["single_warning"]["hazard_map"] if abs(r["kappa"] - kappa) < 1e-12]
        return rows[0]["lam"][channel] if rows else None
    return get


def common_df(res):
    dfs = {res["cppr_seed_stats"][ds]["paired_t"]["df"] for ds in res["cppr_seed_stats"]}
    return dfs.pop() if len(dfs) == 1 else None


# ------------------------------------------------------------------------ expected values
def _rows_s4() -> list:
    facts = {   # active, seeds, |D|, |D| share (%), D members issuing +1, late joiners in D (%), Tier 0 (%)
        "bitcoinalpha": (3629, 1188, 3453, 95.2, 3123, 93.1, 7.2),
        "bitcoinotc": (5684, 1621, 5191, 91.3, 4462, 88.2, 12.9),
        "wiki-rfa": (10152, 627, 2312, 22.8, 1865, 19.5, 84.3)}
    rows = []
    for ds, (active, seeds, n_d, d_pct, raters, late_pct, t0_pct) in facts.items():
        cov = ("tier_coverage", ds)
        rows += [(f"active/{ds}", EV4, "active accounts (a training edge at either end)", ds, Value(active, 0),
                  at(*cov, "n_active")),
                 (f"seeds/{ds}", EV4, "seeds S0 (majority-positive in-edges in snapshots 0 to 4)", ds,
                  Value(seeds, 0), at(*cov, "n_seeds")),
                 (f"D/{ds}", EV4, "endorsed set D", ds, Value(n_d, 0), at(*cov, "n_D")),
                 (f"D_share/{ds}", EV4, "D as a share of active accounts", ds, Value(d_pct, 1, **PCT),
                  at(*cov, "D_share_of_active")),
                 (f"D_raters/{ds}", EV4, "members of D that issued a +1", ds, Value(raters, 0),
                  at(*cov, "n_D_positive_raters")),
                 (f"late/{ds}", EV4, "late joiners in D (share of accounts first active after snapshot 4)", ds,
                  Value(late_pct, 1, **PCT), at(*cov, "late_in_D_share")),
                 (f"tier0/{ds}", EV4, "Tier 0 share of active accounts", ds, Value(t0_pct, 1, **PCT),
                  at(*cov, "tier0_share"))]
    return rows


def _rows_s5() -> list:
    vpe = {"bitcoinalpha": (0.055, 0.047), "bitcoinotc": (0.072, 0.031), "wiki-rfa": (0.364, 0.093)}
    diff = {"bitcoinalpha": (0.055, 0.041), "bitcoinotc": (0.048, 0.037), "wiki-rfa": (0.322, 0.057)}
    t = {"bitcoinalpha": Value(3.00, 2), "bitcoinotc": Value(2.87, 2), "wiki-rfa": Value(12.7, 1)}
    mcc = {"bitcoinalpha": (-0.053, 0.070), "bitcoinotc": (-0.077, 0.074), "wiki-rfa": (-0.066, 0.010)}
    rows = []
    for ds in NETWORKS:
        st = ("cppr_seed_stats", ds)
        rows += [(f"vpe/{ds}", EV5, "Plain VPE whitewash gain, mean +- sd over seeds", ds, MeanSd(*vpe[ds], 3),
                  both(at(*st, "whitewash_plain_vpe"), at(*st, "whitewash_plain_vpe_sd"))),
                 (f"diff/{ds}", EV5, "paired Plain VPE - CPPR whitewash gain, mean +- sd", ds,
                  MeanSd(*diff[ds], 3), both(at(*st, "paired_t", "mean"), at(*st, "paired_t", "sd"))),
                 (f"t/{ds}", EV5, "paired t statistic", ds, t[ds], at(*st, "paired_t", "t")),
                 (f"dmcc/{ds}", EV5, "per-seed CPPR - Plain Erasure test MCC, range", ds, Range(*mcc[ds], 3),
                  both(at(*st, "d_mcc_cppr_minus_plain_min"), at(*st, "d_mcc_cppr_minus_plain_max")))]
    rows.insert(1, ("vpe_seeds/bitcoinalpha", EV5, "Plain VPE whitewash gain per seed (seeds 0 to 4)",
                    "bitcoinalpha", Seq((0.100, -0.000, 0.030, 0.036, 0.108), 3),
                    at("cppr_seed_stats", "bitcoinalpha", "by_seed", "whitewash_plain_vpe")))
    rows.append(("df", EV5, "degrees of freedom of the paired t statistics", "all three", Value(4, 0), common_df))
    return rows


def _rows_s6() -> list:
    eng = ("single_warning", "engine")
    rows = [("N", EV6, "engine Erlang order N", "engine", Value(2, 0), at(*eng, "N")),
            ("phi", EV6, "engine release rate phi", "engine", Value(0.5, 1), at(*eng, "phi")),
            ("L_f", EV6, "engine witness lag L_f (snapshots)", "engine", Value(1, 0), at(*eng, "L_f")),
            ("H", EV6, "engine witness-evidence expiry H (snapshots)", "engine", Value(6, 0), at(*eng, "H")),
            ("A", EV6, "engine horizon A (snapshots)", "engine", Value(13.28, 2), at(*eng, "A"))]
    for lam, where, pct, durable in ((2.0, "lambda = 2", 64.7, 35), (0.53, "lambda = 0.53", 46.8, 53),
                                     (0.10, "lambda = 0.10", 16.4, 84)):
        rows += [(f"R1/{lam:g}", EV6, "revoked fraction R at lag 1, engine rules (expiry H, closure A)", where,
                  Value(pct, 1, **PCT), grid(lam, 1, "revoked_engine_rules")),
                 (f"durable1/{lam:g}", EV6, "durable share at lag 1, engine rules", where,
                  Value(durable, 0, **PCT), grid(lam, 1, "durable_engine_rules"))]
    rows += [("lam_full", f"{EV6}, {SUPP}", "hazard lambda^1 of one fully reliable witness (kappa 1.0)", "engine",
              Value(0.53, 2), hazard(FULL)),
             ("lam_floor", f"{EV6}, {SUPP}", "hazard lambda^1 of one witness at the floor (kappa 0.20)", "engine",
              Value(0.10, 2), hazard(FLOOR)),
             ("witnesses", SUPP, "fully reliable witnesses needed for lambda^1 >= 1.9", "engine", Value(7, 0),
              at("single_warning", "witnesses_to_reach", "n"))]
    for lag, pct in ((0, 82.5), (1, 64.7), (3, 34.6)):
        rows.append((f"paper_R/{lag}", PAPER, f"R at lag {lag}, Eq. 8 (no expiry)", f"lambda = {CEILING:g}",
                     Value(pct, 1, **PCT), grid(CEILING, lag, "revoked_no_expiry")))
    return rows


EXPECTED = _rows_s4() + _rows_s5() + _rows_s6()
NOTES = {}                                  # key -> traced cause of a mismatch (none at present)


def reproduction_rows(result: dict) -> list:
    out = []
    analysed = result.get("tier_coverage", {})
    for key, source, quantity, ds, expect, get in EXPECTED:
        if ds in DS_NAMES and ds not in analysed:
            continue
        try:
            m = get(result)
        except (KeyError, TypeError, IndexError, ValueError):
            m = None
        out.append(dict(key=key, source=source, quantity=quantity, dataset=DS_NAMES.get(ds, ds),
                        expected=expect.expected(), measured=expect.measured(m), status=status(expect, m)))
    return out


# ------------------------------------------------------------------------ tables of new numbers
def _pct(x, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _ci(v, ci, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{digits}f}" if ci is None else f"{v:+.{digits}f} [{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


COVERAGE_ROWS = (
    ("active accounts", lambda c: f"{c['n_active']:,}"),
    ("seeds S0", lambda c: f"{c['n_seeds']:,}"),
    ("endorsed set D (share of active accounts)", lambda c: f"{c['n_D']:,} ({_pct(c['D_share_of_active'])})"),
    ("members of D that issued a +1", lambda c: f"{c['n_D_positive_raters']:,}"),
    ("late joiners (first active after snapshot 4) in D",
     lambda c: f"{c['n_late_in_D']:,}/{c['n_late']:,} ({_pct(c['late_in_D_share'])})"),
    ("tier mix of active accounts, Tiers 0 / 1 / 2 / 3", lambda c: " / ".join(_pct(x) for x in c["tier_shares"])),
    ("endorsed accounts (Tier 1 to 3)", lambda c: f"{c['n_endorsed']:,}"),
)

CPPR_ROWS = (
    ("Plain VPE whitewash gain, per seed",
     lambda b: ", ".join(f"{x:+.3f}" for x in b["by_seed"]["whitewash_plain_vpe"])),
    ("Plain VPE whitewash gain, mean [95% CI] (sd)",
     lambda b: f"{_ci(b['whitewash_plain_vpe'], b['ci95']['whitewash_plain_vpe'])} "
               f"({b['whitewash_plain_vpe_sd']:.3f})"),
    ("CPPR whitewash gain, per seed", lambda b: ", ".join(f"{x:+.3f}" for x in b["by_seed"]["whitewash_cppr"])),
    ("CPPR whitewash gain, mean [95% CI] (sd)",
     lambda b: f"{_ci(b['whitewash_cppr'], b['ci95']['whitewash_cppr'])} ({b['whitewash_cppr_sd']:.3f})"),
    ("paired Plain VPE - CPPR, mean [95% CI] (sd)",
     lambda b: f"{_ci(b['whitewash_paired_diff'], b['ci95']['whitewash_paired_diff'])} "
               f"({b['paired_t']['sd']:.3f})"),
    ("paired t (df), two-sided p", lambda b: _t_text(b["paired_t"])),
    ("CPPR - Plain Erasure test MCC, per seed",
     lambda b: ", ".join(f"{x:+.3f}" for x in b["by_seed"]["d_mcc_cppr_minus_plain"])),
    ("CPPR - Plain Erasure test MCC, mean [95% CI]",
     lambda b: _ci(b["d_mcc_cppr_minus_plain"], b["ci95"]["d_mcc_cppr_minus_plain"])),
)


def _t_text(pt: dict) -> str:
    if pt["t"] is None:
        return f"undefined (all differences equal), df {pt['df']}"
    return f"{pt['t']:.2f} (df {pt['df']}), p = {pt['p_two_sided']:.4f}"


def _by_network(result: dict, section: str, rows) -> list:
    dss = [ds for ds in NETWORKS if ds in result[section]]
    lines = ["| Quantity | " + " | ".join(DS_NAMES[ds] for ds in dss) + " |", "|---" * (len(dss) + 1) + "|"]
    for label, get in rows:
        lines.append(f"| {label} | " + " | ".join(get(result[section][ds]) for ds in dss) + " |")
    return lines


def _grid_table(sw: dict) -> list:
    lags = sorted({r["lag"] for r in sw["grid"]})
    lines = ["| lambda | " + " | ".join(f"lag {s}" for s in lags) + " |", "|---" * (len(lags) + 1) + "|"]
    for lam in sorted({r["lam"] for r in sw["grid"]}):
        cells = {r["lag"]: r for r in sw["grid"] if r["lam"] == lam}
        lines.append(f"| {lam:g} | " + " | ".join(
            f"{100 * cells[s]['revoked_engine_rules']:.1f} / {100 * cells[s]['revoked_no_expiry']:.1f}"
            for s in lags) + " |")
    return lines


def _hazard_table(sw: dict) -> list:
    lines = ["| witness | kappa | h | nu | lambda^1 (gamma 1.0) | lambda^2 (gamma 0.8) | "
             "R at lags " + " / ".join(str(c["lag"]) for c in sw["hazard_map"][0]["revoked"])
             + ": channel 1; positive rating; negative rating |", "|---" * 7 + "|"]
    for row in sw["hazard_map"]:
        cells = row["revoked"]
        per = lambda key: " / ".join(f"{100 * c[key]:.1f}" for c in cells)              # noqa: E731
        ch1 = " / ".join(f"{100 * c['channel'][0]:.1f}" for c in cells)
        lines.append(f"| {row['label']} | {row['kappa']:.2f} | {row['h']:.2f} | {row['nu']:.4f} | "
                     f"{row['lam'][0]:.4f} | {row['lam'][1]:.4f} | {ch1}; {per('positive_rating')}; "
                     f"{per('negative_rating')} |")
    return lines


# ------------------------------------------------------------------------ notes for the paper (generated)
def _join(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def coverage_note(result: dict) -> str:
    cov = result["tier_coverage"]
    dss = [ds for ds in NETWORKS if ds in cov]
    share = lambda ds: f"{DS_NAMES[ds]} ({_pct(cov[ds]['D_share_of_active'])})"      # noqa: E731
    more = [share(ds) for ds in dss if cov[ds]["D_share_of_active"] > 0.5]
    half = [share(ds) for ds in dss if cov[ds]["D_share_of_active"] == 0.5]
    fewer = [share(ds) for ds in dss if cov[ds]["D_share_of_active"] < 0.5]
    clauses = ([f"more than half of the active accounts on {_join(more)}"] if more else []) \
        + ([f"exactly half on {_join(half)}"] if half else []) + ([f"fewer than half on {_join(fewer)}"] if fewer else [])
    late = _join([f"{_pct(cov[ds]['late_in_D_share'])} on {DS_NAMES[ds]}" for ds in dss])
    tier0 = _join([f"{_pct(cov[ds]['tier0_share'])} on {DS_NAMES[ds]}" for ds in dss])
    return (f"Legacy endorsed set D (keeper.credibility.tiers.legacy_bfs). D covers {', and '.join(clauses)}. "
            "During its single pass, one positive rating from a member of D admits the recipient, and a ring of "
            "accounts that rate each other positively after that rating enters D as well. Share of the accounts "
            f"first active after snapshot 4 that are in D: {late}. Tier 0 share of the active accounts: {tier0}.")


def _routing_pair(row: dict, lag: int, suffix: str = "") -> tuple:
    """(negative-rating, positive-rating) revoked fractions of one hazard-map row at ``lag``, in percent."""
    cell = [c for c in row["revoked"] if c["lag"] == lag][0]
    return f"{100 * cell[f'negative_rating{suffix}']:.1f}%", f"{100 * cell[f'positive_rating{suffix}']:.1f}%"


def revocation_note(result: dict) -> str:
    sw = result["single_warning"]
    eng, lag, reach = sw["engine"], 1, sw["witnesses_to_reach"]
    r = {lam: (grid(lam, lag, "revoked_engine_rules")(result), grid(lam, lag, "revoked_no_expiry")(result))
         for lam in (CEILING, 0.53, 0.10)}
    rows = {row["kappa"]: row for row in sw["hazard_map"]}
    full, floor = rows[FULL], rows[FLOOR]
    (fn, fp), (ln, lp) = _routing_pair(full, lag), _routing_pair(floor, lag)
    (fn0, fp0), (ln0, lp0) = _routing_pair(full, lag, "_no_expiry"), _routing_pair(floor, lag, "_no_expiry")
    gap = {lam: 100 * (r[lam][1] - r[lam][0]) for lam in r}
    expiry = ("so the expiry matters only for a weak hazard" if abs(gap[CEILING]) < 0.05 <= abs(gap[0.10]) else
              f"differences of {gap[CEILING]:.2f}, {gap[0.53]:.2f} and {gap[0.10]:.2f} points")
    return (f"Single warning. The engine keeps witness evidence live for H = {eng['H']} snapshots and closes a "
            f"record at A = {eng['A']:.2f}. Under these rules, a warning issued one snapshot after a rating "
            f"revokes {100 * r[CEILING][0]:.1f}% of it at the ceiling hazard lambda = 2, the hazard of Fig. 1c "
            f"and the lag census. One fully reliable witness gives lambda^1 = {full['lam'][0]:.3f}, and lambda = "
            f"0.53 revokes {100 * r[0.53][0]:.1f}%. One witness at the floor 0.20 gives lambda^1 = "
            f"{floor['lam'][0]:.3f}, and lambda = 0.10 revokes {100 * r[0.10][0]:.1f}%. Eq. 8 has no expiry and "
            f"gives {100 * r[CEILING][1]:.1f}%, {100 * r[0.53][1]:.1f}% and {100 * r[0.10][1]:.1f}%, {expiry}. "
            "The engine mixes a rating's two channels by its routing. One warning at lag 1 therefore revokes "
            f"{fn} of a negative and {fp} of a positive rating from a fully reliable witness, and {ln} and {lp} "
            f"at the floor. Without the expiry these become {fn0} and {fp0}, and {ln0} and {lp0}. Reaching "
            f"lambda^1 >= {reach['target']:g} takes {reach['n']} fully reliable witnesses.")


def _interval_items(st: dict, dss: list, field: str, keep) -> list:
    return [f"{DS_NAMES[ds]} ({_ci(st[ds][field], st[ds]['ci95'][field])})" for ds in dss
            if keep(st[ds]["ci95"][field])]


def cppr_note(result: dict) -> str:
    st = result["cppr_seed_stats"]
    dss = [ds for ds in NETWORKS if ds in st]
    items, excl, incl = [], [], []
    for ds in dss:
        b = st[ds]
        ci = b["ci95"]["whitewash_paired_diff"]
        items.append(f"{DS_NAMES[ds]}: {_ci(b['whitewash_paired_diff'], ci)}, t = {_t_text(b['paired_t'])}")
        (excl if ci[0] > 0.0 or ci[1] < 0.0 else incl).append(DS_NAMES[ds])
    verdict = " and ".join(([f"excludes 0 on {_join(excl)}"] if excl else [])
                           + ([f"includes 0 on {_join(incl)}"] if incl else []))
    ref = _interval_items(st, dss, "whitewash_plain_vpe", lambda ci: ci[0] <= 0.0 <= ci[1])
    ref_text = (f"The Plain VPE whitewash gain, the reference of the attack cut, has an interval that includes 0 "
                f"on {_join(ref)}." if ref else "The Plain VPE whitewash gain has an interval that excludes 0 on "
                "every network.")
    residual = _interval_items(st, dss, "whitewash_cppr", lambda ci: ci[0] > 0.0)
    residual_text = (f"The CPPR whitewash gain has an interval above 0 on {_join(residual)}." if residual else
                     "No network has a CPPR whitewash gain whose interval lies above 0.")
    signs = [f"{DS_NAMES[ds]} ({st[ds]['d_mcc_cppr_minus_plain_min']:+.3f} to "
             f"{st[ds]['d_mcc_cppr_minus_plain_max']:+.3f})" for ds in dss
             if st[ds]["d_mcc_cppr_minus_plain_min"] < 0.0 < st[ds]["d_mcc_cppr_minus_plain_max"]]
    mcc = (f"The per-seed CPPR - Plain Erasure test MCC takes both signs on {_join(signs)}." if signs else
           "The per-seed CPPR - Plain Erasure test MCC keeps one sign on every network.")
    return (f"CPPR seed-level statistics, {st[dss[0]]['label']}. The paired Plain VPE - CPPR whitewash gain, "
            f"with its 95% headline interval (ruling R30) and paired t statistic, is {'; '.join(items)}. The "
            f"interval of the paired gain {verdict}. {ref_text} {residual_text} {mcc}")


def paper_notes(result: dict) -> list:
    """Markdown lines of the "Notes for the paper" section, every finding computed from ``result``."""
    return ["Every item below is computed from the values when this summary is written.", "",
            f"- {coverage_note(result)}", f"- {revocation_note(result)}", f"- {cppr_note(result)}"]


def render_markdown(result: dict) -> str:
    prov = result.get("provenance", {})
    rows = reproduction_rows(result)
    bad = [r for r in rows if r["status"] != "match"]
    sw = result["single_warning"]
    eng = sw["engine"]
    lines = ["# Defense re-analysis (Task 2.2)", "",
             f"Generated by `python -m keeper.analysis.defense_extra` (git {str(prov.get('git_commit'))[:7]}"
             f"{', uncommitted changes' if prov.get('git_dirty') else ''}, {prov.get('utc')}). Tier coverage is a "
             f"census of each training graph, and the single-warning numbers are closed forms; neither has an "
             f"interval. The CPPR intervals follow ruling R30: the hull of a {prov.get('n_boot')}-replicate "
             f"bootstrap over seeds and the t-interval over the per-seed values (df = 4). The fields are "
             f"signed, so ruling R31 clips none of them. Full values: `results/analysis/defense_extra.json`.", "",
             "## Reproduction of the review's numbers", "",
             f"{len(rows) - len(bad)} of {len(rows)} match at the precision they are stated in; {len(bad)} do not.",
             "", "| Source | Quantity | Dataset | Expected | Measured | Status |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['source']} | {r['quantity']} | {r['dataset']} | {r['expected']} | {r['measured']} | "
              f"{r['status']} |" for r in rows]
    if bad:
        lines += ["", "### Mismatches", ""]
        lines += [f"- `{r['key']}`: expected {r['expected']}, measured {r['measured']}. "
                  f"{NOTES.get(r['key'], 'Cause not traced.')}" for r in bad]
    lines += ["", "## Legacy tier coverage", "",
              "Tiers from `keeper.credibility.tiers.legacy_bfs`; D replicated from the seeded pass and checked "
              "against it.", ""] + _by_network(result, "tier_coverage", COVERAGE_ROWS)
    lines += ["", "## Single-warning revocation", "",
              f"Engine: N = {eng['N']}, phi = {eng['phi']}, L_f = {eng['L_f']}, H = {eng['H']}, A = {eng['A']:.2f}, "
              f"lambda_max = {eng['lambda_max']:g}, gamma = {tuple(eng['gamma'])}. Revoked fraction R (%) of a "
              "rating whose sender is warned `lag` snapshots after it: engine rules (evidence expires after H, "
              "record closes at A) / Eq. 8 without expiry.", ""] + _grid_table(sw)
    lines += ["", "Hazard of one warning through the engine's formula, and what it revokes under the engine "
              "rules (channel 1 alone, then mixed by a positive rating's routing (0.8, 0.2) and a negative "
              "rating's (0.2, 0.8)):", ""] + _hazard_table(sw)
    reach = sw["witnesses_to_reach"]
    lines += ["", f"Fully reliable witnesses needed for lambda^1 >= {reach['target']:g}: {reach['n']} "
              f"(lambda^1 = {', '.join(f'{x:.3f}' for x in reach['lam1_by_count'])} for 1 to "
              f"{len(reach['lam1_by_count'])} witnesses)."]
    lines += ["", f"## CPPR seed-level statistics ({PRE_REVISION_TITLE})", ""]
    lines += _by_network(result, "cppr_seed_stats", CPPR_ROWS)
    lines += ["", "## Notes for the paper", ""] + paper_notes(result)
    return "\n".join(lines) + "\n"

