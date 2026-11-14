"""Generate the revision's supplement tables (results/paper/tables/<name>.tex) and the side file
results/paper/revision_supp_tables.json from stored result files only.  No number is typed by hand.

    python -m keeper.paperkit.make_revision_supp_tables [--only NAME ...] [--check]

A table whose source file does not exist yet is skipped with a message (exit code 0), so the generator can be
rerun as results arrive.  `--check` regenerates in memory and exits 1 when any output on disk differs.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX_DIR = "results/paper/tables"
VALUES = "results/paper/revision_supp_tables.json"
E1, LAUND, TIERS, E9 = ("results/analysis/" + n for n in ("e1.json", "laundering_extra.json",
                                                         "tiers_report.json", "e9a_e8.json"))
NETS = {"bitcoinalpha": "Bitcoin-Alpha", "bitcoinotc": "Bitcoin-OTC", "wiki-rfa": "Wiki-RfA",
        "epinions-dated": "Epinions-Dated"}
POLICY = {"arrival": "arrival", "exponential": "exponential", "erlang2": "Erlang-2", "erlang3": "Erlang-3",
          "erlang5": "Erlang-5", "quarantine": "quarantine", "no_commit": "no commit"}
WITNESS = ("off", "capped", "uncapped")
MB_TEXT = (
    r"Witnesses are unweighted (\emph{off}, the stored rule) or weighted by their BTP credibility tier "
    r"with (\emph{capped}) or without (\emph{uncapped}) the per-edge cap."
)


class Missing(Exception):
    """A source file that does not exist yet."""


# ------------------------------------------------------------------ cells and formatting

class C:
    """One table cell: the text printed, the underlying value recorded in the side file."""
    def __init__(self, name, text, value=None):
        self.name, self.text, self.value = name, text, text if value is None else value


def num(x, dec=3, signed=False):
    """Fixed decimals; a negative value or an explicit sign goes in math mode; None prints n/a."""
    if x is None:
        return "n/a"
    s = f"{abs(x):.{dec}f}"
    if float(s) == 0:
        return s
    if x < 0:
        return f"$-{s}$"
    return f"$+{s}$" if signed else s


def small(x, dec=4):
    """Like num, but a nonzero magnitude below 1e-3 prints as a power of ten."""
    if x is None or x == 0 or abs(x) >= 1e-3:
        return num(x, dec)
    m, e = f"{abs(x):.1e}".split("e")
    return f"${'-' if x < 0 else ''}{m}\\times10^{{{int(e)}}}$"


def pct(x, dec=1):
    return "n/a" if x is None else f"{100 * x:.{dec}f}"


def pval(p):
    return "n/a" if p is None else (r"$<10^{-4}$" if p < 1e-4 else f"{p:.3g}")


def cn(name, x, dec=3, signed=False):
    return C(name, num(x, dec, signed), x)


def cs(name, x, dec=4):
    return C(name, small(x, dec), x)


def cp(name, x, dec=1):
    return C(name, pct(x, dec), x)


def cq(name, x, dec=3):
    """A share printed in percent with dec decimals (a negative value in math mode); None prints n/a."""
    return C(name, num(None if x is None else 100 * x, dec), x)


def two_level(left, groups, sub):
    """Header lines: the group names over their sub-columns, a rule under each group, then the column names."""
    top = " & " * len(left) + " & ".join(rf"\multicolumn{{{len(sub)}}}{{c}}{{\textbf{{{g}}}}}" for g in groups)
    rules = "".join(rf"\cmidrule(lr){{{len(left) + 1 + i * len(sub)}-{len(left) + (i + 1) * len(sub)}}}"
                    for i in range(len(groups)))
    return [top + r" \\", rules] + header(*left, *(sub * len(groups)))


def cci(name, est, ci, dec=3, signed=False, scale=1.0):
    """'est [lo, hi]'; scale 100 turns a share into percent."""
    f = (lambda v: pct(v, dec)) if scale == 100 else (lambda v: num(v, dec, signed))
    return C(name, f"{f(est)} [{f(ci[0])}, {f(ci[1])}]", [est, ci[0], ci[1]])


def lab(name, text):
    return C(name, text)


class Float:
    """One float: caption, tabular column spec, header lines, body rows (lists of C, or None for a rule)."""
    def __init__(self, suffix, caption, colspec, header, rows, wide=True, size="scriptsize"):
        self.suffix, self.caption, self.colspec, self.header = suffix, caption, colspec, header
        self.rows, self.wide, self.size = rows, wide, size

    def render(self, name):
        env = "table*" if self.wide else "table"
        out = [rf"\begin{{{env}}}[!t]", r"\centering", rf"\{self.size}", r"\setlength{\tabcolsep}{3pt}",
               rf"\caption{{{self.caption}}}", rf"\label{{tab:s_{name}{self.suffix}}}",
               rf"\begin{{tabular}}{{{self.colspec}}}", r"\toprule", *self.header, r"\midrule"]
        for r in self.rows:
            out.append(r"\midrule" if r is None else " & ".join(c.text for c in r) + r" \\")
        return "\n".join(out + [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}", ""])

    def cells(self):
        return {"|".join(str(c.value if isinstance(c.value, str) else c.text) for c in r[:self.keylen]):
                {c.name: {"value": c.value, "text": c.text} for c in r} for r in self.rows if r}

    keylen = 1


def header(*names):
    return [" & ".join(rf"\textbf{{{n}}}" for n in names) + r" \\"]


def load(root, rel, sources):
    path = Path(root) / rel
    if not path.exists():
        raise Missing(rel)
    sources.add(rel)
    return json.loads(path.read_text())


def present(d):
    return [k for k in NETS if k in d]


def netname(k):
    return NETS[k]


# ------------------------------------------------------------------ 1. release policies

RELEASE_COLUMNS = (("L1", "latency_age1"), ("SSR", "silent_sender_revocation"),
                   ("FO", "silent_sender_revocation_future_only"), ("HMR", "honest_mass_revoked"))


def t_release_policies(root, src):
    """One float: a row per release policy and witness setting, the four columns once per network."""
    d = load(root, E1, src)["e1a"]
    nets = present(d)
    rows = []
    for p in d[nets[0]]["metrics"]:
        for w in WITNESS:
            r = [lab("policy", POLICY[p]), lab("witnesses", w)]
            r += [cp(f"{ds}_{key}", d[ds]["metrics"][p][w][key]) for ds in nets for _, key in RELEASE_COLUMNS]
            rows.append(r)
        rows.append(None)
    rows.pop()
    cap = (
        r"Release policies of the streaming engine without attack, at the same mean delay of four snapshots: "
        r"arrival, exponential, Erlang-$N$ ($N = 2, 3, 5$), quarantine and never commit, each under one "
        r"witness setting. "
        + MB_TEXT
        + r" Per network, in percent: L1, mass durable (released from escrow) at age one snapshot; SSR, revoked "
          r"mass of senders flagged later; FO, the same for senders first warned after the rating; HMR, revoked "
          r"mass of other senders. The main paper's release-policy table shows the capped rows."
    )
    h = two_level(["Policy", "Witnesses"], [netname(ds) for ds in nets], [n for n, _ in RELEASE_COLUMNS])
    f = Float("", cap, "@{}ll" + "rrrr" * len(nets) + "@{}", h, rows)
    f.keylen = 2
    return [f]


# ------------------------------------------------------------------ 2. witness attack

WITNESS_LAG = 1   # the registered lag of H4; the cells at lags 0, 3, 7 and 14 stay in results/analysis/e1.json


def t_witness_attack(root, src):
    """One float: a row per lag-1 attack (k, g, ring, farming), three columns per network."""
    d = load(root, E1, src)["e1b"]
    nets = present(d)
    cell = {(ds, r["k"], r["g"], r["ring"], r["farming"], r["mb"]): r["total"] for ds in nets for r in d[ds]
            if r["lag"] == WITNESS_LAG}
    yn = lambda b: "yes" if b else "no"
    rows = []
    for k, g, ring, farming in sorted({key[1:5] for key in cell}):
        r = [C("k", str(k)), C("g", str(g)), C("ring", yn(ring)), C("farming", yn(farming))]
        for ds in nets:
            off, on = (cell.get((ds, k, g, ring, farming, mb)) or {} for mb in ("off", "on"))
            r += [cq(f"{ds}_off_mean", off.get("mean")), cq(f"{ds}_on_mean", on.get("mean")),
                  cq(f"{ds}_on_p95", on.get("p95"))]
        rows.append(r)
    cap = (
        r"Fake witnesses against an honest warning one snapshot after it (lag $1$), under SCEM's two-stage "
        r"release. Per row, $k$ fakes warn about the honest rater, $g$ credible honest accounts endorse them, "
        r"Ring marks fakes linked by a positive ring, and Farm marks fakes that first build reliability by "
        r"rating one another. Per network: the share of the honest warning revoked with minus without the "
        r"attack, in percent, as the mean with unweighted (Off) and tier-weighted (On) witnesses and as the "
        r"95th percentile with weighting (On p95). Other lags: released result files."
    )
    h = two_level(["$k$", "$g$", "Ring", "Farm"], [netname(ds) for ds in nets], ["Off", "On", "On p95"])
    f = Float("", cap, "@{}rrll" + "rrr" * len(nets) + "@{}", h, rows)
    f.keylen = 4
    return [f]


# ------------------------------------------------------------------ 3. window (H5)

def _blocks(node):
    return node["blocks"] if "blocks" in node else node


def _window_designs(d):
    """(label, blocks) per commitment-window design: h5 holds the single-record study, the registered test of H5,
    and h5_batched the descriptive batched one (results/analysis/e1.json)."""
    return [("Single record", _blocks(d["h5"])), ("Batched", _blocks(d["h5_batched"]))]


def t_window(root, src):
    d = load(root, E1, src)
    designs = _window_designs(d)
    first = designs[0][1]
    labels = sorted({k.split("/")[0] for k in first}, key=int)
    floats = []
    for lbl in labels:
        rows = []
        keys = [k for k in first if k.startswith(lbl + "/")]
        for key in keys:
            ds = key.split("/")[1]
            for p, b0 in first[key].items():
                r = [lab("network", netname(ds)), lab("policy", POLICY[p]), cn("A", b0["A"], 2)]
                for dl, blocks in designs:
                    b = blocks.get(key, {}).get(p)
                    r += [C(f"{dl}_eligible", "n/a" if b is None else str(b["n_eligible"]),
                            None if b is None else b["n_eligible"]),
                          cs(f"{dl}_max_abs_diff", None if b is None else b["max_abs"]),
                          lab(f"{dl}_status", "n/a" if b is None else b.get("status", "n/a"))]
                r.append(cn("never_commit_mean", (b0["no_commit"] or {}).get("mean")))
                rows.append(r)
        multi = len(designs) > 1
        cap = (
            rf"Commitment-window check (H5) on the {lbl}-snapshot stream, per network and committing release policy; $A$: commitment age in snapshots. Per design ("
            + " and ".join(f"\\emph{{{dl}}}" for dl, _ in designs)
            + r"): Elig., eligible victim records (witness lag plus witness life at least $A$, closing inside the "
              r"stream); Max, largest absolute change of the revoked share caused by the late witnesses (the exact "
              r"property requires at most $10^{-12}$); Status, holds, defect or untested (no eligible record). "
              r"Never commit: mean change under the never-commit policy on the first design's records. "
            + ("Single record attacks one record at a time; Batched releases all late witnesses of a snapshot together and is descriptive, without status." if multi else "")
        )
        top = [r" & & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{dl}}}" for dl, _ in designs) + r" & \\"]
        cm = "".join(rf"\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(designs)))
        h = top + [cm] + [" & ".join(["\\textbf{Network}", "\\textbf{Policy}", "\\textbf{$A$}"]
                                     + ["\\textbf{Elig.}", "\\textbf{Max}", "\\textbf{Status}"] * len(designs)
                                     + ["\\textbf{Never commit}"]) + r" \\"]
        f = Float(f"_{lbl}", cap, "@{}llr" + "rrl" * len(designs) + "r@{}", h, rows)
        f.keylen = 2
        floats.append(f)
    return floats


# ------------------------------------------------------------------ 4. granularity

def t_granularity(root, src):
    d = load(root, E1, src)["e1d"]
    rows = []
    for ds in present(d):
        for cut in sorted(d[ds], key=int):
            e = d[ds][cut]
            m = e["metrics"]["erlang2"]
            rows.append([lab("network", netname(ds)), C("snapshots", cut, int(cut)),
                         cn("days_per_snapshot", e["days_per_snapshot"], 1),
                         cn("age_days", e["policies"]["erlang2"]["horizon_days"], 0),
                         cp("ssr_off", m["off"]["silent_sender_revocation"]),
                         cp("ssr_capped", m["capped"]["silent_sender_revocation"]),
                         cp("hmr_off", m["off"]["honest_mass_revoked"]),
                         cp("hmr_capped", m["capped"]["honest_mass_revoked"])])
    cap = (
        r"Granularity of the stream: each network cut into 22, 44 or 88 snapshots, Erlang-2 release. Days: "
        r"snapshot length in days; Age: commitment age in days. SSR, HMR: revoked share (percent) of the "
        r"rating mass of silent senders (default rule) and of honest senders, with witnesses off (stored rule) or weighted "
        r"by capped BTP tier."
    )
    h = [r"\textbf{Network} & \textbf{Snapshots} & \textbf{Days} & \textbf{Age} & \multicolumn{2}{c}{\textbf{SSR (\%)}}"
         r" & \multicolumn{2}{c}{\textbf{HMR (\%)}} \\", r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r" & & & & off & capped & off & capped \\"]
    f = Float("", cap, "@{}lrrrrrrr@{}", h, rows)
    f.keylen = 2
    return [f]


# ------------------------------------------------------------------ 5-7. laundering re-analysis

YARDSTICKS = (("laplace", "Laplace"), ("raw", "Raw"), ("base_rate", "Base rate"), ("logreg", "Logreg"))


def t_yardsticks(root, src):
    d = load(root, LAUND, src)
    rows = []
    for ds in present(d):
        a, s = d[ds]["amplification"], d[ds]["amp_split"]
        r = [lab("network", netname(ds))]
        r += [cci(k, a[k], a["ci95"][k], 2) for k, _ in YARDSTICKS]
        r += [cci(k, s[k], s["ci95"][k], 2) for k in ("edgeless", "edge_keeping")]
        rows.append(r)
    cap = (
        r"Amplification of the retraining gain over four counting yardsticks: the mean gain divided by the "
        r"mean shift of a yardstick computed from the target's remaining ratings, namely Laplace (add-one "
        r"smoothed positive share), Raw (positive share), Base rate (a prior at the network's positive share) "
        r"or Logreg (a degree-feature logistic regression). Edgeless and Keeping split the Laplace "
        r"amplification into targets left without ratings and the others. Above 1: the gain exceeds the "
        r"yardstick's shift."
    )
    h = header("Network", *(n for _, n in YARDSTICKS), "Edgeless", "Keeping")
    return [Float("", cap, "@{}lrrrrrr@{}", h, rows)]


def t_partial_deletion(root, src):
    d = load(root, LAUND, src)
    ks = (("k1_of_2", "1 of 2"), ("k1_of_3", "1 of 3"), ("k2_of_3", "2 of 3"))
    rows_g, rows_s = [], []
    for ds in present(d):
        p = d[ds]["partial_deletion_u1"]
        rows_g.append([lab("network", netname(ds))]
                      + [cci(k, p[k], p["ci95"][k]) for k, _ in ks]
                      + [cci(k, p[k], p["ci95"][k]) for k in ("full_of_2", "full_of_3")])
        rows_s.append([lab("network", netname(ds))]
                      + [cci(k, p["share_of_full"][k], p["share_of_full"]["ci95"][k], scale=100) for k, _ in ks])
    cap_g = (
        r"Partial deletion of warnings under inference-only unlinking (U1): mean gain in platform percentile "
        r"(as a fraction of the platform) after deleting 1 of 2, 1 of 3 and 2 of 3 warnings, and after "
        r"deleting all of them (Full) for targets with two and with three warnings."
    )
    cap_s = (
        r"Share of the full deletion gain reached by partial deletion (U1): mean over seeds of the partial "
        r"gain divided by the same targets' full gain, in percent, after deleting 1 of 2, 1 of 3 and 2 of 3 "
        r"warnings."
    )
    return [Float("_gain", cap_g, "@{}lrrrrr@{}", header("Network", *(n for _, n in ks), "Full of 2", "Full of 3"),
                  rows_g),
            Float("_share", cap_s, "@{}lrrr@{}", header("Network", *(n for _, n in ks)), rows_s)]


def t_percentiles(root, src):
    d = load(root, LAUND, src)
    rows_p, rows_c, rows_w = [], [], []
    for ds in present(d):
        e, pc, t = d[ds], d[ds]["percentile"], d[ds]["tau_mcc"]
        ci = pc["ci95"]
        cw = d[ds]["cohort_weighted"]
        rows_w.append([lab("network", netname(ds)), cp("share_pos", cw["cohort_share_pos"])]
                      + [cci(k, cw[k], cw["ci95"][k], dec=1, scale=100) for k in
                         ("pct_before", "pct_after_u1", "pct_after_u2", "pct_after_u2_nopos", "pct_after_u2_pos")]
                      + [cci("gain_u2", cw["gain_u2"], cw["ci95"]["gain_u2"], signed=True)])
        rows_p.append([lab("network", netname(ds)),
                       cci("before", pc["before_mean"], ci["before_mean"], scale=100),
                       cci("after_u1", pc["after_mean_u1"], ci["after_mean_u1"], scale=100),
                       cci("after_u2", pc["after_mean_u2"], ci["after_mean_u2"], scale=100),
                       cci("leave_bottom5", pc["escape_bottom5"], ci["escape_bottom5"], scale=100),
                       cci("leave_bottom10", pc["escape_bottom10"], ci["escape_bottom10"], scale=100)])
        x, q = e["edgeless"], e["edgeless"]
        rows_c.append([lab("network", netname(ds)),
                       C("tau_range", f"{t['range'][0]:.3f} to {t['range'][1]:.3f}", t["range"]),
                       C("cohort", str(x["cohort"]), x["cohort"]), C("n_edgeless", str(x["n"]), x["n"]),
                       cp("edgeless_share", x["share"]),
                       cci("cold_start_pct", q["cold_start_pct_mean"], q["ci95"]["cold_start_pct_mean"],
                           scale=100)])
    cap_w = (
        r"Cohort-weighted laundering outcomes. The sample is stratified (about half endorsed), so each cell "
        r"weights the means of endorsed and other targets by their cohort shares (Endorsed, percent): "
        r"platform percentile before deletion, after inference-only unlinking (U1) and after full retraining "
        r"(U2); the percentile after U2 without and with an endorsement; and the gain under U2."
    )
    cap_p = (
        r"Platform percentile (percent of active accounts with a lower reputation) of the sampled laundered "
        r"targets, as plain means over the stratified sample (cohort-weighted: previous table): before "
        r"deletion, after U1 and after U2, and the share of targets that start in the lowest 5\% (Leave 5\%) "
        r"or lowest 10\% (Leave 10\%) and leave it after U1."
    )
    cap_c = (
        r"Decision threshold and targets left without ratings. Tau: range over seeds of the MCC-maximizing "
        r"threshold. Cohort: targets in the cohort; Edgeless: number and share (percent) left without any "
        r"rating after deletion; Cold start: their common post-deletion platform percentile (percent)."
    )
    return [Float("_cohort", cap_w, "@{}lrrrrrrr@{}", header("Network", "Endorsed", "Before", "After U1", "After U2",
                                                             "U2, no endors.", "U2, endorsed", "Gain U2"), rows_w),
            Float("_percentile", cap_p, "@{}lrrrrr@{}", header("Network", "Before", "After U1", "After U2",
                                                              "Leave 5\\%", "Leave 10\\%"), rows_p),
            Float("_edgeless", cap_c, "@{}lrrrrr@{}", header("Network", "Tau range", "Cohort", "Edgeless $n$",
                                                            "Edgeless (\\%)", "Cold start (\\%)"), rows_c)]


# ------------------------------------------------------------------ 8. credibility layer

METHODS = (("bfs", "Legacy BFS"), ("uncapped", "BTP uncapped"), ("capped", "BTP capped"),
           ("uncapped_neg_all", "BTP uncapped, all raters"), ("capped_neg_all", "BTP capped, all raters"))


def t_credibility(root, src):
    d = load(root, TIERS, src)["networks"]
    rows_a, rows_t, rows_d = [], [], []
    for ds in present(d):
        n = d[ds]
        for v in ("capped", "uncapped"):
            x = n["variants"][v]
            c, probe = x["calibration"], x["star_size_probe"]
            adm = probe["max_admitted"]
            rows_a.append([lab("network", netname(ds)), lab("variant", v),
                           cn("eps", c["eps"], 2), C("K", str(c["K"]), c["K"]),
                           C("n_ref", str(c["n_ref"]), c["n_ref"]), C("n_seeds", str(c["n_seeds"]), c["n_seeds"]),
                           cp("retention", x["overlap"]["retention"]),
                           C("guarantee", "n/a" if probe["constant"] is None else str(probe["constant"]),
                             probe["constant"]),
                           C("max_admitted", (r"$\ge$ " if probe["lower_bound"] else "") + str(adm), adm)])
        for rule, shares in (("legacy", n["legacy"]["tier_shares"]), ("capped BTP", x_shares(n))):
            rows_t.append([lab("network", netname(ds)), lab("rule", rule)]
                          + [cp(f"tier{i}", s) for i, s in enumerate(shares)])
        own = n["demotion_census"]["own"]["methods"]
        for m, mname in METHODS:
            r = {row["k"]: row for row in own[m]["rows"]}
            rows_d.append([lab("network", netname(ds)), lab("method", mname)]
                          + [cp(f"dropped_k{k}", r[k]["share_dropped"]) for k in (1, 5, 10)]
                          + [cp(f"tier0_k{k}", r[k]["share_to_tier0"]) for k in (1, 5, 10)])
    cap_a = (
        r"Calibration and admission guarantee of the credibility layer, per network and BTP variant: capped "
        r"(KEEPER's default; an endorsement carries at most a fixed multiple of the per-step trust) or "
        r"uncapped. $\varepsilon$: tolerance chosen on honest data; $K$: propagation steps; $n_\mathrm{ref}$: "
        r"reference accounts; Seeds: seed accounts; Retention: percent of legacy Tier $\ge 1$ accounts kept "
        r"in Tier $\ge 1$; Guarantee: proved bound on fakes admitted per attack edge; Max: largest star of "
        r"fakes admitted in the probe ($\ge$: lower bound, the probe did not reach the end of the star)."
    )
    cap_t = (
        r"Tier shares of active accounts (percent) under the legacy breadth-first tiers or capped BTP. Tier 0 "
        r"holds unendorsed, possibly fake accounts; Tiers 1 to 3 are the terciles of the endorsed accounts."
    )
    cap_d = (
        r"Demotion census: free fake accounts bad-mouth a credible victim, per network and tiering method "
        r"(BTP variants as in the calibration table; ``all raters'' counts every rater's negative ratings, "
        r"the default only trusted raters'). For $k = 1, 5, 10$ fresh accounts each rating one of 100 victims "
        r"from the method's own Tier $\ge 2$: Drop, percent of victims whose tier falls; Tier 0, percent that "
        r"land in Tier 0."
    )
    ks = ["$k{=}1$", "$k{=}5$", "$k{=}10$"]
    h_d = [r"\textbf{Network} & \textbf{Method} & \multicolumn{3}{c}{\textbf{Drop (\%)}} & "
           r"\multicolumn{3}{c}{\textbf{Tier 0 (\%)}} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r" & & " + " & ".join(ks * 2) + r" \\"]
    fa = Float("_calibration", cap_a, "@{}llrrrrrrr@{}",
               header("Network", "Variant", "$\\varepsilon$", "$K$", "$n_\\mathrm{ref}$", "Seeds",
                      "Retention (\\%)", "Guarantee", "Max"), rows_a)
    ft = Float("_tiers", cap_t, "@{}llrrrr@{}", header("Network", "Rule", "Tier 0", "Tier 1", "Tier 2", "Tier 3"),
               rows_t, wide=False)
    fd = Float("_demotion", cap_d, "@{}llrrrrrr@{}", h_d, rows_d)
    for f in (fa, ft, fd):
        f.keylen = 2
    return [fa, ft, fd]


def x_shares(n):
    return n["variants"]["capped"]["btp"]["tier_shares"]


# ------------------------------------------------------------------ 9. self term (H8) and cold start (E8)

def t_self_term(root, src):
    d = load(root, E9, src)
    h8, sec, e8 = d["hypotheses"]["H8"]["networks"], d["secondary"], d["e8"]
    nets = present(h8)
    rows_h = []
    for ds in nets:
        a, b = h8[ds]["components"]["a"], h8[ds]["components"]["b"]
        rows_h.append([lab("network", netname(ds)),
                       cci("gain_on", a["estimate"], a["ci95"], signed=True), C("p_gain", pval(a["p_value"]),
                                                                                  a["p_value"]),
                       lab("decision_gain", a["decision"]),
                       cci("ratio_on_off", b["estimate"], b["ci95"], 2), C("p_ratio", pval(b["p_value"]),
                                                                          b["p_value"]),
                       lab("reading_ratio", b["reading"])])
    rows_s, rows_m = [], []
    for ds in nets:
        for g, gname in (("drawn", "50 drawn"), ("ge4", "$\\ge 4$ warnings")):
            for t in ("off", "on"):
                q = lambda k: sec[ds][f"{k}_{g}_{t}"]
                rows_s.append([lab("network", netname(ds)), lab("targets", gname), lab("self_term", t),
                               cci("gain", q("gain")["estimate"], q("gain")["ci95"], signed=True),
                               cci("pct_before", q("pct_before")["estimate"], q("pct_before")["ci95"], scale=100),
                               cci("pct_after", q("pct_after")["estimate"], q("pct_after")["ci95"], scale=100)])
        m = {t: sec[ds][f"base_t1_mcc_{t}"] for t in ("off", "on")}
        rows_m.append([lab("network", netname(ds))] + [cci(f"mcc_{t}", m[t]["estimate"], m[t]["ci95"])
                                                         for t in ("off", "on")])
    rows_e, rows_l = [], []
    for ds in nets:
        s = e8[ds]["summary"]
        ci = lambda k: (s[k]["estimate"], s[k]["ci95"])
        rows_e.append([lab("network", netname(ds)), lab("m", "none (uncalibrated)"),
                       cci("drawn", *ci("laundering_pct_drawn_uncalibrated"), scale=100),
                       cci("ge4", *ci("laundering_pct_ge4_uncalibrated"), scale=100), lab("newcomer", "n/a")])
        for m_ in d["settings"]["ms"]:
            rows_e.append([lab("network", netname(ds)), C("m", str(m_), m_),
                           cci("drawn", *ci(f"laundering_pct_drawn_m{m_}"), scale=100),
                           cci("ge4", *ci(f"laundering_pct_ge4_m{m_}"), scale=100),
                           cci("newcomer", *ci(f"newcomer_change_m{m_}"), signed=True, scale=100)])
            rows_l.append([lab("network", netname(ds)), C("m", str(m_), m_),
                           cci("mcc_base", *ci(f"low_degree_mcc_base_m{m_}")),
                           cci("mcc_calibrated", *ci(f"low_degree_mcc_calibrated_m{m_}")),
                           cci("delta_mcc", *ci(f"low_degree_delta_mcc_m{m_}"), signed=True)])
    cap_h = (
        r"Self-term test (H8) over the 50 drawn targets. Gain: mean laundering gain (rise in platform "
        r"percentile, as a fraction of the platform) with the self term on, with its one-sided $p$-value; "
        r"Decision is met when its interval excludes 0. Ratio: gain with the self term on divided by the gain "
        r"with it off, with its $p$-value; Reading is met when the ratio's lower bound is at least 0.5, and "
        r"inconclusive otherwise."
    )
    cap_s = (
        r"Secondary outcomes of the self-term study, per network, target group (the 50 drawn targets, or the "
        r"targets with at least four warnings) and self term (off, on). Gain: rise in platform percentile as "
        r"a fraction of the platform; Before, After: platform percentile (percent) before and after deletion."
    )
    cap_m = (
        r"Test MCC of the base model with the self term off and on."
    )
    cap_e = (
        r"Cold-start analysis (E8), laundered targets, per network and calibration: none, or $m = 1, 2, 4$ "
        r"prior pseudo-ratings. Platform percentile (percent) of the 50 drawn laundered targets and of the "
        r"targets with at least four warnings, and change of the newcomers' percentile in percentage points."
    )
    cap_l = (
        r"Cold-start analysis (E8), low-degree accounts: MCC on low-degree accounts of the base model and of "
        r"the model calibrated with strength $m$, and their difference (Delta)."
    )
    fl = [Float("_h8", cap_h, "@{}lrrlrrl@{}", header("Network", "Gain", "$p$", "Decision", "Ratio", "$p$",
                                                       "Reading"), rows_h),
          Float("_secondary", cap_s, "@{}lllrrr@{}", header("Network", "Targets", "Self term", "Gain", "Before (\\%)",
                                                           "After (\\%)"), [r for r in rows_s if r]),
          Float("_mcc", cap_m, "@{}lrr@{}", header("Network", "Self term off", "Self term on"), rows_m, wide=False),
          Float("_e8_percentile", cap_e, "@{}llrrr@{}", header("Network", "$m$", "50 drawn (\\%)",
                                                              "$\\ge 4$ warnings (\\%)", "Newcomers"), rows_e),
          Float("_e8_mcc", cap_l, "@{}lrrrr@{}", header("Network", "$m$", "MCC base", "MCC calibrated",
                                                       "$\\Delta$ MCC"), rows_l)]
    for f in fl:
        f.keylen = 3 if f.suffix == "_secondary" else 2 if f.suffix in ("_e8_percentile", "_e8_mcc") else 1
    return fl


TABLES = {"s_release_policies": t_release_policies, "s_witness_attack": t_witness_attack, "s_window": t_window,
          "s_granularity": t_granularity, "s_yardsticks": t_yardsticks, "s_partial_deletion": t_partial_deletion,
          "s_percentiles": t_percentiles, "s_credibility": t_credibility, "s_self_term": t_self_term}


# ------------------------------------------------------------------ driver

def build_one(root, name):
    """(tex, side-file entry) of one table; raises Missing when a source file does not exist yet."""
    src = set()
    floats = TABLES[name](root, src)
    tex = "\n".join(f.render(name) for f in floats)
    sha = {f: hashlib.sha256((Path(root) / f).read_bytes()).hexdigest() for f in sorted(src)}
    cells = {f"tab:s_{name}{f.suffix}": f.cells() for f in floats}
    return tex, {"sources": sha, "cells": cells}


def build(root=ROOT, only=None):
    """({name: tex}, {name: entry}, [skipped names with the missing file])."""
    texs, entries, skipped = {}, {}, []
    for name in only or TABLES:
        try:
            texs[name], entries[name] = build_one(root, name)
        except Missing as e:
            skipped.append((name, str(e)))
    return texs, entries, skipped


def dump(entries):
    return json.dumps({"generator": "keeper/paperkit/make_revision_supp_tables.py", "tables": entries},
                      indent=1, sort_keys=True) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(TABLES), metavar="NAME")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    root = Path(args.root)
    texs, entries, skipped = build(root, args.only)
    for name, rel in skipped:
        print(f"skipped {name}: source {rel} does not exist yet", file=sys.stderr)
    vpath = root / VALUES
    old = json.loads(vpath.read_text())["tables"] if vpath.exists() else {}
    merged = {**old, **entries}
    if args.check:
        bad = [f"{n}.tex" for n, t in texs.items() if not (root / TEX_DIR / f"{n}.tex").exists()
               or (root / TEX_DIR / f"{n}.tex").read_text() != t]
        bad += [VALUES] if any(old.get(n) != e for n, e in entries.items()) else []
        if bad:
            print("STALE:", ", ".join(bad), file=sys.stderr)
            return 1
        print("revision supplement tables up to date")
        return 0
    (root / TEX_DIR).mkdir(parents=True, exist_ok=True)
    for n, t in texs.items():
        (root / TEX_DIR / f"{n}.tex").write_text(t)
    vpath.parent.mkdir(parents=True, exist_ok=True)
    vpath.write_text(dump(merged))
    print("wrote " + ", ".join(sorted(texs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
