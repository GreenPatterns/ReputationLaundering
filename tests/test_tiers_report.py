"""keeper.analysis.tiers_report, the --report of keeper.credibility.tiers (Task 3.1; rulings R33b, R37, R38, R56).

The CLI refuses to overwrite its output before any work (ruling R27). On toys: the Sybil rows and the
max_trust ranking by B(u), the cap-size sensitivity, the star-size probe (the attacker's best response,
pinned from below as well as above), the demotion census and the guard comparison of ruling R56, the
calibration table, and a whole network block of the four variants (capped and uncapped, each under the
credible-negatives guard and under the all-raters guard). On the three networks: the capped best star under
both guards. CPU only.
"""
import copy
import math

import numpy as np
import pytest

from keeper.credibility import tiers
from tests.credibility_common import (HONEST_N, NETWORKS, TOY_EPS, graph, honest_rows, max_final_trust_endorsers,
                                      max_inflow_endorsers, with_rows)

VARIANT_SPECS = {"uncapped": (None, "trusted"), "capped": (tiers.KEEPER_EDGE_CAP_UNITS, "trusted"),
                 "uncapped_neg_all": (None, "all"), "capped_neg_all": (tiers.KEEPER_EDGE_CAP_UNITS, "all")}


class WorkStarted(Exception):
    """Raised by the first step of the report: reaching it means the overwrite guard let the run go on."""


def _work_started(*args, **kwargs):
    raise WorkStarted


def test_report_cli_refuses_an_existing_output_before_its_work(tmp_path, monkeypatch):
    from keeper.analysis import tiers_report as TR
    from keeper.laundering import core as C
    monkeypatch.delenv(C.OVERWRITE_ENV, raising=False)
    monkeypatch.setattr(TR, "analyse", _work_started)
    out = tmp_path / "tiers_report.json"
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        tiers.main(["--report", "--out", str(out)])
    assert out.read_text() == "{}"
    with pytest.raises(WorkStarted):
        tiers.main(["--report", "--out", str(out), "--force"])
    with pytest.raises(SystemExit):                                   # nothing to do without --report
        tiers.main(["--out", str(tmp_path / "new.json")])


def test_report_sybil_rows_and_max_trust_endorsers_by_hand():
    from keeper.analysis import tiers_report as TR
    # Ring: attack edge a reaches fake a mod s. Star: every attack edge reaches the hub, which endorses the rest.
    assert TR.sybil_rows([7, 8], [10, 11, 12], "ring") == [(7, 10, 1), (8, 11, 1), (10, 11, 1), (11, 12, 1),
                                                           (12, 10, 1)]
    assert TR.sybil_rows([7, 8], [10, 11, 12], "star") == [(7, 10, 1), (8, 10, 1), (10, 11, 1), (10, 12, 1)]
    with pytest.raises(ValueError):
        TR.sybil_rows([7], [10, 11], "chain")
    # The ranking: the highest score first, ties to the smaller id.
    assert TR.top_accounts(np.array([0.1, 0.3, 0.1, 0.1]), 3).tolist() == [1, 0, 2]
    # Ruling R33b: max_trust ranks honest accounts by B(u) (keeper.credibility.tiers.endorser_inflow_scores).
    G = graph(honest_rows(0))
    cal = tiers.calibrate_btp(G, HONEST_N, TOY_EPS, edge_cap_units=tiers.KEEPER_EDGE_CAP_UNITS)
    assert TR.max_trust_endorsers(G, HONEST_N, cal, 5).tolist() == max_inflow_endorsers(
        honest_rows(0), 5, tiers.KEEPER_EDGE_CAP_UNITS)


@pytest.mark.parametrize("neg_from", ["trusted", "all"])
def test_report_cap_sensitivity_on_a_toy(neg_from):
    # For each cap size, the retention over the sensitivity grid, the eps the preregistered rule picks (the
    # standard grid, extended downward if needed) and the per-edge constant floor(K * edge_cap_units), under the
    # given guard (ruling R56). keep = 0.95 makes the rule extend its grid at edge_cap_units = 1 (the toy's capped
    # retention stays below under either guard).
    from keeper.analysis import tiers_report as TR
    G = graph(honest_rows(0))
    rows = TR.cap_sensitivity(G, HONEST_N, keep=0.95, units_grid=(1.0, 2.0), eps_grid=(0.5, 0.2, 0.05, 0.01),
                              neg_from=neg_from)
    K = tiers.calibrate_btp(G, HONEST_N, 0.2).K
    assert [row["edge_cap_units"] for row in rows] == [1.0, 2.0] and rows[0]["rule_extended"] is True
    for row in rows:
        units = row["edge_cap_units"]
        rule_eps, rule_table = tiers.choose_eps(G, HONEST_N, keep=0.95, edge_cap_units=units,
                                                fallback_grid=tiers.EPS_GRID_EXTENDED, neg_from=neg_from)
        assert (row["rule_eps"], row["rule_met"], row["rule_extended"]) == (rule_eps, rule_table["met"],
                                                                              rule_table["extended"])
        assert row["neg_from"] == neg_from and row["per_edge_constant"] == math.floor(K * units + TR.CAP_TOL)
        assert [r["eps"] for r in row["rows"]] == [0.5, 0.2, 0.05, 0.01]
        for r in row["rows"]:                                         # the rows of choose_eps with that cap
            assert r == tiers.choose_eps(G, HONEST_N, (r["eps"],), 0.95, edge_cap_units=units,
                                         neg_from=neg_from)[1]["rows"][0]
    # At eps = 0.5 the two guards retain different shares on this toy, so the guard does reach these rows.
    other = TR.cap_sensitivity(G, HONEST_N, keep=0.95, units_grid=(1.0,), eps_grid=(0.5,),
                               neg_from="all" if neg_from == "trusted" else "trusted")
    assert other[0]["rows"][0]["retention"] != rows[0]["rows"][0]["retention"]


def test_report_uncapped_variant_extends_its_eps_grid_when_needed():
    # Task 3.1b: the preregistered rule extends the eps grid to 0.02, 0.01 and 0.005 for every variant, not only the
    # capped ones. On this toy, with keep = 0.99, the uncapped rule meets the target only at eps = 0.02.
    from keeper.analysis import tiers_report as TR
    G = graph(honest_rows(3))
    v = TR.variant_block(G, HONEST_N, tiers.legacy_bfs(G, HONEST_N), None, keep=0.99, ring_g=(1,), ring_s=(3,), draws=1)
    assert (v["eps"], v["choose_eps"]["met"], v["choose_eps"]["extended"]) == (0.02, True, True)
    assert v["calibration"]["eps"] == 0.02 and v["neg_from"] == "trusted"


# The toy's best response: seed 9 saturates the cap from step 0 (B = K c_e), and its star of 8 fakes is
# admitted whole: 2K - 4 with K = 6, against the constant floor(2K) = 12. (On the three networks, with K = 12,
# 13 and 14, the best stars sit at 2K - 3.)
TOY_PROBE_MAX = 8


@pytest.mark.parametrize("neg_from", ["trusted", "all"])
def test_report_star_size_probe_on_a_toy(neg_from):
    # The attacker's best response under KEEPER's cap (R33b): one attack edge from each of the five accounts with
    # the largest B(u) into a star of every size from 2 to floor(K units) + 4. Admission never exceeds the cap or
    # the constant, and, pinned from below, the best star comes within 4 fakes of the constant. The fakes receive
    # no negative rating, so both guards (ruling R56) admit the same stars.
    from keeper.analysis import tiers_report as TR
    rows = honest_rows(0)
    G = graph(rows)
    cal = tiers.calibrate_btp(G, HONEST_N, TOY_EPS, edge_cap_units=tiers.KEEPER_EDGE_CAP_UNITS)
    probe = TR.star_size_probe(G, HONEST_N, cal, neg_from=neg_from)
    assert probe["neg_from"] == neg_from
    constant = math.floor(cal.K * cal.edge_cap_units + TR.CAP_TOL)
    assert probe["constant"] == constant and probe["sizes"] == list(range(2, constant + 5))
    assert probe["endorsers"] == max_inflow_endorsers(rows, TR.PROBE_ENDORSERS, tiers.KEEPER_EDGE_CAP_UNITS)
    for entry in probe["per_endorser"]:
        assert [row["s"] for row in entry["rows"]] == probe["sizes"] and entry["lower_bound"] is False
        for row in entry["rows"]:
            assert row["btp_admitted"] <= row["cap"] <= row["cap_constant"] == constant and row["bound_holds"]
            assert row["legacy_admitted"] is None                     # the probe skips the legacy pass
        best = max(entry["rows"], key=lambda row: row["btp_admitted"])
        assert (entry["max_admitted"], entry["best_s"]) == (best["btp_admitted"], best["s"])
    assert probe["max_admitted"] == max(entry["max_admitted"] for entry in probe["per_endorser"])
    # An independent count with keeper.credibility.tiers alone, for the best endorser and size.
    u, s = probe["argmax_endorser"], probe["argmax_s"]
    fakes = list(range(HONEST_N, HONEST_N + s))
    star = [(u, fakes[0], +1, 19)] + [(fakes[0], f, +1, 19) for f in fakes[1:]]
    levels = tiers.btp_tiers(graph(rows + star), HONEST_N + s, calibration=cal, neg_from=neg_from)[0]
    assert int((levels[fakes] > 0).sum()) == probe["max_admitted"] == TOY_PROBE_MAX
    assert constant - 4 <= probe["max_admitted"] <= constant


def test_report_inflow_ranked_endorser_admits_at_least_as_many_fakes():
    # R33b: the account with the largest B(u) admits at least as many fakes into its best star as the account
    # with the largest final trust per out-edge, t^(K)(u) / (dout(u) + 1), the ranking R33b replaced.
    from keeper.analysis import tiers_report as TR
    rows = honest_rows(0)
    G = graph(rows)
    cal = tiers.calibrate_btp(G, HONEST_N, TOY_EPS, edge_cap_units=tiers.KEEPER_EDGE_CAP_UNITS)
    sizes = range(2, math.floor(cal.K * cal.edge_cap_units + TR.CAP_TOL) + 5)
    by_inflow = TR.best_star(G, HONEST_N, cal, max_inflow_endorsers(rows, 1, cal.edge_cap_units)[0], sizes)
    by_final_trust = TR.best_star(G, HONEST_N, cal, max_final_trust_endorsers(rows, 1)[0], sizes)
    assert by_inflow["max_admitted"] >= by_final_trust["max_admitted"] > 0


def test_report_uncapped_probe_labels_size_limited_maxima_as_lower_bounds():
    # Uncapped caps run to hundreds, so the probe uses a sparse size grid. A best star admitted whole whose next
    # size was not probed may hide a larger best response, so it is labelled a lower bound ("at least").
    from keeper.analysis import tiers_report as TR
    G = graph(honest_rows(0))
    cal = tiers.calibrate_btp(G, HONEST_N, TOY_EPS)
    probe = TR.star_size_probe(G, HONEST_N, cal, sizes=(2, 5, 10, 20))
    assert probe["constant"] is None and probe["sizes"] == [2, 5, 10, 20]
    for entry in probe["per_endorser"]:
        expect = entry["max_admitted"] == entry["best_s"] and entry["best_s"] + 1 not in probe["sizes"]
        assert entry["lower_bound"] is expect
    assert any(entry["lower_bound"] for entry in probe["per_endorser"])      # the label is exercised
    overall = max(probe["per_endorser"], key=lambda entry: entry["max_admitted"])
    assert (probe["max_admitted"], probe["lower_bound"]) == (overall["max_admitted"], overall["lower_bound"])
    assert list(TR.PROBE_SIZES_UNCAPPED) == sorted(TR.PROBE_SIZES_UNCAPPED) and TR.PROBE_SIZES_UNCAPPED[0] == 2


def census_by_hand(Tr, n, methods, victims, k):
    """Per method: (victims whose tier drops, victims in Tier 0, other honest accounts whose tier moves, summed
    over the streams) when k fresh Sybils each rate one victim negatively in the last training snapshot, one victim
    per stream. The oracle of demotion_census."""
    base = {name: fn(Tr, n) for name, (_, fn) in methods.items()}
    out = {name: [0, 0, 0] for name in methods}
    for v in victims:
        A = with_rows(Tr, [(n + i, v, -1) for i in range(k)], int(np.max(Tr.snap)))
        for name, (_, fn) in methods.items():
            levels = fn(A, n + k)
            others = np.arange(n) != v
            out[name][0] += int(levels[v] < base[name][v])
            out[name][1] += int(levels[v] == 0)
            out[name][2] += int((levels[:n][others] != base[name][others]).sum())
    return out


@pytest.mark.parametrize("pool", ["own", "common"])
def test_report_demotion_census_on_a_toy(pool):
    # Ruling R56's census: k fresh Sybils per victim each rate it negatively once, one victim per stream. With pool
    # "own" each method's victims come from its own Tier >= 2 accounts; with "common" every method faces the same
    # victims, drawn from the accounts in Tier >= 2 under every method. Under the credible-negatives guard no
    # victim's tier and no other account's tier moves; under BFS and under the all-raters guard victims drop.
    from keeper.analysis import tiers_report as TR
    G = graph(honest_rows(0))
    methods = TR.census_methods(G, HONEST_N, [("capped", tiers.KEEPER_EDGE_CAP_UNITS, "trusted", TOY_EPS),
                                              ("capped_neg_all", tiers.KEEPER_EDGE_CAP_UNITS, "all", TOY_EPS)])
    assert list(methods) == ["bfs", "capped", "capped_neg_all"]
    assert [methods[name][0]["neg_from"] for name in methods] == ["all", "trusted", "all"]
    # The toy's credible accounts hold 6 to 11 trusted positive ratings, so 5 Sybils are needed to flip some.
    census = TR.demotion_census(G, HONEST_N, methods, k_grid=(1, 5), n_victims=6, seed=1, pool=pool)
    assert (census["pool"], census["k_grid"], census["n_victims"], census["seed"]) == (pool, [1, 5], 6, 1)
    base = {name: fn(G, HONEST_N) for name, (_, fn) in methods.items()}
    common = np.flatnonzero(np.all([levels >= 2 for levels in base.values()], axis=0))
    for name, entry in census["methods"].items():
        candidates = np.flatnonzero(base[name] >= 2) if pool == "own" else common
        victims = entry["victims"]
        assert entry["n_pool"] == len(candidates) and len(set(victims)) == 6 and victims == sorted(victims)
        assert victims == sorted(np.random.default_rng(1).choice(candidates, size=6, replace=False).tolist())
        for k in (1, 5):
            want = census_by_hand(G, HONEST_N, {name: methods[name]}, victims, k)[name]
            row = next(r for r in entry["rows"] if r["k"] == k)
            assert (row["n_dropped"], row["n_to_tier0"], row["collateral_total"]) == tuple(want), (name, k)
            assert row["share_dropped"] == row["n_dropped"] / 6 and row["n_victims"] == 6
    if pool == "common":                                              # one draw, the same victims for every method
        assert len({tuple(entry["victims"]) for entry in census["methods"].values()}) == 1
    trusted = census["methods"]["capped"]
    assert trusted["unchanged"] is True and all(r["n_dropped"] == r["collateral_total"] == 0 for r in trusted["rows"])
    for name in ("bfs", "capped_neg_all"):                            # pinned from below: the attack does work
        assert census["methods"][name]["unchanged"] is False
        assert next(r for r in census["methods"][name]["rows"] if r["k"] == 5)["n_to_tier0"] > 0, name
    with pytest.raises(ValueError):
        TR.demotion_census(G, HONEST_N, methods, k_grid=(1,), n_victims=6, seed=1, pool="all")


def test_report_guard_change_on_a_toy():
    # The honest Tier >= 1 accounts under the credible-negatives guard against the all-raters guard, each at its own
    # eps: the counts, the share whose tier differs, and the tier transitions over the active accounts.
    from keeper.analysis import tiers_report as TR
    G = graph(honest_rows(0))
    units = tiers.KEEPER_EDGE_CAP_UNITS
    change = TR.guard_change(G, HONEST_N, ("capped", units, "trusted", 0.2), ("capped_neg_all", units, "all", 0.5))
    t = tiers.btp_tiers(G, HONEST_N, 0.2, edge_cap_units=units, neg_from="trusted")[0]
    a = tiers.btp_tiers(G, HONEST_N, 0.5, edge_cap_units=units, neg_from="all")[0]
    union = (t > 0) | (a > 0)
    assert (change["eps_trusted"], change["eps_all"], change["same_eps"]) == (0.2, 0.5, False)
    assert (change["n_tier1_trusted"], change["n_tier1_all"], change["n_union"]) == (
        int((t > 0).sum()), int((a > 0).sum()), int(union.sum()))
    assert (change["n_trusted_only"], change["n_all_only"]) == (int(((t > 0) & (a == 0)).sum()),
                                                               int(((a > 0) & (t == 0)).sum()))
    assert change["n_tier_changed"] == int((union & (t != a)).sum()) > 0
    assert change["share_tier_changed"] == change["n_tier_changed"] / change["n_union"]
    assert change["share_tier1_membership_changed"] == (change["n_trusted_only"] + change["n_all_only"]) / change["n_union"]
    active = np.unique(G.edges)
    assert change["transitions"] == [[int(((a[active] == i) & (t[active] == j)).sum()) for j in range(4)]
                                     for i in range(4)]


@pytest.fixture(scope="module")
def toy_block():
    """One network block of the random toy, shared by the parametrized checks below."""
    from keeper.analysis import tiers_report as TR
    G = graph(honest_rows(0))
    return G, TR.network_block(G, HONEST_N, T_rows=21, ring_g=(1, 2), ring_s=(3, 10, 50), draws=2,
                               census_kw=dict(k_grid=(1, 5), n_victims=5))


def test_report_calibration_table_on_a_toy(toy_block):
    # One row per network and variant: the numbers the preregistration takes (eps, K, n_ref, theta0, the guarantee,
    # the best star) next to the share of honest Tier >= 1 accounts that the guard of ruling R56 moves.
    from keeper.analysis import tiers_report as TR
    _, block = toy_block
    rows = TR.calibration_table({"toy": block})
    assert [(r["network"], r["variant"]) for r in rows] == [("toy", name) for name, _, _ in TR.VARIANTS]
    for r in rows:
        v = block["variants"][r["variant"]]
        assert (r["role"], r["neg_from"], r["edge_cap_units"], r["eps"]) == (v["role"], v["neg_from"],
                                                                           v["edge_cap_units"], v["eps"])
        assert (r["met"], r["extended"], r["retention"]) == (v["choose_eps"]["met"], v["choose_eps"]["extended"],
                                                             v["overlap"]["retention"])
        assert {key: r[key] for key in ("K", "n_ref", "theta0", "edge_cap", "n_seeds")} == {
            key: v["calibration"][key] for key in ("K", "n_ref", "theta0", "edge_cap", "n_seeds")}
        assert (r["guarantee_per_edge"], r["best_star"]) == (v["star_size_probe"]["constant"],
                                                             v["star_size_probe"]["max_admitted"])
        change = block["guard_change"].get(r["variant"])
        assert r["share_tier_changed_vs_all"] == (None if change is None else change["share_tier_changed"])
    assert [r["share_tier_changed_vs_all"] is None for r in rows] == [False, False, True, True]
    # On this toy the guard moves no tier at the chosen eps, so both shares are 0; distinct values show that each
    # column reads its own field of guard_change.
    fake = copy.deepcopy(block)
    for i, change in enumerate(fake["guard_change"].values()):
        change.update(share_tier_changed=0.25 + i, share_tier1_membership_changed=0.5 + i)
    for r in TR.calibration_table({"toy": fake}):
        change = fake["guard_change"].get(r["variant"])
        if change is not None:
            assert (r["share_tier_changed_vs_all"], r["share_tier1_membership_changed_vs_all"]) == (
                change["share_tier_changed"], change["share_tier1_membership_changed"])


@pytest.mark.parametrize("variant", list(VARIANT_SPECS))
def test_report_network_block_on_a_toy(variant, toy_block):
    # KEEPER's default is capped BTP (tiers.KEEPER_EDGE_CAP_UNITS, ruling R38) under the credible-negatives guard
    # (ruling R56). The ablations drop the cap, the guard, or both (the last is Task 3.1's approved rule).
    from keeper.analysis import tiers_report as TR
    units, neg_from = VARIANT_SPECS[variant]
    assert TR.VARIANTS == tuple((name, u, nf) for name, (u, nf) in VARIANT_SPECS.items())
    G, block = toy_block
    assert set(block["variants"]) == set(VARIANT_SPECS)
    v = block["variants"][variant]
    assert (v["edge_cap_units"], v["neg_from"], v["role"]) == (units, neg_from, TR.variant_role(units, neg_from))
    assert TR.variant_role(tiers.KEEPER_EDGE_CAP_UNITS, "trusted") == "KEEPER default (rulings R38, R56)"
    eps, table = tiers.choose_eps(G, HONEST_N, edge_cap_units=units, fallback_grid=tiers.EPS_GRID_EXTENDED,
                                  neg_from=neg_from)
    assert v["eps"] == eps and v["choose_eps"] == table
    cal = tiers.calibrate_btp(G, HONEST_N, eps, edge_cap_units=units)
    assert v["calibration"] == dict(eps=cal.eps, K=cal.K, theta0=cal.theta0, n_ref=cal.n_ref, n_seeds=cal.n_seeds,
                                    t_seed=cal.t_seed, edge_cap_units=units, edge_cap=cal.edge_cap, neg_from=neg_from)
    active = np.unique(G.edges)
    levels = tiers.btp_tiers(G, HONEST_N, calibration=cal, neg_from=neg_from)[0]
    assert v["btp"]["neg_from"] == neg_from
    assert v["btp"]["tier_counts"] == np.bincount(levels[active], minlength=4).tolist()
    ov = v["overlap"]
    assert ov["n_both"] + ov["n_legacy_only"] == ov["n_legacy_tier1"] == block["legacy"]["n_endorsed"]
    assert ov["retention"] == ov["n_both"] / ov["n_legacy_tier1"]
    assert sum(v["btp"]["tier_counts"]) == block["legacy"]["n_active"] == v["btp"]["n_active"]
    causal_cal = tiers.calibrate_btp_causal(G, HONEST_N, eps, edge_cap_units=units)
    causal_levels = tiers.causal_tiers(G, HONEST_N, calibration=causal_cal, neg_from=neg_from)
    causal = v["causal"]
    assert causal["tier1_counts"] == (causal_levels > 0).sum(axis=1).tolist()
    assert len(causal["tier1_counts"]) == len(causal["K"]) == len(causal["theta0"]) == len(causal["n_ref"]) == 21
    test = v["sybil_test"]
    cases = test["cases"]
    assert len(cases) == (2 + 1) * 2 * 2 * 3                          # (2 draws + max_trust) x topologies x g x s
    assert {(c["selection"], c["topology"]) for c in cases} == {
        (selection, topology) for selection in ("random", "max_trust") for topology in ("ring", "star")}
    caps = {}
    for c in cases:
        assert c["btp_admitted"] <= c["btp_trusted"] <= c["cap"] == math.floor(c["bound_over_theta0"] + TR.CAP_TOL)
        assert c["sybil_trust"] <= c["bound"] + 1e-15 and c["bound_holds"] and c["cap_holds"]
        assert (c["theta0"], c["K"], c["n_ref"], c["edge_cap"]) == (cal.theta0, cal.K, cal.n_ref, cal.edge_cap)
        assert c["n_active"] == cal.n_ref + c["s"]
        for name in ("legacy", "btp", "cap"):
            assert c[f"{name}_per_edge"] == c["cap" if name == "cap" else f"{name}_admitted"] / c["g"]
        if units is None:
            assert c["cap_constant"] is None and c["cap_constant_holds"] is None
        else:                                                         # R37: floor(K g edge_cap_units)
            assert c["cap"] <= c["cap_constant"] == math.floor(cal.K * c["g"] * units + TR.CAP_TOL)
            assert c["cap_constant_holds"] is True
        if c["selection"] == "random":                                   # pool accounts are in D
            assert c["legacy_admitted"] == c["s"]
        caps.setdefault((c["selection"], c["draw"], c["g"]), set()).add(c["cap"])
    assert all(len(group) == 1 for group in caps.values())            # one cap per endorser set: any s, topology
    growth = test["uncalibrated_cap_growth"]
    assert [row["s"] for row in growth] == [3, 10, 50]
    if units is None:
        assert growth[0]["cap"] < growth[-1]["cap"]
    else:                                                             # the cap ties the count to K alone
        assert all(row["cap"] <= math.floor(row["K"] * units + TR.CAP_TOL) for row in growth)
    assert all(row["all_bounds_hold"] and row["all_caps_hold"] for row in test["summary"])
    assert all(row["all_cap_constants_hold"] is (None if units is None else True) for row in test["summary"])
    assert all(row["limited_by_region_size"] is (row["btp_admitted_max"] == row["s"]) for row in test["summary"])
    assert test["endorsers"]["max_trust"] == TR.max_trust_endorsers(G, HONEST_N, cal, 2).tolist()   # R33b
    probe = v["star_size_probe"]
    assert probe["endorsers"] == TR.max_trust_endorsers(G, HONEST_N, cal, TR.PROBE_ENDORSERS).tolist()
    if units is None:                                                 # sparse sizes: maxima may be lower bounds
        assert probe["constant"] is None and probe["sizes"] == list(TR.PROBE_SIZES_UNCAPPED)
    else:
        assert probe["max_admitted"] <= probe["constant"] == math.floor(cal.K * units + TR.CAP_TOL)
        assert probe["lower_bound"] is False
    # The network-level blocks of ruling R56: every variant enters the demotion census (both victim pools) under its
    # own calibration, each guarded variant is compared with its all-raters twin, and the cap sensitivity runs under
    # both guards.
    assert set(block["demotion_census"]) == set(TR.CENSUS_POOLS) == {"own", "common"}
    for pool, census in block["demotion_census"].items():
        entry = census["methods"][variant]
        assert census["pool"] == pool and set(census["methods"]) == {"bfs"} | set(VARIANT_SPECS)
        assert (entry["method"], entry["edge_cap_units"], entry["neg_from"], entry["eps"]) == ("btp", units, neg_from,
                                                                                              eps)
        if neg_from == "trusted":
            assert entry["unchanged"] is True                         # the exact property: no tier moves
    if neg_from == "trusted":
        twin = next(name for name, (u, nf) in VARIANT_SPECS.items() if u == units and nf == "all")
        assert (block["guard_change"][variant]["trusted_variant"], block["guard_change"][variant]["all_variant"]) == (
            variant, twin)
    assert set(block["cap_sensitivity"]) == set(tiers.NEG_FROM)
    assert all(row["neg_from"] == neg_from for row in block["cap_sensitivity"][neg_from])


# The attacker's best star under KEEPER's cap on the three networks (ruling R33b): one attack edge from the account
# with the largest B(u) into stars of 2 to floor(2K) + 4 fakes: (eps, that account, the best star, floor(2K)). Task
# 3.1 measured 21, 23 and 25 fakes (2K - 3) under the all-raters guard. Ruling R56 keeps each network's eps
# (tests/test_credibility_btp.py), and the fakes receive no negative rating, so the guard admits the same stars.
BEST_STAR = {"bitcoinalpha": (0.02, 34, 21, 24), "bitcoinotc": (0.01, 1, 23, 26), "wiki-rfa": (0.5, 53, 25, 28)}


@pytest.mark.parametrize("neg_from", ["all", "trusted"])
def test_capped_best_star_on_the_three_networks(neg_from, training_graphs):
    from keeper.analysis import tiers_report as TR
    assert set(BEST_STAR) == set(NETWORKS)
    for ds, (eps, endorser, best, constant) in BEST_STAR.items():
        Tr, n = training_graphs[ds]
        cal = tiers.calibrate_btp(Tr, n, eps, edge_cap_units=tiers.KEEPER_EDGE_CAP_UNITS)
        assert TR.max_trust_endorsers(Tr, n, cal, 1).tolist() == [endorser]
        assert math.floor(cal.K * cal.edge_cap_units + TR.CAP_TOL) == constant
        star = TR.best_star(Tr, n, cal, endorser, range(2, constant + 5), neg_from=neg_from)
        assert (star["max_admitted"], star["best_s"], star["lower_bound"]) == (best, best, False), ds
