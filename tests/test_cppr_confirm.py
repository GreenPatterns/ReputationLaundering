"""Tests for keeper.analysis.cppr_confirm, cppr_runs and cppr_endpoints (Task 4.2b, E3/E4 confirmatory analysis).

Hand-computed cases: each endpoint (H1, H2w, H2b, H3b) on small tables, the control adjustment, the leave-one-seed-out
best response (with a case where the in-sample argmax differs from it, and a bootstrap replicate that draws a seed
twice), the p-values, the three-way reading and "contradicted" (ruling R62(b)), and components with fewer than 80% of
their seeds. The bootstrap primitives that return replicates reproduce the Phase 2 intervals. Synthetic run files, built
by tests/cppr_confirm_fixtures.py in tmp_path, exercise every integrity field (each one named, a failing run excluded
and counted), H3a, the r0 check between E4 and E3, missing runs and partial run sets, the unused-target sensitivity with
the stored seeds' victims (ruling R62(a)) and its replay check, bit-identical reruns and the CLI's overwrite guard
(ruling R27). One test replays the driver's draws on the three training graphs and checks the protocol's counts. CPU
only; nothing is written under results/.
"""
import copy
import inspect
import json
import math
import os
from functools import partial

import numpy as np
import pytest
from scipy.stats import t as student_t

from keeper.analysis import cppr_confirm as CC
from keeper.analysis import cppr_endpoints as E
from keeper.analysis.bootstrap import (column_means_stat, crossed_bootstrap, crossed_replicates, nested_bootstrap,
                                       nested_replicates, percentile_interval, seed_t_interval)
from keeper.analysis.intervals import Boot
from keeper.analysis import cppr_runs as R
from keeper.cluster import cppr_tasks as T
from keeper.cluster import protocol_settings as PS
from keeper.laundering import core as C
from tests import cppr_confirm_fixtures as F

N_BOOT = 300
ALPHA_DS = "bitcoinalpha"
STORED = frozenset(range(0, 8))           # synthetic "seeds 0-4" edge targets for the sensitivity
STORED_VICTIMS = frozenset(range(100, 104))   # and bad-mouth victims (ruling R62(a))


def fake_replay(ds, seeds, manifest=None):
    """The synthetic draws: STORED and STORED_VICTIMS for seeds 0-4, the fixtures' targets for the other seeds."""
    return {s: dict(edge=sorted(STORED), trusted=sorted(STORED_VICTIMS)) if s in R.STORED_SEEDS
            else {k: F.targets(ds, s)[k] for k in ("edge", "trusted")} for s in seeds}


def index(rows) -> dict:
    """(arm, method, kind) -> {target: row} from (arm, method, kind, target, r0, gain) tuples."""
    out = {}
    for arm, method, kind, target, r0, gain in rows:
        out.setdefault((arm, method, kind), {})[target] = dict(r0=r0, gain=gain)
    return out


def analyse(root, prereg, **kw):
    kw.setdefault("replay", fake_replay)
    return CC.analyse(str(root), str(prereg), networks=(ALPHA_DS,), n_boot=N_BOOT, **kw)


# ------------------------------------------------------------------ bootstrap primitives that return replicates
CELLS = {5: {1: 0.1, 2: 0.4, 3: -0.2}, 6: {1: 0.3, 3: 0.0, 4: 0.5}, 7: {2: 0.2, 4: 0.1}}


def test_crossed_replicates_reproduce_the_crossed_bootstrap():
    point, reps, drawn = crossed_replicates(CELLS, n_boot=500, seed=3)
    assert reps.shape == (500,) and drawn.shape == (500, 3)
    assert (point, *percentile_interval(reps)) == crossed_bootstrap(CELLS, n_boot=500, seed=3)


def test_nested_replicates_reproduce_the_nested_bootstrap():
    point, reps, drawn = nested_replicates(CELLS, n_boot=500, seed=3)
    assert reps.shape == (500,) and drawn.shape == (500, 3)
    assert (point, *percentile_interval(reps)) == nested_bootstrap(CELLS, n_boot=500, seed=3)


@pytest.mark.parametrize("replicates", [crossed_replicates, nested_replicates])
def test_a_seed_aware_combine_sees_the_drawn_seeds_with_the_same_draws(replicates):
    def by_identity(per_seed, drawn):
        return drawn.mean(axis=1).astype(float)              # a statistic of the drawn seeds' identities alone
    point, reps, drawn = replicates(CELLS, combine=by_identity, n_boot=50, seed=1, seed_aware=True)
    assert point == 1.0                                      # on the data the seeds sit at positions 0, 1 and 2
    assert np.array_equal(reps, drawn.mean(axis=1))
    assert np.array_equal(drawn, replicates(CELLS, n_boot=50, seed=1)[2])


def test_percentile_interval_refuses_more_than_one_percent_undefined_replicates():
    info = {}
    assert percentile_interval(np.r_[np.arange(199.0), np.nan], info=info) == pytest.approx([4.95, 193.05])
    assert info["undefined_share"] == pytest.approx(0.005)            # 1 of 200 replicates is undefined
    with pytest.raises(FloatingPointError):
        percentile_interval(np.r_[np.arange(197.0), np.nan, np.nan, np.nan])   # 1.5%


# ------------------------------------------------------------------ endpoints by hand
H1_SEEDS = {5: index([("direct", "plain", "edge", 1, 0.5, 0.10), ("direct", "plain", "edge", 2, 0.4, 0.06),
                      ("direct", "cppr", "edge", 1, 0.5, 0.02), ("direct", "cppr", "edge", 2, 0.4, 0.00),
                      ("control", "cppr", "edge", 1, 0.5, 0.01), ("control", "cppr", "edge", 2, 0.4, -0.01)]),
            6: index([("direct", "plain", "edge", 1, 0.3, 0.08), ("direct", "plain", "edge", 3, 0.6, 0.04),
                      ("direct", "cppr", "edge", 1, 0.3, 0.01), ("direct", "cppr", "edge", 3, 0.6, 0.03),
                      ("control", "cppr", "edge", 1, 0.3, 0.00), ("control", "cppr", "edge", 3, 0.6, 0.01)])}


def test_h1_difference_and_cut_by_hand():
    h1 = E.h1_cells(H1_SEEDS)
    assert h1["a"].cells == {5: {1: [pytest.approx(0.08)], 2: [pytest.approx(0.06)]},
                             6: {1: [pytest.approx(0.07)], 3: [pytest.approx(0.01)]}}
    a = E.difference_component(h1["a"], "H1/a", Boot("toy", N_BOOT), 2)
    assert a["estimate"] == pytest.approx(0.055)                       # seed means 0.07 and 0.04
    assert a["by_seed"] == {5: pytest.approx(0.07), 6: pytest.approx(0.04)}
    assert a["ci95_seed_t"] == pytest.approx(seed_t_interval([0.07, 0.04]))
    b = E.ratio_component(h1["b"], "H1/b", Boot("toy", N_BOOT), 2)
    # gain_adj(cppr) = cppr - control: seed 5 (0.01, 0.01), seed 6 (0.01, 0.02); Plain Erasure 0.08 and 0.06
    assert (b["numerator_mean"], b["denominator_mean"]) == (pytest.approx(0.0125), pytest.approx(0.07))
    assert b["estimate"] == pytest.approx(1 - 0.0125 / 0.07)
    assert b["by_seed"] == {5: pytest.approx(0.875), 6: pytest.approx(0.75)}
    assert b["rule"] == "95% interval lower bound >= 0.8"


def test_h2w_difference_and_suppression_by_hand():
    rows = {5: index([("whitewash", "vpe", "edge", 1, 0.5, 0.05), ("control", "vpe", "edge", 1, 0.5, 0.01),
                      ("whitewash", "cppr", "edge", 1, 0.5, 0.01), ("control", "cppr", "edge", 1, 0.5, 0.00),
                      ("whitewash", "vpe", "edge", 2, 0.4, 0.03), ("control", "vpe", "edge", 2, 0.4, -0.01),
                      ("whitewash", "cppr", "edge", 2, 0.4, 0.00), ("control", "cppr", "edge", 2, 0.4, 0.01)]),
            6: index([("whitewash", "vpe", "edge", 1, 0.5, 0.06), ("control", "vpe", "edge", 1, 0.5, 0.00),
                      ("whitewash", "cppr", "edge", 1, 0.5, 0.02), ("control", "cppr", "edge", 1, 0.5, 0.01)])}
    h2w = E.h2w_cells(rows)
    assert h2w["a"].cells == {5: {1: [pytest.approx(0.03)], 2: [pytest.approx(0.05)]}, 6: {1: [pytest.approx(0.05)]}}
    a = E.difference_component(h2w["a"], "H2w/a", Boot("toy", N_BOOT), 2)
    assert a["estimate"] == pytest.approx(0.045)
    b = E.ratio_component(h2w["b"], "H2w/b", Boot("toy", N_BOOT), 2)
    assert (b["numerator_mean"], b["denominator_mean"]) == (pytest.approx(0.005), pytest.approx(0.05))
    assert b["estimate"] == pytest.approx(0.9)


def test_h2b_uses_losses_and_the_control_rows_of_kind_trusted():
    rows = {5: index([("badmouth", "vpe", "trusted", 101, 0.7, -0.05), ("control", "vpe", "trusted", 101, 0.7, 0.01),
                      ("badmouth", "cppr", "trusted", 101, 0.7, -0.01), ("control", "cppr", "trusted", 101, 0.7, 0.0),
                      ("control", "vpe", "edge", 101, 0.7, 0.5), ("control", "cppr", "edge", 101, 0.7, -0.5)]),
            6: index([("badmouth", "vpe", "trusted", 102, 0.6, -0.04), ("control", "vpe", "trusted", 102, 0.6, 0.0),
                      ("badmouth", "cppr", "trusted", 102, 0.6, -0.02),
                      ("control", "cppr", "trusted", 102, 0.6, -0.01)])}
    h2b = E.h2b_cells(rows)
    # losses: seed 5 (0.05 + 0.01) - (0.01 - 0) = 0.05; seed 6 (0.04 - 0) - (0.02 - 0.01) = 0.03
    assert h2b["a"].cells == {5: {101: [pytest.approx(0.05)]}, 6: {102: [pytest.approx(0.03)]}}
    assert h2b["b"].cells == {5: {101: [pytest.approx(0.01), pytest.approx(0.06)]},
                              6: {102: [pytest.approx(0.01), pytest.approx(0.04)]}}
    b = E.ratio_component(h2b["b"], "H2b/b", Boot("toy", N_BOOT), 2)
    assert b["estimate"] == pytest.approx(1 - 0.01 / 0.05)


def test_a_target_missing_from_one_arm_is_dropped_and_counted():
    rows = index([("direct", "plain", "edge", t, 0.5, 0.05) for t in (1, 2, 3)]
                 + [("direct", "cppr", "edge", t, 0.5, 0.01) for t in (1, 2)])
    h1 = E.h1_cells({5: rows})
    assert sorted(h1["a"].cells[5]) == [1, 2] and h1["a"].dropped == 1
    assert h1["b"].cells == {} and h1["b"].dropped == 3                # no control row at all


def test_overshoot_counts_targets_whose_cut_exceeds_1_2_among_positive_plain_gains():
    # cuts: target 1: 1 - (-0.03 - 0.0) / 0.1 = 1.3 (over); target 2: 1 - (0.0 - 0.01) / 0.1 = 1.1; target 3: plain <= 0
    rows = index([("direct", "plain", "edge", 1, 0.5, 0.1), ("direct", "cppr", "edge", 1, 0.5, -0.03),
                  ("control", "cppr", "edge", 1, 0.5, 0.0), ("direct", "plain", "edge", 2, 0.4, 0.1),
                  ("direct", "cppr", "edge", 2, 0.4, 0.0), ("control", "cppr", "edge", 2, 0.4, 0.01),
                  ("direct", "plain", "edge", 3, 0.3, -0.01), ("direct", "cppr", "edge", 3, 0.3, 0.0),
                  ("control", "cppr", "edge", 3, 0.3, 0.0)])
    share, left_out = E.overshoot_cells({5: rows})
    assert share.cells == {5: {1: [1.0], 2: [0.0]}} and left_out == 1
    account = E.account_cells({5: index([("direct", "plain", "account", 9, 0.5, 0.04),
                                         ("direct", "cppr", "account", 9, 0.5, 0.01)])})
    assert account["plain_minus_cppr"].cells == {5: {9: [pytest.approx(0.03)]}} and account["gain_vpe"].cells == {}


def test_utility_is_the_direct_arms_cppr_mcc_minus_plain_and_ungated_per_seed():
    def run(seed, plain, cppr, ungated):
        result = dict(test_metrics=dict(direct=dict(plain=F.metrics(plain), cppr=F.metrics(cppr))),
                      ungated_baseline=F.metrics(ungated))
        return R.Run(R.RunKey("e3", T.CAPPED, ALPHA_DS, seed), result, {})
    runs = {5: run(5, 0.20, 0.25, 0.30), 6: run(6, 0.22, 0.23, 0.20)}
    out = CC.utility_for(runs, Boot(ALPHA_DS, N_BOOT), "mcc/", 10, ungated=True)
    assert out["cppr_minus_plain"]["by_seed"] == {5: pytest.approx(0.05), 6: pytest.approx(0.01)}
    assert out["cppr_minus_ungated"]["by_seed"] == {5: pytest.approx(-0.05), 6: pytest.approx(0.03)}
    assert out["cppr_minus_plain"]["unit_of_analysis"] == "seed" and "p_value" not in out["cppr_minus_plain"]


def test_control_adjustment_subtracts_the_e3_control_gain_and_requires_equal_r0():
    e4 = index([("whitewash", "cppr", "edge", 1, 0.5, 0.03), ("whitewash", "cppr", "edge", 2, 0.4, 0.02)])
    e3 = index([("control", "cppr", "edge", 1, 0.5, 0.01), ("control", "cppr", "edge", 2, 0.4, -0.01),
                ("whitewash", "cppr", "edge", 1, 0.5, 0.9)])        # E3's own whitewash rows are not the control
    adjusted, problems = E.adjusted_gains(e4, e3)
    assert adjusted == {1: pytest.approx(0.02), 2: pytest.approx(0.03)} and problems == []
    e3_shifted = index([("control", "cppr", "edge", 1, 0.5 + 1e-15, 0.01), ("control", "cppr", "edge", 2, 0.4, -0.01)])
    adjusted, problems = E.adjusted_gains(e4, e3_shifted)
    assert list(adjusted) == [2] and len(problems) == 1 and "r0" in problems[0]
    adjusted, problems = E.adjusted_gains(e4, index([("control", "cppr", "edge", 1, 0.5, 0.01)]))
    assert list(adjusted) == [1] and len(problems) == 1 and "target 2" in problems[0]


# ------------------------------------------------------------------ H3b: leave-one-seed-out best response
def test_h3b_leave_one_seed_out_choice_differs_from_the_in_sample_argmax():
    # BFS cell A gains 10, 0, 0 and cell B 3, 3, 3 over seeds 5, 6, 7; capped BTP gains nothing.
    cells = {5: {1: [10.0, 3.0, 0.0, 0.0]}, 6: {2: [0.0, 3.0, 0.0, 0.0]}, 7: {3: [0.0, 3.0, 0.0, 0.0]}}
    comp = E.h3b_component(E.Paired(cells, 0), "H3b", Boot("toy", N_BOOT), 3, methods=("bfs", "btpc"),
                           cell_labels=("A", "B"))
    assert comp["in_sample_choice"] == {"bfs": "A", "btpc": "A"}       # A has the larger mean over all seeds (10/3)
    assert {s: v["choice"]["bfs"] for s, v in comp["by_seed"].items()} == {5: "B", 6: "A", 7: "A"}
    assert {s: v["endpoint"] for s, v in comp["by_seed"].items()} == {5: 3.0, 6: 0.0, 7: 0.0}
    assert comp["estimate"] == pytest.approx(1.0)                      # the in-sample choice would give 10/3
    assert comp["ci95_seed_t"] == pytest.approx(seed_t_interval([3.0, 0.0, 0.0]))


def test_h3b_bootstrap_choice_leaves_out_every_copy_of_the_drawn_seed():
    # positions 0 and 1 drew seed 0 (BFS: A 10, B 3), position 2 drew seed 1 (A 0, B 3); capped BTP gains nothing
    seed0, seed1 = [[10.0, 3.0], [0.0, 0.0]], [[0.0, 3.0], [0.0, 0.0]]
    means = np.array([[seed0, seed0, seed1]])                         # [B, S, methods, cells]
    endpoint, choice, contribution = E.loso_endpoints(means, np.array([[0, 0, 1]]))
    assert choice[0, :, 0].tolist() == [1, 1, 0]                      # by position, seed 0 would pick A (mean 5)
    assert endpoint[0].tolist() == [3.0, 3.0, 0.0]
    endpoint, choice, _ = E.loso_endpoints(means, np.array([[0, 0, 0]]))
    assert np.isnan(endpoint).all() and (choice == -1).all()          # one distinct seed: no choice is defined


def _h3b_replicate_by_loops(cells, seeds, targets, seed_draw, target_draw):
    """One crossed H3b replicate written out with loops: cell means on the drawn targets, then each position's
    leave-one-seed-out choice among the positions holding another seed, the contributions and their mean."""
    means = []
    for s in seed_draw:
        rows = [cells[seeds[s]][targets[j]] for j in target_draw if targets[j] in cells[seeds[s]]]
        means.append(np.mean(rows, axis=0).reshape(2, 2) if rows else np.full((2, 2), np.nan))
    endpoints = []
    for i, s in enumerate(seed_draw):
        others = [k for k, o in enumerate(seed_draw) if o != s]
        contribution = []
        for m in range(2):
            loso = [np.mean([means[k][m][c] for k in others if not np.isnan(means[k][m][c])] or [np.nan])
                    for c in range(2)]
            finite = [c for c in range(2) if not np.isnan(loso[c])]
            contribution.append(means[i][m][max(finite, key=lambda c: (loso[c], -c))] if finite else np.nan)
        endpoints.append(contribution[0] - contribution[1])
    defined = [e for e in endpoints if not np.isnan(e)]
    return np.mean(defined) if defined else np.nan


def test_h3b_bootstrap_replicates_match_an_explicit_recomputation():
    rng = np.random.default_rng(0)
    cells = {s: {int(t): [float(x) for x in rng.normal(size=4)] for t in rng.choice(12, 6, replace=False)}
             for s in (5, 6, 7, 8)}
    seeds, targets = sorted(cells), sorted(set().union(*cells.values()))
    point, reps, drawn = crossed_replicates(cells, column_means_stat, partial(E.h3b_combine, n_methods=2), n_boot=60,
                                            seed=4, seed_aware=True)
    draw = np.random.default_rng(4)                                    # crossed_bootstrap's draw order, one chunk
    seed_draws = draw.integers(0, len(seeds), size=(60, len(seeds)))
    target_draws = draw.integers(0, len(targets), size=(60, len(targets)))
    assert np.array_equal(seed_draws, drawn)
    expected = [_h3b_replicate_by_loops(cells, seeds, targets, seed_draws[b], target_draws[b]) for b in range(60)]
    assert np.allclose(reps, expected, equal_nan=True, rtol=0, atol=1e-12)
    data = _h3b_replicate_by_loops(cells, seeds, targets, range(len(seeds)), range(len(targets)))
    assert point == pytest.approx(data, abs=1e-12)


def test_every_component_records_its_bootstrap_streams():
    cells = {s: {t: [0.05 + 0.001 * s + 0.002 * t] for t in range(4)} for s in range(5, 15)}
    boot = Boot(ALPHA_DS, N_BOOT)
    comp = E.difference_component(E.Paired(cells), "H1/a", boot, 10)
    assert comp["bootstrap_streams"] == dict(crossed=boot.seed_for("H1/a", "crossed"),
                                             nested=boot.seed_for("H1/a", "nested"))


# ------------------------------------------------------------------ p-values, readings, evaluability
def test_p_values_by_hand():
    values = [0.07, 0.04]
    t = 0.055 / (np.std(values, ddof=1) / math.sqrt(2))
    assert E.p_seed_t(values) == pytest.approx(student_t.sf(t, 1))
    assert (E.p_seed_t([0.1, 0.1]), E.p_seed_t([-0.1, -0.1]), E.p_seed_t([0.1])) == (0.0, 1.0, None)
    reps = np.array([-1.0, 0.0, 1.0, 2.0, np.nan])
    assert E.p_bootstrap(reps, 0.0) == pytest.approx(3 / 5)            # 2 of the 4 defined replicates are <= 0
    assert E.p_bootstrap(reps, 1.0, strict=True) == pytest.approx(3 / 5)
    num, den = np.array([0.01, 0.015]), np.array([0.08, 0.06])
    ratio = num.mean() / den.mean()
    se = ((num - ratio * den) / den.mean()).std(ddof=1) / math.sqrt(2)
    assert E.p_ratio_seed_t(num, den, 0.8) == pytest.approx(student_t.sf(((1 - ratio) - 0.8) / se, 1))


def test_component_p_value_is_the_larger_of_the_seed_t_and_bootstrap_p_values():
    cells = {s: {t: [0.05 + 0.001 * s + 0.002 * t] for t in range(4)} for s in range(5, 15)}
    comp = E.difference_component(E.Paired(cells, 0), "x", Boot("toy", N_BOOT), 10)
    assert comp["p_bootstrap"] == pytest.approx(1 / (N_BOOT + 1))     # every replicate is positive
    assert comp["p_value"] == max(comp["p_seed_t"], comp["p_bootstrap"])
    assert comp["decision"] == "met" and comp["sensitivity_nested"]["decision"] == "met"


@pytest.mark.parametrize("ci, reading, contradicted", [
    ([0.85, 1.10], "met", False), ([0.80, 0.95], "met", False), ([0.50, 0.90], "inconclusive", False),
    ([0.50, 0.80], "inconclusive", False), ([0.30, 0.79], "partial", False), ([-0.20, 0.40], "partial", False),
    ([-0.50, 0.0], "partial", False), ([-0.50, -0.01], "partial", True)])
def test_three_way_reading_of_the_threshold(ci, reading, contradicted):
    """Ruling R62(b): partial is not a contradiction; only an interval wholly below 0 is."""
    d = E.threshold_decision(ci, 0.8, True)
    assert (d["reading"], d["decision"], d["met"], d["contradicted"]) == (reading, reading, reading == "met",
                                                                          contradicted)
    assert E.threshold_decision(ci, 0.8, False)["decision"] == "not evaluable"


@pytest.mark.parametrize("ci, decision", [([0.01, 0.2], "met"), ([0.0, 0.2], "not met"), ([-0.1, 0.2], "not met"),
                                          ([-0.3, 0.0], "not met"), ([-0.3, -0.01], "contradicted")])
def test_difference_decision_and_contradiction(ci, decision):
    d = E.difference_decision(ci, True)
    assert (d["decision"], d["met"], d["contradicted"]) == (decision, decision == "met", decision == "contradicted")


def test_a_component_wholly_on_the_wrong_side_is_contradicted():
    cells = {s: {t: [-0.05 - 0.001 * t] for t in range(3)} for s in range(5, 15)}
    comp = E.difference_component(E.Paired(cells, 0), "x", Boot("toy", N_BOOT), 10)
    assert comp["decision"] == "contradicted" and comp["contradicted"] and not comp["met"]
    summary = E.network_summary({"a_difference": comp})
    assert summary["contradicted"] == ["a_difference"] and summary["status"] == "interval rule not met"


def test_a_component_with_fewer_than_80_percent_of_its_seeds_is_not_evaluable():
    def comp(n_seeds):
        cells = {s: {1: [0.1 + 0.01 * s], 2: [0.2]} for s in range(5, 5 + n_seeds)}
        return E.difference_component(E.Paired(cells, 0), "x", Boot("toy", N_BOOT), 10)
    seven, eight = comp(7), comp(8)
    assert (seven["n_seeds"], seven["evaluable"], seven["decision"], seven["met"]) == (7, False, "not evaluable", False)
    assert seven["estimate"] is not None                               # still reported, without a decision
    assert eight["evaluable"] and eight["decision"] == "met"
    summary = E.network_summary({"a": seven, "b": eight})
    assert (summary["evaluable"], summary["p_value"], summary["status"]) == (False, None, "not evaluable")


def test_hypothesis_p_value_is_the_largest_over_components_and_networks():
    met = dict(evaluable=True, met=True, contradicted=False, decision="met")
    net = E.network_summary({"a": dict(met, p_value=0.01), "b": dict(met, p_value=0.03)})
    assert (net["p_value"], net["interval_rule_met"], net["status"]) == (0.03, True, "interval rule met")
    hyp = E.hypothesis_summary({"x": net, "y": dict(net, p_value=0.002)}, ("x", "y"))
    assert (hyp["p_value"], hyp["status_before_holm"]) == (0.03, "interval rule met on every network")
    partial = E.hypothesis_summary({"x": net}, ("x", "y"))
    assert (partial["evaluable"], partial["p_value"], partial["networks_missing"]) == (False, None, ["y"])


# ------------------------------------------------------------------ H3a
def test_h3a_flags_admission_above_the_guarantee_and_a_wrong_or_missing_guarantee():
    check = lambda adm, g=1: E.h3a_check("run", adm, K=12, attack_edges=g, edge_cap_units=2.0,  # noqa: E731
                                         arms=("whitewash",))
    assert check({"whitewash": dict(admitted_per_group=[24, 3, 0], guarantee_per_group=24)}) == (3, [])
    n, v = check({"whitewash": dict(admitted_per_group=[25, 3], guarantee_per_group=24)})
    assert n == 2 and [(x["group"], x["admitted"]) for x in v] == [(0, 25)]
    _, v = check({"whitewash": dict(admitted_per_group=[3], guarantee_per_group=30)})
    assert len(v) == 1 and "floor(K g edge_cap_units) = 24" in v[0]["reason"]
    _, v = check({"whitewash": dict(admitted_per_group=[0], guarantee_per_group=None)})
    assert len(v) == 1 and "guarantee" in v[0]["reason"]
    _, v = check({})
    assert len(v) == 1 and "no admission record" in v[0]["reason"]
    assert check({"whitewash": dict(admitted_per_group=[0, 0], guarantee_per_group=0)}, g=0) == (2, [])
    assert len(check({"whitewash": dict(admitted_per_group=[1, 0], guarantee_per_group=0)}, g=0)[1]) == 1


# ------------------------------------------------------------------ run set and integrity checks
def test_expected_runs_match_the_task_generator(tmp_path):
    root = str(tmp_path)
    rows = T.e3_tasks(root) + T.e4_tasks(root)
    keys = R.expected_runs()
    assert sorted(k.stem for k in keys) == sorted(r.id for r in rows) and len(keys) == 345
    assert {k.result_file(root) for k in keys} == {r.done_path for r in rows}


def mutate_config(result, dump, section, **values):
    """Change config[section] in the result and in its dump alike, as a driver bug would."""
    for obj in (result, dump):
        obj["config"][section].update(values)


def set_config(result, dump, **values):
    for obj in (result, dump):
        obj["config"].update(values)


FAILURES = [   # (name, mutation of an E3 capped-BTP run, the one field it violates)
    ("epochs", lambda r, d: r.update(epochs=2), "epochs"),
    ("rho", lambda r, d: r.update(rho=0.05), "rho"),
    ("granularity", lambda r, d: r.update(granularity=2), "granularity"),
    ("k_fake", lambda r, d: r.update(k_fake=50), "k_fake"),
    ("gate_norm", lambda r, d: r.update(gate_norm="snapshot"), "gate_norm"),
    ("config gate_norm", lambda r, d: set_config(r, d, gate_norm="snapshot"), "config.gate_norm"),
    ("methods", lambda r, d: set_config(r, d, methods=["plain_vpe", "cppr"]), "config.methods"),
    ("arms", lambda r, d: set_config(r, d, arms=["direct", "whitewash", "badmouth"]), "config.arms"),
    ("tiers method", lambda r, d: mutate_config(r, d, "tiers", method="bfs"), "config.tiers.method"),
    ("eps", lambda r, d: mutate_config(r, d, "tiers", eps=0.05), "config.tiers.eps"),
    ("K", lambda r, d: mutate_config(r, d, "tiers", K=13), "config.tiers.K"),
    ("n_ref", lambda r, d: mutate_config(r, d, "tiers", n_ref=3630), "config.tiers.n_ref"),
    ("edge_cap_units", lambda r, d: mutate_config(r, d, "tiers", edge_cap_units=None), "config.tiers.edge_cap_units"),
    ("neg_from all", lambda r, d: mutate_config(r, d, "tiers", neg_from="all"), "config.tiers.neg_from"),
    ("neg_from absent", lambda r, d: [o["config"]["tiers"].pop("neg_from") for o in (r, d)], "config.tiers.neg_from"),
    ("attack edges", lambda r, d: mutate_config(r, d, "attack", attack_edges=1), "config.attack.attack_edges"),
    ("attack k_fake", lambda r, d: mutate_config(r, d, "attack", k_fake=50), "config.attack.k_fake"),
    ("topology", lambda r, d: mutate_config(r, d, "attack", sybil_topology="star"), "config.attack.sybil_topology"),
    ("reference calibration", lambda r, d: mutate_config(r, d, "attack", reference_calibration={"eps": 0.02}),
     "config.attack.reference_calibration"),
    ("hold_test_snapshot", lambda r, d: set_config(r, d, hold_test_snapshot=False), "config.hold_test_snapshot"),
    ("prereg", lambda r, d: mutate_config(r, d, "preregistration", sha256="0" * 64), "config.preregistration.sha256"),
    ("git_dirty", lambda r, d: r["provenance"].update(git_dirty=True), "provenance.git_dirty"),
    ("git_untracked", lambda r, d: r["provenance"].update(git_untracked=1), "provenance.git_untracked"),
    ("git_untracked False", lambda r, d: r["provenance"].update(git_untracked=False), "provenance.git_untracked"),
    ("git_commit", lambda r, d: r["provenance"].update(git_commit=None), "provenance.git_commit"),
    ("dataset", lambda r, d: r.update(ds="bitcoinotc"), "ds"),
    ("dump of another run", lambda r, d: d.update(result="e3_bfs_bitcoinalpha_s5.json"), "dump.result"),
    ("dump config", lambda r, d: d["config"]["tiers"].update(eps=0.5), "dump.config"),
    ("dump rows", lambda r, d: d.update(n_rows=d["n_rows"] + 1), "dump.n_rows"),
]


@pytest.mark.parametrize("name, mutate, field", FAILURES, ids=[f[0] for f in FAILURES])
def test_each_integrity_failure_names_its_field(tmp_path, name, mutate, field):
    _, sha = F.write_prereg(tmp_path)
    key = R.RunKey("e3", T.CAPPED, ALPHA_DS, 5)
    result, dump = F.build_run(tmp_path, key, F.Model(), sha)
    assert R.integrity_failures(key, result, dump, sha) == []
    mutate(result, dump)
    assert [f["field"] for f in R.integrity_failures(key, result, dump, sha)] == [field]


def test_every_violated_field_of_a_run_is_listed(tmp_path):
    """A run like the 2-epoch smoke runs of an older commit: each violation is listed, not just the first."""
    _, sha = F.write_prereg(tmp_path)
    key = R.RunKey("e3", T.CAPPED, ALPHA_DS, 5)
    result, dump = F.build_run(tmp_path, key, F.Model(), sha)
    result.update(epochs=2)
    result["provenance"].update(git_untracked=1)
    for obj in (result, dump):
        obj["config"]["tiers"].pop("neg_from")
        obj["config"]["preregistration"]["sha256"] = "0" * 64
    fields = [f["field"] for f in R.integrity_failures(key, result, dump, sha)]
    assert sorted(fields) == sorted(["epochs", "config.tiers.neg_from", "config.preregistration.sha256",
                                     "provenance.git_untracked"])


def test_e4_runs_check_their_cell_methods_and_reference_calibration(tmp_path):
    _, sha = F.write_prereg(tmp_path)
    paired = R.RunKey("e4", T.CAPPED, ALPHA_DS, 5, PS.E4_PAIRED_RUN[0])          # trains Plain VPE as well
    bfs = R.RunKey("e4", T.BFS, ALPHA_DS, 5, PS.E4_CELLS[3])                    # max-B(u) star s = 3, BFS tiers
    for key in (paired, bfs):
        result, dump = F.build_run(tmp_path, key, F.Model(), sha)
        assert R.integrity_failures(key, result, dump, sha) == []
    set_config(result, dump, methods=["plain_vpe", "cppr"])
    mutate_config(result, dump, "tiers", neg_from="trusted")                    # BFS has no guard
    result["config"]["attack"]["reference_calibration"]["eps"] = 0.05           # --pool-eps of the uncapped eps
    dump["config"] = copy.deepcopy(result["config"])
    mutate_config(result, dump, "attack", endorser_select="random")
    fields = [f["field"] for f in R.integrity_failures(bfs, result, dump, sha)]
    assert sorted(fields) == sorted(["config.methods", "config.tiers.neg_from", "config.attack.endorser_select",
                                     "config.attack.reference_calibration.eps"])


def test_row_level_dump_defects_are_integrity_failures(tmp_path):
    _, sha = F.write_prereg(tmp_path)
    key = R.RunKey("e3", T.CAPPED, ALPHA_DS, 5)
    _, dump = F.build_run(tmp_path, key, F.Model(), sha)
    index, failures = R.index_rows(dump)
    assert failures == [] and set(index) >= {("direct", "plain", "edge"), ("control", "cppr", "trusted")}
    broken = copy.deepcopy(dump)
    broken["rows"].append(list(broken["rows"][0]))                       # a duplicate target row
    F.rows_of(broken, "control", "cppr", "edge")[0][4] += 1e-9          # r0 differs from the target's other rows
    F.rows_of(broken, "whitewash", "vpe", "edge")[0][6] = None          # a gain that is not finite
    _, failures = R.index_rows(broken)
    assert len(failures) == 3, failures


# ------------------------------------------------------------------ end to end on synthetic runs
def test_registered_synthetic_runs_pass_and_meet_every_rule(tmp_path):
    prereg = F.build_results(tmp_path)
    out = analyse(tmp_path, prereg)
    i = out["integrity"]
    assert out["status"] == "ok" and i["failures"] == [] and i["missing_runs"] == [] and i["excluded_runs"] == {}
    assert i["n_used"] == dict(e3=25, e4=90) and i["complete"] and CC.exit_code(out) == 0
    assert out["h3a"]["ok"] and out["h3a"]["n_runs"] == 10 + 40 and out["h3a"]["violations"] == []
    for hyp in ("H1", "H2w", "H2b", "H3b"):
        net = out["hypotheses"][hyp]["networks"][ALPHA_DS]
        assert net["status"] == "interval rule met", (hyp, net)
        assert out["hypotheses"][hyp]["evaluable"] is False             # two registered networks were not analysed
    cut = out["hypotheses"]["H1"]["networks"][ALPHA_DS]["components"]["b_cut"]
    assert cut["reading"] == "met" and cut["n_seeds"] == 10 and cut["n_targets"] > 15
    assert out["sensitivities"]["unused_targets"]["replay_mismatches"] == []
    assert set(out) >= {"schema", "status", "protocol", "settings", "integrity", "h3a", "hypotheses", "sensitivities",
                        "secondary", "descriptive", "method"}


def test_missing_runs_are_listed_and_never_imputed(tmp_path):
    gone = {"e3_btpc_bitcoinalpha_s14", "e4_bfs_rand_ring_k3_g1_bitcoinalpha_s9"}
    prereg = F.build_results(tmp_path, skip=gone)
    os.remove(R.RunKey("e3", T.BFS, ALPHA_DS, 13).dump_file(str(tmp_path)))
    out = analyse(tmp_path, prereg)
    missing = {(m["run"], m["reason"]) for m in out["integrity"]["missing_runs"]}
    assert missing == {("e3_btpc_bitcoinalpha_s14", "result missing"),
                       ("e4_bfs_rand_ring_k3_g1_bitcoinalpha_s9", "result missing"),
                       ("e3_bfs_bitcoinalpha_s13", "target dump missing")}
    assert out["integrity"]["n_missing"] == dict(e3=2, e4=1) and not out["integrity"]["complete"]
    assert out["status"] == "ok" and CC.exit_code(out) == 0
    h1 = out["hypotheses"]["H1"]["networks"][ALPHA_DS]["components"]["a_difference"]
    assert (h1["n_seeds"], h1["evaluable"], 14 in h1["seeds"]) == (9, True, False)


def test_a_run_failing_integrity_is_excluded_and_counted(tmp_path):
    def misconfigured(key, result, dump):
        if key.stem in ("e3_btpc_bitcoinalpha_s14", "e3_btpc_bitcoinalpha_s13", "e3_btpc_bitcoinalpha_s12"):
            result.update(epochs=2)
    prereg = F.build_results(tmp_path, mutate=misconfigured)
    out = analyse(tmp_path, prereg)
    i = out["integrity"]
    assert out["status"] == "integrity_failure" and CC.exit_code(out) == 2
    assert i["excluded_runs"] == {f"e3_btpc_bitcoinalpha_s{s}": ["epochs"] for s in (12, 13, 14)}
    assert i["n_excluded"] == dict(e3=3, e4=0) and i["n_used"] == dict(e3=22, e4=90) and not i["complete"]
    net = out["hypotheses"]["H1"]["networks"][ALPHA_DS]
    assert net["components"]["a_difference"]["seeds"] == list(range(5, 12))      # never used
    assert (net["evaluable"], net["status"]) == (False, "not evaluable")           # 7 of 10 seeds < 80%
    assert out["h3a"]["n_runs"] == 7 + 40                                          # nor checked for H3a


@pytest.mark.parametrize("present", ["nothing", "some e3 seeds"])
def test_a_partial_run_set_runs_and_marks_thin_components_not_evaluable(tmp_path, present):
    keep = set() if present == "nothing" else {f"e3_btpc_bitcoinalpha_s{s}" for s in range(5, 13)}
    prereg = F.build_results(tmp_path, skip={k.stem for k in R.expected_runs((ALPHA_DS,))} - keep)
    out = analyse(tmp_path, prereg)
    assert out["status"] == "ok" and len(out["integrity"]["missing_runs"]) == 115 - len(keep)
    h1 = out["hypotheses"]["H1"]["networks"][ALPHA_DS]
    h3b = out["hypotheses"]["H3b"]["networks"][ALPHA_DS]["components"]["difference"]
    assert h3b["evaluable"] is False and h3b["decision"] == "not evaluable"
    if keep:                                                                       # 8 of 10 seeds: evaluable
        assert h1["evaluable"] and h1["components"]["a_difference"]["n_seeds"] == 8
    else:
        assert not h1["evaluable"] and h1["components"]["b_cut"]["estimate"] is None
    from keeper.analysis.cppr_confirm_md import render_markdown
    assert "Missing runs (" in render_markdown(json.loads(json.dumps(C._finite(out), default=C._json_default)))


def test_a_missing_protocol_file_is_an_integrity_failure(tmp_path):
    F.build_results(tmp_path)
    out = analyse(tmp_path, tmp_path / "docs" / "absent.md")
    assert out["status"] == "integrity_failure" and CC.exit_code(out) == 2
    assert [f["field"] for f in out["integrity"]["failures"]] == ["protocol"]


def test_an_r0_mismatch_between_an_e4_run_and_its_e3_control_excludes_the_e4_run(tmp_path):
    stem = "e4_btpc_maxb_star_k50_g1_bitcoinalpha_s7"

    def shift_r0(key, result, dump):
        if key.stem == stem:
            F.rows_of(dump, "whitewash", "cppr", "edge")[0][4] += 1e-12
    prereg = F.build_results(tmp_path, mutate=shift_r0)
    out = analyse(tmp_path, prereg)
    assert out["status"] == "integrity_failure" and CC.exit_code(out) == 2
    [failure] = out["integrity"]["failures"]
    assert failure["run"] == stem and failure["field"] == "r0 against E3" and "r0" in failure["detail"]
    assert out["integrity"]["excluded_runs"] == {stem: ["r0 against E3"]}
    assert out["integrity"]["n_used"] == dict(e3=25, e4=89)
    cells = out["descriptive"][ALPHA_DS]["e4_cells"]["btpc"]["maxb_star_k50_g1"]
    assert 7 not in cells["seeds"] and cells["n_seeds"] == 4


def test_the_unused_target_sensitivity_drops_the_stored_seeds_targets_and_victims(tmp_path):
    prereg = F.build_results(tmp_path)
    out = analyse(tmp_path, prereg)
    unused = out["sensitivities"]["unused_targets"]
    assert unused["stored_seed_targets"][ALPHA_DS] == dict(edge=len(STORED), trusted=len(STORED_VICTIMS))

    def reused(kind, seeds, stored):
        return sum(len(set(F.targets(ALPHA_DS, s)[kind]) & stored) for s in seeds)
    edge_cells = reused("edge", PS.SEEDS_E3, STORED)
    h1 = unused["hypotheses"]["H1"][ALPHA_DS]["components"]["a_difference"]
    assert h1["n_pairs_removed"] == edge_cells and h1["n_pairs"] == 10 * F.N_EDGE - edge_cells
    victim_cells = reused("trusted", PS.SEEDS_E3, STORED_VICTIMS)                  # ruling R62(a)
    h2b = unused["hypotheses"]["H2b"][ALPHA_DS]["components"]["a_difference"]
    assert victim_cells > 0 and h2b["n_pairs_removed"] == victim_cells
    h3b = unused["hypotheses"]["H3b"][ALPHA_DS]["components"]["difference"]
    assert h3b["n_pairs_removed"] == reused("edge", PS.SEEDS_5, STORED)


def test_a_replay_that_differs_from_a_dump_is_reported(tmp_path):
    prereg = F.build_results(tmp_path)

    def off_by_one(ds, seeds, manifest=None):
        drawn = fake_replay(ds, seeds, manifest)
        drawn[6]["trusted"] = drawn[6]["trusted"][1:] + [999]
        return drawn
    mismatches = analyse(tmp_path, prereg, replay=off_by_one)["sensitivities"]["unused_targets"]["replay_mismatches"]
    assert {(m["run"], m["kind"], m["n_common"]) for m in mismatches} == {
        (f"e3_{tier}_bitcoinalpha_s6", "trusted", F.N_EDGE - 1) for tier in ("btpc", "bfs", "btpu")}


def test_reruns_are_bit_identical(tmp_path):
    prereg = F.build_results(tmp_path)
    first, second = analyse(tmp_path, prereg), analyse(tmp_path, prereg)
    assert json.dumps(C._finite(first), sort_keys=True, default=C._json_default) == \
        json.dumps(C._finite(second), sort_keys=True, default=C._json_default)


# ------------------------------------------------------------------ CLI
class WorkStarted(Exception):
    """Raised by the first step of the analysis: reaching it means the overwrite guard let the run go on."""


def _work_started(*args, **kwargs):
    raise WorkStarted


def cli(tmp_path, prereg, *extra):
    return ["--results-root", str(tmp_path), "--prereg", str(prereg), "--networks", ALPHA_DS, "--n-boot", str(N_BOOT),
            "--out", str(tmp_path / "out" / "cppr_confirm.json"), "--md", str(tmp_path / "out" / "cppr_confirm.md"),
            *extra]


def test_cli_refuses_an_existing_output_before_its_work(tmp_path, monkeypatch):
    monkeypatch.delenv(C.OVERWRITE_ENV, raising=False)
    monkeypatch.setattr(CC, "analyse", _work_started)
    out, md = tmp_path / "out" / "cppr_confirm.json", tmp_path / "out" / "cppr_confirm.md"
    out.parent.mkdir()
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        CC.main(cli(tmp_path, tmp_path / "prereg.md"))
    assert out.read_text() == "{}" and not md.exists()
    out.unlink()
    md.write_text("stored")
    with pytest.raises(FileExistsError):
        CC.main(cli(tmp_path, tmp_path / "prereg.md"))
    with pytest.raises(WorkStarted):
        CC.main(cli(tmp_path, tmp_path / "prereg.md", "--force"))


def _without_provenance(path):
    with open(path) as f:
        data = json.load(f)
    data.pop("provenance")
    return data


def test_cli_writes_json_and_markdown_and_reruns_identically(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "seed_targets", fake_replay)
    prereg = F.build_results(tmp_path)
    assert CC.main(cli(tmp_path, prereg)) == 0
    out, md = tmp_path / "out" / "cppr_confirm.json", tmp_path / "out" / "cppr_confirm.md"
    first = _without_provenance(out)
    with open(out) as f:
        prov = json.load(f)["provenance"]
    assert prov["n_boot"] == N_BOOT and "git_commit" in prov
    assert any(k.endswith("e3_btpc_bitcoinalpha_s5.json") for k in prov["inputs_sha256"])
    assert any(k.endswith("preregistration.md") for k in prov["inputs_sha256"])
    text = md.read_text()
    assert "H1" in text and "H3b" in text and "Missing runs" in text and "not the registered 10,000" in text
    assert CC.main(cli(tmp_path, prereg, "--force")) == 0
    assert _without_provenance(out) == first


def test_cli_exit_status_2_on_an_integrity_failure_and_3_on_an_h3a_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "seed_targets", fake_replay)

    def dirty(key, result, dump):
        if key.stem == "e3_bfs_bitcoinalpha_s8":
            result["provenance"]["git_dirty"] = True
    prereg = F.build_results(tmp_path / "a", mutate=dirty)
    assert CC.main(cli(tmp_path / "a", prereg)) == 2
    with open(tmp_path / "a" / "out" / "cppr_confirm.json") as f:
        assert json.load(f)["status"] == "integrity_failure"
    assert "e3_bfs_bitcoinalpha_s8: provenance.git_dirty" in (tmp_path / "a" / "out" / "cppr_confirm.md").read_text()

    def over(key, result, dump):
        if key.stem == "e4_btpc_rand_star_k50_g1_bitcoinalpha_s6":
            result["sybil_admission"]["whitewash"]["admitted_per_group"][4] = 25
    prereg = F.build_results(tmp_path / "b", mutate=over)
    assert CC.main(cli(tmp_path / "b", prereg)) == 3
    with open(tmp_path / "b" / "out" / "cppr_confirm.json") as f:
        data = json.load(f)
    assert data["status"] == "code_defect" and data["h3a"]["code_defect"]
    [violation] = data["h3a"]["violations"]
    assert (violation["run"], violation["group"], violation["admitted"]) == (
        "e4_btpc_rand_star_k50_g1_bitcoinalpha_s6", 4, 25)


# ------------------------------------------------------------------ the driver's draws, replayed on the real graphs
# Section 2.1: distinct E3 edge targets (seeds 5-14) and those also drawn in seeds 0-4; likewise the bad-mouth victims
# (the counts of the Task 4.2b replay that led to ruling R62).
PROTOCOL_COUNTS = {"bitcoinalpha": (80, 44), "bitcoinotc": (126, 38), "wiki-rfa": (98, 36)}
VICTIM_COUNTS = {"bitcoinalpha": (142, 12), "bitcoinotc": (145, 10), "wiki-rfa": (125, 22)}


@pytest.mark.parametrize("ds", PS.NETWORKS)
def test_the_replayed_draws_reproduce_the_protocols_counts(ds):
    drawn = R.seed_targets(ds, list(R.STORED_SEEDS) + list(PS.SEEDS_E3))
    for kind, counts in (("edge", PROTOCOL_COUNTS), ("trusted", VICTIM_COUNTS)):
        stored = set().union(*(set(drawn[s][kind]) for s in R.STORED_SEEDS))
        e3 = set().union(*(set(drawn[s][kind]) for s in PS.SEEDS_E3))
        assert (len(e3), len(e3 & stored)) == counts[ds], kind
    assert all(len(drawn[s]["edge"]) == len(drawn[s]["trusted"]) == R.N_EDGE for s in drawn)


def test_the_replay_uses_the_drivers_defaults_and_workload():
    from keeper.cppr import cp_vpe_gpu as D
    from keeper.cppr import vpe as V
    from keeper.cppr import vpe_frontier
    defaults = {k: p.default for k, p in inspect.signature(D.run_single_seed).parameters.items()}
    assert (defaults["rho"], defaults["n_edge"], defaults["n_acc"], defaults["g"], defaults["k_fake"],
            defaults["gate_norm"]) == (R.RHO, R.N_EDGE, R.N_ACC, R.GRANULARITY, R.E3_K_FAKE, R.GATE_NORM)
    assert R.workload_function() is (V.workload if hasattr(V, "workload") else vpe_frontier.workload)
