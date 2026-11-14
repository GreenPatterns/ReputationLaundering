"""Tests for keeper.analysis.laundering_extra (Task 2.1, re-analysis of stored laundering results).

The unit cases are hand-computable: counting yardsticks; the crossed (ruling R29) and nested (R21)
bootstraps, undefined replicates and keyed cells; the seed-t, hull and Clopper-Pearson intervals of ruling
R30; edgeless detection after channel C2; escape from the bottom of the platform ranking; the
leave-one-out degree features of the logistic-regression yardstick (rulings R20 and R28); the extraction
of target rows and partial-deletion gains from small synthetic probe4 records; and the generated notes.
The CLI refuses to overwrite its outputs (ruling R27). One regression test reads the stored Bitcoin-Alpha
records and checks the review's numbers. CPU only.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from keeper.analysis.bootstrap import (clopper_pearson, column_means_stat, combine_ratio, hier_bootstrap,
                                       hull, ratio_seed_t_interval, ratio_stat, resolve_design,
                                       seed_t_interval)
from keeper.analysis.laundering_extra import (
    Boot, DegreeLogReg, amplification_block, base_rate_shift, c2_logreg_shift, ci_design_for, degree_table,
    degrees_after_c2, edge_degree_counts, edgeless_after_c2, escaped_from_bottom, fit_degree_logreg,
    hier_bootstrap_mean, laplace_shift, panel_score, partial_deletion_values, raw_shift, target_rows)
from keeper.analysis.laundering_extra_md import Count, PlusMinus, Range, Value, status
from keeper.laundering import metrics as M


# ------------------------------------------------------------------ counting yardsticks
def test_shifts_match_hand_values():
    assert abs(laplace_shift(npos=0, nneg=1) - (1 / 2 - 1 / 3)) < 1e-12
    assert abs(base_rate_shift(npos=0, nneg=1, p=0.9) - (1.8 / 2 - 1.8 / 3)) < 1e-12


def test_raw_shift_hand_values():
    assert raw_shift(npos=0, nneg=1) == 1.0                      # 0/0 after deletion reads as 1.0
    assert abs(raw_shift(npos=1, nneg=2) - (1 - 1 / 3)) < 1e-12


@pytest.mark.parametrize("npos", range(0, 4))
@pytest.mark.parametrize("nneg", range(1, 4))
def test_shifts_match_the_stored_counting_convention(npos, nneg):
    """probe4 stores count_gain via metrics.counting_gain(n_pos, k, k, kind); the port must agree."""
    assert abs(laplace_shift(npos, nneg) - float(M.counting_gain(npos, nneg, nneg, "laplace"))) < 1e-12
    assert abs(raw_shift(npos, nneg) - float(M.counting_gain(npos, nneg, nneg, "raw"))) < 1e-12
    assert abs(base_rate_shift(npos, nneg, p=0.5) - laplace_shift(npos, nneg)) < 1e-12


def test_shifts_vectorise():
    got = laplace_shift(np.array([0, 1]), np.array([1, 2]))
    assert np.allclose(got, [1 / 2 - 1 / 3, 2 / 3 - 2 / 5])


# ------------------------------------------------------------------ bootstrap designs (R21, R29)
DESIGNS = ("crossed", "nested")
# 0.975 quantiles of Student's t in closed form: 2 df, (2p - 1) / sqrt(2p(1 - p)); 1 df (Cauchy), tan(pi (p - 1/2))
T975_DF2, T975_DF1 = 0.95 / math.sqrt(2 * 0.975 * 0.025), math.tan(math.pi * 0.475)


def keyed(groups):
    """seed -> {"t0": v0, "t1": v1, ...}: the same targets in every seed, keyed by target."""
    return {s: {f"t{i}": v for i, v in enumerate(vals)} for s, vals in groups.items()}


def test_hier_bootstrap_brackets_mean():
    data = {0: [0.3, 0.4, 0.5], 1: [0.35, 0.45, 0.55]}
    m, lo, hi = hier_bootstrap_mean(data, n_boot=2000, seed=1)
    assert lo < m < hi and abs(m - 0.425) < 1e-9


def test_auto_design_is_crossed_for_aligned_data_and_nested_for_unkeyed_lists():
    lists = {0: [0.3, 0.4, 0.5], 1: [0.35, 0.45, 0.55]}
    assert resolve_design(lists, "auto") == "nested" and resolve_design(keyed(lists), "auto") == "crossed"
    assert resolve_design({0: [0.1], 1: [0.3]}, "auto") == "crossed"           # one value per seed aligns
    assert hier_bootstrap_mean(lists, n_boot=300, seed=1) == hier_bootstrap_mean(lists, n_boot=300, seed=1,
                                                                                 design="nested")


def test_hier_bootstrap_requires_an_explicit_design_and_records_it():
    """The re-review's case: unkeyed lists of four values over five seeds. hier_bootstrap needs the design
    spelled out; only hier_bootstrap_mean keeps "auto", and info["design"] reports which design ran."""
    lists = {s: [0.0, 1.0, 0.2, 0.9] for s in range(5)}
    with pytest.raises(TypeError):
        hier_bootstrap(keyed(lists), n_boot=10, seed=0)
    with pytest.raises(ValueError):
        hier_bootstrap(keyed(lists), n_boot=10, seed=0, design="auto")
    info = {}
    hier_bootstrap(keyed(lists), n_boot=10, seed=0, design="crossed", info=info)
    assert info["design"] == "crossed"
    info = {}
    hier_bootstrap_mean(lists, n_boot=200, seed=0, info=info)
    assert info["design"] == "nested"
    info = {}
    hier_bootstrap_mean(keyed(lists), n_boot=200, seed=0, info=info)
    assert info["design"] == "crossed"


def test_crossed_design_requires_cells_keyed_by_target():
    with pytest.raises(ValueError):
        hier_bootstrap_mean({0: [1.0, 2.0], 1: [1.0, 2.0]}, n_boot=10, seed=0, design="crossed")
    with pytest.raises(ValueError):
        hier_bootstrap_mean({0: [1.0]}, n_boot=10, seed=0, design="pooled")
    assert hier_bootstrap_mean({0: [1.0, 2.0, 3.0], 1: [1.0, 2.0]}, n_boot=10, seed=0, design="nested")[0] == 1.75


@pytest.mark.parametrize("design", DESIGNS)
def test_hier_bootstrap_is_seeded(design):
    data = keyed({0: [0.1, 0.5, 0.9], 1: [0.2, 0.3, 0.7], 2: [0.0, 0.4, 0.6]})
    first = hier_bootstrap_mean(data, n_boot=500, seed=3, design=design)
    assert first == hier_bootstrap_mean(data, n_boot=500, seed=3, design=design)
    assert first != hier_bootstrap_mean(data, n_boot=500, seed=4, design=design)


@pytest.mark.parametrize("design", DESIGNS)
def test_hier_bootstrap_ratio_statistics(design):
    """Per-seed column means s0 = (2, 2), s1 = (2, 4)."""
    groups = keyed({0: [[1.0, 2.0], [3.0, 2.0]], 1: [[2.0, 4.0], [2.0, 4.0]]})
    mean_of_ratios, lo, hi = hier_bootstrap(groups, stat=ratio_stat, n_boot=300, seed=0, design=design)
    assert abs(mean_of_ratios - (1.0 + 0.5) / 2) < 1e-12 and lo <= mean_of_ratios <= hi
    ratio_of_means, lo, hi = hier_bootstrap(groups, stat=column_means_stat, combine=combine_ratio,
                                            n_boot=300, seed=0, design=design)
    assert abs(ratio_of_means - 2.0 / 3.0) < 1e-12 and lo <= ratio_of_means <= hi


@pytest.mark.parametrize("design", DESIGNS)
def test_hier_bootstrap_drops_empty_seeds_and_handles_one_value_per_seed(design):
    assert hier_bootstrap_mean({0: {"a": 1.0, "b": 3.0}, 1: {}}, n_boot=100, seed=0, design=design)[0] == 2.0
    m, lo, hi = hier_bootstrap_mean({0: [0.1], 1: [0.3], 2: [0.5]}, n_boot=1000, seed=0, design=design)
    assert abs(m - 0.3) < 1e-12 and 0.1 <= lo < m < hi <= 0.5


def test_hier_bootstrap_degenerates_when_every_value_is_one():
    assert hier_bootstrap_mean(keyed({0: [1.0, 1.0], 1: [1.0, 1.0]}), n_boot=100, seed=0) == (1.0, 1.0, 1.0)
    assert hier_bootstrap_mean({0: [1.0, 1.0], 1: [1.0]}, n_boot=100, seed=0, design="nested") == (1.0, 1.0, 1.0)


def test_crossed_interval_when_values_depend_only_on_the_target():
    """Targets t0 and t1 score 0 and 1 in all five seeds.

    Crossed: one target resample serves every seed, so a replicate is the mean of two draws from {0, 1}:
    0 or 1 with probability 1/4 each, so the 95% interval is exactly [0, 1]. Nested: each seed draws its
    own pair, the replicate averages five such means, and P(replicate <= 0.1) = 1.07% < 2.5% (likewise
    above 0.9), so the interval lies inside [0.1, 0.9].
    """
    data = keyed({s: [0.0, 1.0] for s in range(5)})
    assert hier_bootstrap_mean(data, n_boot=2000, seed=0, design="crossed") == (0.5, 0.0, 1.0)
    m, lo, hi = hier_bootstrap_mean(data, n_boot=2000, seed=0, design="nested")
    assert m == 0.5 and 0.1 <= lo < m < hi <= 0.9


def test_crossed_aligns_targets_by_key_not_by_insertion_order():
    a = {0: {"a": 0.0, "b": 1.0, "c": 0.5}, 1: {"a": 0.2, "b": 0.9, "c": 0.4}}
    b = {0: {"c": 0.5, "b": 1.0, "a": 0.0}, 1: {"b": 0.9, "a": 0.2, "c": 0.4}}
    assert hier_bootstrap_mean(a, n_boot=300, seed=5) == hier_bootstrap_mean(b, n_boot=300, seed=5)


def test_missing_cells_are_skipped():
    """Target z is missing in seed 1; a replicate drawing z ten times out of ten has probability 1e-10."""
    cells = {0: {**{f"t{i}": 1.0 for i in range(9)}, "z": 0.0}, 1: {f"t{i}": 1.0 for i in range(9)}}
    info = {}
    m, lo, hi = hier_bootstrap(cells, n_boot=2000, seed=0, design="crossed", info=info)
    assert abs(m - (0.9 + 1.0) / 2) < 1e-12 and info["undefined_share"] == 0.0 and lo <= m <= hi


def test_more_than_one_percent_undefined_replicates_are_refused():
    """Target b exists only in seed 0: drawing (b, b) and seeds (1, 1) leaves no cell, probability 1/16."""
    info = {}
    with pytest.raises(FloatingPointError):
        hier_bootstrap({0: {"a": 1.0, "b": 0.0}, 1: {"a": 1.0}}, n_boot=8000, seed=2, design="crossed", info=info)
    assert abs(info["undefined_share"] - 1 / 16) < 0.01


@pytest.mark.parametrize("design, share", [("crossed", 3 / 16), ("nested", 15 / 64)])
def test_a_per_seed_zero_over_zero_ratio_makes_the_replicate_undefined(design, share):
    """Seed 0's target b is 0/0 while seed 1 always has a ratio. Crossed: undefined when the targets drawn
    are (b, b) and seed 0 is drawn at least once, 1/4 * 3/4. Nested: each of the two seed slots is seed 0
    drawing (b, b) with probability 1/2 * 1/4, so 1 - (7/8)^2 = 15/64."""
    groups = {0: {"a": [1.0, 1.0], "b": [0.0, 0.0]}, 1: {"a": [1.0, 1.0], "b": [1.0, 1.0]}}
    info = {}
    with pytest.raises(FloatingPointError):
        hier_bootstrap(groups, stat=ratio_stat, n_boot=8000, seed=4, design=design, info=info)
    assert abs(info["undefined_share"] - share) < 0.02


# ------------------------------------------------------------------ seed-t, hull and exact intervals (R30)
def test_seed_t_interval_by_hand():
    """Per-seed values 1, 2, 3: mean 2, sd 1, so 2 +- t(0.975, 2) / sqrt(3). Values 1 and 3: sd sqrt(2),
    standard error 1, so 2 +- t(0.975, 1)."""
    half = T975_DF2 / np.sqrt(3)
    assert seed_t_interval([1.0, 2.0, 3.0]) == pytest.approx([2 - half, 2 + half], abs=1e-9)
    assert seed_t_interval([1.0, None, 3.0]) == pytest.approx([2 - T975_DF1, 2 + T975_DF1], abs=1e-9)
    assert seed_t_interval([5.0]) is None


def test_ratio_seed_t_interval_by_hand():
    """num (1, 2, 3) over den (1, 1, 1): R = 2 with residuals -1, 0, 1. Over den (2, 2, 2): R = 1 with
    residuals (num - den) / 2 = -0.5, 0, 0.5."""
    half = T975_DF2 / np.sqrt(3)
    assert ratio_seed_t_interval([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == pytest.approx([2 - half, 2 + half])
    assert ratio_seed_t_interval([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == pytest.approx([1 - half / 2, 1 + half / 2])


def test_hull():
    assert hull([0.1, 0.5], [0.2, 0.7], None) == [0.1, 0.7] and hull(None) is None


def test_block_headline_is_the_seed_t_interval_for_a_seed_only_field():
    """One value per seed (1, 2, 3): every bootstrap replicate is a mean of three of them, inside [1, 3],
    while the t-interval is 2 +- 2.484, so the hull is the t-interval."""
    d = _toy_block(lambda b: b.stat("x", {0: [1.0], 1: [2.0], 2: [3.0]}, support=None))
    half = T975_DF2 / np.sqrt(3)
    assert d["ci95_seed_t"]["x"] == pytest.approx([2 - half, 2 + half])
    assert 1.0 <= d["ci95_bootstrap"]["x"][0] and d["ci95_bootstrap"]["x"][1] <= 3.0
    assert d["ci95"]["x"] == d["ci95_seed_t"]["x"] and d["ci95_source"]["x"].startswith("hull")


def test_block_headline_is_clopper_pearson_for_a_zero_or_one_field_at_its_boundary():
    def fill(b):
        b.stat("escape", keyed({0: [1.0, 1.0, 1.0], 1: [1.0, 1.0, 1.0]}), unit="targets", support=(0.0, 1.0))
        b.stat("over", keyed({0: [0.0, 0.0], 1: [0.0, 0.0]}), unit="evaluations", support=(0.0, 1.0))
        b.stat("frac", keyed({0: [1.0, 0.0], 1: [1.0, 1.0]}), unit="targets", support=(0.0, 1.0))  # 0.75
    d = _toy_block(fill)
    assert d["ci95"]["escape"] == clopper_pearson(3, 3) and "3 distinct targets" in d["ci95_source"]["escape"]
    assert d["ci95_bootstrap"]["escape"] == [1.0, 1.0]
    assert d["ci95"]["over"] == clopper_pearson(0, 4) and "4 evaluations" in d["ci95_source"]["over"]
    assert d["ci95_source"]["frac"].startswith("hull")


def test_headline_interval_is_clipped_to_each_declared_support():
    """Per-seed values whose t-interval (t(0.975, 2) = 4.303) crosses the support: a proportion with seed
    means 0, 0, 0.5 (half-width 0.717); thresholds 0.02, 0.03, 0.30; a mean count 0, 5, 0 of a cohort of
    10 and 10, 10, 5 near its top; a magnitude 0, 0, 0.01; a signed shift -1, 0, 3, which stays as it is."""
    def fill(b):
        b.stat("proportion", keyed({0: [0.0, 0.0], 1: [0.0, 0.0], 2: [1.0, 0.0]}), unit="targets",
               support=(0.0, 1.0))
        b.stat("tau", {0: [0.02], 1: [0.03], 2: [0.30]}, support=(0.0, 1.0))
        b.stat("count_low", {0: [0.0], 1: [5.0], 2: [0.0]}, support=(0.0, 10.0))
        b.stat("count_high", {0: [10.0], 1: [10.0], 2: [5.0]}, support=(0.0, 10.0))
        b.stat("magnitude", {0: [0.0], 1: [0.0], 2: [0.01]}, support=(0.0, None))
        b.stat("signed", {0: [-1.0], 1: [0.0], 2: [3.0]}, support=None)
    d = _toy_block(fill)
    for field in ("proportion", "tau", "count_low", "magnitude"):
        assert d["ci95"][field][0] == 0.0 and d["ci95_unclipped"][field][0] < 0.0
        assert d["ci95"][field][1] == d["ci95_unclipped"][field][1] and "clipped" in d["ci95_source"][field]
    assert d["ci95"]["count_high"][1] == 10.0 and d["ci95_unclipped"]["count_high"][1] > 10.0
    assert d["ci95"]["signed"][0] < -1.0 and "signed" not in d["ci95_unclipped"]
    assert d["ci95_support"]["magnitude"] == [0.0, None] and d["ci95_support"]["signed"] is None


def test_block_stat_requires_a_declared_support():
    from keeper.analysis.laundering_extra import Block
    with pytest.raises(TypeError):
        Block("toy", Boot("toy", n_boot=10)).stat("x", {0: [1.0], 1: [2.0]})


def _toy_block(fill):
    from keeper.analysis.laundering_extra import Block
    b = Block("toy", Boot("toy", n_boot=500))
    fill(b)
    return b.to_dict()


def test_clopper_pearson_at_the_boundary():
    lo, hi = clopper_pearson(10, 10)
    assert abs(lo - 0.025 ** (1 / 10)) < 1e-12 and hi == 1.0
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / 10))) < 1e-12


# ------------------------------------------------------------------ edgeless detection (C2)
def test_edgeless_after_c2():
    """C2 deletes every negative in-edge of the target; edgeless means nothing incident is left.

    node 0: only a negative in-edge                  -> edgeless
    node 3: a negative and a positive in-edge        -> keeps 4->3
    node 5: two negative in-edges and an out-edge    -> keeps 5->4
    node 6: a negative in-edge and a negative self-loop (a negative in-edge of 6 itself) -> edgeless
    node 7: a positive self-loop only                -> keeps it
    """
    edges = np.array([[1, 0], [2, 3], [4, 3], [1, 5], [2, 5], [5, 4], [1, 6], [6, 6], [7, 7]])
    is_neg = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0], bool)
    got = edgeless_after_c2(edges, is_neg, np.array([0, 3, 5, 6, 7]), n_nodes=8)
    assert got.tolist() == [True, False, False, True, False]


def test_edgeless_matches_the_review_definition_without_self_loops():
    """review_isolated.py: no out-edge and no positive in-edge in the pre-deletion training graph."""
    rng = np.random.default_rng(0)
    edges = rng.integers(0, 30, size=(200, 2))
    edges = edges[edges[:, 0] != edges[:, 1]]
    is_neg = rng.random(len(edges)) < 0.3
    nodes = np.arange(30)
    outdeg = np.bincount(edges[:, 0], minlength=30)
    pos_in = np.bincount(edges[~is_neg, 1], minlength=30)
    want = (outdeg == 0) & (pos_in == 0)
    assert (edgeless_after_c2(edges, is_neg, nodes, n_nodes=30) == want).all()


# ------------------------------------------------------------------ percentile escape
def test_escaped_from_bottom():
    before = np.array([0.01, 0.04, 0.06, 0.09, 0.20])
    after = np.array([0.05, 0.50, 0.01, 0.11, 0.30])
    assert escaped_from_bottom(before, after, 0.05).tolist() == [True, True]      # 0.05 >= 0.05 leaves
    assert escaped_from_bottom(before, after, 0.10).tolist() == [False, True, False, True]
    assert escaped_from_bottom(before, after, 0.0).tolist() == []


# ------------------------------------------------------------------ logistic-regression yardstick
GRAPH = np.array([[0, 2], [1, 2], [2, 0], [3, 2], [1, 0]])
GRAPH_POS = np.array([1, 0, 1, 1, 0], bool)            # 0->2 +, 1->2 -, 2->0 +, 3->2 +, 1->0 -


def test_degree_table():
    deg = degree_table(GRAPH, GRAPH_POS, n_nodes=4)       # columns: pos_in, neg_in, pos_out, neg_out
    assert deg.tolist() == [[1, 1, 1, 0], [0, 0, 0, 2], [2, 1, 1, 0], [0, 0, 1, 0]]


def test_edge_degree_counts_leave_one_out_on_target_in_degrees():
    """R20: v's in-degrees exclude the edge's own label; u's out-degrees are taken as they are."""
    got = edge_degree_counts(GRAPH, GRAPH_POS, n_nodes=4, loo="in")
    assert got.tolist() == [[1, 1, 1, 0],    # 0->2 +: pos_in(2) 2-1, neg_in(2) 1, pos_out(0) 1, neg_out(0) 0
                            [2, 0, 0, 2],    # 1->2 -: pos_in(2) 2, neg_in(2) 1-1, pos_out(1) 0, neg_out(1) 2
                            [0, 1, 1, 0],    # 2->0 +: pos_in(0) 1-1, neg_in(0) 1, pos_out(2) 1, neg_out(2) 0
                            [1, 1, 1, 0],    # 3->2 +
                            [1, 0, 0, 2]]    # 1->0 -: pos_in(0) 1, neg_in(0) 1-1, neg_out(1) 2


def test_edge_degree_counts_leave_one_out_on_both_endpoints():
    got = edge_degree_counts(GRAPH, GRAPH_POS, n_nodes=4, loo="in_and_out")
    assert got.tolist() == [[1, 1, 0, 0], [2, 0, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1]]
    with pytest.raises(ValueError):
        edge_degree_counts(GRAPH, GRAPH_POS, n_nodes=4, loo="none")


def test_degrees_after_c2_remove_the_warnings_from_both_ends():
    deg = degree_table(GRAPH, GRAPH_POS, n_nodes=4)
    after = degrees_after_c2(deg, GRAPH, ~GRAPH_POS, v=2)  # deletes 1->2 (-)
    assert after[2].tolist() == [2, 0, 1, 0] and after[1].tolist() == [0, 0, 0, 1]
    assert deg[2].tolist() == [2, 1, 1, 0]                 # the input table is not modified


def test_panel_score_and_c2_shift_by_hand():
    """With coef (0, -1, 0, 0) and intercept 0 a pair scores sigmoid(-log1p(neg_in(v)))."""
    model = DegreeLogReg(coef=np.array([0.0, -1.0, 0.0, 0.0]), intercept=0.0)
    deg = degree_table(GRAPH, GRAPH_POS, n_nodes=4)
    sources = np.array([0, 1, 2, 3])                      # v = 2 is itself a source: its self-pair is dropped
    before = 1 / (1 + np.exp(np.log1p(1.0)))               # sigmoid(-log 2) = 1/3
    assert abs(panel_score(model, deg, 2, sources) - before) < 1e-12
    assert abs(c2_logreg_shift(model, deg, GRAPH, ~GRAPH_POS, 2, sources) - (0.5 - before)) < 1e-12


def test_panel_score_uses_rater_out_degrees_and_drops_the_self_pair():
    model = DegreeLogReg(coef=np.array([0.0, 0.0, 0.0, -1.0]), intercept=0.0)
    deg = degree_table(GRAPH, GRAPH_POS, n_nodes=4)
    sig = lambda x: 1 / (1 + np.exp(-x))                   # noqa: E731
    want = np.mean([sig(0.0), sig(-np.log1p(2.0)), sig(0.0)])     # sources 0, 1, 3 (2 is v)
    assert abs(panel_score(model, deg, 2, np.array([0, 1, 2, 3])) - want) < 1e-12


def test_fit_degree_logreg_learns_the_sign_of_in_degrees():
    """Twenty raters each rate node 0 positively and node 1 negatively."""
    raters = np.arange(2, 22)
    edges = np.concatenate([np.column_stack([raters, np.zeros(20, int)]),
                            np.column_stack([raters, np.ones(20, int)])])
    is_pos = np.r_[np.ones(20, bool), np.zeros(20, bool)]
    model = fit_degree_logreg(edges, is_pos, n_nodes=22, loo="in")
    assert model.coef[0] > 0 > model.coef[1] and model.n_iter < model.max_iter and model.n_train == 40


def test_fit_degree_logreg_defaults_to_leave_one_out_on_both_endpoints():
    """R28: the headline yardstick removes the edge's own label from v's in- and u's out-degrees."""
    raters = np.arange(2, 22)
    edges = np.column_stack([raters, np.zeros(20, int)])
    is_pos = np.arange(20) % 2 == 0
    model = fit_degree_logreg(edges, is_pos, n_nodes=22)
    assert model.loo == "in_and_out" and model.summary()["loo"] == "in_and_out"
    assert edge_degree_counts(edges, is_pos, n_nodes=22).tolist() == \
        edge_degree_counts(edges, is_pos, n_nodes=22, loo="in_and_out").tolist()


# ------------------------------------------------------------------ probe4 record extraction
def fake_target(v, n_neg, n_pos=0, g1=0.30, g2=0.32, g1x=0.31, pb=0.03, pa1=0.20, curve=(),
                dmcc=-0.01, dmcc_placebo=0.02, d_src=0.001):
    return dict(
        target=v, n_neg=n_neg, n_pos=n_pos, stratum=M.stratum_of(n_pos), r0=0.5, pct_before=pb,
        count_gain=dict(laplace=float(M.counting_gain(n_pos, n_neg, n_neg, "laplace")),
                        raw=float(M.counting_gain(n_pos, n_neg, n_neg, "raw"))),
        direct=dict(ops=dict(
            unlink=dict(rep=0.97, gain=g1, d_src=d_src, d_tgt=10.0, pct_after=pa1),
            unlink_frozen_x=dict(gain=g1x),
            exact_retrain=dict(rep=0.96, gain=g2, pct_after=0.25, d_t1_mcc=dmcc))),
        direct_greedy_curve=[dict(k=k, cost=dict(edges=e), unlink_gain=g) for k, e, g in curve],
        placebo=None if dmcc_placebo is None else dict(ops=dict(exact_retrain=dict(d_t1_mcc=dmcc_placebo))))


def fake_record():
    return dict(targets=[
        fake_target(10, 1, g1x=0.50),
        fake_target(11, 2, n_pos=1, g1x=0.40, curve=[(1, 1, 0.10)], dmcc_placebo=None),
        fake_target(12, 3, g1x=0.60, curve=[(1, 1, 0.05), (2, 1, 0.07)]),   # greedy stopped after one edge
        fake_target(13, 3, g1x=0.30, curve=[(1, 1, 0.02), (2, 2, 0.20)]),
        fake_target(99, 2, curve=[(1, 1, 9.0)])])                          # a calibration target


def test_target_rows_exclude_calibration_targets_and_flatten_the_fields():
    rows = target_rows(fake_record(), calibration={99}, edgeless={10, 12})
    assert [r["target"] for r in rows] == [10, 11, 12, 13]
    assert [r["edgeless"] for r in rows] == [True, False, True, False]
    r = rows[1]
    assert (r["g_u1"], r["g_u2"], r["g_u1_frozen"], r["stratum"], r["n_pos"]) == (0.30, 0.32, 0.40, "pos", 1)
    assert (r["rep_u1"], r["pct_u1"], r["pct_u2"], r["d_tgt_u1"], r["dmcc_u2"]) == (0.97, 0.20, 0.25, 10.0, -0.01)
    assert r["dmcc_placebo_u2"] is None and rows[0]["dmcc_placebo_u2"] == 0.02
    assert rows[2]["curve"] == {1: 0.05, 2: 0.07} and rows[2]["curve_short"] == 1 and rows[3]["curve_short"] == 0


def test_target_rows_refuse_a_count_gain_that_disagrees_with_the_port():
    rec = fake_record()
    rec["targets"][0]["count_gain"]["laplace"] += 1e-3
    with pytest.raises(ValueError):
        target_rows(rec, calibration=set(), edgeless=set())


def test_partial_deletion_values_from_greedy_curves():
    """Cells keyed by target, so the crossed bootstrap can align a target across seeds."""
    got = partial_deletion_values(target_rows(fake_record(), calibration={99}, edgeless=set()))
    assert got == {"k1_of_2": {11: 0.10}, "k1_of_3": {12: 0.05, 13: 0.02}, "k2_of_3": {12: 0.07, 13: 0.20},
                   "full_of_2": {11: 0.40}, "full_of_3": {12: 0.60, 13: 0.30}}


def test_partial_deletion_values_refuse_an_incomplete_curve():
    rec = dict(targets=[fake_target(12, 3, curve=[(1, 1, 0.05)])])
    with pytest.raises(ValueError):
        partial_deletion_values(target_rows(rec, calibration=set(), edgeless=set()))


def test_amplification_block_is_the_seed_mean_of_per_seed_ratios_of_means():
    rows = {0: target_rows(dict(targets=[fake_target(1, 1, g2=0.2), fake_target(2, 2, g2=0.4)]), set(), set()),
            1: target_rows(dict(targets=[fake_target(1, 1, g2=0.3), fake_target(2, 2, g2=0.3)]), set(), set())}
    logreg, target_only = {1: 0.1, 2: 0.2}, {1: 0.3, 2: 0.3}
    lap = (laplace_shift(0, 1) + laplace_shift(0, 2)) / 2
    block = amplification_block(rows, p_pos=0.9, logreg_shift=logreg, boot=Boot("toy", n_boot=200),
                                sensitivity={"logreg_target_loo": target_only})
    assert abs(block["laplace"] - (0.3 / lap + 0.3 / lap) / 2) < 1e-12
    assert abs(block["logreg"] - (0.3 / 0.15 + 0.3 / 0.15) / 2) < 1e-12
    assert abs(block["logreg_target_loo"] - 1.0) < 1e-12
    assert block["by_seed"]["raw"] == pytest.approx([0.3 / 1.0, 0.3 / 1.0])
    fields = {"laplace", "raw", "base_rate", "logreg", "logreg_target_loo"}
    for key in ("ci95", "ci95_bootstrap", "ci95_nested", "ci95_seed_t", "ci95_source"):
        assert set(block[key]) == fields


def test_escape_counts_by_hand():
    """q = 0.05. Seed 0: target 1 escapes, target 2 stays below, target 3 starts above. Seed 1: targets 1
    and 3 escape, target 2 starts above. Below q: 4 evaluations of 3 targets, 3 escapes; targets 1 and 3
    escape in every seed where they start below q, target 2 does not.

    q = 0.10. Seed 0 as before. Seed 1: target 2 (0.06) now starts below and escapes (0.50); target 3
    ends at 0.06 and stays below. Below q: 5 evaluations of 3 targets, 3 escapes; only target 1 escapes
    every time."""
    from keeper.analysis.laundering_extra import _escape_counts

    def row(t, before, after):
        return dict(target=t, pct_before=before, pct_u1=after)
    rows = {0: [row(1, 0.01, 0.20), row(2, 0.03, 0.04), row(3, 0.50, 0.60)],
            1: [row(1, 0.02, 0.30), row(2, 0.06, 0.50), row(3, 0.01, 0.06)]}
    assert _escape_counts(rows, 0.05) == dict(level=0.05, n_below=4, n_escaped=3, n_targets_below=3,
                                              n_targets_always_escaped=2)
    assert _escape_counts(rows, 0.10) == dict(level=0.10, n_below=5, n_escaped=3, n_targets_below=3,
                                              n_targets_always_escaped=1)


def test_ci_design_is_crossed_only_when_every_seed_has_the_same_targets():
    assert ci_design_for({0: [3, 1, 2], 1: [1, 2, 3]}) == ("crossed", None)
    design, reason = ci_design_for({0: [1, 2, 3], 1: [1, 2, 4]})
    assert design == "nested" and "differ" in reason


def test_frac_over_counts_each_target_once_with_the_any_evaluation_rule():
    """Two targets over three seeds. With every rater-term shift below 0.003 the share is 0 and its interval
    is Clopper-Pearson over the 2 distinct targets, not the 6 evaluations. One evaluation above 0.003
    makes target 1 count as exceeding, and the share (1/6) gets the clipped hull."""
    from keeper.analysis.laundering_extra import logit_block

    def rows(d_src_first):
        return {s: target_rows(dict(targets=[fake_target(1, 1, d_src=d_src_first(s)),
                                             fake_target(2, 2, d_src=-0.002)]), set(), set()) for s in range(3)}
    g = logit_block(rows(lambda s: 0.001), Boot("toy", n_boot=200))
    assert g["frac_over_0.003"] == 0.0 and g["ci95"]["frac_over_0.003"] == clopper_pearson(0, 2)
    assert "2 distinct targets" in g["ci95_source"]["frac_over_0.003"]
    assert g["targets_over_0.003"] == dict(n_targets=2, n_targets_any_over=0)
    g = logit_block(rows(lambda s: 0.004 if s == 0 else 0.001), Boot("toy", n_boot=200))
    assert g["targets_over_0.003"] == dict(n_targets=2, n_targets_any_over=1)
    assert g["ci95_source"]["frac_over_0.003"].startswith("hull") and g["ci95"]["frac_over_0.003"][0] >= 0.0


def test_boot_falls_back_to_the_nested_design():
    rows = {s: target_rows(dict(targets=[fake_target(1, 1, g2=0.1 + s), fake_target(2, 2)]), set(), set())
            for s in range(3)}
    crossed, nested = Boot("toy", n_boot=200), Boot("toy", n_boot=200, design="nested")
    block_c = amplification_block(rows, 0.9, {1: 0.1, 2: 0.2}, crossed)
    block_n = amplification_block(rows, 0.9, {1: 0.1, 2: 0.2}, nested)
    assert block_c["ci95_nested"] == block_n["ci95_nested"] == block_n["ci95_bootstrap"]
    assert block_c["laplace"] == block_n["laplace"]
    with pytest.raises(ValueError):
        Boot("toy", design="pooled")


# ------------------------------------------------------------------ markdown expectation checks
def test_value_matches_at_the_stated_precision():
    assert status(Value(5.6, 1, scale=100), 0.0556) == "match"
    assert status(Value(5.6, 1, scale=100), 0.05549) == "mismatch"   # 5.549% is 5.5%, not 5.6%
    assert status(Value(1.38, 2), 1.3842) == "match"
    assert status(Value(-0.0006, 4), -0.000646) == "match"


def test_range_count_and_plus_minus():
    assert status(Range(0.935, 0.973, 3), (0.934537, 0.973379)) == "match"
    assert status(Range(34, 48, 0, scale=100), (0.3414, 0.4749)) == "mismatch"
    assert status(Count(81, 135), (81, 135)) == "match" and status(Count(81, 135), (80, 135)) == "mismatch"
    assert status(PlusMinus(0.9733, 0.0001), 0.97338) == "match"
    assert status(PlusMinus(0.9733, 0.0001), 0.97345) == "mismatch"


def _notes_block(lap_ci, edge_ci, c2, placebo, mean_ci, short, over):
    """A minimal per-dataset result with the fields the generated notes read."""
    return dict(
        amplification=dict(laplace=1.4, raw=0.4, base_rate=0.9, logreg=2.0, logreg_target_loo=1.9,
                           ci95=dict(laplace=lap_ci, raw=[0.3, 0.5], base_rate=[0.7, 1.1], logreg=[1.5, 2.5],
                                     logreg_target_loo=[1.4, 2.4])),
        amp_split=dict(edgeless=1.8, edge_keeping=1.1, pos=1.0, nopos=2.0,
                       ci95=dict(edgeless=[1.3, 2.3], edge_keeping=edge_ci, pos=[0.8, 1.3], nopos=[1.5, 2.5])),
        logit={"rater_term_mean": -0.002, "rater_term_mean_abs": 0.0075, "n_over_0.003": over, "n_eval": 240,
               "ci95": {"rater_term_mean_abs": [0.004, 0.012]}},
        max_dmcc=dict(c2_u2=c2, placebo_u2=placebo, mean_c2_u2=sum(mean_ci) / 2, ci95=dict(mean_c2_u2=mean_ci)),
        partial_deletion_u1=dict(n_curve_points_short=short, n_curve_points=250))


def test_paper_notes_generate_the_negative_findings():
    from keeper.analysis.laundering_extra_md import amplification_findings, claim_checks, paper_notes
    result = {"bitcoinalpha": _notes_block([1.13, 1.67], [0.95, 1.42], 0.07, 0.11, [-0.012, 0.003], 2, 135),
              "wiki-rfa": _notes_block([1.88, 2.18], [1.79, 2.00], 0.05, 0.04, [-0.013, -0.002], 0, 0)}
    includes, below = amplification_findings(result)
    assert "Bitcoin-Alpha, Laplace yardstick, edge-keeping targets: 1.10 [0.95, 1.42]" in includes
    assert "Bitcoin-Alpha, base-rate prior yardstick, all targets: 0.90 [0.70, 1.10]" in includes
    assert not any(x.startswith("Wiki-RfA, Laplace yardstick, edge-keeping") for x in includes)
    assert "Wiki-RfA, raw fraction yardstick, all targets: 0.40 [0.30, 0.50]" in below
    unsupported, supported, not_checked = claim_checks(result)
    bad = {(cid, ds) for cid, ds, _ in unsupported}
    assert {("P2", "bitcoinalpha"), ("P3", "bitcoinalpha"), ("P3", "wiki-rfa"), ("P4", "wiki-rfa"),
            ("P5", "wiki-rfa")} == bad
    assert {("P1", "bitcoinalpha"), ("P1", "wiki-rfa"), ("P4", "bitcoinalpha"), ("P5", "bitcoinalpha")} == \
        {(cid, ds) for cid, ds, _ in supported}
    assert ("P2", "bitcoinotc", "dataset not analysed") in not_checked                # OTC is not in result
    text = "\n".join(paper_notes(result))
    assert "135/240 (56%)" in text and "placebo 0.0400, C2 0.0500" in text and "Bitcoin-Alpha 2 of 250" in text
    assert "supported only if the 95% headline interval lies within +-0.007" in text
    assert "P3, Bitcoin-Alpha: point estimate -0.0045 (inside +-0.007), interval [-0.0120, +0.0030]" in text
    assert paper_notes({}).count("- none") == 3


def test_claims_with_missing_fields_are_listed_as_not_checked():
    from keeper.analysis.laundering_extra_md import claim_checks, paper_notes
    block = _notes_block([1.13, 1.67], [0.95, 1.42], 0.07, 0.11, [-0.012, 0.003], 0, 0)
    del block["logit"]
    unsupported, supported, not_checked = claim_checks({"bitcoinalpha": block})
    assert ("P2", "bitcoinalpha") in {(cid, ds) for cid, ds, _ in not_checked}
    assert all(cid != "P2" for cid, _, _ in unsupported + supported)
    assert "P2, Bitcoin-Alpha (missing field" in "\n".join(paper_notes({"bitcoinalpha": block}))


# ------------------------------------------------------------------ CLI guard (ruling R27) and stored data
class WorkStarted(Exception):
    """Raised by the first step of the analysis: reaching it means the overwrite guard let the run go on."""


def _work_started(*args, **kwargs):
    raise WorkStarted


def test_cli_refuses_an_existing_output_before_its_work(tmp_path, monkeypatch):
    from keeper.analysis import laundering_extra as LE
    from keeper.laundering import core as C
    monkeypatch.delenv(C.OVERWRITE_ENV, raising=False)
    monkeypatch.setattr(LE, "analyse_dataset", _work_started)
    out, md = tmp_path / "extra.json", tmp_path / "extra.md"
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        LE.main(["--out", str(out), "--md", str(md), "--datasets", "bitcoinalpha"])
    assert out.read_text() == "{}" and not md.exists()
    out.unlink()
    md.write_text("stored")
    with pytest.raises(FileExistsError):
        LE.main(["--out", str(out), "--md", str(md), "--datasets", "bitcoinalpha"])
    with pytest.raises(WorkStarted):
        LE.main(["--out", str(out), "--md", str(md), "--datasets", "bitcoinalpha", "--force"])


STORED = Path(__file__).resolve().parents[1] / "results" / "laundering" / "probe4_bitcoinalpha_s0.json"


@pytest.mark.skipif(not STORED.exists(), reason="stored probe4 results are not present")
def test_bitcoinalpha_reproduces_the_review_on_stored_results():
    """The port on the real stored records: review_evidence.md sections 1 to 3 for Bitcoin-Alpha."""
    from keeper.analysis.laundering_extra import Inputs, analyse_dataset
    out = analyse_dataset("bitcoinalpha", Inputs(), n_boot=200)
    assert (out["edgeless"]["n"], out["edgeless"]["cohort"]) == (81, 135)
    amp = out["amplification"]
    assert (round(amp["laplace"], 2), round(amp["raw"], 2), round(amp["base_rate"], 2)) == (1.38, 0.41, 0.86)
    assert out["percentile"]["escape_bottom5"] == 1.0 == out["percentile"]["escape_bottom10"]
    assert (out["logit"]["n_over_0.003"], out["logit"]["n_eval"]) == (135, 240)
    assert abs(out["shares"]["u1_over_u2"] - 0.99245) < 5e-6
    assert out["n_eval"] == [48] * 5 and out["calibration_excluded"] == [3684, 3724]
    assert out["ci_design"] == "crossed" and out["amplification"]["logreg_model"]["loo"] == "in_and_out"


# ------------------------------------------------------------------ cohort-weighted (post-stratified) estimates
def test_post_stratified_stat_weights_the_stratum_means_by_their_cohort_shares():
    """Column 0 holds the value, column 1 flags the positive-history stratum; NaN cells are missing targets."""
    from keeper.analysis.laundering_extra import post_stratified_stat
    batch = np.array([[[0.1, 0.0], [0.3, 0.0], [0.9, 1.0], [np.nan, np.nan]]])        # one replicate, four targets
    assert post_stratified_stat(0.2)(batch) == pytest.approx([0.2 * 0.9 + 0.8 * 0.2])
    assert post_stratified_stat(0.5)(batch) == pytest.approx([0.5 * 0.9 + 0.5 * 0.2])


def test_cohort_weighted_block_reproduces_the_probe_s_post_stratified_mean():
    """The block's point estimates equal keeper.laundering.metrics.post_stratified_mean on the same rows."""
    from keeper.analysis.laundering_extra import cohort_weighted_block
    rec = dict(targets=[fake_target(10 + i, 2, g2=0.40, pa1=0.20) for i in range(6)]            # no endorsement
                       + [fake_target(20 + i, 2, n_pos=1, g2=0.20, pa1=0.60) for i in range(6)])
    rows = {s: target_rows(rec, set(), set()) for s in (0, 1)}
    sizes = dict(pos=1, nopos=9)                                    # the cohort: 10% with an endorsement
    block = cohort_weighted_block(rows, sizes["pos"] / 10, Boot("x", 200, design="crossed"))
    strata = [r["stratum"] for r in rows[0]]
    for field, key in (("gain_u2", "g_u2"), ("pct_before", "pct_before"), ("pct_after_u1", "pct_u1")):
        want = M.post_stratified_mean([r[key] for r in rows[0]], strata, sizes)
        assert block[field] == pytest.approx(want), field
    assert block["gain_u2"] == pytest.approx(0.1 * 0.20 + 0.9 * 0.40)       # the plain sample mean would be 0.30
    assert block["pct_after_u1_pos"] == pytest.approx(0.60) and block["pct_after_u1_nopos"] == pytest.approx(0.20)
    assert block["cohort_share_pos"] == pytest.approx(0.1)
    assert block["ci95"]["gain_u2"][0] <= block["gain_u2"] <= block["ci95"]["gain_u2"][1]
