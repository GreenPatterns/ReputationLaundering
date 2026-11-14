"""Task 4.9: Holm's step-down, every decision branch, the code-defect flag, partial inputs and bit-identical reruns of
keeper.analysis.decisions."""
import json
import os
import tempfile

import pytest

from keeper.analysis import decisions as D
from keeper.cluster import protocol_settings as PS
from keeper.laundering import core as C


@pytest.fixture
def scratch():
    """A temporary folder inside the repository (nothing is written outside it)."""
    with tempfile.TemporaryDirectory(dir=C.ROOT, prefix=".scratch_test_") as folder:
        yield folder


def comp(ci, p=0.001, evaluable=True, decision=None):
    return dict(estimate=None if ci is None else sum(ci) / 2, ci95=ci, p_value=p, evaluable=evaluable, n_seeds=5,
                n_seeds_expected=5, decision=decision)


def e9a_doc(a=(0.1, 0.3), b=(0.6, 0.9), p=0.001, overrides=None) -> dict:
    """An e9a_e8.json with H8's components on every network; overrides maps (ds, component) to a component."""
    nets = {ds: dict(components=dict(a=comp(list(a), p), b=comp(list(b), p))) for ds in PS.NETWORKS}
    for (ds, name), c in (overrides or {}).items():
        nets[ds]["components"][name] = c
    return dict(status="ok", hypotheses=dict(H8=dict(networks=nets, p_value=p)),
                integrity=dict(missing_runs=[dict(run="e9a_bitcoinotc_s7_on", reason="result missing")]))


def write(root, name, doc):
    path = os.path.join(root, D.SOURCES[name])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f)


# ------------------------------------------------------------------------ Holm
def test_holm_by_hand_with_the_step_down_stopping_point():
    p = dict(A=0.001, B=0.0075, C=0.008, D=0.03, E=0.2, F=0.5, G=0.9, H=None)
    adj = D.holm(p)
    assert adj["A"] == pytest.approx(0.008)
    assert adj["B"] == pytest.approx(0.0525)              # 7 x 0.0075 >= 0.05: the step-down stops here
    assert adj["C"] == pytest.approx(0.0525)              # 6 x 0.008 = 0.048 < 0.05, yet not rejected after B
    assert adj["D"] == pytest.approx(0.15) and adj["E"] == pytest.approx(0.8)
    assert adj["F"] == adj["G"] == adj["H"] == 1.0
    assert [n for n, q in adj.items() if q < D.ALPHA] == ["A"]


def test_holm_ties_share_one_adjusted_value_and_missing_p_values_are_conservative():
    assert D.holm(dict(A=0.01, B=0.01)) == dict(A=0.02, B=0.02)
    partial = D.holm(dict(A=0.004, B=0.02, C=None))
    final = D.holm(dict(A=0.004, B=0.02, C=0.001))
    assert all(final[k] <= partial[k] for k in ("A", "B"))


# ------------------------------------------------------------------------ component readings
@pytest.mark.parametrize("ci, reading, contradicted", [
    ([0.6, 0.9], "met", False), ([0.3, 0.7], "inconclusive", False), ([0.1, 0.4], "partial", False),
    ([-0.5, -0.1], "partial", True)])
def test_three_way_reading_of_a_removal_threshold(ci, reading, contradicted):
    c = D.read_component(comp(ci), (">=", 0.5))
    assert (c["reading"], c["contradicted"], c["met"]) == (reading, contradicted, reading == "met")


@pytest.mark.parametrize("rule, ci, reading", [
    ((">", 0.0), [0.1, 0.3], "met"), ((">", 0.0), [-0.1, 0.3], "not met"), ((">", 0.0), [-0.3, -0.1], "contradicted"),
    (("<", 0.6), [0.40, 0.55], "met"), (("<", 0.6), [0.50, 0.70], "not met"), (("<", 0.6), [0.65, 0.80], "contradicted")])
def test_difference_and_upper_bound_rules(rule, ci, reading):
    assert D.read_component(comp(ci), rule)["reading"] == reading


def test_a_component_without_results_or_with_too_few_seeds_is_not_evaluable():
    assert D.read_component(None, (">", 0.0))["reading"] == "missing"
    assert D.read_component(comp([0.1, 0.3], evaluable=False), (">", 0.0))["reading"] == "not evaluable"
    assert D.read_component(comp(None), (">", 0.0))["evaluable"] is False


# ------------------------------------------------------------------------ hypothesis decisions
def decide_h8(scratch, doc) -> dict:
    write(scratch, "e9a_e8", doc)
    return D.analyse(scratch)["hypotheses"]["H8"]


def test_supported_when_every_component_is_met_and_holm_passes(scratch):
    h8 = decide_h8(scratch, e9a_doc(p=0.001))
    assert h8["decision"] == "supported" and h8["p_holm"] == pytest.approx(0.008) and not h8["provisional"]
    assert h8["readings"]["bitcoinalpha"] == dict(a="met", b="removal met")


def test_not_supported_by_holm_alone_is_provisional(scratch):
    h8 = decide_h8(scratch, e9a_doc(p=0.01))               # 8 x 0.01 with seven hypotheses pending
    assert (h8["decision"], h8["provisional"]) == ("not supported", True)


def test_not_supported_by_an_inconclusive_threshold_with_its_reading(scratch):
    h8 = decide_h8(scratch, e9a_doc(overrides={("wiki-rfa", "b"): comp([0.3, 0.7])}))
    assert (h8["decision"], h8["provisional"]) == ("not supported", False)
    assert h8["readings"]["wiki-rfa"]["b"] == "removal not established" and h8["null_reading_applies"]


def test_a_contradicted_component_makes_the_hypothesis_contradicted(scratch):
    h8 = decide_h8(scratch, e9a_doc(overrides={("bitcoinotc", "a"): comp([-0.3, -0.1])}))
    assert h8["decision"] == "contradicted" and h8["contradicted"] == ["bitcoinotc/a"]


def test_a_component_with_too_few_seeds_leaves_the_hypothesis_not_evaluable(scratch):
    h8 = decide_h8(scratch, e9a_doc(overrides={("bitcoinalpha", "b"): comp([0.6, 0.9], evaluable=False)}))
    assert h8["decision"] == "not evaluable" and h8["p_value"] is None and h8["p_holm"] is None
    assert h8["missing_runs"] == ["e9a_bitcoinotc_s7_on (result missing)"]


def test_a_reading_that_differs_from_the_module_is_noted(scratch):
    h8 = decide_h8(scratch, e9a_doc(overrides={("bitcoinalpha", "a"): comp([0.1, 0.3], decision="not met")}))
    assert any("bitcoinalpha/a" in note for note in h8["notes"])


# ------------------------------------------------------------------------ partial inputs, defects, reruns
def test_missing_inputs_make_their_hypotheses_pending_and_the_run_still_succeeds(scratch):
    assert D.main(["--results-root", scratch]) == D.EXIT_OK
    with open(os.path.join(scratch, "results", "analysis", "decisions.json")) as f:
        result = json.load(f)
    assert result["pending"] == list(D.FAMILY) + ["H3a", "H4", "H5"]
    assert all(not i["present"] for i in result["inputs"].values()) and not result["holm"]["complete"]
    with open(os.path.join(scratch, "results", "analysis", "decisions.md")) as f:
        assert "Pending (input missing): H1, H2w" in f.read()


@pytest.mark.parametrize("source, doc", [
    ("cppr_confirm", dict(h3a=dict(ok=False, n_runs=1, n_groups=3, violations=[dict(run="e4_x", group=0)]))),
    ("e1", dict(h4=dict(verdict="supported"), h5=dict(verdict="defect", defect=True), missing=[]))])
def test_a_violated_exact_property_is_a_code_defect_and_exits_non_zero(scratch, source, doc):
    write(scratch, source, doc)
    assert D.main(["--results-root", scratch]) == D.EXIT_CODE_DEFECT
    with open(os.path.join(scratch, "results", "analysis", "decisions.json")) as f:
        assert json.load(f)["code_defect"] is True


def test_reruns_are_bit_identical(scratch):
    write(scratch, "e9a_e8", e9a_doc())
    out = os.path.join(scratch, "results", "analysis")
    texts = []
    for _ in range(2):
        D.main(["--results-root", scratch, "--force"])
        texts.append([open(os.path.join(out, n), "rb").read() for n in ("decisions.json", "decisions.md")])
    assert texts[0] == texts[1]
    with pytest.raises(FileExistsError):                  # R27: no overwrite without --force
        D.main(["--results-root", scratch])


def test_the_readings_agree_with_the_producing_modules_on_their_own_output(scratch):
    """Schema check: the CPPR (Alpha only) and E2 analyses of their tests' synthetic runs."""
    from keeper.analysis import e2_analysis as A2
    from keeper.cluster.e2_tasks import e2_runs
    from tests import cppr_confirm_fixtures as F
    from tests import test_cppr_confirm as TC
    from tests import test_e2_analysis as TE
    runs = os.path.join(scratch, "runs")
    write(scratch, "cppr_confirm", json.loads(json.dumps(C._finite(TC.analyse(runs, F.build_results(runs))),
                                                         default=C._json_default)))
    e2 = A2.analyse([TE.record(*k) for k in e2_runs()], dict(path="x", sha256=TE.SHA), n_boot=200)
    write(scratch, "e2", json.loads(json.dumps(C._finite(e2), default=C._json_default)))
    result = D.analyse(scratch)
    hyps = result["hypotheses"]
    assert all(not b["notes"] for b in hyps.values())
    assert hyps["H1"]["decision"] == "not evaluable"                          # OTC and Wiki-RfA not analysed
    assert hyps["H1"]["readings"]["bitcoinalpha"]["b_cut"] == "removal met"
    assert hyps["H6a"]["p_value"] == e2["hypotheses"]["H6a"]["p_value"] and hyps["H6a"]["decision"] == "supported"
    assert result["exact"]["H3a"]["decision"] == "holds" and result["pending"] == ["H7", "H8", "H4", "H5"]
