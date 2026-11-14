"""Task 4.4: keeper.analysis.e1_analysis. The census, H5's eligibility, the H4 and H5 rules on constructed numbers,
shard merging and the integrity checks against keeper.cluster.protocol_settings. CPU only; no network is loaded."""
import copy
import json

import numpy as np
import pytest

from keeper.analysis import e1_analysis as A
from keeper.cluster import protocol_settings as ps
from keeper.credibility import tiers
from keeper.laundering import core as C
from keeper.scem import e1_runner as R


def roundtrip(obj):
    return json.loads(json.dumps(C._finite(obj), default=C._json_default))


def test_census_is_mean_p95_and_max_over_every_unit():
    assert A.census([0, 1, 2, 3, 4]) == dict(n=5, mean=2.0, p95=3.8, max=4.0)
    assert A.census([]) == dict(n=0, mean=None, p95=None, max=None)


# ------------------------------------------------------------------ H4


def h4_rows(mean_by_cell, mb="on"):
    return [dict(k=k, g=0, lag=1, ring=ring, farming=False, mb=mb, total=A.census([mean_by_cell[k, ring]] * 3))
            for (k, ring) in mean_by_cell]


def test_h4_holds_at_the_threshold_and_fails_just_above_it():
    at = {(k, ring): 0.01 for k in (5, 50) for ring in (False, True)}
    above = {**at, (50, False): 0.0100001}
    result = A.h4_rule({ds: h4_rows(at) for ds in ps.NETWORKS})
    assert result["verdict"] == "supported"
    assert all(c["holds"] for b in result["networks"].values() for c in b["cells"])
    result = A.h4_rule({"bitcoinalpha": h4_rows(above), "bitcoinotc": h4_rows(at), "wiki-rfa": h4_rows(at)})
    assert result["verdict"] == "not_supported" and result["networks"]["bitcoinalpha"]["status"] == "fails"


def test_h4_reads_only_mb_on_and_is_not_evaluable_without_a_cell_or_a_network():
    at = {(k, ring): 0.0 for k in (5, 50) for ring in (False, True)}
    off_only = {ds: h4_rows({key: 0.5 for key in at}, mb="off") + h4_rows(at) for ds in ps.NETWORKS}
    assert A.h4_rule(off_only)["verdict"] == "supported"                 # M-B off rows are ignored
    missing_cell = {(5, False): 0.0, (5, True): 0.0, (50, True): 0.0}
    result = A.h4_rule({ds: h4_rows(missing_cell) for ds in ps.NETWORKS})
    assert result["verdict"] == "not_evaluable"
    assert A.h4_rule({ds: h4_rows(at) for ds in ps.NETWORKS[:2]})["verdict"] == "not_evaluable"


# ------------------------------------------------------------------ H5


def test_h5_eligibility():
    t_k = np.array([0, 5, 6])
    assert A.eligible(t_k, 14, 13.28, 1, 21).tolist() == [True, True, False]     # t_k + 15 <= 20
    assert A.eligible(t_k, 7, 13.28, 1, 21).tolist() == [False] * 3               # 8 < A
    assert A.eligible(t_k, 3, 4.0, 1, 21).tolist() == [True] * 3                  # quarantine: 4 >= 4
    assert A.eligible(t_k, 56, 18.42, 1, 88).tolist() == [True, True, True]


def h5_doc(diff_erlang2=0.0, contrast=0.02, ineligible_diff=0.3):
    """Units at lag 14 (t_k 2, eligible for every Erlang policy, not for exponential) and lag 3 (quarantine only),
    plus a record ineligible at both lags (t_k 18) whose difference is large."""
    pols = R.policy_table(R.E1C_POLICIES)
    base = {p: [0.1, 0.1] for p in R.E1C_POLICIES}

    def unit(lag, t_k, diffs):
        attack = {p: [b + diffs.get(p, 0.0), b + ineligible_diff] for p, b in ((p, 0.1) for p in R.E1C_POLICIES)}
        return dict(cell=0, batch=0, lag=lag, t_k=[t_k, 18], attack=attack, clean=base)

    units = [unit(14, 2, {"erlang2": diff_erlang2, "no_commit": contrast}), unit(3, 2, {"no_commit": contrast})]
    return roundtrip(dict(stream=dict(K=21), L_f=1, policies=pols, lags=[3, 14], units=units))


def test_h5_policies_holds_untested_defect_and_weak_contrast():
    rows = A.h5_policies(h5_doc())
    assert rows["exponential"]["status"] == "untested" and rows["exponential"]["n_eligible"] == 0
    for p in ("erlang2", "erlang3", "erlang5"):
        assert rows[p]["status"] == "holds" and rows[p]["n_eligible"] == 1
    assert rows["quarantine"]["n_eligible"] == 2 and rows["quarantine"]["no_commit"]["mean"] == pytest.approx(0.02)
    assert A.h5_policies(h5_doc(diff_erlang2=2e-12))["erlang2"]["status"] == "defect"
    assert A.h5_policies(h5_doc(diff_erlang2=5e-13))["erlang2"]["status"] == "holds"
    assert A.h5_policies(h5_doc(diff_erlang2=-1e-9))["erlang2"]["status"] == "defect"    # |attack - clean|
    assert A.h5_policies(h5_doc(diff_erlang2=-5e-13))["erlang2"]["status"] == "holds"
    assert A.h5_policies(h5_doc(contrast=0.005))["erlang2"]["status"] == "contrast_below_threshold"


def test_h5_verdict():
    good = {(label, ds): A.h5_policies(h5_doc()) for label in ("21", "88") for ds in ps.NETWORKS}
    result = A.h5_rule(good)
    assert result["verdict"] == "supported" and not result["defect"] and len(result["untested"]) == 6
    bad = {**good, ("88", "wiki-rfa"): A.h5_policies(h5_doc(diff_erlang2=1e-6))}
    assert A.h5_rule(bad)["verdict"] == "defect" and A.h5_rule(bad)["defect"]
    weak = {**good, ("21", "bitcoinotc"): A.h5_policies(h5_doc(contrast=0.0))}
    assert A.h5_rule(weak)["verdict"] == "not_supported"
    partial = {key: block for key, block in good.items() if key[0] == "21"}
    result = A.h5_rule(partial)
    assert result["verdict"] == "pending" and not result["defect"]
    assert result["pending"] == [f"88/{ds}" for ds in ps.NETWORKS]


def test_h5_verdict_comes_from_the_single_record_studies_and_batched_numbers_are_descriptive():
    singles = {(study, ds): h5_doc() for study in A.SINGLE_LABEL for ds in ps.NETWORKS}
    batched = {(study, ds): h5_doc(diff_erlang2=0.05) for study in A.STREAM_LABEL for ds in ps.NETWORKS}
    result = A.analyse({**singles, **batched})
    assert result["h5"]["verdict"] == "supported" and not result["h5"]["defect"]
    row = result["h5_batched"]["88/wiki-rfa"]["erlang2"]
    assert (row["n_eligible"], row["share_nonzero"]) == (1, 1.0) and row["max_abs"] == pytest.approx(0.05)
    assert result["h5_batched"]["21/bitcoinalpha"]["quarantine"]["share_nonzero"] == 0.0
    only_batched = A.analyse(batched)["h5"]
    assert only_batched["verdict"] == "pending" and len(only_batched["pending"]) == 6 and not only_batched["defect"]
    bad = {**singles, ("e1d_h5_single", "wiki-rfa"): h5_doc(diff_erlang2=-1e-9)}
    assert A.analyse({**bad, **batched})["h5"]["defect"]
    md = A.render_md(dict(result, protocol_sha256="0" * 64,
                          code=dict(git_commit="abc", git_commit_single="def", any_dirty=False)))
    batched_md = md.split("## Batched attacks (descriptive): several records attacked in one stream")[1]
    line = next(x for x in batched_md.splitlines() if x.startswith("| 88/wiki-rfa | erlang2 |"))
    assert line.endswith("| 1 | 1.000 | 0.0500 | 0.0500 |")


# ------------------------------------------------------------------ merging and integrity


def calibration(ds="bitcoinalpha"):
    n_ref = [0] + [100 * t for t in range(1, 21)]
    return dict(variant="capped", eps=ps.EPS_CAPPED[ds], edge_cap_units=ps.EDGE_CAP_UNITS, neg_from=ps.NEG_FROM,
                t_seed=4, K_t=[tiers.default_K(n) for n in n_ref], n_ref_t=n_ref, theta0_t=[None] * 21, n_seeds=10,
                K_full=ps.K_EXPECTED[ds], n_ref_full=ps.N_REF_EXPECTED[ds], theta0_full=0.0)


def e1b_doc(index=0, count=1, units=None, commit="abc", ds="bitcoinalpha", study="e1b"):
    cells = [dict(index=0, n_batches=2), dict(index=1, n_batches=1)]
    units = [dict(cell=c, batch=b) for c, b in ((0, 0), (0, 1), (1, 0))] if units is None else units
    return roundtrip(dict(study=study, ds=ds, shard=dict(index=index, count=count), engine_defaults=R.engine_defaults(),
                          preregistration=dict(path="p", sha256="sha"), provenance=dict(git_commit=commit),
                          stream=dict(K=21, t_seed=4, n_ref=ps.N_REF_EXPECTED[ds]),
                          policies=R.policy_table(["erlang2"]), calibration={"capped": calibration(ds)}, cells=cells,
                          n_units=3, units=units))


def test_merge_shards_needs_every_shard_and_every_unit_once():
    whole = A.merge_shards([e1b_doc()])[("e1b", "bitcoinalpha")]
    parts = [e1b_doc(0, 2, [dict(cell=0, batch=1), dict(cell=1, batch=0)]), e1b_doc(1, 2, [dict(cell=0, batch=0)])]
    assert A.merge_shards(parts)[("e1b", "bitcoinalpha")]["units"] == whole["units"]
    with pytest.raises(A.IntegrityError, match="shards"):
        A.merge_shards(parts[:1])
    with pytest.raises(A.IntegrityError, match="exactly once"):
        A.merge_shards([parts[0], e1b_doc(1, 2, [dict(cell=0, batch=1)])])
    single = [e1b_doc(0, 2, [dict(cell=0, batch=1)], study="e1c_single"),
              e1b_doc(1, 2, [dict(cell=0, batch=0), dict(cell=1, batch=0)], study="e1c_single")]
    assert A.merge_shards(single)[("e1c_single", "bitcoinalpha")]["units"] == whole["units"]
    with pytest.raises(A.IntegrityError, match="exactly once"):
        A.merge_shards([single[0], e1b_doc(1, 2, [dict(cell=0, batch=1)], study="e1c_single")])


def test_integrity_accepts_the_registered_settings_and_names_each_departure():
    good = A.merge_shards([e1b_doc()])
    assert A.assert_integrity(good, "sha") == dict(git_commit="abc", git_commit_single=None, any_dirty=False)
    for path, value, message in [(("calibration", "capped", "eps"), 0.05, "eps"),
                                 (("calibration", "capped", "neg_from"), "all", "neg_from"),
                                 (("calibration", "capped", "K_full"), 11, "whole-stream"),
                                 (("stream", "n_ref"), 3628, "n_ref"),
                                 (("engine_defaults", "L_f"), 2, "engine defaults"),
                                 (("policies", "erlang2", "phi"), 0.25, "policy erlang2")]:
        doc = copy.deepcopy(good[("e1b", "bitcoinalpha")])
        node = doc
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with pytest.raises(A.IntegrityError, match=message):
            A.assert_integrity({("e1b", "bitcoinalpha"): doc}, "sha")
    with pytest.raises(A.IntegrityError, match="protocol sha256"):
        A.assert_integrity(good, "other")
    two = {**good, **A.merge_shards([e1b_doc(commit="def", ds="bitcoinotc")])}
    with pytest.raises(A.IntegrityError, match="2 commits"):
        A.assert_integrity(two, "sha")


def test_single_record_studies_may_come_from_their_own_commit():
    good = A.merge_shards([e1b_doc(), e1b_doc(commit="def", study="e1c_single")])
    assert A.assert_integrity(good, "sha") == dict(git_commit="abc", git_commit_single="def", any_dirty=False)
    two = {**good, **A.merge_shards([e1b_doc(commit="ghi", ds="bitcoinotc", study="e1c_single")])}
    with pytest.raises(A.IntegrityError, match="single-record outputs come from 2 commits"):
        A.assert_integrity(two, "sha")
    h5_88 = copy.deepcopy(good[("e1c_single", "bitcoinalpha")])
    h5_88["study"] = "e1d_h5_single"
    with pytest.raises(A.IntegrityError, match="expected 88, 19"):             # the 88-snapshot stream's K, t_seed
        A.assert_integrity({("e1d_h5_single", "bitcoinalpha"): h5_88}, "sha")
