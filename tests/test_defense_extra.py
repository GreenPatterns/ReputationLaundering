"""Tests for keeper.analysis.defense_extra (Task 2.2, defense re-analysis of the stored results).

The unit cases are hand-computable:
* legacy tier coverage: seeds, the endorsed set D, late joiners and the tier mix on a small graph, and the
  guard that the replicated D agrees with keeper.credibility.tiers.legacy_bfs;
* single-warning revocation: the closed form without expiry (Eq. 8, revocation_curve_C) against its N = 2
  formula, the engine-rule form (witness evidence live for H snapshots, closure at A) against an
  independent per-stage derivation, their agreement when H and A are unbounded, and the SCEM streaming
  engine itself on a one-rating stream, which the engine-rule form must reproduce for both channel
  routings;
* the engine's hazard map: one witness at reliability 1.0 and at the floors, the per-contributor cap, and
  the number of fully reliable witnesses that reaches 1.9;
* the paired seed-level statistics of the pre-revision CPPR runs (t with df = n - 1, R30 intervals).
The CLI refuses to overwrite its outputs (ruling R27). Regression tests read the stored data and check
the review's numbers. CPU only.
"""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import t as student_t

from keeper.analysis import defense_extra as DE
from keeper.analysis.intervals import Boot
from keeper.laundering import signed as S
from keeper.scem import scem as SC

ROOT = Path(__file__).resolve().parents[1]


def graph(rows):
    """A training graph from (u, v, sign, snapshot) rows, in the given (stored) order."""
    rows = np.asarray(rows, dtype=np.int64)
    return SimpleNamespace(edges=rows[:, :2], labels=S.one_hot_from_sign(rows[:, 2]), snap=rows[:, 3])


# Node 0 is the only seed. Its positive rating of node 4 admits 4, and the ring 4 -> 5 -> 6 -> 4 follows;
# node 7's positive rating of node 3 does not count (7 is outside D). Tiers: [0, 0, 0, 0, 3, 1, 1, 0].
RING = [(1, 0, +1, 0), (2, 0, +1, 1), (0, 3, -1, 2),
        (0, 4, +1, 5), (4, 5, +1, 6), (5, 6, +1, 6), (6, 4, +1, 6), (7, 3, +1, 7), (3, 7, +1, 8)]


# ------------------------------------------------------------------ legacy tier coverage
def test_endorsed_set_replicates_the_seeded_pass():
    seeds, members = DE.endorsed_set(graph(RING), 8)
    assert seeds == {0} and members == {0, 4, 5, 6}


def test_first_snapshots_take_the_earliest_edge_at_either_end():
    first = DE.first_snapshots(graph(RING), 9)
    assert first.tolist() == [0, 0, 1, 2, 5, 6, 6, 7, -1]          # node 8 has no training edge


def test_tier_coverage_by_hand():
    cov = DE.tier_coverage(graph(RING), 8)
    assert (cov["n_active"], cov["n_seeds"], cov["n_D"]) == (8, 1, 4)
    assert cov["D_share_of_active"] == 0.5
    assert cov["n_D_positive_raters"] == 4                          # 0, 4, 5 and 6 each issue a +1
    assert (cov["n_late"], cov["n_late_in_D"], cov["late_in_D_share"]) == (4, 3, 0.75)   # 4, 5, 6, 7
    assert cov["tier_counts"] == [5, 2, 0, 1]
    assert cov["tier_shares"] == [0.625, 0.25, 0.0, 0.125] and cov["tier0_share"] == 0.625


def test_tier_coverage_refuses_a_replicated_set_that_disagrees_with_legacy_bfs(monkeypatch):
    monkeypatch.setattr(DE, "endorsed_set", lambda Tr, n: ({0}, {0, 4}))      # misses 5 and 6
    with pytest.raises(AssertionError, match="legacy_bfs"):
        DE.tier_coverage(graph(RING), 8)


# ------------------------------------------------------------------ single-warning revocation
PHI, LF = 0.5, 1.0


def no_expiry_by_hand(lag: float, lam: float) -> float:
    """N = 2: R = exp(-eta) ((1 - r^2) + eta (1 - r)), eta = phi (lag + L_f), r = phi / (phi + lam)."""
    eta, r = PHI * (lag + LF), PHI / (PHI + lam)
    return math.exp(-eta) * ((1 - r * r) + eta * (1 - r))


def engine_rules_by_hand(lag: float, lam: float, window: float = 6.0) -> float:
    """N = 2, per stage: a record still in stage n at the warning's maturity survives the hazard window
    if it completes its remaining stages before being revoked, or is neither revoked nor complete when
    the witness evidence expires."""
    x0, a = lag + LF, PHI + lam
    one_left = PHI / a * (1 - math.exp(-a * window)) + math.exp(-a * window)
    two_left = ((PHI / a) ** 2 * (1 - math.exp(-a * window) * (1 + a * window))
                + math.exp(-PHI * window) * (1 + PHI * window) * math.exp(-lam * window))
    eta = PHI * x0
    done = 1 - math.exp(-eta) * (1 + eta)
    return 1 - (done + math.exp(-eta) * two_left + eta * math.exp(-eta) * one_left)


def test_engine_constants_come_from_the_engine():
    assert (DE.ENGINE["N"], DE.ENGINE["phi"], DE.ENGINE["L_f"], DE.ENGINE["H"]) == (2, 0.5, 1, 6)
    assert DE.ENGINE["gamma"] == (1.0, 0.8) and DE.ENGINE["lambda_max"] == 2.0 and DE.ENGINE["cap_c"] == 1.0
    assert abs(DE.horizon_A() - 13.2767) < 1e-4


@pytest.mark.parametrize("lam", [0.10, 0.53, 1.0, 2.0])
@pytest.mark.parametrize("lag", [0, 1, 3])
def test_no_expiry_closed_form_matches_its_n2_formula(lam, lag):
    assert abs(DE.revoked_no_expiry(lag, lam) - no_expiry_by_hand(lag, lam)) < 1e-12


def test_no_expiry_at_the_ceiling_hazard_by_hand():
    assert abs(DE.revoked_no_expiry(1, 2.0) - 1.76 * math.exp(-1.0)) < 1e-12      # 64.7%


@pytest.mark.parametrize("lam", [0.10, 0.53, 1.0, 2.0])
@pytest.mark.parametrize("lag", [0, 1, 3])
def test_engine_rule_form_matches_the_per_stage_derivation(lam, lag):
    assert abs(DE.revoked_engine_rules(lag, lam) - engine_rules_by_hand(lag, lam)) < 1e-12


def test_engine_rule_form_without_expiry_or_closure_is_the_closed_form():
    for lam in (0.1, 0.53, 2.0):
        for lag in (0, 1, 3, 20):
            unbounded = DE.revoked_engine_rules(lag, lam, H=1e6, A=math.inf)
            assert abs(unbounded - DE.revoked_no_expiry(lag, lam)) < 1e-12


def test_expiry_lowers_the_revoked_fraction_of_a_weak_hazard_only():
    assert DE.revoked_no_expiry(1, 0.10) - DE.revoked_engine_rules(1, 0.10) > 0.005
    assert abs(DE.revoked_no_expiry(1, 2.0) - DE.revoked_engine_rules(1, 2.0)) < 1e-5


def test_nothing_is_revoked_without_a_hazard_or_after_closure():
    assert DE.revoked_engine_rules(1, 0.0) == pytest.approx(0.0, abs=1e-15)
    assert DE.revoked_no_expiry(1, 0.0) == pytest.approx(0.0, abs=1e-15)
    assert DE.revoked_engine_rules(13, 2.0) == 0.0          # matures at age 14 > A: the record has closed
    with pytest.raises(ValueError):
        DE.revoked_engine_rules(-1, 2.0)


def engine_revoked(lag: int, sign: int, tau: int) -> tuple:
    """SCEM engine on one rating 0 -> 1 at snapshot tau, warned by a fresh witness 2 at tau + lag."""
    out = SC.run_scem_streaming_weights_np(np.array([[0, 1], [2, 0]]), np.array([sign, -1]),
                                           np.array([tau, tau + lag]), 3, K_snap=21)
    return float(out["revoked_haz_final"][0]), float(1.0 - out["omega_final"][0]), float(out["pi_final"][0])


@pytest.mark.parametrize("tau, kappa", [(5, 0.20), (0, 0.35)])        # the floors after / up to snapshot 4
@pytest.mark.parametrize("lag", [0, 1, 3])
@pytest.mark.parametrize("sign, routing", [(+1, "positive"), (-1, "negative")])
def test_engine_rule_form_reproduces_the_streaming_engine(tau, kappa, lag, sign, routing):
    lam = DE.witness_hazard([kappa])["lam"]
    want = DE.revoked_by_routing(lag, lam, DE.ENGINE["psi"][routing])
    revoked, lost, provisional = engine_revoked(lag, sign, tau)
    assert provisional == 0.0                                    # the record has closed by snapshot 21
    assert abs(revoked - want) < 1e-12 and abs(lost - want) < 1e-12


# ------------------------------------------------------------------ hazard map
def hazard_by_hand(h: float, gamma: float) -> float:
    nu = math.log1p(math.exp(h)) - math.log(2.0)
    return 2.0 * (1.0 - math.exp(-gamma * nu / 2.0))


def test_one_witness_hazard_by_hand():
    full, floor = DE.witness_hazard([1.0]), DE.witness_hazard([0.20])
    assert full["h"] == 1.0 and abs(full["nu"] - (math.log1p(math.e) - math.log(2.0))) < 1e-15
    assert all(abs(a - hazard_by_hand(1.0, g)) < 1e-15 for a, g in zip(full["lam"], (1.0, 0.8)))
    assert all(abs(a - hazard_by_hand(0.2, g)) < 1e-15 for a, g in zip(floor["lam"], (1.0, 0.8)))
    assert round(full["lam"][0], 2) == 0.53 and round(floor["lam"][0], 2) == 0.10


def test_hazard_is_zero_without_evidence_and_each_witness_is_capped():
    assert DE.witness_hazard([])["lam"] == [0.0, 0.0]
    assert DE.witness_hazard([1.5])["h"] == 1.0                  # one contributor adds at most c = 1
    assert DE.witness_hazard([1.0, 1.0])["h"] == 2.0


def test_witnesses_needed_for_a_hazard():
    assert DE.witnesses_to_reach(1.9) == 7                       # 6 give 1.859, 7 give 1.915
    assert DE.witnesses_to_reach(0.5) == 1
    assert DE.witnesses_to_reach(2.0) is None                    # lambda stays below lambda_max = 2


def test_revocation_grid_rows():
    rows = DE.revocation_grid((0.10, 2.0), (0, 1))
    assert [(r["lam"], r["lag"]) for r in rows] == [(0.10, 0), (0.10, 1), (2.0, 0), (2.0, 1)]
    r = rows[1]
    assert r["revoked_engine_rules"] == DE.revoked_engine_rules(1, 0.10)
    assert r["revoked_no_expiry"] == DE.revoked_no_expiry(1, 0.10)
    assert r["durable_engine_rules"] == 1.0 - r["revoked_engine_rules"]


def test_hazard_map_rows_mix_the_channels_by_routing():
    row = DE.hazard_map([(1.0, "fully reliable witness")], (1,))[0]
    lam = DE.witness_hazard([1.0])["lam"]
    assert row["lam"] == lam and row["kappa"] == 1.0
    cell = row["revoked"][0]
    per_channel = [DE.revoked_engine_rules(1, x) for x in lam]
    assert cell["lag"] == 1 and cell["channel"] == per_channel
    assert abs(cell["positive_rating"] - (0.8 * per_channel[0] + 0.2 * per_channel[1])) < 1e-15
    assert abs(cell["negative_rating"] - (0.2 * per_channel[0] + 0.8 * per_channel[1])) < 1e-15


# ------------------------------------------------------------------ CPPR seed-level statistics
def test_paired_t_by_hand():
    out = DE.paired_t([1.0, 2.0, 3.0, 4.0, 5.0], [0.0] * 5)
    assert out["n"] == 5 and out["df"] == 4 and out["mean"] == 3.0
    assert abs(out["sd"] - math.sqrt(2.5)) < 1e-15
    assert abs(out["t"] - 3.0 / (math.sqrt(2.5) / math.sqrt(5))) < 1e-12
    assert abs(out["p_two_sided"] - 2 * student_t.sf(out["t"], 4)) < 1e-15


def test_paired_t_is_undefined_for_identical_differences():
    out = DE.paired_t([1.0, 2.0], [0.0, 1.0])
    assert out["t"] is None and out["p_two_sided"] is None


def toy_runs(vpe, cppr, dmcc, ds="toy", gate="count"):
    return [{"ds": ds, "seed": s, "gate_norm": gate,
             "inject_then_erase_whitewash": {"gain_plain_vpe": a, "gain_cp_vpe": b},
             "direct_laundering": {"d_t1_mcc_cp_vs_plain": m}} for s, (a, b, m) in enumerate(zip(vpe, cppr, dmcc))]


def test_cppr_seed_block_by_hand():
    vpe, cppr, dmcc = [0.10, 0.00, 0.03, 0.04, 0.11], [0.00, 0.01, -0.02, 0.0, 0.02], [-0.05, 0.07, 0.0, 0.01, 0.02]
    out = DE.cppr_seed_block(toy_runs(vpe, cppr, dmcc), Boot("toy", n_boot=400))
    diff = [a - b for a, b in zip(vpe, cppr)]
    assert out["label"] == DE.PRE_REVISION
    assert out["whitewash_paired_diff"] == pytest.approx(np.mean(diff), abs=1e-15)
    assert out["by_seed"]["whitewash_paired_diff"] == pytest.approx(diff, abs=1e-15)
    assert out["paired_t"]["t"] == pytest.approx(DE.paired_t(vpe, cppr)["t"], abs=1e-12)
    assert out["whitewash_plain_vpe_sd"] == pytest.approx(np.std(vpe, ddof=1), abs=1e-15)
    assert (out["d_mcc_cppr_minus_plain_min"], out["d_mcc_cppr_minus_plain_max"]) == (-0.05, 0.07)
    lo, hi = out["ci95"]["whitewash_paired_diff"]
    t_lo, t_hi = out["ci95_seed_t"]["whitewash_paired_diff"]
    assert lo <= t_lo and hi >= t_hi                                 # R30: the hull contains the seed-t
    assert out["ci95_support"]["whitewash_paired_diff"] is None      # signed: not clipped (R31)


def test_cppr_runs_are_checked():
    good = toy_runs([0.1] * 5, [0.0] * 5, [0.0] * 5)
    DE.check_cppr_runs("toy", good)
    with pytest.raises(ValueError, match="gate"):
        DE.check_cppr_runs("toy", toy_runs([0.1] * 5, [0.0] * 5, [0.0] * 5, gate="snapshot"))
    with pytest.raises(ValueError, match="seed"):
        DE.check_cppr_runs("toy", good[1:] + good[:1])
    with pytest.raises(ValueError, match="other"):
        DE.check_cppr_runs("bitcoinalpha", good)


# ------------------------------------------------------------------ CLI guard (ruling R27)
class WorkStarted(Exception):
    """Raised by the first step of the analysis: reaching it means the overwrite guard let the run go on."""


def _work_started(*args, **kwargs):
    raise WorkStarted


def test_cli_refuses_an_existing_output_before_its_work(tmp_path, monkeypatch):
    from keeper.laundering import core as C
    monkeypatch.delenv(C.OVERWRITE_ENV, raising=False)
    monkeypatch.setattr(DE, "analyse", _work_started)
    out, md = tmp_path / "defense.json", tmp_path / "defense.md"
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        DE.main(["--out", str(out), "--md", str(md)])
    assert out.read_text() == "{}" and not md.exists()
    out.unlink()
    md.write_text("stored")
    with pytest.raises(FileExistsError):
        DE.main(["--out", str(out), "--md", str(md)])
    with pytest.raises(WorkStarted):
        DE.main(["--out", str(out), "--md", str(md), "--force"])


# ------------------------------------------------------------------ stored data: the review's numbers
CPPR_STORED = ROOT / "results" / "laundering" / "cpgate_count_bitcoinalpha_s0.json"


@pytest.mark.skipif(not CPPR_STORED.exists(), reason="stored CPPR runs are not present")
def test_cppr_seed_statistics_reproduce_the_review():
    expected = {"bitcoinalpha": (0.055, 0.041, 3.00, -0.053, 0.070), "bitcoinotc": (0.048, 0.037, 2.87, -0.077, 0.074),
                "wiki-rfa": (0.322, 0.057, 12.69, -0.066, 0.010)}
    for ds, (mean, sd, t, lo, hi) in expected.items():
        out = DE.cppr_seed_block(DE.load_cppr_runs(ds, DE.Inputs()), Boot(ds, n_boot=200))
        assert round(out["whitewash_paired_diff"], 3) == mean and round(out["paired_t"]["sd"], 3) == sd
        assert round(out["paired_t"]["t"], 2) == t and out["paired_t"]["df"] == 4
        assert (round(out["d_mcc_cppr_minus_plain_min"], 3), round(out["d_mcc_cppr_minus_plain_max"], 3)) == (lo, hi)
    alpha = DE.cppr_seed_block(DE.load_cppr_runs("bitcoinalpha", DE.Inputs()), Boot("bitcoinalpha", n_boot=200))
    assert [round(x, 3) for x in alpha["by_seed"]["whitewash_plain_vpe"]] == [0.100, -0.000, 0.030, 0.036, 0.108]


def test_single_warning_table_reproduces_the_review_and_the_paper():
    lag1 = {lam: DE.revoked_engine_rules(1, lam) for lam in (2.0, 0.53, 0.10)}
    assert [round(100 * lag1[lam], 1) for lam in (2.0, 0.53, 0.10)] == [64.7, 46.8, 16.4]
    assert [round(100 * (1 - lag1[lam])) for lam in (2.0, 0.53, 0.10)] == [35, 53, 84]
    assert [round(100 * DE.revoked_no_expiry(s, 2.0), 1) for s in (0, 1, 3)] == [82.5, 64.7, 34.6]


def test_bitcoinalpha_tier_coverage_reproduces_the_review():
    from keeper.laundering import core as C
    D = C.load("bitcoinalpha")
    cov = DE.tier_coverage(C.train_split(C.full_graph(D), D.tt), D.n)
    assert (cov["n_active"], cov["n_seeds"], cov["n_D"], cov["n_D_positive_raters"]) == (3629, 1188, 3453, 3123)
    assert round(100 * cov["D_share_of_active"], 1) == 95.2 and round(100 * cov["late_in_D_share"], 1) == 93.1
    assert round(100 * cov["tier0_share"], 1) == 7.2


# ------------------------------------------------------------------ markdown: reproduction table and notes
def test_mean_sd_and_sequence_match_at_the_stated_precision():
    from keeper.analysis.defense_extra_md import MeanSd, Seq
    assert MeanSd(0.055, 0.047, 3).check((0.05486, 0.04716))
    assert not MeanSd(0.055, 0.047, 3).check((0.0556, 0.04716)) and not MeanSd(0.055, 0.047, 3).check((0.055, 0.0476))
    assert MeanSd(0.055, 0.047, 3).expected() == "+0.055 +- 0.047"
    seq = Seq((0.100, -0.000, 0.030, 0.036, 0.108), 3)
    assert seq.check([0.10034, -0.00004, 0.03005, 0.03584, 0.10812])
    assert not seq.check([0.10034, -0.00004, 0.03005, 0.03584]) and not seq.check([0.1006, 0, 0.03, 0.036, 0.108])


@pytest.fixture(scope="module")
def analysed():
    """The whole analysis on the stored data (200 bootstrap replicates), as main() hands it to the renderer."""
    import json
    from keeper.laundering import core as C
    result = DE.analyse(DE.NETWORKS, DE.Inputs(), n_boot=200)
    return json.loads(json.dumps(C._finite(result), default=C._json_default))


@pytest.mark.skipif(not CPPR_STORED.exists(), reason="stored CPPR runs are not present")
def test_every_review_number_is_reproduced_and_shown_next_to_its_expected_value(analysed):
    from keeper.analysis.defense_extra_md import render_markdown, reproduction_rows
    rows = reproduction_rows(analysed)
    assert len(rows) >= 40
    assert [r for r in rows if r["status"] != "match"] == []
    text = render_markdown(analysed)
    for r in rows:
        assert f"| {r['expected']} | {r['measured']} | match |" in text


@pytest.mark.skipif(not CPPR_STORED.exists(), reason="stored CPPR runs are not present")
def test_notes_follow_the_values(analysed):
    import copy
    from keeper.analysis.defense_extra_md import paper_notes
    text = "\n".join(paper_notes(analysed))
    assert "pre-revision runs (superseded by E3)" in text
    assert "Wiki-RfA: +0.322" in text and "excludes 0" in text
    flipped = copy.deepcopy(analysed)
    block = flipped["cppr_seed_stats"]["wiki-rfa"]
    block["ci95"]["whitewash_paired_diff"] = [-0.01, 0.6]
    for ds in DE.NETWORKS:                        # every per-seed MCC difference made positive
        flipped["cppr_seed_stats"][ds]["d_mcc_cppr_minus_plain_min"] = 0.001
    text2 = "\n".join(paper_notes(flipped))
    assert "Wiki-RfA: +0.322 [-0.010, +0.600]" in text2 and "includes 0" in text2
    assert "both signs" in text and "both signs" not in text2
    assert "more than half of the active accounts on Bitcoin-Alpha (95.2%) and Bitcoin-OTC (91.3%)" in text
    assert "fewer than half on Wiki-RfA (22.8%)" in text
    flipped["tier_coverage"]["bitcoinalpha"]["D_share_of_active"] = 0.3
    text3 = "\n".join(paper_notes(flipped))
    assert "fewer than half on Bitcoin-Alpha (30.0%) and Wiki-RfA (22.8%)" in text3
    # Plain VPE reference gain with an interval including 0 (Alpha); CPPR residual gain above 0 (OTC)
    assert "the reference of the attack cut, has an interval that includes 0 on Bitcoin-Alpha (+0.055 [" in text
    assert "The CPPR whitewash gain has an interval above 0 on Bitcoin-OTC (+0.024 [" in text
    flipped["cppr_seed_stats"]["bitcoinalpha"]["ci95"]["whitewash_plain_vpe"] = [0.01, 0.1]
    flipped["cppr_seed_stats"]["bitcoinotc"]["ci95"]["whitewash_cppr"] = [-0.01, 0.04]
    text4 = "\n".join(paper_notes(flipped))
    assert "The Plain VPE whitewash gain has an interval that excludes 0 on every network." in text4
    assert "No network has a CPPR whitewash gain whose interval lies above 0." in text4
