"""SCEM audit values of the paper, regenerated from code.

Silent-sender trace (paper Table III, Block 2): keeper.scem.audit.silent_sender_trace(), the Round-5
audit's item 6, integrates the engine's closed-form release (keeper.scem.scem) for a sender whose
witness warning matures at age 2 under the ceiling hazard lambda = 2. The paper prints its omega_k,
pi_k and surviving mass to three decimals and R_k = 1 - omega_k at closure. The audit's stored record
rounds to four decimals, and round(0.3525, 3) is 0.352, so the published 0.353 is checked on the
unrounded trace. Fig. 6 draws the same R from keeper.scem.scem.revocation_curve_C at witness lag 1.
Lag census (Table "census"): keeper.scem.scem.measure_dataset_evidence_lags.
The audit's CPPR items need cp_vpe_scem.py, which is in no commit of the repository; they must fail
with a clear error. keeper.scem.audit.main() first calls keeper.data.provenance.verify_all_provenance(),
which must not overwrite a stored record unless forced.
"""
import json
from pathlib import Path

import pytest

from keeper.data import provenance as prov
from keeper.scem import audit
from keeper.scem import scem as SC

ROOT = Path(__file__).resolve().parents[1]
ROUND5 = ROOT / "results" / "legacy" / "round5_audit_results.json"
SCEM_AGGREGATE = ROOT / "results" / "legacy" / "probe4_scem_aggregate.json"
TRACE_SYMBOLIC = ROOT / "results" / "scem" / "stage_b" / "trace_symbolic.json"
PROVENANCE = ROOT / "results" / "legacy" / "provenance_verification.json"

# Table III, Block 2: (omega_k, pi_k, omega_k + pi_k) by age; at closure the paper prints R_k instead
# of the surviving mass. The five values of the plan: omega at ages 1, 2, 4 and at closure, and R_k.
PUBLISHED_TRACE = {
    0.0: ("0.000", "1.000", "1.000"),
    1.0: ("0.090", "0.910", "1.000"),
    2.0: ("0.264", "0.736", "1.000"),
    4.0: ("0.351", "0.007", "0.359"),
    13.28: ("0.353", "0.000", None),
}
PUBLISHED_R_CLOSURE = "0.647"
CLOSURE = 13.28


def trace_by_age():
    return {row["age_snapshots"]: row for row in audit.silent_sender_trace()}


def test_published_trace_values_come_from_code():
    rows = trace_by_age()
    five = [rows[1.0]["durable_omega"], rows[2.0]["durable_omega"], rows[4.0]["durable_omega"],
            rows[CLOSURE]["durable_omega"], 1.0 - rows[CLOSURE]["durable_omega"]]
    assert [f"{v:.3f}" for v in five] == ["0.090", "0.264", "0.351", "0.353", PUBLISHED_R_CLOSURE]
    for age, (omega, pi, surviving) in PUBLISHED_TRACE.items():
        row = rows[age]
        assert f"{row['durable_omega']:.3f}" == omega, age
        assert f"{row['provisional_pi']:.3f}" == pi, age
        if surviving is not None:
            assert f"{row['surviving_mass_omega_surv']:.3f}" == surviving, age


def test_trace_closure_equals_revocation_curve_of_fig6():
    omega_closure = trace_by_age()[CLOSURE]["durable_omega"]
    _, r_lag1 = SC.revocation_curve_C(1.0, N=2, phi=0.5, L_f=1.0, lam1=2.0)
    assert abs(float(r_lag1) - (1.0 - omega_closure)) < 1e-9


def test_audit_trace_block_reproduces_stored_record():
    stored = json.loads(ROUND5.read_text())["silent_sender_trace_and_newcomer_audit"]
    assert audit.compute_silent_sender_trace_and_newcomer_audit() == stored


def test_symbolic_example_reproduces_stored_record():
    """stage_b's `trace` output: the Section F example (L_f = 0.5), which shares omega(1) = 0.090 with Block 2."""
    assert SC.verify_symbolic_example() == json.loads(TRACE_SYMBOLIC.read_text())


@pytest.mark.parametrize("ds", ["bitcoinalpha", "bitcoinotc", "wiki-rfa"])
def test_lag_census_reproduces_stored_values(ds):
    stored = json.loads(SCEM_AGGREGATE.read_text())["scem_verification"]["dataset_evidence_lags"][ds]
    assert SC.measure_dataset_evidence_lags(str(ROOT / "data" / "cyberdata" / f"{ds}.csv")) == stored


@pytest.mark.parametrize("item", ["evaluate_node_id_disjoint_otc_and_clustered_se", "evaluate_cppr_pool_feasibility"])
def test_items_needing_cp_vpe_scem_fail_clearly(item):
    with pytest.raises(NotImplementedError, match="cp_vpe_scem.py"):
        getattr(audit, item)()


def test_provenance_refuses_to_overwrite_its_default_output(tmp_path, monkeypatch):
    # The default output is OUT_DIR/provenance_verification.json. OUT_DIR points at tmp_path, so a
    # missing guard overwrites the stand-in, never the stored record.
    monkeypatch.setattr(prov, "OUT_DIR", str(tmp_path))
    out = tmp_path / "provenance_verification.json"
    out.write_text("stored")
    with pytest.raises(FileExistsError):
        prov.verify_all_provenance()
    assert out.read_text() == "stored"


def test_provenance_cli_refuses_without_force(tmp_path):
    out = tmp_path / "provenance_verification.json"
    out.write_text("stored")
    with pytest.raises(SystemExit) as stop:
        prov.main(["--out", str(out)])
    assert stop.value.code == 2 and out.read_text() == "stored"


def test_provenance_force_rewrites_and_matches_stored_record(tmp_path):
    out = tmp_path / "provenance_verification.json"
    out.write_text("stored")
    got = prov.verify_all_provenance(out_path=str(out), force=True)
    assert json.loads(out.read_text()) == got
    stored = json.loads(PROVENANCE.read_text())
    assert got["primary_dataset_csvs"] == stored["primary_dataset_csvs"]
    assert got["epinions_join_spec"] == stored["epinions_join_spec"]
    # The three probe4_epinions-dated_s*.partial.jsonl files are not stored in the repository; the four
    # artifacts that are stored hash as recorded.
    artifacts = got["epinions_shipped_artifacts"]
    assert set(artifacts) == {"dataset_census.json", "calibration_epinions-dated_s0.json",
                              "verification_epinions-dated_s0.json", "incidental_epinions-dated_s0.json"}
    assert all(artifacts[name] == stored["epinions_shipped_artifacts"][name] for name in artifacts)
