"""E2 analysis (Task 4.5; keeper.analysis.e2_analysis) on synthetic run records with known answers. CPU only.

Each synthetic run of (network, seed, model) gives its targets a known gain: the attack stream's proxy is the clean
twin's plus direction x gain. Targets recur across seeds (crossed design). The tests check the direction sign, the
pairing with the clean twin, the H6a and H6b estimates, the 80% evaluability rule, the hypothesis p-value, a
contradicted component, the integrity checks and the command line.
"""
import json

import numpy as np
import pytest

from keeper.analysis import e2_analysis as A
from keeper.cluster import protocol_settings as ps
from keeper.cluster.e2_tasks import e2_runs, task_id
from keeper.scem import e2_driver as DRV
from keeper.scem import scem as SC

SHA = "a" * 64
BASE = {"s3r": 0.08, "scem": 0.01, "scem_nohaz": 0.05, "scem_phi05": 0.02, "scem_n1": 0.03}
N_BOOT = 300


def targets_of(seed: int) -> list:
    return list(range(100 + seed % 4, 115 + seed % 4)) + list(range(200 + seed % 2, 215 + seed % 2))


def gains_of(ds: str, seed: int, model: str) -> np.ndarray:
    """The known gains of a run's 30 targets: the model's base plus noise of its own (network, seed, model)."""
    rng = np.random.default_rng([ps.NETWORKS.index(ds), seed, ps.E2_MODELS.index(model)])
    return BASE[model] + 0.004 * rng.standard_normal(30)


def settings_of(model: str):
    if model == "s3r":
        return None
    want = DRV.SCEM_MODELS[model][1]
    return dict(N=want["N"], phi_fixed=want["phi_fixed"], hazard=want["hazard"],
                horizon_A=SC.erlang_horizon_A(want["N"], want["horizon_phi"]))


def record(ds: str, seed: int, stream: str, model: str) -> dict:
    direction = [1] * 15 + [-1] * 15
    clean = 0.5 + 0.01 * np.arange(30)
    proxy = clean if stream == "clean" else clean + np.array(direction) * gains_of(ds, seed, model)
    rules = dict(eps=ps.EPS_CAPPED[ds], edge_cap_units=ps.EDGE_CAP_UNITS, neg_from=ps.NEG_FROM)
    mb = None if model == "s3r" else dict(rules, calibration={"K": [12]}, levels_sha256="clean" if stream == "clean"
                                          else "attack", honest_levels_changed=0)
    return dict(experiment="E2", ds=ds, seed=seed, model_seed=seed, stream=stream, lag=A.STREAM_LAGS[stream],
                model=model, epochs=ps.EPOCHS, settings=settings_of(model),
                learned_phi=None if model == "s3r" else 0.3 + 0.001 * seed, mb=mb,
                plan=dict(rules, sha256=f"plan-{ds}-{seed}", clean_levels_sha256="clean"),
                targets=targets_of(seed), kind=["whitewash"] * 15 + ["badmouth"] * 15, direction=direction,
                proxy=proxy.tolist(), percentile=(proxy / 2).tolist(), panel=dict(sources_sha256="panel"),
                test_mcc=0.3, prereg=dict(path="docs/preregistration.md", sha256=SHA),
                provenance=dict(git_commit="c0ffee", git_dirty=False), _file=f"{task_id(ds, seed, stream, model)}.json")


@pytest.fixture
def records():
    return [record(*key) for key in e2_runs()]


def index_of(records):
    return A.run_set(records)[0]


def expected_contrast(ds: str, kind: str, model: str, reference: str, seeds) -> float:
    part = slice(0, 15) if kind == "whitewash" else slice(15, 30)
    return float(np.mean([np.mean(gains_of(ds, s, model)[part] - gains_of(ds, s, reference)[part]) for s in seeds]))


def test_run_set_has_315_registered_runs(records):
    index, missing, unexpected, duplicates = A.run_set(records)
    assert len(e2_runs()) == 315 == len(index) and not (missing or unexpected or duplicates)
    assert sum(1 for k in e2_runs() if k[1] >= 10) == 3 * 5 * 2 * 3        # seeds 10-14: 2 streams x 3 H6 models


def test_gain_takes_the_direction_sign():
    attack = dict(targets=[1, 2], kind=["whitewash", "badmouth"], direction=[1, -1], proxy=[0.6, 0.3])
    clean = dict(targets=[1, 2], kind=["whitewash", "badmouth"], direction=[1, -1], proxy=[0.5, 0.5])
    index = {("x", 5, "lag1", "m"): attack, ("x", 5, "clean", "m"): clean}
    assert A.run_gains(index, "x", 5, "lag1", "m", "whitewash") == pytest.approx({1: 0.1})
    assert A.run_gains(index, "x", 5, "lag1", "m", "badmouth") == pytest.approx({2: 0.2})   # a drop is a gain
    assert A.run_gains(index, "x", 5, "lag3", "m", "badmouth") is None


def test_gains_pair_each_attack_run_with_its_own_clean_twin(records):
    index = index_of(records)
    ds = ps.NETWORKS[0]
    before = A.cells(index, ds, "lag1", "whitewash", "scem_nohaz", [5], "scem")
    index[(ds, 5, "clean", "scem_phi05")]["proxy"] = [9.0] * 30          # another model's twin: no effect
    assert A.cells(index, ds, "lag1", "whitewash", "scem_nohaz", [5], "scem") == before
    index[(ds, 5, "clean", "scem")]["proxy"] = (np.array(index[(ds, 5, "clean", "scem")]["proxy"]) + 0.1).tolist()
    after = A.cells(index, ds, "lag1", "whitewash", "scem_nohaz", [5], "scem")
    assert after[5] == pytest.approx({t: g + 0.1 for t, g in before[5].items()})


def test_h6a_and_h6b_estimate_the_known_contrasts(records):
    hyps = A.hypotheses(index_of(records), N_BOOT)
    for h, (model, reference) in A.HYPOTHESES.items():
        block = hyps[h]
        for ds in ps.NETWORKS:
            for kind in ("whitewash", "badmouth"):
                c = block["components"][ds][kind]
                assert c["n_seeds"] == c["n_seeds_expected"] == 10 and c["n_pairs"] == 150
                assert c["estimate"] == pytest.approx(expected_contrast(ds, kind, model, reference, range(5, 15)))
                assert c["ci95"][0] <= c["ci95_seed_t"][0] and c["ci95"][1] >= c["ci95_seed_t"][1]   # the hull
                assert c["met"] and c["decision"] == "met" and c["p_value"] == max(c["p_seed_t"], c["p_bootstrap"])
                assert c["p_bootstrap"] == pytest.approx(1 / (N_BOOT + 1))
        comps = [c for by_kind in block["components"].values() for c in by_kind.values()]
        assert block["evaluable"] and block["interval_rule_met"] and len(comps) == 6
        assert block["p_value"] == max(c["p_value"] for c in comps)


def test_targets_recur_across_seeds_and_are_crossed(records):
    c = A.hypotheses(index_of(records), N_BOOT)["H6a"]["components"][ps.NETWORKS[0]]["whitewash"]
    assert c["n_targets"] == 18 < c["n_pairs"]                       # 15 targets per seed, 18 distinct over 10 seeds


def test_a_component_needs_80_percent_of_its_seeds(records):
    ds = ps.NETWORKS[1]
    drop = {(ds, s, "lag1", "scem") for s in (12, 13)}
    eight = A.hypotheses(index_of([r for r in records if A.key_of(r) not in drop]), N_BOOT)
    assert eight["H6a"]["components"][ds]["badmouth"]["n_seeds"] == 8 and eight["H6a"]["evaluable"]
    drop |= {(ds, 14, "lag1", "scem")}
    kept = [r for r in records if A.key_of(r) not in drop]
    seven = A.hypotheses(index_of(kept), N_BOOT)
    for h in ("H6a", "H6b"):
        c = seven[h]["components"][ds]["whitewash"]
        assert c["n_seeds"] == 7 and not c["evaluable"] and c["decision"] == A.NOT_EVALUABLE
        assert not seven[h]["evaluable"] and seven[h]["p_value"] is None
        assert seven[h]["status_before_holm"] == A.NOT_EVALUABLE
    assert set(A.run_set(kept)[1]) == {task_id(*k) for k in drop}


def test_a_contradicted_component_is_reported(records, monkeypatch):
    ds = ps.NETWORKS[2]
    for r in records:
        if r["ds"] == ds and r["model"] == "scem_nohaz" and r["stream"] == "lag1":
            clean = np.array(record(ds, r["seed"], "clean", "scem_nohaz")["proxy"])
            r["proxy"] = (clean + np.array(r["direction"]) * (BASE["scem"] - 0.03)).tolist()
    h6a = A.hypotheses(index_of(records), N_BOOT)["H6a"]
    assert h6a["components"][ds]["whitewash"]["contradicted"] and not h6a["interval_rule_met"]
    assert f"{ds}/whitewash" in h6a["contradicted"] and h6a["status_before_holm"] == "interval rule not met"


def test_secondary_outcomes_cover_lag3_and_the_variants(records):
    index = index_of(records)
    sec = A.secondary(index, N_BOOT)
    ds = ps.NETWORKS[0]
    lag3 = sec[ds]["lag3"]["whitewash"]
    assert set(lag3) == set(ps.E2_MODELS) and lag3["scem_n1"]["gain"]["n_seeds_expected"] == 5
    assert lag3["scem_n1"]["minus_scem"]["estimate"] == pytest.approx(
        expected_contrast(ds, "whitewash", "scem_n1", "scem", range(5, 10)))
    assert "minus_scem" not in sec[ds]["lag1"]["whitewash"]["s3r"]          # that is H6b
    assert sec[ds]["lag1"]["badmouth"]["s3r"]["gain"]["n_seeds"] == 10
    phi = A.per_run(index, "learned_phi", ["scem"])[ds]["scem"]["lag1"]
    assert phi["n_seeds"] == 10 and phi["mean"] == pytest.approx(0.3 + 0.001 * 9.5)


@pytest.mark.parametrize("breaks, message", [
    (lambda r: r["prereg"].update(sha256="b" * 64), "protocol sha256"),
    (lambda r: r["provenance"].update(git_dirty=True), "git state not clean"),
    (lambda r: r["plan"].update(eps=0.05), "plan has"),
    (lambda r: r["mb"].update(edge_cap_units=None), "M-B has"),
    (lambda r: r["settings"].update(hazard=False), "built SCEM module"),
    (lambda r: r.update(epochs=2), "epochs"),
    (lambda r: r["plan"].update(sha256="other"), "the stream plan"),
    (lambda r: r["mb"].update(calibration={"K": [13]}), "frozen M-B calibration"),
    (lambda r: r["panel"].update(sources_sha256="other"), "the panel"),
    (lambda r: r.update(targets=r["targets"][::-1]), "clean-twin settings"),
    (lambda r: r["provenance"].update(git_commit="beef"), "the code commit"),
    (lambda r: r.pop("plan"), "malformed record"),
])
def test_integrity_failures_are_caught(records, breaks, message):
    assert A.integrity(records, SHA, []) == []
    target = next(r for r in records if A.key_of(r) == (ps.NETWORKS[0], 6, "lag1", "scem"))
    breaks(target)
    failures = A.integrity(records, SHA, [])
    assert failures and any(message in f for f in failures), failures


def test_a_clean_twin_must_use_the_clean_levels(records):
    target = next(r for r in records if A.key_of(r) == (ps.NETWORKS[0], 6, "clean", "scem_n1"))
    target["mb"]["levels_sha256"] = "attack"
    assert any("clean twin's M-B levels" in f for f in A.integrity(records, SHA, []))


def write_runs(root, records):
    folder = root.joinpath("results", "scem", "e2")
    folder.mkdir(parents=True)
    for r in records:
        (folder / r["_file"]).write_text(json.dumps({k: v for k, v in r.items() if k != "_file"}))


def test_cli_writes_the_report_and_refuses_to_overwrite(records, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("KEEPER_OVERWRITE", raising=False)
    prereg = tmp_path / "prereg.md"
    prereg.write_text("protocol\n")
    sha = __import__("hashlib").sha256(prereg.read_bytes()).hexdigest()
    for r in records:
        r["prereg"]["sha256"] = sha
    write_runs(tmp_path, records[:-1])                       # one run missing
    argv = ["--results-root", str(tmp_path), "--prereg", str(prereg), "--n-boot", str(N_BOOT)]
    assert A.main(argv) == 0
    out = json.loads((tmp_path / "results" / "analysis" / "e2.json").read_text())
    assert out["runs"]["missing"] == [records[-1]["_file"][:-5]] and out["runs"]["used"] == 314
    assert out["hypotheses"]["H6a"]["evaluable"] and "provenance" in out
    md = (tmp_path / "results" / "analysis" / "e2.md").read_text()
    assert "## H6a" in md and "## H6b" in md and "Missing (1)" in md
    with pytest.raises(SystemExit) as stop:
        A.main(argv)
    assert stop.value.code == 2 and "exists" in capsys.readouterr().err


def test_cli_writes_nothing_when_integrity_fails(records, tmp_path, capsys):
    prereg = tmp_path / "prereg.md"
    prereg.write_text("another protocol\n")
    write_runs(tmp_path, records)
    assert A.main(["--results-root", str(tmp_path), "--prereg", str(prereg), "--n-boot", "50"]) == 2
    assert "protocol sha256" in capsys.readouterr().err
    assert not (tmp_path / "results" / "analysis").exists()
