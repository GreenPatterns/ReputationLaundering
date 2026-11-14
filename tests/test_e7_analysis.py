"""Task 4.7: the E7 task list (keeper.cluster.e7_tasks) and analysis (keeper.analysis.e7_analysis), on synthetic runs.

  A1 the task list: 120 runs in the format of keeper.cluster.cppr_tasks, commands that the drivers' parsers accept
  A2 the numbers: CPPR gains and MCC per arm and method; witness gains pair each attack with the clean twin of the same
     M-B setting, and off_minus_on pairs M-B off with M-B on by seed and target; the attacked-only blocks
  A3 integrity: the protocol's sha256, the git state, a setting, the plan and the clean-twin pairing stop the analysis
     (exit 2) and nothing is written; missing runs are listed
"""
import hashlib
import json
import shlex
from collections import Counter

import numpy as np
import pytest

from keeper.analysis import e7_analysis as A
from keeper.cluster import e7_tasks as T
from keeper.cluster import protocol_settings as ps
from keeper.cppr import cp_vpe_gpu as G
from keeper.cppr import witness_arm as WA
from keeper.cppr.dumps import TARGET_COLUMNS
from keeper.scem import scem as SC

RESULTS = "/data/keeper"
PY = ps.PYTHON + " -m "
DS, SEEDS = "bitcoinalpha", (5, 6, 7)
EDGE, ACC, WW = [11, 12, 13], [21, 22], [11, 12, 13]
SETTINGS = dict(N=2, phi_fixed=None, hazard=True, horizon_A=SC.erlang_horizon_A(2, 0.125), lambda_max=2.0,
                phi_min=0.125, phi_max=0.5)
MB = dict(eps=0.02, edge_cap_units=2.0, neg_from="trusted", calibration=dict(K=[12] * 3, theta0=[0.1] * 3,
                                                                          n_ref=[9] * 3))
GIT = dict(git_commit="abc123", git_dirty=False, git_untracked=0)


# ------------------------------------------------------------------ A1. the task list
def test_the_task_list_holds_the_registered_runs_and_commands_the_drivers_parse():
    rows = T.e7_tasks(RESULTS)
    assert len(rows) == 120 == len({r.id for r in rows}) == len(T.cppr_runs()) + len(T.witness_runs())
    assert [r.id.rsplit("_", 2)[1] for r in rows][::40] == ["wiki-rfa", "bitcoinotc", "bitcoinalpha"]
    assert Counter(r.id.rsplit("_", 1)[1] for r in rows) == {f"s{s}": 24 for s in range(5, 10)}
    for r in rows:
        assert r.done_path == f"{RESULTS}/results/cppr/e7/{r.id}.json"
        assert r.log_path == f"{RESULTS}/results/cppr/e7/logs/{r.id}.log"
        if r.id.startswith("e7_cppr"):
            a = G.build_parser().parse_args(shlex.split(r.command[len(PY + "keeper.cppr.cp_vpe_gpu "):]))
            protocol = G.protocol_from_args(a)                   # the run's protocol is valid
            g = protocol.attack_edges
            assert r.id == T.cppr_id(a.ds, a.seed, g) and protocol.backbone == "rating_scem"
            assert (protocol.tiers, a.btp_eps, protocol.hold_test_snapshot, a.epochs) == \
                ("btp", ps.EPS_CAPPED[a.ds], True, 150)
            assert (",".join(protocol.arms), ",".join(protocol.methods)) == ps.E7_CPPR[g]
            assert a.pool_eps == (ps.EPS_CAPPED[a.ds] if g else None) and (a.k_fake, a.sybil_topology) == (3, "ring")
            assert a.dump_targets == f"{RESULTS}/results/cppr/e7/targets" and a.prereg == "docs/preregistration.md"
        else:
            a = WA.build_parser().parse_args(shlex.split(r.command[len(PY + "keeper.cppr.witness_arm "):]))
            assert r.id == T.witness_id(a.ds, a.seed, a.stream, a.mb) and (a.epochs, a.out) == (150, r.done_path)


# ------------------------------------------------------------------ synthetic runs
def cppr_result(seed: int, g: int, sha: str) -> tuple:
    """(result, target dump) of an E7 CPPR run whose gains and MCC are known functions of seed and target."""
    arms, methods = ps.E7_CPPR[g]
    name = T.cppr_id(DS, seed, g)
    ref = dict(eps=0.02, K=12, n_ref=3629, edge_cap_units=2.0) if g else None
    config = dict(gate_norm="count", arms=arms.split(","), methods=methods.split(","), hold_test_snapshot=True,
                  ungated_baseline=False, preregistration=dict(sha256=sha),
                  attack=dict(attack_edges=g, endorser_select="random", sybil_topology="ring", k_fake=3,
                              reference_calibration=ref),
                  tiers=dict(method="btp", eps=0.02, K=12, n_ref=3629, edge_cap_units=2.0, neg_from="trusted"),
                  backbone=dict(name="rating_scem", model="scem", settings=SETTINGS,
                                mb=dict(arms=dict(base=MB, whitewash=MB))))
    rows, held = [], dict(base=dict(t1=dict(mcc=0.5 + seed / 100)))
    keys = {"plain": "plain_erasure", "vpe": "plain_vpe", "cppr": "cppr"}
    for arm in arms.split(","):
        held[arm] = {}
        for m in ("plain", "vpe", "cppr"):
            if keys[m] not in methods.split(",") or (arm == "whitewash" and m == "plain"):
                continue
            held[arm][m] = dict(t1=dict(mcc=0.4 + seed / 100 + g / 10 + len(m) / 1000))
            for kind, targets in ((("edge", EDGE), ("account", ACC)) if arm == "direct" else (("edge", WW),)):
                for t in targets:
                    gain = gain_of(arm, m, kind, seed, t, g)
                    rows.append([arm, m, kind, t, 0.25, 0.25 + gain, gain] + [0] * 5 + [False])
    result = dict(ds=DS, seed=seed, epochs=150, rho=0.02, granularity=1, k_fake=3, gate_norm="count", config=config,
                  provenance=dict(GIT), test_metrics=held, sybil_admission=dict(whitewash=dict(snap_f=[7, 8, 9])),
                  dumps=dict(targets=dict(path=f"/x/targets/{name}_targets.json", n_rows=len(rows))))
    return result, dict(result=f"{name}.json", config=config, columns=list(TARGET_COLUMNS), n_rows=len(rows), rows=rows)


def gain_of(arm, m, kind, seed, t, g) -> float:
    return {"plain": 1.0, "vpe": 0.5, "cppr": 0.1}[m] * (1 + g) + seed / 10 + t / 1000 + (kind == "account") / 100


def proxy_of(stream, mb, seed, t) -> float:
    """Clean: 0.3 + t/1000; an attack adds (2 if M-B off else 1) * (1 + g) * seed / 100 to attacked targets 11, 12."""
    base = 0.3 + t / 1000
    if stream == "clean" or t not in (11, 12):
        return base
    return base + (2 if mb == "off" else 1) * (1 + int(stream[1])) * seed / 100


def witness_result(seed: int, stream: str, mb: str, sha: str) -> dict:
    """The witness plan lists its targets in draw order, which is not sorted (as in the real runs)."""
    g = ps.E7_WITNESS_STREAMS[stream]
    order = EDGE[::-1]
    plan = dict(targets=order, victims=[[1, 11, 6], [2, 12, 7], [3, 13, 2]], sha256="plan")
    attack = None if g is None else dict(g=g, k=5, lag=1, sybil_ring=True, rng=[20261027, seed, 7, g],
                                         attacked_targets=[11, 12], endorsers=[[4]] * 2 if g else [[]] * 2)
    return dict(experiment="E7", arm="witness", ds=DS, seed=seed, model_seed=seed, stream=stream, g=g, mb=mb,
                epochs=150, backbone="rating_scem", model="scem", gate_norm="count", settings=SETTINGS,
                plan=plan, attack=attack, stream_info=dict(sha256=f"stream-{stream}"),
                gate=dict(method="btp", eps=0.02, edge_cap_units=2.0, neg_from="trusted"),
                mb_info=dict(MB) if mb == "on" else None, targets=order,
                proxy=[proxy_of(stream, mb, seed, t) for t in order], panel=dict(sources_sha256="panel"),
                test_mcc=0.3 + seed / 100, prereg=dict(sha256=sha), provenance=dict(GIT))


@pytest.fixture
def root(tmp_path):
    """A results root with every E7 run of Bitcoin-Alpha for seeds 5-7 (seeds 8-9 missing), and a protocol file."""
    prereg = tmp_path / "prereg.md"
    prereg.write_text("protocol\n")
    sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
    folder = tmp_path / "results" / "cppr" / "e7"
    (folder / "targets").mkdir(parents=True)
    for seed in SEEDS:
        for g in ps.E7_CPPR:
            result, dump = cppr_result(seed, g, sha)
            (folder / f"{T.cppr_id(DS, seed, g)}.json").write_text(json.dumps(result))
            (folder / "targets" / f"{T.cppr_id(DS, seed, g)}_targets.json").write_text(json.dumps(dump))
        for stream in ps.E7_WITNESS_STREAMS:
            for mb in ps.E7_MB:
                (folder / f"{T.witness_id(DS, seed, stream, mb)}.json").write_text(
                    json.dumps(witness_result(seed, stream, mb, sha)))
    return tmp_path


def _analyse(root, n_boot=200):
    return A.analyse(str(root), dict(path="p", sha256=hashlib.sha256((root / "prereg.md").read_bytes()).hexdigest()),
                     networks=(DS,), n_boot=n_boot)


# ------------------------------------------------------------------ A2. the numbers
def test_the_numbers_on_synthetic_runs(root):
    report = _analyse(root)
    net = report["networks"][DS]
    mean_seed = np.mean(SEEDS) / 10
    for block, g, kind, targets, methods in (("direct_edge", 0, "edge", EDGE, ("plain", "vpe", "cppr")),
                                             ("direct_account", 0, "account", ACC, ("plain", "vpe", "cppr")),
                                             ("whitewash_g0", 0, "edge", WW, ("vpe", "cppr")),
                                             ("whitewash_g1", 1, "edge", WW, ("vpe", "cppr"))):
        for m in methods:
            arm = "direct" if block.startswith("direct") else "whitewash"
            want = np.mean([gain_of(arm, m, kind, 0, t, g) for t in targets]) + mean_seed
            assert net[block][f"gain_{m}"] == pytest.approx(want), (block, m)
            assert net[block]["n_cells"][f"gain_{m}"] == 3 * len(targets)
            assert net[block]["ci95"][f"gain_{m}"][0] <= net[block][f"gain_{m}"] <= net[block]["ci95"][f"gain_{m}"][1]
    assert set(net["whitewash_g1"]) >= {"gain_vpe", "gain_cppr"} and "gain_plain" not in net["whitewash_g0"]
    mcc = net["mcc"]
    assert mcc["mcc_base"] == pytest.approx(0.5 + np.mean(SEEDS) / 100)
    assert mcc["mcc_direct_plain"] == pytest.approx(0.4 + np.mean(SEEDS) / 100 + 0.005)
    assert mcc["mcc_whitewash_g1_cppr"] == pytest.approx(0.4 + np.mean(SEEDS) / 100 + 0.1 + 0.004)

    for g in (0, 1):
        w, attacked = net[f"witness_g{g}"], net[f"witness_g{g}_attacked"]
        on = [(1 + g) * s / 100 * 2 / 3 for s in SEEDS]           # two of three targets attacked
        assert w["gain_mb_on"] == pytest.approx(np.mean(on))
        assert w["gain_mb_off"] == pytest.approx(2 * np.mean(on))
        assert w["off_minus_on"] == pytest.approx(np.mean(on))
        assert attacked["gain_mb_on"] == pytest.approx((1 + g) * np.mean(SEEDS) / 100)
        assert attacked["off_minus_on"] == pytest.approx((1 + g) * np.mean(SEEDS) / 100)
        assert w["n_cells"]["off_minus_on"] == 9 and attacked["n_cells"]["off_minus_on"] == 6
    assert net["mcc_witness"]["mcc_g1_mboff"] == pytest.approx(0.3 + np.mean(SEEDS) / 100)
    assert len(report["runs"]["missing"]) == 16 and report["runs"]["used"] == 24 and report["runs"]["expected"] == 40
    assert "## bitcoinalpha" in A.markdown(report)


def test_the_cli_writes_both_outputs_and_refuses_to_overwrite(root, capsys):
    argv = ["--results-root", str(root), "--prereg", str(root / "prereg.md"), "--n-boot", "100"]
    assert A.main(argv) == A.EXIT_OK                               # the other networks' runs are only missing
    assert A.main(argv) == A.EXIT_EXISTS and "exists" in capsys.readouterr().err
    report = json.loads((root / "results" / "analysis" / "e7.json").read_text())
    assert report["schema"] == A.SCHEMA and set(report["networks"][DS]) >= {"direct_edge", "witness_g1", "mcc"}
    assert report["runs"]["expected"] == 120 and (root / "results" / "analysis" / "e7.md").is_file()


# ------------------------------------------------------------------ A3. integrity
def _rewrite(root, name, change):
    path = root / "results" / "cppr" / "e7" / f"{name}.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("name, change, message", [
    (T.cppr_id(DS, 5, 0), lambda d: d["config"]["preregistration"].update(sha256="other"), "preregistration.sha256"),
    (T.cppr_id(DS, 6, 1), lambda d: d["provenance"].update(git_dirty=True), "git_dirty"),
    (T.cppr_id(DS, 6, 1), lambda d: d["config"]["backbone"]["mb"]["arms"]["base"].update(eps=0.05), "M-B eps"),
    (T.cppr_id(DS, 7, 1), lambda d: d["sybil_admission"]["whitewash"].update(snap_f=[1, 2, 3]), "snap_f"),
    (T.witness_id(DS, 5, "g1", "on"), lambda d: d.update(prereg=dict(sha256="other")), "prereg.sha256"),
    (T.witness_id(DS, 5, "g1", "off"), lambda d: d["plan"].update(sha256="other"), "witness plan"),
    (T.witness_id(DS, 6, "clean", "off"), lambda d: d["provenance"].update(git_commit="def456"), "clean twin"),
    (T.witness_id(DS, 7, "g0", "off"), lambda d: d.update(stream_info=dict(sha256="x")), "M-B on and off"),
    (T.witness_id(DS, 7, "g0", "on"), lambda d: d["attack"].update(k=50), "attack.k"),
])
def test_an_integrity_failure_stops_the_analysis(root, capsys, name, change, message):
    _rewrite(root, name, change)
    with pytest.raises(ValueError, match=message):
        _analyse(root)
    assert A.main(["--results-root", str(root), "--prereg", str(root / "prereg.md"), "--n-boot", "100"]) == \
        A.EXIT_INTEGRITY                                           # the other networks' runs are only missing
    assert not (root / "results" / "analysis").exists() and message in capsys.readouterr().err


def test_witness_targets_must_be_the_cppr_direct_targets_as_a_set(root):
    """Draw order does not matter (the fixture's plans are unsorted); a different target set stops the analysis."""
    _analyse(root)
    for stream in ps.E7_WITNESS_STREAMS:
        for mb in ps.E7_MB:
            _rewrite(root, T.witness_id(DS, 6, stream, mb),
                     lambda d: d.update(targets=[11, 12, 14], plan=dict(d["plan"], targets=[11, 12, 14])))
    with pytest.raises(ValueError, match="witness targets are not the CPPR runs' direct edge targets"):
        _analyse(root)
