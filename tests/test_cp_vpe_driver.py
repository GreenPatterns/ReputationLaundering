"""The CPPR driver keeper.cppr.cp_vpe_gpu end to end on the CPU, with every model replaced by a stub (Task 3.2).

No model is trained: tests/cppr_common.StubModel stands in for LocalGDTE, and its embedding is a deterministic
function of the training inputs. The driver's graphs, tiers, requests, reputations, test metrics, linkage and
dumps therefore run for real, and the frozen copy of the old driver (tests/cp_vpe_legacy.py) is the oracle.

  D1 protocol flags parse and validate; a changed protocol needs --out; guards stop a run before training
  D2 linkage_attack equals its frozen copy, and verdict_scores exposes the features it scores
  D3 dump rows: per target and arm, and per pooled verdict and candidate rater
  D4 default flags reproduce the old driver: the same trainings, in the same order, on the same inputs, and
     the same output values (the CPU form of the GPU regression gate)
  D5 the full protocol records its configuration, scores every arm on the original test snapshot, keeps the
     fakes within the per-edge bound, and writes both dumps
  D6 running a subset of the arms leaves the others' numbers unchanged
  D7 rulings R39 and R43: BFS, capped and uncapped runs draw the same random endorsers from one pool, timing
     included; under BFS they are credible at their attack edge and every region is lifted
Task 4.1b (Phase 4 hardening):
  D8 an endorser pool too small for g stops the run before the base model trains: the random pools (R43), and
     the ranked candidates of every group (R44)
  D9 dump folders are made before training; the result is written first, then the dumps; the provenance adds the
     untracked-path count and the library versions
  D10 --methods parses and validates; a changed list is a changed protocol (needs --out)
  D11 --methods cppr: CPPR's outputs and dump rows equal the default run's, with base and CPPR trainings only
  D12 other method lists: the trainings drop accordingly, and fields that need an absent method are absent
  D13 the control arm's models are also scored on the bad-mouth victims, without training: block gains and
      per-victim dump rows; its gains on the whitewash targets keep their meaning
  D14 each injected arm's sybil_admission records the snap_f the stored rng drew for every group
Task 4.1c (rulings R43, R44, the base-model cache, deferred minors):
  D15 --pool-eps is required with attack edges and equals --btp-eps in capped BTP runs; BFS needs no --btp-eps
  D16 to D18 (the base-model cache) are in tests/test_cp_vpe_base_cache.py
  D19 the result folder and the cache folder are made before any training
The stubs fixture is in tests/conftest.py.

Run from the repository root (CPU is enough):
    CUDA_VISIBLE_DEVICES="" <ps python> -m pytest -q tests/test_cp_vpe_driver.py
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from keeper.analysis import tiers_report
from keeper.cppr import cp_vpe_gpu as G
from keeper.cppr import dumps as Dm
from keeper.cppr import linkage as L
from keeper.cppr import vpe as V
from keeper.credibility import tiers
from keeper.laundering import core as C
from keeper.laundering import signed as S
from tests import cp_vpe_legacy as O
from tests.cppr_common import ALPHA_EPS, StubModel, alpha_calibration, background

INF = float("inf")
ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ D1. flags, validation and guards
def test_protocol_flags_parse_and_validate():
    parser = G.build_parser()
    args = parser.parse_args(["--tiers", "btp", "--btp-eps", "0.02", "--btp-edge-cap", "none", "--btp-neg-from", "all",
                              "--attack-edges", "2", "--endorser-select", "max_trust", "--sybil-topology", "star",
                              "--arms", "badmouth,whitewash", "--control-arm", "--hold-test-snapshot",
                              "--ungated-baseline", "--dump-targets", "d1", "--dump-verdicts", "d2", "--prereg",
                              "docs/p.md", "--pool-eps", "0.02"])
    protocol = G.protocol_from_args(args)
    assert protocol == G.RunProtocol(tiers="btp", btp_eps=0.02, btp_edge_cap=None, btp_neg_from="all", attack_edges=2,
                                     endorser_select="max_trust", sybil_topology="star",
                                     arms=("whitewash", "badmouth", "control"), hold_test_snapshot=True,
                                     ungated_baseline=True, dump_targets="d1", dump_verdicts="d2", prereg="docs/p.md",
                                     pool_eps=0.02)
    assert not protocol.is_legacy()
    default = parser.parse_args([])
    assert default.btp_edge_cap == tiers.KEEPER_EDGE_CAP_UNITS == 2.0 and default.attack_edges == 0
    assert default.btp_neg_from == tiers.KEEPER_NEG_FROM == "trusted"            # ruling R56: the guard by default
    assert G.protocol_from_args(default) == G.RunProtocol() and G.RunProtocol().arms == G.DEFAULT_ARMS
    assert G.RunProtocol().btp_neg_from == "trusted" and G.RunProtocol(btp_neg_from="all").is_legacy()
    with pytest.raises(SystemExit):
        parser.parse_args(["--btp-neg-from", "untrusted"])
    assert G.parse_edge_cap("NONE") is None and G.parse_edge_cap("4") == 4.0
    for bad in ("0", "-1", "nan", "inf", "two"):
        with pytest.raises(argparse.ArgumentTypeError):
            G.parse_edge_cap(bad)
    assert G.parse_arms("control,direct") == ("direct", "control")              # canonical order
    assert G.parse_arms("direct", control=True) == ("direct", "control")
    assert G.parse_arms("control", control=True) == ("control",)
    for bad in ("", "direct,direct", "direct,sybil"):
        with pytest.raises(ValueError):
            G.parse_arms(bad)
    for bad in (dict(tiers="btp"), dict(endorser_select="max_trust", attack_edges=1), dict(tiers="btp", btp_eps=0.0),
                dict(tiers="trust"), dict(attack_edges=-1), dict(sybil_topology="mesh"), dict(arms=()),
                dict(arms=("direct", "direct")), dict(btp_edge_cap=0.0), dict(tiers="btp", btp_eps=INF),
                dict(btp_neg_from="none")):
        with pytest.raises(ValueError):
            G.RunProtocol(**bad)
    # Dumps and the preregistration hash add outputs only: the computation stays that of the stored runs.
    assert G.RunProtocol(dump_targets="d", dump_verdicts="d", prereg="p").is_legacy()
    for changed in (dict(tiers="btp", btp_eps=0.02), dict(attack_edges=1, pool_eps=0.02), dict(sybil_topology="star"),
                    dict(arms=("direct",)), dict(hold_test_snapshot=True), dict(ungated_baseline=True)):
        assert not G.RunProtocol(**changed).is_legacy()


def test_config_block_records_the_guard_only_for_btp_tiers():
    # Ruling R56: neg_from sits next to eps, K and theta0 in the config block. It is BTP's guard, so it is null under
    # bfs tiers, which take no tier calibration; the endorser protocol's reference calibration is recorded apart.
    cal = tiers.BTPCalibration(seeds=(0,), K=3, eps=0.5, n_ref=5, edge_cap_units=2.0)
    for protocol, want in ((G.RunProtocol(tiers="btp", btp_eps=0.5), "trusted"),
                           (G.RunProtocol(tiers="btp", btp_eps=0.5, btp_neg_from="all"), "all")):
        cfg = G.config_block(protocol, "count", 3, cal, None)["tiers"]
        assert (cfg["neg_from"], cfg["eps"], cfg["K"], cfg["theta0"]) == (want, 0.5, 3, 0.1), protocol
    bfs = G.RunProtocol(pool_eps=0.5, attack_edges=1, endorser_select="max_trust")
    cfg = G.config_block(bfs, "count", 3, None, None, reference=cal)
    assert (cfg["tiers"]["neg_from"], cfg["tiers"]["eps"]) == (None, None)
    assert cfg["attack"]["reference_calibration"]["eps"] == 0.5


def test_the_driver_builds_its_btp_tiers_with_the_protocols_guard(stubs, tmp_path, monkeypatch):
    # Ruling R56: run_single_seed's one tier function (cred_levels and every cred_inj) applies --btp-neg-from. The run
    # stops at its first training, after the tier function is built.
    built = []

    class Stop(Exception):
        pass

    def spy(*args, **kwargs):
        built.append(real(*args, **kwargs))
        return built[-1]

    def stop(*args, **kwargs):
        raise Stop

    real = G.CredibilityTiers
    monkeypatch.setattr(G, "CredibilityTiers", spy)
    monkeypatch.setattr(G, "train_gated_model", stop)
    for neg_from in ("trusted", "all"):
        with pytest.raises(Stop):
            G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / f"{neg_from}.json"),
                              protocol=G.RunProtocol(tiers="btp", btp_eps=ALPHA_EPS, btp_neg_from=neg_from))
        assert (built[-1].method, built[-1].neg_from) == ("btp", neg_from)


def test_a_changed_protocol_needs_an_explicit_output_name(stubs):
    # The default names (cp_vpe_<ds>_s<seed>.json, cpgate_<norm>_<ds>_s<seed>.json) belong to the stored protocol.
    with pytest.raises(ValueError, match="--out"):
        G.run_single_seed("bitcoinalpha", 0, protocol=G.RunProtocol(attack_edges=1, pool_eps=ALPHA_EPS))
    assert stubs == []


def test_existing_outputs_and_a_missing_preregistration_stop_the_run_before_training(stubs, tmp_path):
    out = tmp_path / "run.json"
    dump_dir = tmp_path / "dumps"
    dump_dir.mkdir()
    (dump_dir / "run_targets.json").write_text("{}")
    with pytest.raises(FileExistsError):                                       # ruling R27 covers the dumps
        G.run_single_seed("bitcoinalpha", 0, out_name=str(out), protocol=G.RunProtocol(dump_targets=str(dump_dir)))
    with pytest.raises(FileNotFoundError, match="preregistration file"):
        G.run_single_seed("bitcoinalpha", 0, out_name=str(out), protocol=G.RunProtocol(prereg=str(tmp_path / "no.md")))
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        G.run_single_seed("bitcoinalpha", 0, out_name=str(out))
    assert stubs == [] and out.read_text() == "{}"


# ------------------------------------------------------------------ D2. linkage
def _releases():
    """Plain VPE and CPPR releases of the Bitcoin-Alpha workload, and a release with single-member pools."""
    bg = background()
    cred = G.compute_credibility_levels(bg.Tr, bg.D.n, seeded=True)
    rng = np.random.default_rng(0)
    res_vpe = V.apply_vpe(bg.D, bg.G, bg.tt, bg.reqs, V.VPEConfig(1, INF, 10), rng)
    res_cp, cred_cp = G.apply_cp_vpe(bg.D, bg.G, bg.tt, bg.reqs, cred, G.CPVPEConfig(1, INF, 10), rng)
    launder = [q for q in bg.reqs if q.tag != "honest"]                       # rho = 0: tiny pools
    res_small = V.apply_vpe(bg.D, bg.G, bg.tt, launder, V.VPEConfig(1, INF, 0), rng)
    pad = lambda res: np.concatenate([cred, np.zeros(len(res.anon_nodes), dtype=np.int64)])  # noqa: E731
    return bg, [(res_vpe, pad(res_vpe)), (res_cp, cred_cp), (res_small, pad(res_small))]


def test_linkage_attack_equals_its_frozen_copy(monkeypatch):
    monkeypatch.setattr(C, "embed", lambda m, Tr, tt: m.z)
    bg, releases = _releases()
    small = releases[2][0]
    assert any(len(small.pools[p["window"]]) < 2 for p in small.pooled)          # the singleton path runs
    for res, cred in releases:
        model = StubModel(res.Tr, cred, 0, "count")
        assert L.linkage_attack(res, model, bg.tt) == O.linkage_attack(res, model, bg.tt)
        records = L.verdict_scores(res, model, bg.tt)
        assert [idx for idx, _, _ in records] == list(range(len(res.pooled)))
        for idx, p, scores in records:
            members = {int(u) for u in res.pools[p["window"]]}
            if len(members) < 2:
                assert scores is None
                continue
            for key in ("structure", "timing", "model", "degree", "activity", "activity_window"):
                assert set(scores[key]) == members, key


# ------------------------------------------------------------------ D3. dump rows
def test_target_rows_count_in_edges_before_and_after_erasure():
    def graph(rows, n):
        rows = np.asarray(rows, dtype=np.int64)
        return SimpleNamespace(edges=rows[:, :2], labels=S.one_hot_from_sign(rows[:, 2]), n=n)
    before = graph([(0, 2, 1), (1, 2, -1), (3, 2, -1), (2, 0, 1), (3, 4, -1), (4, 3, 1)], 5)
    after = graph([(0, 2, 1), (2, 1, 1), (9, 2, -1), (4, 0, 1)], 10)          # 9: a pool node
    rows = Dm.target_rows("direct", "plain", "edge", [2, 3, 4], [0.25, 0.5, 0.5], np.array([0.75, 0.5, 0.25]),
                          before, after)
    col = {c: i for i, c in enumerate(Dm.TARGET_COLUMNS)}
    two, three, four = rows
    assert [two[col[c]] for c in ("arm", "method", "kind", "target", "r0", "r_after", "gain")] == [
        "direct", "plain", "edge", 2, 0.25, 0.75, 0.5]
    assert [two[col[c]] for c in ("n_neg_before", "n_pos_before", "n_neg_after", "n_pos_after", "n_out_after",
                                  "edgeless_after")] == [2, 1, 1, 1, 1, False]
    assert [three[col[c]] for c in ("n_neg_before", "n_pos_before", "n_neg_after", "n_pos_after", "n_out_after",
                                    "edgeless_after", "gain")] == [0, 1, 0, 0, 0, True, 0.0]
    # 4 keeps an out-edge but no in-edge: not edgeless (edgeless means no training edge at all)
    assert [four[col[c]] for c in ("n_neg_before", "n_pos_before", "n_neg_after", "n_pos_after", "n_out_after",
                                   "edgeless_after", "gain")] == [1, 0, 0, 0, 1, False, -0.25]


def test_verdict_rows_hold_every_candidate_with_its_linkage_features(monkeypatch):
    monkeypatch.setattr(C, "embed", lambda m, Tr, tt: m.z)
    bg, releases = _releases()
    for (res, cred), method in zip(releases, ("vpe", "cppr", "vpe")):
        model = StubModel(res.Tr, cred, 0, "count")
        rows = Dm.verdict_rows("direct", method, res, model, bg.tt, cred)
        col = {c: i for i, c in enumerate(Dm.VERDICT_COLUMNS)}
        assert len(rows) == sum(len(res.pools[p["window"]]) for p in res.pooled)
        assert all(len(r) == len(Dm.VERDICT_COLUMNS) for r in rows)
        by_key = {(r[col["verdict"]], r[col["candidate"]]): r for r in rows}
        for idx, p, scores in L.verdict_scores(res, model, bg.tt):
            pool = res.pools[p["window"]]
            for u in pool:
                r = by_key[(idx, int(u))]
                assert (r[col["arm"]], r[col["method"]], r[col["target"]], r[col["sign"]]) == (
                    "direct", method, p["target"], p["sign"])
                assert (r[col["pool_id"]], r[col["window"]], r[col["pool_tier"]], r[col["pool_size"]]) == (
                    p["window"], p.get("raw_window", p["window"]), p.get("cred_level"), len(pool))
                assert r[col["label"]] == int(int(u) in p["raters"]) and r[col["candidate_tier"]] == int(cred[u])
                for c, key in (("structure", "structure"), ("timing", "timing"), ("model_prob", "model"),
                               ("degree", "degree"), ("activity", "activity"), ("activity_window", "activity_window")):
                    assert r[col[c]] == (None if scores is None else scores[key][int(u)])
            assert sum(by_key[(idx, int(u))][col["label"]] for u in pool) == len(set(p["raters"]))


# ------------------------------------------------------------------ D4. the default protocol is the old driver
def test_default_flags_reproduce_the_old_driver(stubs, tmp_path):
    old = O.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "old.json"), gate_norm="count")
    calls_old = list(stubs)
    stubs.clear()
    new = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "new.json"), gate_norm="count")
    assert len(calls_old) == 8 and stubs == calls_old          # same trainings, same order, identical inputs
    for key, value in old.items():
        if key != "runtime_s":
            assert new[key] == value, key
    assert set(new) - set(old) == {"config", "sybil_admission"}
    assert new["config"]["tiers"]["method"] == "bfs" and new["config"]["tiers"]["K"] is None
    assert new["config"]["tiers"]["neg_from"] is None                 # the guard of ruling R56 is BTP's, not BFS's
    assert new["config"]["attack"]["attack_edges"] == 0 and new["config"]["preregistration"] is None
    assert new["sybil_admission"]["whitewash"]["n_admitted"] == 0      # without attack edges no fake gets in
    stored = json.loads((tmp_path / "new.json").read_text())
    assert stored["provenance"]["argv"] and stored["config"] == json.loads(json.dumps(new["config"]))


# ------------------------------------------------------------------ D5. the full protocol
def test_full_protocol_records_its_configuration_and_holds_one_test_set(stubs, tmp_path, monkeypatch):
    seen = []
    real_test_metrics = C.test_metrics

    def spy(m, z, D, Gr):
        rows, t1, t2 = C.task_masks(Gr, D.tt)
        seen.append((Gr.rid[rows].tobytes(), t1.tobytes(), t2.tobytes()))
        return real_test_metrics(m, z, D, Gr)

    monkeypatch.setattr(C, "test_metrics", spy)
    prereg = tmp_path / "preregistration.md"
    prereg.write_text("settings fixed before any run\n")
    dump_dir = tmp_path / "dumps"
    protocol = G.RunProtocol(tiers="btp", btp_eps=ALPHA_EPS, attack_edges=1, endorser_select="max_trust",
                             sybil_topology="star", arms=G.ARMS, hold_test_snapshot=True, ungated_baseline=True,
                             dump_targets=str(dump_dir), dump_verdicts=str(dump_dir), prereg=str(prereg),
                             pool_eps=ALPHA_EPS)
    out = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "full.json"), protocol=protocol)
    cal = alpha_calibration()
    bg = background()

    assert len(stubs) == 1 + 3 + 2 * 3 + 1                    # base; direct; whitewash, badmouth, control; ungated
    rows0, t1_0, t2_0 = C.task_masks(bg.G, bg.tt)
    assert len(seen) == 11 and set(seen) == {(bg.G.rid[rows0].tobytes(), t1_0.tobytes(), t2_0.tobytes())}

    cfg = out["config"]
    assert cfg["tiers"] == dict(method="btp", eps=ALPHA_EPS, K=cal.K, theta0=cal.theta0, edge_cap_units=2.0,
                                edge_cap=cal.edge_cap, neg_from="trusted", n_ref=cal.n_ref, n_seeds=cal.n_seeds,
                                t_seed=4, calibration_used_for="tiers and endorser selection")
    assert {k: cfg["attack"][k] for k in ("attack_edges", "endorser_select", "sybil_topology", "k_fake")} == dict(
        attack_edges=1, endorser_select="max_trust", sybil_topology="star", k_fake=3)
    assert cfg["attack"]["endorser_assignment"] == "ranked" and cfg["attack"]["reference_calibration"] == {
        k: cfg["tiers"][k] for k in ("eps", "K", "theta0", "edge_cap_units", "edge_cap", "n_ref", "n_seeds", "t_seed")}
    assert cfg["arms"] == list(G.ARMS) and cfg["hold_test_snapshot"] and cfg["ungated_baseline"]
    assert cfg["gate_norm"] == "count"
    assert cfg["preregistration"] == dict(path=str(prereg), sha256=hashlib.sha256(prereg.read_bytes()).hexdigest())

    inflow = tiers.endorser_inflow_scores(bg.Tr, bg.D.n, cal)
    replay = np.random.default_rng(3000)                       # the victims are drawn after the whitewash snapshots
    [replay.integers(G.FIRST_FAKE_SNAPSHOT, bg.tt) for _ in bg.edge_t]
    victims = replay.choice(bg.trusted, size=15, replace=False).tolist()
    for arm, targets in (("whitewash", list(bg.edge_t)), ("badmouth", victims)):
        adm = out["sybil_admission"][arm]
        assert adm["guarantee_per_group"] == math.floor(cal.K * 1 * 2.0) == 24
        assert len(adm["admitted_per_group"]) == 15 and max(adm["admitted_per_group"]) <= 24
        # Ruling R44: group i's endorser is the account ranked i + 1 by B(u) among the arm's non-targets.
        ranking = [u for u in tiers_report.top_accounts(inflow, bg.D.n).tolist() if u not in set(targets)]
        assert adm["endorsers"] == [[u] for u in ranking[:15]] and adm["endorser_ranks"] == [[i] for i in range(1, 16)]
        assert inflow[adm["endorsers"][0][0]] == pytest.approx(cal.K * cal.edge_cap, rel=1e-12)   # a max-B account
        assert adm["endorsers_in_D_at_attack"] == [[True]] * 15        # seeds: in legacy's D from the start
        assert adm["pool_size"] is None and adm["share_removed_by_timing"] is None   # max_trust draws no pool
        assert adm["pool_criteria"] is None
    assert set(out["inject_then_erase_control"]) >= {"gain_plain_vpe", "gain_cp_vpe"}
    assert set(out["sybil_admission"]) == {"whitewash", "badmouth"}
    assert set(out["test_metrics"]) == {"base", "direct", "whitewash", "badmouth", "control"}
    assert out["test_metrics"]["direct"]["plain"]["t1"]["mcc"] == out["direct_laundering"]["t1_mcc_plain"]
    assert out["ungated_baseline"]["t1"]["n"] == out["test_metrics"]["base"]["t1"]["n"]

    stored = json.loads((tmp_path / "full.json").read_text())
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert stored["provenance"]["git_commit"] == git_head and isinstance(stored["provenance"]["git_dirty"], bool)
    assert stored["provenance"]["code_sha256"] and stored["provenance"]["dataset"] == "bitcoinalpha"
    targets = json.loads((dump_dir / "full_targets.json").read_text())
    # 150 rows, plus 30 since Task 4.1b: the control arm's two models on the 15 bad-mouth victims
    assert targets["columns"] == list(Dm.TARGET_COLUMNS) and targets["n_rows"] == len(targets["rows"]) == 180
    assert targets["config"] == json.loads(json.dumps(cfg)) and targets["provenance"] == stored["provenance"]
    arms_methods = {(r[0], r[1]) for r in targets["rows"]}
    assert arms_methods == {("direct", "plain"), ("direct", "vpe"), ("direct", "cppr")} | {
        (a, m) for a in ("whitewash", "badmouth", "control") for m in ("vpe", "cppr")}
    verdicts = json.loads((dump_dir / "full_verdicts.json").read_text())
    assert verdicts["columns"] == list(Dm.VERDICT_COLUMNS) and verdicts["n_rows"] == len(verdicts["rows"]) > 0
    assert verdicts["provenance"] == stored["provenance"]
    assert {(r[0], r[1]) for r in verdicts["rows"]} == arms_methods - {("direct", "plain")}
    assert out["dumps"] == dict(targets=dict(path=str(dump_dir / "full_targets.json"), n_rows=180),
                                verdicts=dict(path=str(dump_dir / "full_verdicts.json"), n_rows=verdicts["n_rows"]))


# ------------------------------------------------------------------ D6. arm subsets
def test_an_arm_subset_leaves_the_other_arms_unchanged(stubs, tmp_path):
    full = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "all.json"))
    stubs.clear()
    only = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "bm.json"),
                             protocol=G.RunProtocol(arms=("badmouth",)))
    assert len(stubs) == 1 + 2
    assert only["inject_then_erase_badmouth"] == full["inject_then_erase_badmouth"]
    assert only["sybil_admission"]["badmouth"] == full["sybil_admission"]["badmouth"]
    assert not {"direct_laundering", "inject_then_erase_whitewash"} & set(only)
    assert only["config"]["arms"] == ["badmouth"]


# ------------------------------------------------------------------ D7. rulings R39 and R43 in the driver
def test_random_endorsers_are_the_same_under_every_tier_method(stubs, tmp_path):
    # Ruling R43: BFS, capped and uncapped runs of a seed draw their random endorsers from one pool and one stream, so
    # they face the same attack edges, with timing (ruling R39) applied under every tier method.
    def whitewash(name, **tier_options):
        protocol = G.RunProtocol(attack_edges=1, arms=("whitewash",), pool_eps=ALPHA_EPS, **tier_options)
        out = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / name), protocol=protocol)
        return out["sybil_admission"]["whitewash"]

    bfs = whitewash("bfs.json")
    assert len(stubs) == 1 + 2
    assert bfs["timing_applied"] and bfs["n_admitted"] == 15 * 3      # every region lifted by legacy
    assert bfs["endorsers_in_D_at_attack"] == [[True]] * 15 and bfs["endorser_ranks"] is None
    assert 1 <= bfs["pool_size"]["min"] <= bfs["pool_size"]["max"] <= bfs["pool_size_untimed"]
    assert 0 < bfs["share_removed_by_timing"] < 1
    bg = background()
    protocol = G.RunProtocol(attack_edges=1, pool_eps=ALPHA_EPS)
    inputs = G.endorser_inputs(protocol, bg.Tr, bg.D.n, None, protocol.reference_calibration(bg.Tr, bg.D.n))
    pool, sizes = G.common_pool(inputs, bg.edge_t)
    assert [(c["criterion"], c["size_before"], c["size"]) for c in bfs["pool_criteria"]] == [
        (name, before, size) for (_, before), (name, size) in zip(sizes, sizes[1:])]
    assert bfs["pool_size_untimed"] == len(pool)
    shared = ("endorsers", "snap_f", "endorsers_in_D_at_attack", "pool_size", "pool_size_untimed",
              "share_removed_by_timing", "timing_applied", "pool_criteria")
    for name, options in (("capped.json", dict(tiers="btp", btp_eps=ALPHA_EPS)),
                          ("uncapped.json", dict(tiers="btp", btp_eps=0.05, btp_edge_cap=None))):
        other = whitewash(name, **options)
        assert {k: other[k] for k in shared} == {k: bfs[k] for k in shared}, name


# ------------------------------------------------------------------ D8. pre-flight pool check (Task 4.1b)
WHITEWASH_POOL = "before training, whitewash arm, pool without its targets (15): "
BADMOUTH_POOL = "before training, badmouth arm, pool without the accounts its targets are drawn from (1118): "


def _short_pool_case(case):
    """(protocol, prefix, message) of a pre-flight failure on Bitcoin-Alpha seed 0 under BFS tiers. The pool sizes come
    from the library: the whitewash arm's common pool before timing and at snapshot 5, and the bad-mouth bound's (the
    pool without the whole trusted cohort, since the victims are drawn only after the whitewash graph)."""
    bg = background()
    n, groups = bg.D.n, len(bg.edge_t)
    protocol = G.RunProtocol(attack_edges=1, pool_eps=ALPHA_EPS)
    inputs = G.endorser_inputs(protocol, bg.Tr, n, None, protocol.reference_calibration(bg.Tr, n))
    sizes = {}
    for arm, exclude in (("whitewash", bg.edge_t), ("badmouth", bg.trusted)):
        pool, criteria = G.common_pool(inputs, exclude)
        sizes[arm] = len(pool), int((inputs.join[pool] <= G.FIRST_FAKE_SNAPSHOT).sum())
        # One literal anchors the library's pools: 2050 accounts pass criterion (a), BFS tier >= 2, which comes before
        # the targets are removed. It depends on neither the BTP guard nor the seed (2050 on seed 5 too).
        assert dict(criteria)["bfs_tier"] == 2050
    (u_ww, t_ww), (u_bm, t_bm) = sizes["whitewash"], sizes["badmouth"]
    assert t_ww < u_ww and t_bm < min(u_bm, t_ww)              # each case below exists
    ranked_ww, ranked_bm = n - groups, n - len(bg.trusted)     # ranked candidates: every account but the excluded
    g, prefix, message = dict(
        untimed=(u_ww + 1, WHITEWASH_POOL, f"only {u_ww} candidate endorsers for {u_ww + 1} attack edges per group"),
        timely=(t_ww + 1, WHITEWASH_POOL,
                f"only {t_ww} candidate endorsers credible at snapshot 5 for {t_ww + 1} attack edges (ruling R39)"),
        badmouth=(t_bm + 1, BADMOUTH_POOL,
                  f"only {t_bm} candidate endorsers credible at snapshot 5 for {t_bm + 1} attack edges (ruling R39)"),
        ranked=(ranked_ww // groups + 1, WHITEWASH_POOL, f"only {ranked_ww} candidate endorsers for {groups} groups "
                f"of {ranked_ww // groups + 1} ranked max-B(u) endorsers (ruling R44)"),
        ranked_badmouth=(ranked_bm // groups + 1, BADMOUTH_POOL, f"only {ranked_bm} candidate endorsers for {groups} "
                         f"groups of {ranked_bm // groups + 1} ranked max-B(u) endorsers (ruling R44)"),
    )[case]
    select = "max_trust" if case.startswith("ranked") else "random"
    return G.RunProtocol(attack_edges=g, endorser_select=select, arms=("whitewash",), pool_eps=ALPHA_EPS), prefix, message


@pytest.mark.parametrize("case", ["untimed", "timely", "badmouth", "ranked", "ranked_badmouth"])
def test_a_short_endorser_pool_stops_the_run_before_the_base_model_trains(stubs, tmp_path, case):
    # Bitcoin-Alpha seed 0, BFS tiers. A random group draws from its arm's common pool (ruling R43) restricted to the
    # accounts in legacy's D at its snap_f, smallest at snapshot 5; ranked endorsers need g candidates per group
    # (ruling R44). Both injected graphs are built in every run, so the bad-mouth arm is checked, against its bound,
    # although only the whitewash arm is trained.
    protocol, prefix, message = _short_pool_case(case)
    with pytest.raises(ValueError, match=f"^{re.escape(prefix + message)}$"):
        G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "short.json"), protocol=protocol)
    assert stubs == [] and not (tmp_path / "short.json").exists()


# ------------------------------------------------------------------ D9. dump folders and write order (Task 4.1b)
def test_dump_folders_are_made_before_training_and_the_result_is_written_before_the_dumps(stubs, tmp_path,
                                                                                         monkeypatch):
    blocker = tmp_path / "a_file"
    blocker.write_text("")
    with pytest.raises(OSError):                      # a dump folder that cannot be made stops the run up front
        G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "blocked.json"),
                          protocol=G.RunProtocol(dump_targets=str(blocker / "dumps")))
    assert stubs == [] and not (tmp_path / "blocked.json").exists()

    dump_dir = tmp_path / "new" / "dumps"
    folder_at_training = []
    train = G.train_gated_model

    def spy_train(*args, **kwargs):
        folder_at_training.append(dump_dir.is_dir())
        return train(*args, **kwargs)

    written = []
    real_dump = C.dump

    def spy_dump(obj, name, D=None, force=False):
        written.append(name)
        if name.endswith("_verdicts.json"):
            raise OSError("disk full")                # a dump that fails must not cost the result
        return real_dump(obj, name, D, force=force)

    monkeypatch.setattr(G, "train_gated_model", spy_train)
    monkeypatch.setattr(C, "dump", spy_dump)
    out = tmp_path / "run.json"
    with pytest.raises(OSError, match="disk full"):
        G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(out),
                          protocol=G.RunProtocol(arms=("whitewash",), dump_targets=str(dump_dir),
                                                 dump_verdicts=str(dump_dir)))
    assert folder_at_training == [True] * 3
    targets, verdicts = dump_dir / "run_targets.json", dump_dir / "run_verdicts.json"
    assert written == [str(out), str(targets), str(verdicts)]
    stored = json.loads(out.read_text())
    assert stored["dumps"]["targets"] == dict(path=str(targets), n_rows=15 * 2)
    assert json.loads(targets.read_text())["n_rows"] == 30 and not verdicts.exists()
    prov = stored["provenance"]
    assert isinstance(prov["git_untracked"], int) and prov["git_untracked"] == sum(
        line.startswith("??") for line in prov["git_status"])
    assert {"python", "numpy", "scipy", "pandas", "torch_scatter", "torch_geometric", "scikit-learn"} <= set(prov)


# ------------------------------------------------------------------ D10. --methods: flags (Task 4.1b)
def test_methods_flag_parses_validates_and_is_a_changed_protocol(stubs):
    parser = G.build_parser()
    assert parser.parse_args([]).methods == "plain_erasure,plain_vpe,cppr"
    assert G.protocol_from_args(parser.parse_args([])).methods == G.METHODS == ("plain_erasure", "plain_vpe", "cppr")
    assert G.protocol_from_args(parser.parse_args(["--methods", "cppr,plain_vpe"])).methods == ("plain_vpe", "cppr")
    assert G.parse_methods(" cppr ") == ("cppr",)
    for bad in ("", "plain_vpe", "plain_erasure,plain_vpe", "cppr,cppr", "cppr,ungated"):
        with pytest.raises(ValueError):
            G.parse_methods(bad)
    for bad in ((), ("plain_vpe",), ("cppr", "cppr"), ("cppr", "retain")):
        with pytest.raises(ValueError):
            G.RunProtocol(methods=bad)
    assert not G.RunProtocol(methods=("cppr",)).is_legacy()
    assert G.RunProtocol(methods=("cppr", "plain_erasure", "plain_vpe")).is_legacy()       # canonical order
    with pytest.raises(ValueError, match="--out"):
        G.run_single_seed("bitcoinalpha", 0, protocol=G.RunProtocol(methods=("cppr",)))
    with pytest.raises(SystemExit):
        G.main(["--methods", "plain_vpe"])                    # the CLI refuses a list without cppr (argparse)
    assert stubs == []


# ------------------------------------------------------------------ D11. --methods cppr
DIRECT_KEYS = ("gain_plain_edge", "gain_vpe_edge", "gain_cp_vpe_edge", "reduction_plain_vpe", "reduction_cp_vpe",
               "gain_plain_account", "gain_vpe_account", "gain_cp_vpe_account", "t1_mcc_plain", "t1_mcc_vpe",
               "t1_mcc_cp_vpe", "d_t1_mcc_cp_vs_plain", "linkage_plain_vpe", "linkage_cp_vpe")   # the stored order


def test_cppr_alone_keeps_cpprs_outputs_with_fewer_trainings(stubs, tmp_path, monkeypatch):
    # Every graph and every draw from rng is the default run's, so CPPR's gains, linkage, test metrics, Sybil
    # admission and dump rows are identical. Plain Erasure and Plain VPE are not retrained; their fields and rows,
    # and the fields that compare them with CPPR, are absent (not zeros).
    built = []
    for name in ("plain_erasure", "apply_vpe"):                # the releases of the methods that are not retrained
        real = getattr(V, name)
        monkeypatch.setattr(V, name, lambda *a, _real=real, _name=name, **k: (built.append(_name), _real(*a, **k))[1])

    def run(name, **options):
        dumps = tmp_path / name
        protocol = G.RunProtocol(attack_edges=1, arms=G.ARMS, hold_test_snapshot=True, dump_targets=str(dumps),
                                 dump_verdicts=str(dumps), pool_eps=ALPHA_EPS, **options)
        built.clear()
        out = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / f"{name}.json"),
                                protocol=protocol)
        return out, {kind: json.loads((dumps / f"{name}_{kind}.json").read_text())["rows"]
                     for kind in ("targets", "verdicts")}

    full, full_rows = run("full")
    calls_full, built_full = list(stubs), list(built)
    stubs.clear()
    cppr, cppr_rows = run("cppr", methods=("cppr",))
    assert built == built_full == ["plain_erasure"] + ["apply_vpe"] * 4     # still built: direct and three arms
    # base; direct plain, vpe, cppr; then vpe and cppr of whitewash, bad-mouth and control: CPPR's keep their inputs
    assert len(calls_full) == 10 and stubs == [calls_full[i] for i in (0, 3, 5, 7, 9)]
    assert list(full["direct_laundering"]) == list(DIRECT_KEYS)
    assert cppr["direct_laundering"] == {k: full["direct_laundering"][k] for k in (
        "gain_cp_vpe_edge", "gain_cp_vpe_account", "t1_mcc_cp_vpe", "linkage_cp_vpe")}
    assert cppr["inject_then_erase_whitewash"] == {"gain_cp_vpe": full["inject_then_erase_whitewash"]["gain_cp_vpe"]}
    assert cppr["inject_then_erase_badmouth"] == {"drop_cp_vpe": full["inject_then_erase_badmouth"]["drop_cp_vpe"]}
    assert cppr["inject_then_erase_control"] == {k: full["inject_then_erase_control"][k] for k in (
        "gain_cp_vpe", "gain_cp_vpe_badmouth", "n_fakes", "n_requests")}
    assert cppr["test_metrics"] == {arm: m if arm == "base" else {"cppr": m["cppr"]}
                                    for arm, m in full["test_metrics"].items()}
    assert cppr["sybil_admission"] == full["sybil_admission"] and cppr["base_t1"] == full["base_t1"]
    for kind in ("targets", "verdicts"):
        assert cppr_rows[kind] == [row for row in full_rows[kind] if row[1] == "cppr"] != []
        assert cppr["dumps"][kind]["n_rows"] == len(cppr_rows[kind])
    assert cppr["config"]["methods"] == ["cppr"] and full["config"]["methods"] == list(G.METHODS)


# ------------------------------------------------------------------ D12. other method lists
@pytest.mark.parametrize("methods,n_trainings,direct,whitewash", [
    (("plain_vpe", "cppr"), 1 + 2 + 2,
     ["gain_vpe_edge", "gain_cp_vpe_edge", "gain_vpe_account", "gain_cp_vpe_account", "t1_mcc_vpe", "t1_mcc_cp_vpe",
      "linkage_plain_vpe", "linkage_cp_vpe"],
     ["gain_plain_vpe", "gain_cp_vpe", "suppression_ratio"]),
    (("plain_erasure", "cppr"), 1 + 2 + 1,
     ["gain_plain_edge", "gain_cp_vpe_edge", "reduction_cp_vpe", "gain_plain_account", "gain_cp_vpe_account",
      "t1_mcc_plain", "t1_mcc_cp_vpe", "d_t1_mcc_cp_vs_plain", "linkage_cp_vpe"],
     ["gain_cp_vpe"]),
])
def test_a_method_list_trains_only_its_methods(stubs, tmp_path, methods, n_trainings, direct, whitewash):
    out = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "subset.json"),
                            protocol=G.RunProtocol(arms=("direct", "whitewash"), methods=methods))
    assert len(stubs) == n_trainings
    assert list(out["direct_laundering"]) == direct and list(out["inject_then_erase_whitewash"]) == whitewash
    assert all(v is not None for k, v in out["direct_laundering"].items() if k.startswith("t1_mcc"))


# ------------------------------------------------------------------ D13. the control arm on the bad-mouth victims
def test_the_control_models_are_also_scored_on_the_badmouth_victims(stubs, tmp_path, monkeypatch):
    # The bad-mouth victims are drawn from rng before the control arm, so the control arm's Plain VPE and CPPR,
    # trained once, are scored on them too, with no extra training. H2b then becomes the control-adjusted endpoint
    # (VPE_bm - VPE_ctl) - (CPPR_bm - CPPR_ctl) per victim. The gains on the whitewash targets keep their meaning.
    trained = []
    real = G._retention_arm

    def spy(*args, **kwargs):
        trained.append(real(*args, **kwargs))
        return trained[-1]

    monkeypatch.setattr(G, "_retention_arm", spy)
    dumps = tmp_path / "dumps"
    out = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "ctl.json"),
                            protocol=G.RunProtocol(arms=("badmouth", "control"), dump_targets=str(dumps)))
    assert len(stubs) == 1 + 2 + 2                           # base, bad-mouth and control: nothing more
    control = trained[-1]                                    # the control arm runs last
    rows = json.loads((dumps / "ctl_targets.json").read_text())["rows"]
    col = {c: i for i, c in enumerate(Dm.TARGET_COLUMNS)}

    def cohort(arm, method, kind):
        picked = [r for r in rows if (r[col["arm"]], r[col["method"]], r[col["kind"]]) == (arm, method, kind)]
        return ([r[col["target"]] for r in picked], np.array([r[col["r0"]] for r in picked]),
                [r[col["r_after"]] for r in picked])

    victims, r0_bm, _ = cohort("badmouth", "cppr", "trusted")
    edge_t, r0_edge, _ = cohort("control", "cppr", "edge")
    assert len(victims) == 15 and edge_t == list(background().edge_t)
    block = out["inject_then_erase_control"]
    assert list(block) == ["gain_plain_vpe", "gain_cp_vpe", "gain_plain_vpe_badmouth", "gain_cp_vpe_badmouth",
                           "n_fakes", "n_requests"]
    panel = C.Panel(background().Tr, n_probe=256, seed=1234)
    for name, label in (("vpe", "plain_vpe"), ("cppr", "cp_vpe")):
        model, z = control[name].model, control[name].z
        on_victims = panel.reputation(model, z, np.array(victims))
        assert block[f"gain_{label}_badmouth"] == float(np.mean(on_victims - r0_bm))           # same r0_trusted
        assert block[f"gain_{label}"] == float(np.mean(panel.reputation(model, z, np.array(edge_t)) - r0_edge))
        targets, r0, r_after = cohort("control", name, "trusted")
        assert targets == victims and np.array_equal(r0, r0_bm) and r_after == on_victims.tolist()


# ------------------------------------------------------------------ D14. snap_f per group
def test_each_injected_arm_records_the_snapshots_the_stored_stream_drew(stubs, tmp_path):
    # Later runs of the same seed (E5) are paired with this one, so the analysis can assert that every group's snap_f
    # matches: the stored rng draws the whitewash groups' snapshots, then the bad-mouth targets, then their groups'.
    out = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "snaps.json"),
                            protocol=G.RunProtocol(arms=("whitewash", "badmouth", "control")))
    bg = background()
    replay = np.random.default_rng(3000)
    whitewash = [int(replay.integers(G.FIRST_FAKE_SNAPSHOT, bg.tt)) for _ in bg.edge_t]
    victims = replay.choice(bg.trusted, size=min(len(bg.edge_t), len(bg.trusted)), replace=False)
    badmouth = [int(replay.integers(G.FIRST_FAKE_SNAPSHOT, bg.tt)) for _ in victims]
    admission = out["sybil_admission"]
    assert admission["whitewash"]["snap_f"] == whitewash and admission["badmouth"]["snap_f"] == badmouth
    assert len(whitewash) == len(badmouth) == 15 and set(admission) == {"whitewash", "badmouth"}   # control: no group


# ------------------------------------------------------------------ D15. --pool-eps (Task 4.1c)
def test_pool_eps_is_required_with_attack_edges_and_equals_btp_eps_in_capped_runs(stubs):
    # --pool-eps sets the endorser protocol's reference capped calibration (eps = EPS, KEEPER's cap, rulings R43 and
    # R44), required whenever endorsers are drawn, under every tier method. In a capped BTP run it is the tier
    # calibration's eps; uncapped and BFS runs keep it apart, and BFS runs need no --btp-eps, max_trust included.
    parser = G.build_parser()
    assert parser.parse_args([]).pool_eps is None and parser.parse_args([]).base_cache is None
    assert G.protocol_from_args(parser.parse_args(["--attack-edges", "1", "--pool-eps", "0.02"])).pool_eps == 0.02
    for options in (dict(), dict(tiers="btp", btp_eps=0.02), dict(tiers="btp", btp_eps=0.05, btp_edge_cap=None)):
        for select in G.ENDORSER_SELECTIONS:
            with pytest.raises(ValueError, match="--pool-eps is required"):
                G.RunProtocol(attack_edges=1, endorser_select=select, **options)
            assert G.RunProtocol(attack_edges=1, endorser_select=select, pool_eps=0.02, **options).pool_eps == 0.02
    for bad in (0.0, -0.02, INF, float("nan")):
        with pytest.raises(ValueError, match="pool_eps must be positive and finite"):
            G.RunProtocol(attack_edges=1, pool_eps=bad)
    with pytest.raises(ValueError, match=r"--pool-eps \(0.01\) must equal --btp-eps \(0.02\)"):
        G.RunProtocol(tiers="btp", btp_eps=0.02, attack_edges=1, pool_eps=0.01)
    assert G.RunProtocol(endorser_select="max_trust", attack_edges=2, pool_eps=0.02).btp_eps is None
    bg = background()
    n, reference = bg.D.n, alpha_calibration()
    bfs = G.RunProtocol(endorser_select="max_trust", attack_edges=1, pool_eps=ALPHA_EPS)
    assert bfs.calibrate(bg.Tr, n) is None and bfs.reference_calibration(bg.Tr, n) == reference
    capped = G.RunProtocol(tiers="btp", btp_eps=ALPHA_EPS, attack_edges=1, pool_eps=ALPHA_EPS)
    assert capped.calibrate(bg.Tr, n) == capped.reference_calibration(bg.Tr, n) == reference
    uncapped = G.RunProtocol(tiers="btp", btp_eps=0.05, btp_edge_cap=None, attack_edges=1, pool_eps=ALPHA_EPS)
    assert uncapped.calibrate(bg.Tr, n).edge_cap_units is None and uncapped.reference_calibration(bg.Tr, n) == reference
    assert G.RunProtocol().reference_calibration(bg.Tr, n) is None
    # A flag that would be ignored is refused (fix round 1): --btp-eps under BFS tiers, --pool-eps without attack edges.
    with pytest.raises(ValueError, match=r"^--btp-eps sets BTP tiers' calibration; a BFS run takes none"):
        G.RunProtocol(btp_eps=0.02, attack_edges=1, pool_eps=0.02)
    for tier_options in (dict(), dict(tiers="btp", btp_eps=0.02)):
        with pytest.raises(ValueError, match=r"^--pool-eps has no use without --attack-edges"):
            G.RunProtocol(pool_eps=0.02, **tier_options)
    for argv in (["--attack-edges", "1"], ["--tiers", "btp", "--btp-eps", "0.02", "--attack-edges", "1",
                                           "--pool-eps", "0.05"], ["--btp-eps", "0.02"], ["--pool-eps", "0.02"]):
        with pytest.raises(SystemExit):                       # the CLI refuses through argparse
            G.main(argv)
    assert stubs == []


# ------------------------------------------------------------------ D19. folders made up front (Task 4.1c)
def test_the_result_and_cache_folders_are_made_before_training(stubs, tmp_path, monkeypatch):
    blocker = tmp_path / "a_file"
    blocker.write_text("")
    for out_name, cache in ((blocker / "run.json", None), (tmp_path / "run.json", blocker / "cache")):
        with pytest.raises(OSError):                          # a folder that cannot be made stops the run up front
            G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(out_name),
                              protocol=G.RunProtocol(arms=("whitewash",), base_cache=cache and str(cache)))
    assert stubs == [] and not (tmp_path / "run.json").exists()
    out, cache = tmp_path / "new" / "results" / "run.json", tmp_path / "new" / "cache"
    seen = []
    train = G.train_gated_model
    monkeypatch.setattr(G, "train_gated_model",
                        lambda *a, **k: (seen.append((out.parent.is_dir(), cache.is_dir())), train(*a, **k))[1])
    G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(out),
                      protocol=G.RunProtocol(arms=("whitewash",), methods=("cppr",), base_cache=str(cache)))
    assert seen == [(True, True)] * 2 and out.exists()
