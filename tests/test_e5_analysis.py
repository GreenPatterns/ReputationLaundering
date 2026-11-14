"""Task 4.3: the E5 task list (keeper.cluster.e5_tasks) and the descriptive E5 analysis (keeper.analysis.e5_analysis).

  A1 the task list: 30 capped-BTP runs that train the baselines alone, each with the flags of its E3 or E4 partner
     (the driver's own parser reads both commands), outputs under results/cppr/e5 and dumps for E6
  A2 the analysis on synthetic runs: gains, paired differences and test-MCC differences, with missing runs listed
  A3 the pairing assertions: a different r0, target set, snap_f, endorsers or configuration stops the analysis
CPU only: CUDA_VISIBLE_DEVICES="" <ps python> -m pytest -q tests/test_e5_analysis.py
"""
import json
import os
import shlex
from pathlib import Path

import numpy as np
import pytest

from keeper.analysis import e5_analysis as A
from keeper.cluster import cppr_tasks, e5_tasks
from keeper.cluster import protocol_settings as ps
from keeper.cppr import cp_vpe_gpu
from keeper.cppr import dumps as Dm

RESULTS = "/data/keeper"
DRIVER_WORDS = 8                  # env -u PYTHONPATH -u VIRTUAL_ENV <python> -m keeper.cppr.cp_vpe_gpu
FIELDS = ("ds", "seed", "epochs", "rho", "granularity", "k_fake", "n_edge", "n_acc", "gate_norm", "tiers", "btp_eps",
          "btp_edge_cap", "btp_neg_from", "attack_edges", "endorser_select", "sybil_topology", "pool_eps",
          "hold_test_snapshot", "prereg")


def parsed(row):
    return cp_vpe_gpu.build_parser().parse_args(shlex.split(row.command)[DRIVER_WORDS:])


# ------------------------------------------------------------------ A1. the task list
def test_e5_has_two_runs_per_network_and_seed_each_with_the_flags_of_its_partner():
    pairs = e5_tasks.pairs(RESULTS)
    assert len(pairs) == len(e5_tasks.e5_tasks(RESULTS)) == 30 and len({p.e5.id for p in pairs}) == 30
    assert [(p.ds, p.seed, p.g) for p in pairs] == [(ds, seed, g) for ds in ("wiki-rfa", "bitcoinotc", "bitcoinalpha")
                                                    for seed in range(5, 10) for g in (0, 1)]
    e3 = {r.id: r for r in cppr_tasks.e3_tasks(RESULTS)}
    e4 = {r.id: r for r in cppr_tasks.e4_tasks(RESULTS)}
    for p in pairs:
        assert p.partner == (e3 if p.g == 0 else e4)[p.partner.id]
        mine, theirs = parsed(p.e5), parsed(p.partner)
        assert {f: getattr(mine, f) for f in FIELDS} == {f: getattr(theirs, f) for f in FIELDS}
        assert (mine.ds, mine.seed, mine.tiers, mine.btp_eps) == (p.ds, p.seed, "btp", ps.EPS_CAPPED[p.ds])
        assert cp_vpe_gpu.protocol_from_args(mine).methods == ("author", "singleton", "withdrawal")
        assert mine.arms == ("direct,whitewash" if p.g == 0 else "whitewash") and mine.base_cache == (
            None if p.g == 0 else f"{RESULTS}/.keeper-cache/base") == theirs.base_cache
        assert (mine.attack_edges, mine.endorser_select, mine.sybil_topology, mine.k_fake) == (
            (0, "random", "ring", 3) if p.g == 0 else (1, "random", "ring", 3))
        assert p.e5.done_path == mine.out == f"{RESULTS}/results/cppr/e5/{p.e5.id}.json"
        assert p.e5.log_path == f"{RESULTS}/results/cppr/e5/logs/{p.e5.id}.log"
        assert (mine.dump_targets, mine.dump_verdicts) == (f"{RESULTS}/results/cppr/e5/targets",
                                                           f"{RESULTS}/results/dumps/e5")
        assert p.e5.id == (f"e5_btpc_g0_{p.ds}_s{p.seed}" if p.g == 0 else f"e5_btpc_rand_ring_k3_g1_{p.ds}_s{p.seed}")


def test_the_e5_command_line_writes_the_task_file_once(tmp_path, capsys):
    out = tmp_path / "e5.tsv"
    assert e5_tasks.main(["--results-root", str(tmp_path), "--out", str(out)]) == 0
    assert out.read_text() == cppr_tasks.render(e5_tasks.e5_tasks(str(tmp_path))) and "wrote 30 tasks" in \
        capsys.readouterr().out
    assert e5_tasks.main(["--results-root", str(tmp_path), "--out", str(out)]) == 1
    with pytest.raises(SystemExit):
        e5_tasks.main(["--results-root", "relative/root", "--out", str(out)])


# ------------------------------------------------------------------ A2, A3. the analysis on synthetic runs
SEEDS = (5, 6, 7)
DS = "bitcoinalpha"
TARGETS = {5: [10, 11, 12], 6: [10, 11, 12, 13], 7: [10, 12, 13]}  # targets recur across seeds, with missing cells
ACCOUNTS = {5: [20, 21], 6: [20, 21, 22], 7: [20, 22]}
LEVEL = dict(author=0.30, singleton=0.10, withdrawal=0.20, plain=0.25, vpe=0.05, cppr=0.01)
MCC = dict(author=0.60, singleton=0.55, withdrawal=0.58, plain=0.57, vpe=0.52, cppr=0.54)
METHODS = {("e5", "direct"): ("author", "singleton", "withdrawal"), ("e5", "whitewash"): ("author", "singleton",
           "withdrawal"), ("partner", "direct"): ("plain", "vpe", "cppr"), ("partner", "whitewash"): ("vpe", "cppr"),
           ("partner", "badmouth"): ("vpe", "cppr")}         # an arm E5 does not run: the analysis ignores it


def gain(method: str, seed: int, target: int, g: int = 0) -> float:
    return LEVEL[method] + 0.01 * seed + 0.001 * target + 0.1 * g


def r0(target: int) -> float:
    return 0.5 + target / 1000


def write(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_run(row, role: str, seed: int, g: int, arms) -> None:
    rows, held = [], {}
    for arm in arms:
        cohorts = [("edge", TARGETS[seed])] + ([("account", ACCOUNTS[seed])] if arm == "direct" else [])
        for method in METHODS[(role, arm)]:
            held.setdefault(arm, {})[method] = {"t1": {"mcc": MCC[method] + 0.01 * seed + 0.1 * g}}
            for kind, targets in cohorts:
                rows += [[arm, method, kind, t, r0(t), r0(t) + gain(method, seed, t, g), gain(method, seed, t, g)]
                         + [0] * 5 + [False] for t in targets]
    attack = dict(attack_edges=g, endorser_select="random", sybil_topology="ring", k_fake=3)
    write(Path(row.done_path), dict(
        config=dict(tiers=dict(method="btp", eps=0.02), attack=attack), test_metrics=held,
        sybil_admission=dict(whitewash=dict(snap_f=[5 + seed, 9], endorsers=[[100 + seed]] if g else None))))
    write(Path(A.target_dump_path(row)), dict(columns=list(Dm.TARGET_COLUMNS), rows=rows))


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A results root with every E5 run of DS and SEEDS and its partner (synthetic numbers)."""
    monkeypatch.setattr(ps, "SEEDS_5", SEEDS)
    for p in e5_tasks.pairs(str(tmp_path)):
        if p.ds == DS:
            arms = ("direct", "whitewash") if p.g == 0 else ("whitewash",)
            write_run(p.e5, "e5", p.seed, p.g, arms)
            write_run(p.partner, "partner", p.seed, p.g, arms + (("badmouth",) if p.g == 0 else ()))
    return tmp_path


def by_hand(values: dict) -> float:
    """The mean of per-seed means."""
    return float(np.mean([np.mean(v) for v in values.values()]))


def test_the_analysis_reports_gains_differences_and_mcc_with_intervals(root):
    report = A.analyse(str(root), networks=(DS,), n_boot=2000)
    blocks = report["networks"][DS]
    assert set(blocks) == {"direct_edge", "direct_account", "whitewash_g0", "whitewash_g1", "mcc_direct",
                           "mcc_whitewash_g0", "mcc_whitewash_g1"}
    assert len(report["pairs"]) == 2 * len(SEEDS) and len(report["missing"]) == 0
    for name, cohort, g in (("direct_edge", TARGETS, 0), ("direct_account", ACCOUNTS, 0),
                            ("whitewash_g1", TARGETS, 1)):
        block = blocks[name]
        for method in ("author", "singleton", "withdrawal", "vpe", "cppr") + (("plain",) * name.startswith("direct")):
            expected = by_hand({s: [gain(method, s, t, g) for t in cohort[s]] for s in SEEDS})
            assert block[f"gain_{method}"] == pytest.approx(expected, abs=1e-12)
            lo, hi = block["ci95"][f"gain_{method}"]
            assert lo <= expected <= hi and block["n_cells"][f"gain_{method}"] == sum(len(v) for v in cohort.values())
        # A paired difference is constant here, so its interval is a point.
        assert block["singleton_minus_cppr"] == pytest.approx(LEVEL["singleton"] - LEVEL["cppr"], abs=1e-12)
        assert block["ci95"]["withdrawal_minus_vpe"] == pytest.approx([LEVEL["withdrawal"] - LEVEL["vpe"]] * 2)
        assert ("author_minus_plain" in block) == (name != "whitewash_g1")
    mcc = blocks["mcc_whitewash_g1"]
    assert mcc["mcc_author"] == pytest.approx(MCC["author"] + 0.06 + 0.1, abs=1e-12)    # seeds 5-7: + 0.01 seed
    assert mcc["author_minus_cppr"] == pytest.approx(MCC["author"] - MCC["cppr"], abs=1e-12)
    assert mcc["by_seed"]["mcc_cppr"] == pytest.approx([MCC["cppr"] + 0.01 * s + 0.1 for s in SEEDS])
    assert set(blocks["mcc_direct"]["ci95"]) == {f"mcc_{m}" for m in LEVEL} | {
        f"{b}_minus_{c}" for b in ("author", "singleton", "withdrawal") for c in ("cppr", "vpe", "plain")}


def test_a_missing_run_is_listed_and_its_seed_left_out(root):
    gone = next(p for p in e5_tasks.pairs(str(root)) if p.ds == DS and p.seed == 6 and p.g == 1)
    os.remove(gone.partner.done_path)
    report = A.analyse(str(root), networks=(DS,), n_boot=2000)
    assert report["missing"] == [dict(e5=gone.e5.id, partner=gone.partner.id, missing=[gone.partner.id])]
    assert report["networks"][DS]["whitewash_g1"]["seeds"]["gain_author"] == [5, 7]
    assert report["networks"][DS]["whitewash_g0"]["seeds"]["gain_author"] == list(SEEDS)


def _edit(path, change) -> None:
    data = json.loads(open(path).read())
    change(data)
    write(Path(path), data)


@pytest.mark.parametrize("case,message", [
    ("r0", "direct/edge target 11: r0"), ("targets", "whitewash/edge targets differ: [13]"),
    ("snap_f", "whitewash snap_f"), ("endorsers", "whitewash endorsers"), ("config", "config.tiers differs")])
def test_a_run_that_does_not_pair_with_its_partner_stops_the_analysis(root, case, message, capsys):
    pairs = [p for p in e5_tasks.pairs(str(root)) if p.ds == DS and p.seed == 6]
    p = pairs[1] if case == "endorsers" else pairs[0]
    col = {c: i for i, c in enumerate(Dm.TARGET_COLUMNS)}

    def change(data):
        if case == "r0":
            for r in data["rows"]:
                if (r[col["arm"]], r[col["kind"]], r[col["target"]]) == ("direct", "edge", 11):
                    r[col["r0"]] += 1e-9
        elif case == "targets":
            data["rows"] = [r for r in data["rows"] if (r[col["arm"]], r[col["target"]]) != ("whitewash", 13)]
        elif case == "snap_f":
            data["sybil_admission"]["whitewash"]["snap_f"][1] = 10
        elif case == "endorsers":
            data["sybil_admission"]["whitewash"]["endorsers"] = [[7]]
        else:
            data["config"]["tiers"]["eps"] = 0.05

    _edit(A.target_dump_path(p.partner) if case in ("r0", "targets") else p.partner.done_path, change)
    with pytest.raises(A.PairingError, match="Section 2.7") as err:
        A.analyse(str(root), networks=(DS,), n_boot=2000)
    assert message in str(err.value) and p.e5.id in str(err.value)
    assert A.main(["--results-root", str(root)]) == A.EXIT_PAIRING
    assert message in capsys.readouterr().err and not (root / "results" / "analysis").exists()


def test_the_command_line_writes_json_and_markdown_once(root, capsys, monkeypatch):
    monkeypatch.setattr(A, "analyse", lambda results_root, _real=A.analyse: _real(results_root, networks=(DS,),
                                                                                 n_boot=2000))
    assert A.main(["--results-root", str(root)]) == A.EXIT_OK
    out = json.loads((root / "results" / "analysis" / "e5.json").read_text())
    md = (root / "results" / "analysis" / "e5.md").read_text()
    assert out["schema"] == A.SCHEMA and out["provenance"]["argv"] is not None and "| gain_author |" in md
    assert A.main(["--results-root", str(root)]) == A.EXIT_EXISTS and "exists" in capsys.readouterr().err
    assert A.main(["--results-root", str(root), "--force"]) == A.EXIT_OK
