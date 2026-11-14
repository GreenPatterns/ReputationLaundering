"""Task 4.8: the H8 and E8 analyses (keeper.analysis.e9a_e8) on synthetic E9a runs with known answers.

The synthetic world replaces the data (e9a_e8.reference) by 40 active accounts, 6 drawn targets and 3 stratum
targets, and writes every registered run (3 networks x seeds 5-9 x self term off/on) as the runner writes it: a result
JSON and an .npz dump whose population rows hold each target's post-U1 proxy at its own position. Proxies and gains
are dyadic, so float32 storage is exact and the self-term-on gains are exactly FACTOR[ds] times the off gains:
Alpha 2 (H8(b) met), OTC 0.75 (met), Wiki-RfA 0.25 (partial). Checked: the H8 endpoints and p-values by hand, the
readings (met, partial, contradicted, not evaluable), the E8 call wiring (per-target populations, target_proxy from
the same array, stored target ids, the pct_after cross-check), every integrity failure, missing runs, bit-identical
reruns and the CLI. CPU only.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import ttest_1samp

from keeper.analysis import bootstrap as B
from keeper.analysis import cold_start as CS
from keeper.analysis import dump_manifest as DM
from keeper.analysis import e9a_e8 as E
from keeper.cluster import e9a_tasks as T
from keeper.cluster import protocol_settings as PS
from keeper.cppr.protocol import sha256_file
from keeper.laundering import metrics as M

N_BOOT = 500
N_NODES = 45
ACTIVE = np.arange(40, dtype=np.int64)
DRAWN, STRATUM = [3, 5, 7, 9, 11, 13], [20, 22, 24]
N_POS = {3: 0, 5: 0, 7: 1, 9: 0, 11: 2, 13: 0, 20: 3, 22: 0, 24: 1}
SIZES = dict(pos=2, nopos=4)
FACTOR = {"bitcoinalpha": 2.0, "bitcoinotc": 0.75, "wiki-rfa": 0.25}
_rng = np.random.default_rng(0)
REF = SimpleNamespace(drawn=np.array(DRAWN), stratum=np.array(STRATUM), sizes=SIZES, active=ACTIVE,
                      n_in=_rng.integers(0, 6, N_NODES), test_rows=np.arange(100, 140, dtype=np.int64),
                      test_dst=_rng.integers(0, N_NODES, 40), test_y=np.arange(40) % 2)


def g_off(seed: int, t: int) -> float:
    return (1 + t % 3) / 64 + (seed - 5) / 256


def write_run(root, ds, seed, st, prereg_sha, factor):
    rng = np.random.default_rng([seed, st == "on", len(ds)])
    base = 0.25 + rng.integers(0, 64, len(ACTIVE)) / 1024
    targets = DRAWN + STRATUM
    population = base + rng.integers(-4, 5, (len(targets), len(ACTIVE))) / 1024
    recs = []
    for i, t in enumerate(targets):
        population[i, t] = base[t] + (factor if st == "on" else 1.0) * g_off(seed, t)
        r, r0 = float(population[i, t]), float(base[t])
        recs.append(dict(target=t, n_neg=1, n_pos=N_POS[t], r0=r0, r=r, gain=r - r0,
                         pct_before=M.platform_percentile(r0, base), pct_after=M.platform_percentile(r, population[i]),
                         n_after=N_POS[t], edgeless=N_POS[t] == 0, group="drawn" if t in DRAWN else "ge4"))
    tid = T.task_id(ds, seed, st)
    dump = os.path.join(root, "results", "dumps", "e9a", f"{tid}.npz")
    os.makedirs(os.path.dirname(dump), exist_ok=True)
    np.savez(dump, active_ids=ACTIVE, base_proxy=base.astype(np.float32), n_in=REF.n_in, test_rows=REF.test_rows,
             test_dst=REF.test_dst, test_y=REF.test_y, test_p=rng.random(40).astype(np.float32),
             target_ids=np.array(targets), target_n=np.array([N_POS[t] for t in targets]),
             population=population.astype(np.float32))
    res = dict(experiment="E9a", ds=ds, seed=seed, self_term=st == "on", epochs=150,
               config=dict(ds=ds, seed=seed, self_term=st == "on", epochs=150, targets_drawn=DRAWN,
                           targets_stratum=STRATUM, strata_sizes=SIZES, smoke=None),
               preregistration=dict(path=PS.PREREG, sha256=prereg_sha),
               base=dict(test_metrics=dict(t1=dict(mcc=0.25 + seed / 100 + (st == "on") / 50))),
               targets=recs, dump=dict(path="/elsewhere/" + os.path.basename(dump), sha256=sha256_file(dump)),
               provenance=dict(git_commit="c0ffee", git_dirty=False, git_untracked=0))
    path = os.path.join(root, "results", "laundering", "e9a", f"{tid}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f)
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Every registered run under tmp_path; returns (root, protocol path)."""
    monkeypatch.setattr(E, "reference", lambda ds: REF)
    prereg = tmp_path / "preregistration.md"
    prereg.write_text("protocol\n")
    for ds in PS.NETWORKS:
        for seed in PS.SEEDS_5:
            for st in T.SELF_TERMS:
                write_run(str(tmp_path), ds, seed, st, sha256_file(str(prereg)), FACTOR[ds])
    return str(tmp_path), str(prereg)


def run_path(root, ds, seed, st):
    return os.path.join(root, "results", "laundering", "e9a", f"{T.task_id(ds, seed, st)}.json")


def edit_run(root, ds, seed, st, change):
    path = run_path(root, ds, seed, st)
    with open(path) as f:
        res = json.load(f)
    change(res)
    with open(path, "w") as f:
        json.dump(res, f)


# ------------------------------------------------------------------ H8 endpoints
def test_crossed_replicates_reproduce_the_crossed_bootstrap():
    cells = {s: {t: [g_off(s, t) + 0.01 * ((s * t) % 5)] for t in DRAWN} for s in PS.SEEDS_5}
    point, reps, _ = B.crossed_replicates(cells, n_boot=300, seed=[1, 2])
    assert (point, *B.percentile_interval(reps)) == B.crossed_bootstrap(cells, n_boot=300, seed=[1, 2])


def test_h8_endpoints_and_p_values_by_hand(world):
    root, prereg = world
    result = E.analyse(root, prereg, n_boot=N_BOOT)
    assert result["status"] == "ok" and result["integrity"]["n_present"] == 30
    a, b = (result["hypotheses"]["H8"]["networks"]["bitcoinalpha"]["components"][k] for k in "ab")
    per_seed = [np.mean([2 * g_off(s, t) for t in DRAWN]) for s in PS.SEEDS_5]
    assert a["estimate"] == pytest.approx(np.mean(per_seed), abs=1e-15)
    assert list(a["by_seed"].values()) == pytest.approx(per_seed, abs=1e-15)
    assert a["p_seed_t"] == pytest.approx(ttest_1samp(per_seed, 0.0, alternative="greater").pvalue)
    assert a["p_bootstrap"] == 1 / (N_BOOT + 1) and a["p_value"] == max(a["p_seed_t"], a["p_bootstrap"])
    assert a["ci95"][0] <= min(a["ci95_bootstrap"][0], a["ci95_seed_t"][0]) and a["decision"] == "met"
    assert (b["estimate"], b["ci95"], b["p_seed_t"], b["p_value"]) == (2.0, [2.0, 2.0], 0.0, 1 / (N_BOOT + 1))
    assert b["reading"] == "met" and b["n_pairs"] == 5 * len(DRAWN) and b["n_pairs_dropped"] == 0
    nets = result["hypotheses"]["H8"]["networks"]
    assert nets["bitcoinotc"]["components"]["b"]["estimate"] == pytest.approx(0.75)
    assert nets["bitcoinotc"]["components"]["b"]["reading"] == "met"
    wiki = nets["wiki-rfa"]["components"]["b"]
    assert wiki["reading"] == "partial" and not wiki["contradicted"] and wiki["p_value"] == 1.0
    h8 = result["hypotheses"]["H8"]
    assert h8["evaluable"] and not h8["interval_rule_met"] and h8["networks_missing"] == []
    assert h8["p_value"] == max(n["p_value"] for n in nets.values()) == 1.0
    assert nets["bitcoinalpha"]["p_value"] == max(a["p_value"], b["p_value"])


def test_ratio_p_value_is_the_linearized_seed_t_against_the_threshold():
    num, den = [0.9, 1.1, 1.0, 1.3], [1.0, 1.2, 0.9, 1.1]
    ratio = np.mean(num) / np.mean(den)
    resid = (np.array(num) - ratio * np.array(den)) / np.mean(den)
    t = (ratio - 0.5) / (resid.std(ddof=1) / 2)
    from scipy.stats import t as student_t
    assert E.p_ratio_seed_t(num, den, 0.5) == pytest.approx(student_t.sf(t, 3))


def test_negative_on_gains_are_contradicted(world, monkeypatch):
    root, prereg = world
    for seed in PS.SEEDS_5:
        write_run(root, "bitcoinalpha", seed, "on", sha256_file(prereg), -1.0)
    comps = E.analyse(root, prereg, n_boot=N_BOOT)["hypotheses"]["H8"]["networks"]["bitcoinalpha"]["components"]
    assert comps["a"]["decision"] == "contradicted" and comps["a"]["contradicted"]
    assert comps["b"]["estimate"] == -1.0 and comps["b"]["reading"] == "partial" and comps["b"]["contradicted"]


def test_missing_runs_are_listed_never_imputed_and_three_missing_seeds_are_not_evaluable(world):
    root, prereg = world
    for seed in (7, 8, 9):
        os.remove(run_path(root, "bitcoinalpha", seed, "on"))
    result = E.analyse(root, prereg, n_boot=N_BOOT)
    assert {m["run"] for m in result["integrity"]["missing_runs"]} == {
        T.task_id("bitcoinalpha", s, "on") for s in (7, 8, 9)}
    assert result["status"] == "ok" and not result["integrity"]["complete"]
    net = result["hypotheses"]["H8"]["networks"]["bitcoinalpha"]
    assert net["components"]["a"]["seeds"] == [5, 6] and not net["components"]["a"]["evaluable"]
    assert net["components"]["b"]["n_pairs_dropped"] == 3 * len(DRAWN)
    assert net["status"] == E.NOT_EVALUABLE and result["hypotheses"]["H8"]["p_value"] is None


# ------------------------------------------------------------------ secondary and E8
def test_secondary_outcomes(world):
    root, prereg = world
    sec = E.analyse(root, prereg, n_boot=N_BOOT)["secondary"]["bitcoinalpha"]
    drawn_off = [[g_off(s, t) for t in DRAWN] for s in PS.SEEDS_5]
    assert sec["gain_drawn_off"]["estimate"] == pytest.approx(np.mean(drawn_off), abs=1e-15)
    strata = [M.stratum_of(N_POS[t]) for t in DRAWN]
    want = [M.post_stratified_mean(g, strata, SIZES) for g in drawn_off]
    assert list(sec["gain_drawn_post_stratified_off"]["by_seed"].values()) == pytest.approx(want)
    assert sec["edgeless_drawn"] == dict(n=6, n_edgeless=4, share=4 / 6, source="graph census")
    assert sec["edgeless_ge4"]["n_edgeless"] == 1
    assert sec["base_t1_mcc_on"]["estimate"] == pytest.approx(0.25 + 0.07 + 0.02)


def test_e8_wiring(world, monkeypatch):
    root, prereg = world
    calls, real = [], CS.evaluate

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)
    monkeypatch.setattr(CS, "evaluate", spy)
    result = E.analyse(root, prereg, n_boot=N_BOOT)
    assert len(calls) == 3 * 5                                    # the self-term-off runs only
    for args, kwargs in calls:
        active, target_ids, target_proxy, population = args[0], args[3], args[4], kwargs["target_population"]
        assert args[9] == CS.MS and target_ids.tolist() == DRAWN + STRATUM
        pos = np.searchsorted(active, target_ids)
        assert np.array_equal(target_proxy, population[np.arange(len(pos)), pos])
    e8 = result["e8"]["bitcoinalpha"]
    assert sorted(e8["by_seed"]) == list(PS.SEEDS_5)
    for seed, rec in e8["by_seed"].items():
        assert rec["target_ids"] == DRAWN + STRATUM and rec["groups"] == ["drawn"] * 6 + ["ge4"] * 3
        assert rec["report"]["laundering_population"] == "per-target post-U1"
        with open(run_path(root, "bitcoinalpha", seed, "off")) as f:
            stored = json.load(f)
        assert rec["report"]["uncalibrated"]["laundering_pct"] == [r["pct_after"] for r in stored["targets"]]
    assert {f"laundering_pct_drawn_m{m}" for m in CS.MS} <= set(e8["summary"])
    assert e8["summary"]["newcomer_change_m1"]["n_seeds"] == 5


# ------------------------------------------------------------------ integrity
def _set(path, value):
    def change(res):
        *head, last = path
        node = res
        for key in head:
            node = node[key]
        node[last] = value
    return change


@pytest.mark.parametrize("change, needle", [
    (_set(("preregistration", "sha256"), "0" * 64), "protocol sha256"),
    (_set(("provenance", "git_dirty"), True), "git_dirty"),
    (_set(("provenance", "git_untracked"), 1), "git_untracked"),
    (_set(("epochs",), 2), "epochs"),
    (_set(("config", "smoke"), 3), "smoke"),
    (_set(("config", "targets_drawn"), DRAWN[::-1]), "drawn targets"),
    (_set(("config", "targets_stratum"), STRATUM[:2]), "stratum"),
    (_set(("targets", 1, "gain"), 0.123), "gain of target 5"),
    (_set(("targets", 0, "pct_after"), 0.5), "pct_after of target 3"),
    (_set(("provenance", "git_commit"), "beef"), "commits"),
    (_set(("dump", "sha256"), "1" * 64), "dump sha256"),
])
def test_each_integrity_failure_is_reported(world, change, needle):
    root, prereg = world
    edit_run(root, "bitcoinalpha", 7, "off", change)
    result = E.analyse(root, prereg, n_boot=N_BOOT)
    assert result["status"] == "integrity_failure" and not result["integrity"]["ok"]
    assert any(needle in f["detail"] for f in result["integrity"]["failures"]), result["integrity"]["failures"]


def test_a_changed_dump_and_a_missing_protocol_are_integrity_failures(world):
    root, prereg = world
    dump = os.path.join(root, "results", "dumps", "e9a", f"{T.task_id('wiki-rfa', 5, 'on')}.npz")
    with open(dump, "ab") as f:
        f.write(b"x")
    details = [f["detail"] for f in E.analyse(root, prereg, n_boot=N_BOOT)["integrity"]["failures"]]
    assert details == ["dump sha256 differs from the recorded one"]
    details = [f["detail"] for f in E.analyse(root, prereg + ".gone", n_boot=N_BOOT)["integrity"]["failures"]]
    assert details[0].startswith("protocol file") and len(details) == 1 + 30 + 1    # every run's sha, the dump


# ------------------------------------------------------------------ reruns and CLI
def test_reruns_are_bit_identical(world):
    root, prereg = world
    first = json.dumps(E.analyse(root, prereg, n_boot=N_BOOT), sort_keys=True)
    assert json.dumps(E.analyse(root, prereg, n_boot=N_BOOT), sort_keys=True) == first


def _without_provenance(path):
    with open(path) as f:
        data = json.load(f)
    data.pop("provenance")
    return data


def test_cli_writes_json_markdown_and_manifest_reruns_identically_and_exits_2_on_failure(world, tmp_path):
    root, prereg = world
    out, md = str(tmp_path / "a" / "e9a_e8.json"), str(tmp_path / "a" / "e9a_e8.md")
    argv = ["--results-root", root, "--prereg", prereg, "--n-boot", str(N_BOOT), "--out", out, "--md", md]
    assert E.main(argv) == 0
    first, text = _without_provenance(out), open(md).read()
    assert "H8 hypothesis p-value" in text and first["hypotheses"]["H8"]["p_value"] == 1.0
    diff, n = DM.check_manifest(os.path.join(root, "results", "dumps", "e9a"))
    assert n == 30 and not any(diff.values())
    with pytest.raises(FileExistsError):
        E.main(argv)
    assert E.main(argv + ["--force"]) == 0
    assert _without_provenance(out) == first and open(md).read() == text
    edit_run(root, "bitcoinotc", 6, "on", _set(("provenance", "git_dirty"), True))
    assert E.main(argv + ["--force"]) == E.EXIT_INTEGRITY
