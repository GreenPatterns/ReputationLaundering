"""E6 linkage (Task 4.6): keeper.analysis.linkage_attackers and keeper.analysis.e6_linkage on small synthetic dumps.

Hand-built pools check the within-pool AUC (ties, single-member pools, the AUC-1 sensitivity) and the tie-split top-1;
keeper.cppr.linkage.linkage_attack itself, fed the same scores, checks the hand-crafted attacker; a recording learner
checks that leave one seed out never trains on the held-out seed; the corrected t is checked by hand. A synthetic
results root (tests/cppr_confirm_fixtures plus verdict dumps and their manifest) checks the run set: determinism,
a tampered dump, a missing manifest and a partial run set.
"""
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import t as student_t

from keeper.analysis import cppr_runs as R
from keeper.analysis import dump_manifest as DM
from keeper.analysis import e6_linkage as E6
from keeper.analysis import linkage_attackers as LA
from keeper.cluster import protocol_settings as PS
from keeper.cppr import dumps as DP
from keeper.cppr import linkage as L
from tests import cppr_confirm_fixtures as F

DS = "bitcoinalpha"


def table(pools, singles=0) -> LA.Table:
    """pools: list of (scores per member, true member indices); every feature column carries the scores."""
    verdict = np.concatenate([np.full(len(s), v) for v, (s, _) in enumerate(pools)])
    label = np.concatenate([np.isin(np.arange(len(s)), list(t)) for s, t in pools])
    X = np.repeat(np.concatenate([np.asarray(s, float) for s, _ in pools])[:, None], len(LA.FEATURES), axis=1)
    return LA.Table(verdict, label, X, singles, singles)


# ------------------------------------------------------------------------ the G3 metric
def test_within_pool_auc_counts_ties_one_half_and_other_true_raters_as_members():
    t = table([([3, 1, 1, 2], [1]), ([2, 2, 1], [0, 1])])
    m = LA.item_metrics(t, t.X[:, 0])
    # pool 0: the true score 1 beats nobody, ties one of three others; pool 1: each true 2 ties the other, beats 1
    assert m["auc"].tolist() == pytest.approx([0.5 / 3, 0.75, 0.75])
    assert m["top1"].tolist() == pytest.approx([0.0, 0.5, 0.5])          # two members tie at pool 1's maximum
    assert m["chance"].tolist() == pytest.approx([0.25, 1 / 3, 1 / 3])
    assert LA.g3(t, t.X[:, 0]) == pytest.approx((0.5 / 3 + 1.5) / 3)


def dump_row(arm, method, verdict, pool_id, members, true, feats, tier=1):
    return [[arm, method, verdict, pool_id, pool_id // 4, tier, len(members), 77, 1, u, tier, int(u in true)]
            + (feats[i] if feats is not None else [None] * 6) for i, u in enumerate(members)]


def test_single_member_pools_are_counted_not_scored_and_other_arms_are_ignored():
    rows = (dump_row("direct", "cppr", 0, 4, [10, 11, 12], {11}, [[0.1 * i] * 6 for i in range(3)])
            + dump_row("direct", "cppr", 1, 8, [13], {13}, None)
            + dump_row("whitewash", "cppr", 0, 4, [10, 11], {10}, [[0.0] * 6] * 2)
            + dump_row("direct", "vpe", 0, 0, [10, 11], {10}, [[0.0] * 6] * 2))
    t, failures = LA.table_from_rows(DP.VERDICT_COLUMNS, rows, "direct", "cppr")
    assert failures == [] and t.n_single_items == 1 and t.n_single_verdicts == 1
    assert t.verdict.tolist() == [0, 0, 0] and t.label.tolist() == [False, True, False]
    bad = dump_row("direct", "cppr", 2, 4, [10, 11], set(), [[0.0] * 6] * 2)          # no true rater
    assert LA.table_from_rows(DP.VERDICT_COLUMNS, rows + bad, "direct", "cppr")[1]


def test_handcrafted_attacker_equals_the_drivers_linkage_attack(monkeypatch):
    rng = np.random.default_rng(3)
    pools = []
    for _ in range(40):
        n = int(rng.integers(2, 7))
        pools.append(([round(float(x), 1) for x in rng.random(n)], {int(rng.integers(n))}))
    t = table(pools)
    t.X[:, 1] = np.round(rng.random(len(t.verdict)), 1)             # timing and model_prob differ from structure
    t.X[:, 2] = rng.random(len(t.verdict))
    records, res = [], SimpleNamespace(pooled=[], pools={})
    for v, (s, true) in enumerate(pools):
        rows = np.flatnonzero(t.verdict == v)
        res.pools[v] = list(range(len(s)))
        res.pooled.append(dict(window=v, raters=sorted(true)))
        records.append((v, res.pooled[-1], {k: {u: float(t.X[r, LA.FEATURES.index(f)]) for u, r in enumerate(rows)}
                                            for k, f in (("structure", "structure"), ("timing", "timing"),
                                                         ("model", "model_prob"))}))
    monkeypatch.setattr(L, "verdict_scores", lambda *a: records)
    driver = L.linkage_attack(res, None, 0)
    score, orientation = LA.handcrafted_score(t)
    assert LA.g3(t, score) == pytest.approx(driver["auc"]["combined"], abs=1e-12)
    assert [list(o.values()) for o in orientation] == [list(o.values()) for o in driver["orientation_per_half"]]


# ------------------------------------------------------------------------ leave one seed out
def seed_tables(n_seeds=4, n_pools=60, flip=None) -> dict:
    """Pools of 3-6 members; the true rater has a lower timing and higher model_prob (opposite in seed ``flip``)."""
    out = {}
    for s in range(5, 5 + n_seeds):
        rng = np.random.default_rng(s)
        rows, verdict, label = [], [], []
        for v in range(n_pools):
            n = int(rng.integers(3, 7))
            true = int(rng.integers(n))
            for u in range(n):
                shift = (-1 if s == flip else 1) * (u == true)
                rows.append([rng.random(), rng.random() - 0.3 * shift, rng.random() + 0.3 * shift,
                             float(rng.integers(1, 5)), float(rng.integers(1, 9)), float(rng.integers(0, 3)), 1.0,
                             float(n), float(s)])                   # pool_tier tags the seed
                verdict.append(v)
                label.append(u == true)
        out[s] = LA.Table(np.asarray(verdict), np.asarray(label), np.asarray(rows), 0, 0)
    return out


def test_single_signal_sign_comes_from_the_other_seeds_only():
    tables = seed_tables(flip=5)
    got = LA.loso_signal_scores(tables, "timing")
    assert got[5][1] == -1.0                       # the other seeds say: a lower timing marks the true rater
    assert LA.g3(tables[5], got[5][0]) < 0.5       # which is wrong on seed 5, where it is higher
    assert all(got[s][1] == -1.0 for s in (6, 7, 8))
    assert LA.loso_signal_scores({5: tables[5]}, "timing")[5] is None


def test_learned_attacker_never_sees_its_held_out_seed(monkeypatch):
    seen = []

    class Recorder:
        def fit(self, X, y):
            seen.append(set(X[:, LA.FEATURES.index("pool_tier")].tolist()))
            return self

        def predict_proba(self, X):
            return np.c_[np.zeros(len(X)), X[:, LA.FEATURES.index("model_prob")]]

    monkeypatch.setattr(LA, "_model", Recorder)
    tables = seed_tables()
    out = LA.loso_learned(tables, importance=False)
    assert [s for s in seen] == [set(map(float, tables)) - {float(s)} for s in tables]
    assert all(out[s]["train_seeds"] == sorted(set(tables) - {s}) for s in tables)


def test_learned_attacker_finds_the_signal_and_reports_the_timing_direction():
    tables = seed_tables(n_pools=120)
    learned = LA.loso_learned(tables)
    for s, t in tables.items():
        assert LA.g3(t, learned[s]["score"]) > 0.6
    block = E6.learned_block({s: {k: v for k, v in d.items() if k != "score"} for s, d in learned.items()})
    assert block["timing_direction"] == "lower"
    assert block["importance_ranking"][0] in ("timing", "model_prob")


def test_max_over_attackers_is_taken_per_seed():
    fam = E6.Family("cppr_btp", "e3", PS.CAPPED, "cppr", "linkage_cp_vpe", (5, 6, 7, 8))
    tables = seed_tables(flip=6)
    runs = {s: SimpleNamespace(table=t, recorded=None) for s, t in tables.items()}
    out = E6.analyse_family(fam, runs, n_boot=200, ds=DS)
    mx = out["max_over_attackers"]
    for s in tables:
        per = {a: b["by_seed"][str(s)] for a, b in out["attackers"].items()}
        assert mx["by_seed"][str(s)] == max(per.values())
        assert per[mx["argmax_by_seed"][str(s)]] == max(per.values())
    assert mx["estimate"] == pytest.approx(np.mean(list(mx["by_seed"].values())))
    assert mx["estimate"] >= max(b["estimate"] for b in out["attackers"].values())
    assert mx["sensitivity_single_member_pools_as_auc_1"]["estimate"] == pytest.approx(mx["estimate"])   # no singles


def test_single_member_sensitivity_scores_them_as_auc_1():
    fam = E6.Family("cppr_btp", "e3", PS.CAPPED, "cppr", "linkage_cp_vpe", (5, 6, 7, 8))
    tables = {s: LA.Table(t.verdict, t.label, t.X, 20, 20) for s, t in seed_tables().items()}
    out = E6.analyse_family(fam, {s: SimpleNamespace(table=t, recorded=None) for s, t in tables.items()}, 200, DS)
    mx = out["max_over_attackers"]
    for s, t in tables.items():
        n = int(t.label.sum())
        want = max((b["by_seed"][str(s)] * n + 20) / (n + 20) for b in out["attackers"].values())
        assert mx["sensitivity_single_member_pools_as_auc_1"]["by_seed"][str(s)] == pytest.approx(want)
    assert out["single_member_item_share"] == pytest.approx(80 / (80 + sum(int(t.label.sum()) for t in tables.values())))


def test_corrected_t_by_hand():
    values = [0.55, 0.58, 0.62, 0.57]
    got = LA.corrected_t(values, 0.60)
    J, mean = 4, 0.58
    sd = math.sqrt(((0.55 - mean) ** 2 + (0.58 - mean) ** 2 + (0.62 - mean) ** 2 + (0.57 - mean) ** 2) / 3)
    factor = 1 + J * (1 / 3)                        # n_test / n_train = 1 / (J - 1) under leave one seed out
    se = math.sqrt(factor * sd ** 2 / J)
    assert got["factor"] == pytest.approx(7 / 3) and got["se"] == pytest.approx(se)
    assert got["t"] == pytest.approx((mean - 0.60) / se)
    assert got["p_below"] == pytest.approx(student_t.cdf((mean - 0.60) / se, 3))
    half = student_t.ppf(0.975, 3) * se
    assert got["ci95"] == pytest.approx([mean - half, mean + half])
    assert LA.corrected_t([0.5], 0.6)["p_below"] is None


# ------------------------------------------------------------------------ the run set
def verdict_rows(seed: int, tier: str) -> list:
    rng = np.random.default_rng((seed, len(tier)))
    rows = []
    for arm, method in (("direct", "cppr"), ("direct", "vpe"), ("whitewash", "cppr")):
        for v in range(30):
            n = int(rng.integers(2, 6))
            members = sorted(int(x) for x in rng.choice(500, n, replace=False))
            true = {members[int(rng.integers(n))]}
            feats = [[float(rng.random()), float(rng.random() - 0.2 * (u in true)), float(rng.random()),
                      float(rng.integers(1, 4)), float(rng.integers(1, 6)), float(rng.integers(0, 3))] for u in members]
            rows += dump_row(arm, method, v, 4 * v + 1, members, true, feats)
        rows += dump_row(arm, method, 30, 200, [600], {600}, None)
    return rows


def build_root(root, skip=(), seeds=PS.SEEDS_E3, manifest=True) -> str:
    """The fixtures' E3/E4 run set on DS plus verdict dumps of the E3 capped and BFS runs of ``seeds``."""
    folder = os.path.join(str(root), "results", "dumps", "e3")

    def add_verdicts(key, result, _dump):
        if key.family != "e3" or key.tier == PS.UNCAPPED or key.seed not in seeds:
            return
        rows = verdict_rows(key.seed, key.tier)
        path = os.path.join(folder, f"{key.stem}_verdicts.json")
        result["dumps"]["verdicts"] = dict(path=path, n_rows=len(rows))
        result["direct_laundering"] = dict(linkage_cp_vpe=dict(auc=dict(combined=0.5), n_verdicts_scored=30))
        F.write_json(path, DP.dump_table(DP.VERDICT_COLUMNS, rows, dict(result=f"{key.stem}.json",
                                                                         config=result["config"])))
    prereg = F.build_results(root, networks=(DS,), mutate=add_verdicts, skip=skip)
    if manifest:
        DM.write_manifest(folder)
    return prereg


def run_main(root, prereg, *extra) -> tuple:
    out, md = os.path.join(str(root), "out", "e6.json"), os.path.join(str(root), "out", "e6.md")
    code = E6.main(["--results-root", str(root), "--prereg", prereg, "--networks", DS, "--n-boot", "200",
                    "--out", out, "--md", md, "--force", *extra])
    with open(out) as f:
        return code, json.load(f)


def test_full_run_set_is_deterministic_and_reads_every_dump(tmp_path):
    prereg = build_root(tmp_path)
    first, _ = E6.analyse(str(tmp_path), prereg, (DS,), n_boot=200)
    second, _ = E6.analyse(str(tmp_path), prereg, (DS,), n_boot=200)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["integrity"]["failures"] == [] and first["status"] == "ok"
    btp = first["networks"][DS]["cppr_btp"]
    assert btp["seeds"] == list(PS.SEEDS_E3) and btp["evaluable"]
    assert btp["n_single_member_items"]["5"] == 1 and btp["n_items"]["5"] == 30
    assert first["h7"]["networks"][DS]["decision"] in ("supported", "not supported", "contradicted")
    assert {m["family"] for m in first["integrity"]["missing_runs"]} == {"pseudonym"}     # no E5 run here
    assert btp["handcrafted_check"]["by_seed"]["5"]["recorded_auc"] == 0.5


def test_a_tampered_verdict_dump_fails_the_manifest_check_and_exits_2(tmp_path):
    prereg = build_root(tmp_path)
    path = os.path.join(str(tmp_path), "results", "dumps", "e3", "e3_btpc_bitcoinalpha_s7_verdicts.json")
    with open(path) as f:
        dump = json.load(f)
    dump["rows"][0][12] = 0.123456                     # one feature changed after the manifest was written
    F.write_json(path, dump)
    code, out = run_main(tmp_path, prereg)
    assert code == E6.EXIT_INTEGRITY and out["status"] == "integrity_failure"
    assert [(f["run"], f["field"]) for f in out["integrity"]["failures"]] == [("e3_btpc_bitcoinalpha_s7",
                                                                               "verdicts.sha256")]
    assert 7 not in out["networks"][DS]["cppr_btp"]["seeds"]
    assert os.path.exists(os.path.join(str(tmp_path), "out", "e6.md"))


def test_a_missing_dump_manifest_is_an_integrity_failure(tmp_path):
    prereg = build_root(tmp_path, seeds=(5, 6), manifest=False)
    code, out = run_main(tmp_path, prereg)
    assert code == E6.EXIT_INTEGRITY
    assert {f["field"] for f in out["integrity"]["failures"]} == {"verdicts.manifest"}


def test_a_partial_run_set_lists_the_missing_runs_and_is_not_evaluable(tmp_path):
    prereg = build_root(tmp_path, skip=("e3_btpc_bitcoinalpha_s13", "e3_btpc_bitcoinalpha_s14"),
                        seeds=range(5, 13))
    code, out = run_main(tmp_path, prereg)
    assert code == E6.EXIT_OK
    missing = {(m["run"], m["reason"]) for m in out["integrity"]["missing_runs"] if m["family"] != "pseudonym"}
    assert missing == {("e3_btpc_bitcoinalpha_s13", "result missing"), ("e3_btpc_bitcoinalpha_s14", "result missing"),
                       ("e3_bfs_bitcoinalpha_s13", "verdict dump missing"),
                       ("e3_bfs_bitcoinalpha_s14", "verdict dump missing")}
    assert out["networks"][DS]["cppr_btp"]["max_over_attackers"]["n_seeds"] == 8     # 8 of 10: still evaluable
    assert out["networks"][DS]["cppr_btp"]["evaluable"]
    prereg = build_root(tmp_path / "fewer", seeds=range(5, 12))
    _, out = run_main(tmp_path / "fewer", prereg)
    assert out["networks"][DS]["cppr_btp"]["max_over_attackers"]["decision"] == E6.NOT_EVALUABLE
    assert out["h7"]["evaluable"] is False and out["h7"]["p_value"] is None


def test_an_e5_run_is_read_with_its_own_arms_and_methods(tmp_path):
    prereg = build_root(tmp_path, seeds=(), manifest=False)
    _, sha = F.write_prereg(tmp_path)
    folder = os.path.join(str(tmp_path), "results", "dumps", "e5")
    for seed in PS.SEEDS_5:
        stem = E6.e5_stem(DS, seed)
        result, _ = F.build_run(tmp_path, R.RunKey("e3", PS.CAPPED, DS, seed), F.Model(), sha)
        result["config"].update(arms=PS.E5_ARMS_G0.split(","), methods=PS.E5_METHODS.split(","))
        rows = [r[:1] + ["singleton"] + r[2:] for r in verdict_rows(seed, "e5") if r[:2] == ["direct", "cppr"]]
        path = os.path.join(folder, f"{stem}_verdicts.json")
        result["dumps"]["verdicts"] = dict(path=path, n_rows=len(rows))
        F.write_json(os.path.join(str(tmp_path), "results", "cppr", "e5", f"{stem}.json"), result)
        F.write_json(path, DP.dump_table(DP.VERDICT_COLUMNS, rows, dict(result=f"{stem}.json",
                                                                         config=result["config"])))
    DM.write_manifest(folder)
    code, out = run_main(tmp_path, prereg, "--families", "pseudonym")
    assert code == E6.EXIT_OK, out["integrity"]["failures"]
    assert out["networks"][DS]["pseudonym"]["seeds"] == list(PS.SEEDS_5)
    assert out["networks"][DS]["pseudonym"]["evaluable"]
