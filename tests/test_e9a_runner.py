"""Task 4.8: the E9a runner (keeper.laundering.e9a_runner) and its task list (keeper.cluster.e9a_tasks).

1. The target lists: probe4's 50 drawn targets (its own draw function) and the separate stratum (strict distrust
   majority, at least 4 distrust in-edges), on a hand-built graph and on Bitcoin-Alpha.
2. A run on Bitcoin-Alpha with stub models (tests/cppr_common.StubModel; no training, CPU): the U1 mask covers every
   distrust in-edge and nothing else; the dumped post-U1 populations are aligned with the active ids and reproduce
   each target's r; the dumped test rows are cold_start.held_out_snapshot's, with the base model's P(trust); the
   result records the dump's sha256, the protocol's sha256 and the target lists; an existing result is refused
   before any model is built.
3. The task list: 30 runs whose commands the runner's own parser accepts.
CPU only.
"""
import dataclasses
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from keeper.analysis import cold_start as CS
from keeper.cluster import cppr_tasks as CT
from keeper.cluster import e9a_tasks as T
from keeper.cluster import protocol_settings as PS
from keeper.laundering import core as C
from keeper.laundering import e9a_runner as R
from keeper.laundering import probe4 as P4
from keeper.laundering import signed as S
from keeper.laundering import unlearn as U
from tests.cppr_common import StubModel

ROOT = C.ROOT
PREREG = os.path.join(ROOT, PS.PREREG)
SEED, SMOKE = 5, 3


def _graph(in_edges: dict, n: int = 30) -> C.Graph:
    """A training graph from {target: (n_distrust, n_trust)} in-edges, each from its own rater."""
    edges, signs, rater = [], [], n - 1
    for v, (k_neg, k_pos) in in_edges.items():
        for sign, k in ((-1, k_neg), (1, k_pos)):
            for _ in range(k):
                edges.append((rater, v))
                signs.append(sign)
                rater -= 1
    e = np.array(edges, dtype=np.int64)
    zero = np.zeros(len(e), dtype=np.int64)
    return C.Graph(e, S.one_hot_from_sign(np.array(signs)), zero, zero, n)


# ------------------------------------------------------------------ 1. target lists
def test_target_lists_by_hand():
    Tr = _graph({1: (1, 0), 2: (3, 2), 3: (2, 2), 4: (4, 3), 5: (5, 0), 6: (4, 4), 7: (4, 5), 8: (6, 1)}, n=60)
    drawn, stratum, sizes = R.target_lists(Tr)
    assert drawn.tolist() == [1, 2]                  # strict majority with at most 3 distrust in-edges (3: a tie)
    assert sizes == dict(pos=1, nopos=1)
    assert stratum.tolist() == [4, 5, 8]             # strict majority, >= 4 distrust (6: a tie; 7: a trust majority)


@pytest.fixture(scope="module")
def alpha():
    D = C.load("bitcoinalpha")
    G = C.full_graph(D)
    return SimpleNamespace(D=D, G=G, Tr=C.train_split(G, D.tt))


def test_target_lists_on_bitcoin_alpha_are_probe4s_draw_and_the_stratum(alpha):
    drawn, stratum, sizes = R.target_lists(alpha.Tr)
    cohort, neg, pos = C.pick_targets(alpha.Tr, max_neg=3, strict=True)
    want, want_sizes = P4.stratified_draw(cohort, {int(v): len(pos.get(int(v), [])) for v in cohort}, 50)
    assert drawn.tolist() == want.tolist() and sizes == want_sizes and len(drawn) == 50
    y = S.class_ids(alpha.Tr.labels)
    n_neg = np.bincount(alpha.Tr.edges[y == 1, 1], minlength=alpha.D.n)
    n_pos = np.bincount(alpha.Tr.edges[y == 0, 1], minlength=alpha.D.n)
    assert stratum.tolist() == np.flatnonzero((n_neg >= 4) & (n_neg > n_pos)).tolist()
    assert len(stratum) == 39 and not set(stratum.tolist()) & set(drawn.tolist())


# ------------------------------------------------------------------ 2. a stub run
@pytest.fixture(scope="module")
def stub_run(tmp_path_factory):
    """One self-term-on run on Bitcoin-Alpha (first 3 targets of each group) with stub models."""
    tmp = tmp_path_factory.mktemp("e9a")
    built = []
    real_unlink = U.unlink

    def make_model(D, Tr, seed, tt, legacy_head=False, strategy=None, self_term=False):
        built.append(dict(seed=seed, self_term=self_term, n_train=len(Tr.edges)))
        m = StubModel(Tr, np.zeros(Tr.n, dtype=np.int64), seed, "count" if self_term else "ungated")
        m.tag = "count" if self_term else "ungated"
        return m

    def unlink(m0, G, Tr, mask, tt, recompute_features=True):
        assert recompute_features
        u = real_unlink(m0, G, Tr, mask, tt, recompute_features=False)       # the real graph cut
        return dataclasses.replace(u, model=StubModel(u.Tr, np.zeros(u.Tr.n, dtype=np.int64), SEED, m0.tag))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "make_model", make_model)
        mp.setattr(C, "train", lambda m, Tr, tt, epochs=150, lr=0.005, wd=1e-5: m)
        mp.setattr(C, "embed", lambda m, Tr, tt: m.z)
        mp.setattr(C, "require_gpu", lambda: None)
        mp.setattr(C, "set_deterministic", lambda on=True: None)
        mp.setattr(C, "apply_memory_profile", lambda D: {"stub": True})
        mp.setattr(U, "unlink", unlink)
        args = SimpleNamespace(ds="bitcoinalpha", seed=SEED, self_term="on", epochs=150, prereg=PREREG,
                               out=str(tmp / "e9a_bitcoinalpha_s5_on.json"), dump_dir=str(tmp / "dumps"),
                               smoke=SMOKE, force=False)
        out = R.run(args)
        with pytest.raises(FileExistsError):                  # refused before any model is built
            R.run(args)
    with open(args.out) as f:
        stored = json.load(f)
    with np.load(stored["dump"]["path"]) as npz:
        z = {k: npz[k] for k in npz.files}
    return SimpleNamespace(out=out, stored=stored, z=z, built=built, args=args, unlink=unlink)


def test_the_model_is_built_once_with_the_flag_and_the_result_is_refused_on_a_rerun(stub_run):
    assert stub_run.built == [dict(seed=SEED, self_term=True, n_train=stub_run.built[0]["n_train"])]
    assert stub_run.stored["self_term"] is True and stub_run.stored["config"]["self_term"] is True


def test_the_u1_mask_covers_every_distrust_in_edge_and_nothing_else(alpha, stub_run):
    y = S.class_ids(alpha.Tr.labels)
    _, neg, pos = C.pick_targets(alpha.Tr, max_neg=3, strict=True)
    for rec in stub_run.stored["targets"]:
        v = rec["target"]
        assert sorted(neg[v]) == np.flatnonzero((alpha.Tr.edges[:, 1] == v) & (y == 1)).tolist()
        assert rec["n_neg"] == len(neg[v]) and rec["n_after"] == rec["n_pos"] == len(pos.get(v, []))
        assert rec["edgeless"] == (rec["n_pos"] == 0)


def test_u1_refuses_a_mask_that_leaves_a_distrust_in_edge(alpha, stub_run, monkeypatch):
    monkeypatch.setattr(U, "unlink", stub_run.unlink)
    first = int(np.flatnonzero(S.class_ids(alpha.Tr.labels) == 1)[0])
    v = int(alpha.Tr.edges[first, 1])
    cx = SimpleNamespace(a=None, D=alpha.D, G=alpha.G, Tr=alpha.Tr, tt=alpha.D.tt, neg={v: [first]}, pos={},
                         forget_mask=lambda rows: np.zeros(len(alpha.Tr.edges), bool))    # a mask missing the edge
    m0 = StubModel(alpha.Tr, np.zeros(alpha.Tr.n, dtype=np.int64), SEED, "ungated")
    m0.tag = "ungated"
    with pytest.raises(RuntimeError, match="distrust in-edge"):
        R.u1(cx, m0, v, None, 0)


def test_the_target_lists_and_hashes_are_recorded(stub_run):
    st, z = stub_run.stored, stub_run.z
    drawn, stratum = st["config"]["targets_drawn"], st["config"]["targets_stratum"]
    assert len(drawn) == len(stratum) == SMOKE and st["config"]["smoke"] == SMOKE
    assert [r["target"] for r in st["targets"]] == drawn + stratum == z["target_ids"].tolist()
    assert [r["group"] for r in st["targets"]] == ["drawn"] * SMOKE + ["ge4"] * SMOKE
    with open(st["dump"]["path"], "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == st["dump"]["sha256"]
    with open(PREREG, "rb") as f:
        assert st["preregistration"]["sha256"] == hashlib.sha256(f.read()).hexdigest()
    assert {"git_commit", "git_dirty", "git_untracked", "code_sha256"} <= set(st["provenance"])


def test_population_rows_are_aligned_with_the_active_ids(alpha, stub_run):
    st, z = stub_run.stored, stub_run.z
    active = np.unique(alpha.Tr.edges.ravel())
    assert np.array_equal(z["active_ids"], active) and z["population"].shape == (2 * SMOKE, len(active))
    panel = C.Panel(alpha.Tr, n_probe=256, seed=1234)
    base = StubModel(alpha.Tr, np.zeros(alpha.Tr.n, dtype=np.int64), SEED, "count")
    base.tag = "count"
    assert np.array_equal(z["base_proxy"], panel.reputation(base, base.z, active))
    for i, rec in enumerate(st["targets"]):
        j = int(np.searchsorted(active, rec["target"]))
        assert z["population"][i, j] == rec["r"] and z["base_proxy"][j] == rec["r0"]
        assert rec["gain"] == rec["r"] - rec["r0"] and z["target_n"][i] == rec["n_after"]
        mask = np.zeros(len(alpha.Tr.edges), bool)
        mask[C.in_edges(alpha.Tr)[0][rec["target"]]] = True
        u = stub_run.unlink(base, alpha.G, alpha.Tr, mask, alpha.D.tt)
        assert np.array_equal(z["population"][i], panel.reputation(u.model, u.model.z, active))
        assert rec["pct_after"] == float((z["population"][i] < rec["r"]).mean())
    assert not np.array_equal(z["population"][0], z["base_proxy"])         # U1 moves the raters too


def test_test_rows_follow_held_out_snapshot_with_the_base_probability(alpha, stub_run):
    z = stub_run.z
    rows, dst, y = CS.held_out_snapshot(alpha.G, alpha.D.tt)
    assert np.array_equal(z["test_rows"], rows) and np.array_equal(z["test_dst"], dst)
    assert np.array_equal(z["test_y"], y) and len(rows) > stub_run.stored["base"]["test_metrics"]["t1"]["n"]
    base = StubModel(alpha.Tr, np.zeros(alpha.Tr.n, dtype=np.int64), SEED, "count")
    e = alpha.G.edges[rows]
    assert np.array_equal(z["test_p"], C.p_trust(base, base.z, e[:, 0], e[:, 1]).numpy())
    assert np.array_equal(z["n_in"], CS.incoming_counts(alpha.Tr.edges[:, 1], alpha.D.n))
    assert z["population"].dtype == np.float32                              # lossless: float32 probabilities


# ------------------------------------------------------------------ 3. task list
def test_task_list_has_the_30_registered_runs_in_order(tmp_path):
    rows = T.e9a_tasks(str(tmp_path))
    assert len(rows) == 30 and len({r.id for r in rows}) == 30
    assert [r.id for r in rows[:2]] == ["e9a_wiki-rfa_s5_off", "e9a_wiki-rfa_s5_on"]
    assert rows[-1].id == "e9a_bitcoinalpha_s9_on"
    parser = R.build_parser()
    for row in rows:
        assert row.done_path == os.path.join(str(tmp_path), "results", "laundering", "e9a", f"{row.id}.json")
        assert row.log_path == os.path.join(str(tmp_path), "results", "laundering", "e9a", "logs", f"{row.id}.log")
        words, k = row.command.split(), len(PS.PYTHON.split())
        assert " ".join(words[:k]) == PS.PYTHON and words[k:k + 2] == ["-m", "keeper.laundering.e9a_runner"]
        a = parser.parse_args(words[k + 2:])
        ds, seed, st = row.id[len("e9a_"):].rsplit("_", 2)
        assert (a.ds, a.seed, a.self_term, a.epochs, a.prereg) == (ds, int(seed[1:]), st, 150, PS.PREREG)
        assert a.out == row.done_path and a.dump_dir == os.path.join(str(tmp_path), "results", "dumps", "e9a")
        assert a.smoke is None and not a.force


def test_task_file_cli_writes_once_and_refuses_a_worktree_root(tmp_path, capsys):
    out = tmp_path / "tasks.tsv"
    assert T.main(["--results-root", str(tmp_path), "--out", str(out)]) == 0
    assert out.read_text() == CT.render(T.e9a_tasks(str(tmp_path)))
    assert T.main(["--results-root", str(tmp_path), "--out", str(out)]) == 1
    (tmp_path / "wt").mkdir()
    (tmp_path / "wt" / ".git").write_text("gitdir: elsewhere\n")
    with pytest.raises(SystemExit):
        T.main(["--results-root", str(tmp_path / "wt"), "--out", str(tmp_path / "x.tsv")])
