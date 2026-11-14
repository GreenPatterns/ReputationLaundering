"""Task 4.3: the Regime-2 baselines of the CPPR driver (keeper.cppr.baselines; protocol Section 2.7, experiment E5).

  B1 each baseline's graph and tiers on a toy: retain with author, per-rating pseudonym, withdrawal counts
  B2 a pseudonym's linkage candidates are the CPPR pool that would have held its rating (same window and tier)
  B3 core.make_model's extra_features hook: off, the code and the model are the old ones; on, the columns are appended
  B4 the count columns survive the protected attach_scem_and_credibility_gate, whose prototypes the original columns
     still get exactly
  B5 --methods with the baselines (stub models): every stored output, training and draw is that of a run without
     them; a baselines-only run builds every stored release and draws the same snap_f, endorsers and r0 (E5's pairing
     with E3 and E4); singleton's dumped candidates are CPPR's pools
  B6 with no baseline selected the driver writes what the driver of tag protocol-2026-11-11 writes (stub runs, default
     and full protocol, and the full protocol with --backbone seq_scem given, Task 4.7), and the protected functions
     are AST-identical to that tag's; the driver has <= 800 lines
CPU only: CUDA_VISIBLE_DEVICES="" <ps python> -m pytest -q tests/test_cppr_baselines.py
"""
import ast
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from keeper.cppr import baselines as B
from keeper.cppr import cp_vpe_gpu as G
from keeper.cppr import dumps as Dm
from keeper.cppr import vpe as V
from keeper.laundering import core as C
from keeper.laundering import signed as S
from tests.cppr_common import ALPHA_EPS

ROOT = Path(__file__).resolve().parents[1]
TAG = "protocol-2026-11-11"                 # the registered protocol's commit: the driver before Task 4.3
PROTECTED = ("CPVPEConfig", "apply_cp_vpe", "attach_scem_and_credibility_gate", "train_gated_model",
             "aggregate_saved_runs")
INF = float("inf")
SEED = 5

# A toy of 8 accounts and 4 snapshots (the last one is the test snapshot): (rater, target, sign, snapshot) per row.
TOY_ROWS = [(0, 1, +1, 0), (2, 1, -1, 0), (3, 4, +1, 1), (0, 4, -1, 1), (5, 1, -1, 2), (6, 5, +1, 2), (5, 6, +1, 2),
            (7, 3, +1, 3)]
TOY_TIERS = np.array([3, 0, 2, 1, 0, 2, 1, 0])
# Rows 1 and 3 are erased by an edge request; account 5 is deleted: its ratings (rows 4, 6) are kept as verdicts, the
# rating about it (row 5) is not. The kept ratings are rows 1, 3, 4 and 6.
TOY_REQUESTS = [V.Request(rows=np.array([1, 3])), V.Request(account=5)]


@pytest.fixture(scope="module")
def toy():
    cfg = json.loads((ROOT / "configs" / "bitcoinalpha.json").read_text())
    edges = np.array([r[:2] for r in TOY_ROWS], dtype=np.int64)
    labels = S.one_hot_from_sign(np.array([r[2] for r in TOY_ROWS]))
    time, snap = np.arange(len(TOY_ROWS), dtype=np.int64) * 10, np.array([r[3] for r in TOY_ROWS], dtype=np.int64)
    D = C.Dataset("bitcoinalpha", edges, labels, time, 8, 4, cfg)
    Gr = C.Graph(edges, labels, time, snap, 8)
    res_cp, _ = G.apply_cp_vpe(D, Gr, D.tt, TOY_REQUESTS, TOY_TIERS, G.CPVPEConfig(1, INF, 10),
                               np.random.default_rng(0))
    return D, Gr, C.train_split(Gr, D.tt), res_cp


@pytest.fixture
def on_cpu(monkeypatch):
    """keeper.laundering.core builds its models on the CPU (as in tests/test_self_term.py)."""
    monkeypatch.setattr(C, "DEV", torch.device("cpu"))
    monkeypatch.setenv(C.ALLOW_CPU_ENV, "1")


def graph_rows(Gr) -> list:
    """(rater, target, sign, snapshot, time, row id) per row."""
    sign = np.where(S.neg_mask(Gr.labels), -1, 1)
    return [tuple(int(x) for x in row) for row in zip(Gr.edges[:, 0], Gr.edges[:, 1], sign, Gr.snap, Gr.time, Gr.rid)]


# ------------------------------------------------------------------ B1, B2. graphs and tiers on the toy
def test_retain_with_author_keeps_the_arms_graph_and_tiers(toy):
    D, Gr, Tr, _ = toy
    rel = B.author_release(D, Gr, D.tt, TOY_TIERS)
    assert rel.res.D is D and rel.res.G is Gr and graph_rows(rel.res.Tr) == graph_rows(Tr)
    assert np.array_equal(rel.cred, TOY_TIERS) and rel.train is None and rel.res.pooled == []


def test_a_pseudonym_re_emits_each_kept_rating_from_its_own_node_with_its_raters_tier(toy):
    D, Gr, Tr, res_cp = toy
    rel = B.singleton_release(D, Gr, D.tt, TOY_REQUESTS, TOY_TIERS, res_cp)
    # Rows 0, 2 and 7 stay; nodes 8..11 re-emit rows 1, 3, 4 and 6 at their own snapshots and times, as new rows.
    assert graph_rows(rel.res.G) == [(0, 1, 1, 0, 0, 0), (3, 4, 1, 1, 20, 2), (7, 3, 1, 3, 70, 7),
                                     (8, 1, -1, 0, 10, 8), (9, 4, -1, 1, 30, 9), (10, 1, -1, 2, 40, 10),
                                     (11, 6, 1, 2, 60, 11)]
    assert rel.res.G.n == rel.res.D.n == 12 and np.array_equal(rel.res.anon_nodes, [8, 9, 10, 11])
    assert graph_rows(rel.res.Tr) == [r for r in graph_rows(rel.res.G) if r[3] < D.tt]
    assert rel.cred.tolist() == TOY_TIERS.tolist() + [2, 3, 2, 2]      # the tiers of raters 2, 0, 5 and 5
    assert rel.train is None
    # B2: each verdict's pool is the CPPR pool of its (window, tier) cell, which holds its rater.
    assert [(p["raters"], p["raw_window"], p["cred_level"], p["window"]) for p in rel.res.pooled] == [
        ([2], 0, 2, 2), ([0], 1, 3, 7), ([5], 2, 2, 10), ([5], 2, 2, 10)]
    assert rel.res.pools is res_cp.pools and rel.res.info["n_levels"] == 4
    for p in rel.res.pooled:
        assert p["raters"][0] in res_cp.pools[p["window"]] and p["pool_size"] == len(res_cp.pools[p["window"]])
    assert {p["window"] for p in rel.res.pooled} <= {p["window"] for p in res_cp.pooled}


def test_withdrawal_counts_are_log1p_of_the_kept_ratings_per_target_and_sign(toy):
    D, Gr, Tr, _ = toy
    expected = np.zeros((8, 2))
    expected[6, 0], expected[1, 1], expected[4, 1] = 1, 2, 1     # row 6 (+) into 6; rows 1, 4 (-) into 1; row 3 into 4
    assert np.array_equal(B.withdrawal_counts(Tr, TOY_REQUESTS, 8), np.log1p(expected).astype(np.float32))
    rel = B.withdrawal_release(D, Gr, D.tt, TOY_REQUESTS, TOY_TIERS)
    assert graph_rows(rel.res.G) == [(0, 1, 1, 0, 0, 0), (3, 4, 1, 1, 20, 2), (7, 3, 1, 3, 70, 7)]  # snapshots kept
    assert rel.res.D is D and np.array_equal(rel.cred, TOY_TIERS) and rel.res.pooled == []
    assert rel.train.func is B.train_withdrawal_model
    assert np.array_equal(rel.train.keywords["counts"], np.log1p(expected).astype(np.float32))
    D_p, G_p = object(), Gr.subset(np.array([True, False, True, False, False, False, False, True]))
    given = B.withdrawal_release(D, Gr, D.tt, TOY_REQUESTS, TOY_TIERS, erased=(D_p, G_p))   # the direct arm's
    assert given.res.D is D_p and given.res.G is G_p


def test_releases_builds_only_the_selected_baselines_in_order(toy):
    D, Gr, _, res_cp = toy
    assert B.releases(D, Gr, D.tt, TOY_REQUESTS, TOY_TIERS, res_cp, ("vpe", "cppr")) == {}
    built = B.releases(D, Gr, D.tt, TOY_REQUESTS, TOY_TIERS, res_cp, ("withdrawal", "cppr", "author"))
    assert list(built) == ["author", "withdrawal"]


# ------------------------------------------------------------------ B3, B4. the model hook and the protected attach
def _function(source: str, name: str) -> ast.FunctionDef:
    return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)


def _tag_source(path: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "show", f"{TAG}:{path}"], check=True, capture_output=True,
                              text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.fail(f"tag {TAG} not found; fetch tags")


def test_make_model_without_extra_features_is_the_tags_code_and_model(toy, on_cpu):
    # The code: the tag's make_model plus one statement guarded by extra_features is not None, and one argument.
    old, new = _function(_tag_source("keeper/laundering/core.py"), "make_model"), _function(
        Path(C.__file__).read_text(), "make_model")
    guarded = [s for s in new.body if isinstance(s, ast.If) and "extra_features" in ast.unparse(s.test)]
    assert len(guarded) == 1 and ast.unparse(guarded[0].test) == "extra_features is not None"
    body = [ast.dump(s) for s in new.body[1:] if s is not guarded[0]]             # [1:]: the docstring
    assert body == [ast.dump(s) for s in old.body[1:]]
    assert [a.arg for a in new.args.args] == [a.arg for a in old.args.args] + ["extra_features"]
    # The model: built without the argument and with None, every parameter and buffer is equal.
    D, _, Tr, _ = toy
    plain, off = C.make_model(D, Tr, SEED, D.tt), C.make_model(D, Tr, SEED, D.tt, extra_features=None)
    assert plain.state_dict().keys() == off.state_dict().keys()
    assert all(torch.equal(v, off.state_dict()[k]) for k, v in plain.state_dict().items())
    counts = B.withdrawal_counts(Tr, TOY_REQUESTS, D.n)
    on = C.make_model(D, Tr, SEED, D.tt, extra_features=counts)
    X, X_on = plain.structural_layer.X_raw, on.structural_layer.X_raw
    assert X_on.shape == (D.n, X.shape[1] + 2) and torch.equal(X_on[:, :-2], X)
    assert torch.equal(X_on[:, -2:], torch.from_numpy(counts))
    assert plain.structural_layer.projection is None and on.structural_layer.projection.in_features == X.shape[1] + 2


def test_the_count_columns_survive_the_protected_attach(toy, on_cpu):
    D, _, Tr, _ = toy
    counts = torch.from_numpy(B.withdrawal_counts(Tr, TOY_REQUESTS, D.n))
    m = B.train_withdrawal_model(D, Tr, SEED, D.tt, TOY_TIERS, epochs=0, gate_norm="count", counts=counts.numpy())
    ref = G.attach_scem_and_credibility_gate(C.make_model(D, Tr, SEED, D.tt), TOY_TIERS, G.CRED_WEIGHTS,
                                             gate_norm="count")
    X = m.structural_layer.X_raw
    assert isinstance(m.s3r, G.SCEM)
    assert torch.equal(X[:, :-2], ref.structural_layer.X_raw)      # the original columns: the tier prototypes
    assert torch.equal(X[:, -2:], counts)                          # the count columns: per node again
    # Without the restore, the protected attach would leave the tier means of the counts (tier 0 holds 1, 4 and 7).
    bare = G.attach_scem_and_credibility_gate(C.make_model(D, Tr, SEED, D.tt, extra_features=counts.numpy()),
                                              TOY_TIERS, G.CRED_WEIGHTS, gate_norm="count")
    assert not torch.equal(bare.structural_layer.X_raw[:, -2:], counts)
    trained = B.train_withdrawal_model(D, Tr, SEED, D.tt, TOY_TIERS, epochs=1, gate_norm="count",
                                       counts=counts.numpy())
    assert torch.equal(trained.structural_layer.X_raw[:, -2:], counts)     # an input, never trained


# ------------------------------------------------------------------ B5. the driver with the baselines (stub models)
STORED = ("plain", "vpe", "cppr")
FULL = dict(tiers="btp", btp_eps=ALPHA_EPS, attack_edges=1, pool_eps=ALPHA_EPS, arms=G.ARMS, hold_test_snapshot=True)


def _run(tmp_path, name: str, methods, **options) -> tuple:
    dumps = tmp_path / name
    protocol = G.RunProtocol(**{**FULL, **options}, methods=methods, dump_targets=str(dumps), dump_verdicts=str(dumps))
    out = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / f"{name}.json"), protocol=protocol)
    return out, {kind: json.loads((dumps / f"{name}_{kind}.json").read_text())["rows"]
                 for kind in ("targets", "verdicts")}


def _r0(rows) -> dict:
    col = {c: i for i, c in enumerate(Dm.TARGET_COLUMNS)}
    return {(r[col["arm"]], r[col["kind"]], r[col["target"]]): r[col["r0"]] for r in rows}


def test_the_baselines_change_no_stored_output_and_no_draw(stubs, tmp_path, monkeypatch):
    built = []
    for module, name in ((V, "plain_erasure"), (V, "apply_vpe"), (G, "apply_cp_vpe")):
        real = getattr(module, name)
        monkeypatch.setattr(module, name, lambda *a, _real=real, _name=name, **k: (built.append(_name),
                                                                                 _real(*a, **k))[1])
    stored, stored_rows = _run(tmp_path, "stored", G.METHODS)
    calls_stored, built_stored = list(stubs), list(built)
    stubs.clear()
    built.clear()
    both, both_rows = _run(tmp_path, "both", G.METHODS + G.BASELINES)
    calls_both = list(stubs)
    stubs.clear()
    built.clear()
    only, only_rows = _run(tmp_path, "only", G.BASELINES)
    assert built == built_stored == ["plain_erasure"] + ["apply_vpe", "apply_cp_vpe"] * 4   # every stored release

    # Trainings: base; direct plain, vpe, cppr, singleton, withdrawal (author is the base model); then per injected
    # or control arm vpe, cppr, author, singleton, withdrawal. The stored ones keep their inputs and their order.
    assert len(calls_stored) == 10 and len(calls_both) == 1 + 5 + 3 * 5 and len(stubs) == 1 + 2 + 3 * 3
    assert [calls_both[i] for i in (0, 1, 2, 3, 6, 7, 11, 12, 16, 17)] == calls_stored
    assert stubs == [calls_both[i] for i in (0, 4, 5, 8, 9, 10, 13, 14, 15, 18, 19, 20)]

    # Every stored field is unchanged and in its order; the baselines add theirs (after each cohort's stored fields).
    names = {"direct_laundering": ["gain_{}_edge", "gain_{}_account", "t1_mcc_{}"],
             "inject_then_erase_whitewash": ["gain_{}"], "inject_then_erase_badmouth": ["drop_{}"],
             "inject_then_erase_control": ["gain_{}", "gain_{}_badmouth"]}
    for block, patterns in names.items():
        assert [k for k in both[block] if k in stored[block]] == list(stored[block])
        assert {k: both[block][k] for k in stored[block]} == stored[block]
        added = {p.format(b) for p in patterns for b in G.BASELINES}
        if block == "direct_laundering":
            added |= {"linkage_singleton", "note_author"}
        assert set(both[block]) - set(stored[block]) == added
        assert {k: only[block][k] for k in added} == {k: both[block][k] for k in added}
    direct = both["direct_laundering"]
    assert direct["gain_author_edge"] == direct["gain_author_account"] == 0.0 and direct["note_author"] == \
        B.AUTHOR_DIRECT_NOTE and direct["t1_mcc_author"] == both["base_t1"]["mcc"]
    for out in (both, only):
        assert out["sybil_admission"] == stored["sybil_admission"] and out["base_t1"] == stored["base_t1"]
    assert both["test_metrics"]["direct"]["author"] == both["test_metrics"]["base"]
    for arm, by_method in stored["test_metrics"].items():
        assert {m: both["test_metrics"][arm][m] for m in by_method} == by_method

    # Dumps: the stored methods' rows are unchanged; each baseline has target rows in every arm; only the pseudonym
    # has verdict rows, and every target's r0 is the stored run's in the baselines-only run (E5's pairing).
    for kind in ("targets", "verdicts"):
        assert [r for r in both_rows[kind] if r[1] in STORED] == stored_rows[kind]
        assert [r for r in both_rows[kind] if r[1] not in STORED] == only_rows[kind]
    assert {(r[0], r[1]) for r in only_rows["targets"]} == {(a, b) for a in G.ARMS for b in G.BASELINES}
    assert {(r[0], r[1]) for r in only_rows["verdicts"]} == {(a, "singleton") for a in G.ARMS}
    assert _r0(only_rows["targets"]) == _r0(stored_rows["targets"])
    col = {c: i for i, c in enumerate(Dm.TARGET_COLUMNS)}
    assert all(r[col["gain"]] == 0.0 and r[col["r_after"]] == r[col["r0"]] for r in only_rows["targets"]
               if r[:2] == ["direct", "author"])

    # B2 in the driver: each pseudonym's dumped candidates are the members of its arm's CPPR pool of that cell.
    vcol = {c: i for i, c in enumerate(Dm.VERDICT_COLUMNS)}
    pools, candidates = {}, {}
    for r in stored_rows["verdicts"]:
        if r[vcol["method"]] == "cppr":
            pools.setdefault((r[vcol["arm"]], r[vcol["pool_id"]]), set()).add(r[vcol["candidate"]])
    for r in only_rows["verdicts"]:
        cell = candidates.setdefault((r[vcol["arm"]], r[vcol["verdict"]]), [r[vcol["pool_id"]], [], 0])
        cell[1].append(r[vcol["candidate"]])
        cell[2] += r[vcol["label"]]
    assert candidates and all(sorted(pools[(arm, pool)]) == sorted(members) and labels == 1
                              for (arm, _), (pool, members, labels) in candidates.items())


def test_a_list_of_baselines_may_leave_cppr_out_but_not_name_an_unknown_method():
    assert G.parse_methods("withdrawal,author") == ("author", "withdrawal")
    assert G.parse_methods("singleton,cppr,plain_vpe") == ("plain_vpe", "cppr", "singleton")
    assert G.RunProtocol(methods=G.BASELINES).method_keys == G.BASELINES
    assert not G.RunProtocol(methods=G.METHODS + G.BASELINES).is_legacy()
    for bad in ("plain_vpe", "author,author", "author,retain", ""):
        with pytest.raises(ValueError):
            G.parse_methods(bad)


# ------------------------------------------------------------------ B6. the default path is the tag's
@pytest.fixture(scope="module")
def tag_driver(tmp_path_factory):
    """The driver of tag protocol-2026-11-11 as a stand-alone module (its imports are the current package's)."""
    path = tmp_path_factory.mktemp("tag") / "cp_vpe_gpu_tag.py"
    path.write_text(_tag_source("keeper/cppr/cp_vpe_gpu.py"))
    spec = importlib.util.spec_from_file_location("cp_vpe_gpu_tag", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_protected_functions_are_those_of_the_tag_and_the_driver_stays_short(tag_driver):
    def nodes(source: str) -> dict:
        return {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    old, new = nodes(Path(tag_driver.__file__).read_text()), nodes(Path(G.__file__).read_text())
    assert all(new[name] == old[name] for name in PROTECTED)
    assert len(Path(G.__file__).read_text().splitlines()) <= 800


def _written(folder: Path) -> dict:
    """Every file a run wrote, without what differs by time, place and host (runtime, provenance, dump paths)."""
    out = {}
    for path in sorted(folder.rglob("*.json")):
        data = json.loads(path.read_text())
        data.pop("provenance"), data.pop("runtime_s", None)
        for d in data.get("dumps", {}).values():
            d["path"] = Path(d["path"]).name
        out[str(path.relative_to(folder))] = data
    return out


@pytest.mark.parametrize("options, backbone", [(None, None), (dict(FULL, ungated_baseline=True), None),
                                               (dict(FULL, ungated_baseline=True), "seq_scem")])
def test_without_baselines_the_driver_writes_what_the_tags_driver_writes(stubs, tmp_path, monkeypatch, capsys,
                                                                          tag_driver, options, backbone):
    """Task 4.7 adds the case --backbone seq_scem given explicitly (the tag's driver never reads the field)."""
    monkeypatch.setattr(tag_driver, "train_gated_model", G.train_gated_model)    # the stub of the stubs fixture
    written, calls, printed = {}, {}, {}
    for name, driver in (("tag", tag_driver), ("now", G)):
        folder = tmp_path / name
        chosen = {} if backbone is None or driver is tag_driver else dict(backbone=backbone)
        protocol = None if options is None else driver.RunProtocol(
            **options, **chosen, dump_targets=str(folder / "dumps"), dump_verdicts=str(folder / "dumps"))
        stubs.clear()
        driver.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(folder / "run.json"), protocol=protocol)
        written[name], calls[name] = _written(folder), list(stubs)
        printed[name] = re.sub(r"\(\d+(\.\d+)?s\)|wrote \S+", "", capsys.readouterr().out)
    assert written["now"] == written["tag"] and len(written["now"]) == (1 if options is None else 3)
    assert calls["now"] == calls["tag"] and printed["now"] == printed["tag"]
