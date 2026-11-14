"""Task 4.7: E7's witness-laundering arm (keeper.cppr.witness_arm; protocol Section 2.9). CPU only.

  W1 the targets are the CPPR driver's direct edge targets of the seed
  W2 the attacked stream: nothing deleted (every original row kept, in order, leading its snapshot); the victims are
     exactly the warnings about the direct targets, sent by their warners; ruling R34's timing and skips; R39's
     endorsers (causal tier >= 2, never W or T) from default_rng((20261027, seed, 7, g))
  W3 a run (stub models): the clean twin and its attacks share the model seed, plan and panel; M-B on passes the
     weights of the run's own stream, off none; the gate gets the capped tiers of the stream; the record carries the
     protocol, the rules and the git state
"""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from keeper.analysis import cppr_runs as R
from keeper.cppr import rating_backbone as RB
from keeper.cppr import witness_arm as WA
from keeper.cppr.protocol import CredibilityTiers
from keeper.credibility import tiers as CT
from keeper.credibility.tiers import T_SEED
from keeper.laundering import core as C
from keeper.laundering import signed as S
from keeper.scem import e2_streams as E
from keeper.scem import witness_attack as WAT
from keeper.scem.witness_cred import level_weights
from tests import test_e2_streams as TS
from tests.cppr_common import StubModel

SEED, TT = 5, TS.T_TRAIN


@pytest.fixture(scope="module")
def Tr():
    return C.train_split(TS.synthetic_graph(), TT)


@pytest.fixture(scope="module")
def targets(Tr):
    return WA.direct_targets(Tr, SEED)


def test_the_targets_are_the_drivers_direct_edge_targets(Tr, targets):
    cohort, neg, _ = C.pick_targets(Tr, max_neg=3, strict=True)
    _, edge_t, _ = R.workload_function()(Tr, cohort, neg, 15, 5, 0.02, 3000 + SEED,
                                         np.random.default_rng(int(1e6 * 0.02) + 17 * SEED))
    assert targets.tolist() == list(edge_t) and len(targets) >= 5


@pytest.mark.parametrize("g", [0, 1])
def test_the_attacked_stream_deletes_nothing_and_follows_r34_and_r39(Tr, targets, g):
    levels = TS.synthetic_levels()
    G, inj = WA.attack_stream(Tr, targets, levels, SEED, g, TT)
    original = G.subset(G.rid >= 0)
    for name in ("edges", "labels", "time", "snap", "rid"):
        assert np.array_equal(getattr(original, name), getattr(Tr, name)), name      # nothing deleted, same order
    assert np.all(np.diff(G.snap) >= 0)
    for s in np.unique(G.snap):                                  # original rows lead every snapshot
        injected = G.rid[G.snap == s] < 0
        assert not (injected[:-1] & ~injected[1:]).any()

    sign = 1 - 2 * S.class_ids(Tr.labels)
    warnings = {(int(u), int(v), int(t)) for (u, v), s, t in zip(Tr.edges, sign, Tr.snap) if s < 0 and v in targets}
    attacked = {(int(Tr.edges[r, 0]), int(Tr.edges[r, 1]), int(Tr.snap[r])) for r in inj.victims}
    skipped = {(int(Tr.edges[r, 0]), int(Tr.edges[r, 1]), int(Tr.snap[r]))
               for r in np.concatenate([inj.skipped_seed_window, inj.skipped_beyond_stream])}
    assert attacked and attacked | skipped == warnings and not attacked & skipped
    assert all(t + WA.LAG <= T_SEED + 1 or t + WA.LAG >= TT for _, _, t in skipped)

    e, s, t = inj.edges, inj.sign, inj.snap
    for i, rec in enumerate(inj.victims):
        warner, target, t_k = int(Tr.edges[rec, 0]), int(Tr.edges[rec, 1]), int(Tr.snap[rec])
        sybils, t_a = inj.sybil_ids[i], max(t_k + WA.LAG - 1, T_SEED + 1)
        assert len(sybils) == WA.K_SYBILS and inj.t_attack[i] == t_a and inj.t_badmouth[i] == t_k + WA.LAG
        bad = inj.badmouth_edge_ids[i]
        assert (e[bad, 0] == sybils).all() and (e[bad, 1] == warner).all() and (s[bad] == -1).all()
        assert (t[bad] == t_k + WA.LAG).all() and t_k + WA.LAG > t_a
        ring = inj.ring_edge_ids[i]
        assert (e[ring, 1] == np.roll(sybils, -1)).all() and (s[ring] == 1).all() and (t[ring] == t_a).all()
        att = inj.attack_edge_ids[i]
        assert len(att) == g and (e[att, 1] == sybils[:g]).all() and (t[att] == t_a).all() and (s[att] == 1).all()
        assert all(levels[t_a, u] >= 2 and u not in (warner, target) for u in e[att, 0])
    replay = WAT.inject_adversarial_witnesses(*E.signed(Tr), Tr.n, WAT.warnings_about(*E.signed(Tr)[:2], targets),
                                              5, 1, g, np.random.default_rng((20261027, SEED, 7, g)), levels, TT,
                                              sybil_ring=True)
    assert np.array_equal(replay.endorsers, inj.endorsers) and np.array_equal(replay.edges, inj.edges)


# ------------------------------------------------------------------ W3. a run with stub models
@pytest.fixture
def stub_run(monkeypatch, tmp_path):
    """run() on the synthetic network named bitcoinalpha; models are StubModel; returns the training calls."""
    graph = TS.synthetic_graph()
    D = C.Dataset("bitcoinalpha", graph.edges, graph.labels, graph.time, graph.n, TT + 1, {})
    calls = []

    def train_rating_model(D_, Tr_, seed, tt, cred, weights, epochs=150, anon=None, pool_w=None, gate_norm="x"):
        calls.append(dict(Tr=Tr_, seed=seed, tt=tt, cred=np.asarray(cred), weights=weights, epochs=epochs,
                          gate_norm=gate_norm))
        m = StubModel(Tr_, cred, seed, gate_norm)
        m.scem = SimpleNamespace(phi=torch.tensor(0.3))
        return m, dict(stub=True)

    for name, value in (("load", lambda ds: D), ("full_graph", lambda D_: graph), ("require_gpu", lambda: None),
                        ("apply_memory_profile", lambda D_: {"stub": True}),
                        ("set_deterministic", lambda on=True: None),
                        ("embed", lambda m, Tr_, tt: m.z)):
        monkeypatch.setattr(C, name, value)
    monkeypatch.setattr(RB, "train_rating_model", train_rating_model)
    monkeypatch.delenv(C.OVERWRITE_ENV, raising=False)
    prereg = tmp_path / "prereg.md"
    prereg.write_text("protocol\n")

    def run(stream, mb):
        path = tmp_path / f"{stream}_{mb}.json"
        WA.run("bitcoinalpha", SEED, stream, mb, 2, str(path), str(prereg))
        return json.loads(path.read_text())
    return SimpleNamespace(run=run, calls=calls, graph=graph, D=D, prereg=prereg)


def test_the_clean_twin_and_its_attacks_pair_and_get_their_own_mb_weights(stub_run):
    runs = {(st, mb): stub_run.run(st, mb) for st in ("clean", "g0") for mb in ("on", "off")}
    Tr = C.train_split(stub_run.graph, TT)
    cal = CT.calibrate_btp(Tr, stub_run.D.n, 0.02, edge_cap_units=2.0)
    for (stream, mb), call in zip(runs, stub_run.calls):
        r = runs[(stream, mb)]
        assert call["seed"] == r["model_seed"] == SEED and call["gate_norm"] == "count" and call["epochs"] == 2
        assert np.array_equal(call["cred"], CredibilityTiers("btp", cal, "trusted")(call["Tr"], call["Tr"].n))
        want = level_weights(E.stream_levels(call["Tr"], Tr, 0.02, TT))
        assert (call["weights"] is None) == (mb == "off") and (mb == "off" or np.array_equal(call["weights"], want))
        assert (r["mb_info"] is None) == (mb == "off") and (r["attack"] is None) == (stream == "clean")
    first = runs[("clean", "on")]
    for r in runs.values():
        for field in ("plan", "panel", "targets", "epochs", "settings"):
            assert r[field] == first[field], field
    assert runs[("g0", "on")]["stream_info"] == runs[("g0", "off")]["stream_info"] != first["stream_info"]
    assert runs[("g0", "on")]["attack"] == runs[("g0", "off")]["attack"]
    assert first["stream_info"]["n_nodes"] == stub_run.D.n and runs[("g0", "on")]["proxy"] != first["proxy"]
    attack = runs[("g0", "on")]["attack"]
    assert attack["n_sybils"] == 5 * len(attack["victims"]) and attack["rng"] == [20261027, SEED, 7, 0]
    assert runs[("g0", "on")]["mb_info"]["honest_levels_changed"] == 0             # g = 0: the guard holds
    sha = hashlib.sha256(stub_run.prereg.read_bytes()).hexdigest()
    assert first["prereg"]["sha256"] == sha and "git_commit" in first["provenance"]
    assert (first["gate"]["eps"], first["gate"]["edge_cap_units"], first["gate"]["neg_from"]) == (0.02, 2.0, "trusted")
    assert first["learned_phi"] == pytest.approx(0.3) and first["targets"] == WA.direct_targets(Tr, SEED).tolist()


def test_an_existing_output_is_refused_before_any_work(stub_run, capsys):
    out = stub_run.prereg.parent / "taken.json"
    out.write_text("stored")
    with pytest.raises(SystemExit) as stop:
        WA.main(["--ds", "bitcoinalpha", "--seed", "5", "--stream", "clean", "--mb", "on", "--prereg",
                 str(stub_run.prereg), "--out", str(out)])
    assert stop.value.code == 2 and "exists" in capsys.readouterr().err
    assert out.read_text() == "stored" and not stub_run.calls
