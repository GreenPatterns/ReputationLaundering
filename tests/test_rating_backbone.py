"""Task 4.7: the rating-level backbone of the CPPR driver (keeper.cppr.rating_backbone; protocol Section 2.9, E7).

  RB1 --backbone parses and validates; seq_scem is the default and the stored protocol, rating_scem is a changed one
      (the byte-identity of the default path against tag protocol-2026-11-11 is tests/test_cppr_baselines.py, B6)
  RB2 the rating-level model is built through core.make_model as the class of stage_b.cmd_model, with M-B weights
  RB3 the protected attach function keeps its rating-level SCEM and gates every structural conv layer
  RB4 pool nodes take their pool's tier, at every snapshot, in the gate and in the M-B levels (CPPR), and a Plain VPE
      window node its pool's weight in both
  RB5 the driver with --backbone rating_scem (stub models): the same trainings on the same graphs and tiers as the
      default backbone, each with the M-B weights of its arm and release, and the backbone recorded in the config
CPU only: CUDA_VISIBLE_DEVICES="" <ps python> -m pytest -q tests/test_rating_backbone.py
"""
import json

import numpy as np
import pytest

from keeper.cppr import cp_vpe_gpu as G
from keeper.cppr import rating_backbone as RB
from keeper.cppr import vpe as V
from keeper.credibility.tiers import T_SEED, TIER_WEIGHTS
from keeper.laundering import core as C
from keeper.scem import e2_streams as E
from keeper.scem import scem as SC
from keeper.scem.witness_cred import level_weights
from tests.cppr_common import ALPHA_EPS, StubModel, background, fingerprint
from tests.test_cppr_baselines import INF, TOY_REQUESTS, TOY_TIERS, on_cpu, toy  # noqa: F401  (fixtures)

W = np.asarray(TIER_WEIGHTS)
RATING = dict(tiers="btp", btp_eps=ALPHA_EPS, hold_test_snapshot=True, backbone="rating_scem")


def gate_node_weights(model) -> np.ndarray:
    """The per-node weights the gate installed on the model's first conv layer (its closure's node_w)."""
    f = model.structural_layer.conv_layers[0].incoming_edges
    cells = dict(zip(f.__code__.co_freevars, (c.cell_contents for c in f.__closure__)))
    return cells["node_w"].cpu().numpy().astype(np.float64)


# ------------------------------------------------------------------ RB1. the option
def test_backbone_parses_defaults_to_the_stored_one_and_validates():
    parse = G.build_parser().parse_args
    assert G.protocol_from_args(parse([])) == G.protocol_from_args(parse(["--backbone", "seq_scem"])) == G.RunProtocol()
    assert G.RunProtocol().backbone == "seq_scem" and G.RunProtocol().is_legacy()
    rating = G.protocol_from_args(parse(["--backbone", "rating_scem", "--tiers", "btp", "--btp-eps", "0.02",
                                         "--hold-test-snapshot"]))
    assert rating.backbone == "rating_scem" and not rating.is_legacy()
    for bad in (dict(tiers="bfs", btp_eps=None), dict(btp_edge_cap=None), dict(btp_neg_from="all"),
                dict(hold_test_snapshot=False), dict(methods=("cppr", "author")), dict(ungated_baseline=True),
                dict(base_cache="/tmp/cache"), dict(backbone="lstm")):
        with pytest.raises(ValueError):
            G.RunProtocol(**{**RATING, **bad})
    with pytest.raises(SystemExit):
        parse(["--backbone", "lstm"])


# ------------------------------------------------------------------ RB2, RB3. build and gate (CPU, no training)
def test_the_rating_model_is_built_through_make_model_and_gated_on_its_conv_layers(toy, on_cpu, monkeypatch):
    D, Gr, Tr, _ = toy
    built, real = [], C.make_model
    monkeypatch.setattr(C, "make_model", lambda *a, **k: built.append(a[2]) or real(*a, **k))
    weights = np.full((D.tt, D.n), 0.5)
    m, settings = RB.train_rating_model(D, Tr, 5, D.tt, TOY_TIERS, weights, epochs=0, gate_norm="count")
    assert built == [5] and C.LocalGDTE.__module__ == "keeper.backbone.local_model"   # the pipeline class is restored
    assert type(m).__module__ == "keeper.backbone.local_model_scem" and isinstance(m.scem, SC.SCEM)
    assert m.s3r is m.scem and (settings["N"], settings["phi_fixed"], settings["hazard"]) == (2, None, True)
    convs = m.structural_layer.conv_layers
    assert len(convs) >= 1 and all(c.incoming_edges.__name__ == RB.GATED_NAME for c in convs)
    assert np.allclose(gate_node_weights(m), W[TOY_TIERS])
    z = C.embed(m, Tr, D.tt)                                     # the gated rating-level forward pass runs
    assert z.shape[0] == D.n and bool(np.isfinite(z.numpy()).all())


def test_check_gated_refuses_a_replaced_module_or_an_ungated_layer(toy, on_cpu):
    D, _, Tr, _ = toy
    m, _ = RB.train_rating_model(D, Tr, 5, D.tt, TOY_TIERS, None, epochs=0, gate_norm="count")
    with pytest.raises(RuntimeError, match="replaced"):
        RB.check_gated(m, object())
    conv = m.structural_layer.conv_layers[0]
    conv.incoming_edges = type(conv).incoming_edges.__get__(conv)
    with pytest.raises(RuntimeError, match="not installed"):
        RB.check_gated(m, m.scem)


# ------------------------------------------------------------------ RB4. pool nodes: gate and M-B
def test_pool_nodes_take_their_pools_tier_in_the_gate_and_in_the_mb_levels(toy, on_cpu):
    D, Gr, Tr, res_cp = toy
    _, cred_cp = G.apply_cp_vpe(D, Gr, D.tt, TOY_REQUESTS, TOY_TIERS, G.CPVPEConfig(1, INF, 10),
                                np.random.default_rng(0))
    n_arm, T = D.n, 8                                            # 8 rows, so that rows after t_seed exist
    arm_levels = np.random.default_rng(1).integers(0, 4, size=(T, n_arm))
    full = RB.extend_levels(arm_levels, cred_cp, n_arm)
    assert full.shape == (T, len(cred_cp)) and np.array_equal(full[:, :n_arm], arm_levels)
    assert len(cred_cp) > n_arm and (full[:, n_arm:] == cred_cp[n_arm:]).all()      # every snapshot
    weights = RB.mb_weights(full)
    assert (weights[:T_SEED + 1] == 1).all() and np.array_equal(weights[T_SEED + 1:, n_arm:],
                                                                 np.tile(W[cred_cp[n_arm:]], (T - T_SEED - 1, 1)))
    m, _ = RB.train_rating_model(res_cp.D, res_cp.Tr, 5, D.tt, cred_cp, None, epochs=0, gate_norm="count")
    assert np.allclose(gate_node_weights(m)[n_arm:], weights[-1, n_arm:])          # gate weight == M-B weight

    res_vpe = V.apply_vpe(D, Gr, D.tt, TOY_REQUESTS, V.VPEConfig(1, INF, 10), np.random.default_rng(0))
    pool_w = {int(w): float(np.mean(W[TOY_TIERS[list(p)]])) for w, p in res_vpe.pools.items() if len(p)}
    cred_vpe = np.concatenate([TOY_TIERS, np.zeros(len(res_vpe.anon_nodes), dtype=np.int64)])
    weights = RB.mb_weights(RB.extend_levels(arm_levels, cred_vpe, n_arm), res_vpe.anon_nodes, pool_w)
    m, _ = RB.train_rating_model(res_vpe.D, res_vpe.Tr, 5, D.tt, cred_vpe, None, epochs=0, gate_norm="count",
                                 plain_vpe_anon_nodes=res_vpe.anon_nodes, plain_vpe_pool_weights=pool_w)
    assert pool_w and np.allclose(gate_node_weights(m)[n_arm:], weights[-1, n_arm:])
    assert all(weights[-1, res_vpe.anon_nodes[w]] == pool_w[w] for w in pool_w)


def test_extend_levels_refuses_tiers_that_do_not_fit_the_arm():
    with pytest.raises(ValueError):
        RB.extend_levels(np.zeros((3, 5), dtype=np.int64), np.zeros(4, dtype=np.int64), 5)
    with pytest.raises(ValueError):
        RB.extend_levels(np.zeros((3, 4), dtype=np.int64), np.zeros(6, dtype=np.int64), 5)


# ------------------------------------------------------------------ RB5. the driver (stub models)
def test_the_driver_trains_the_same_models_with_the_rating_backbone(stubs, tmp_path, monkeypatch):
    rating_calls = []

    def train_rating_model(D, Tr, seed, tt, cred, weights, epochs=150, anon=None, pool_w=None, gate_norm="snapshot"):
        stubs.append(fingerprint("gated", D.name, int(D.n), int(D.T), Tr, np.asarray(cred), seed, tt, epochs,
                                 None if anon is None else np.asarray(anon), pool_w, gate_norm))
        rating_calls.append(dict(Tr=Tr, cred=np.asarray(cred), weights=weights, anon=anon, pool_w=pool_w))
        return StubModel(Tr, cred, seed, gate_norm, pool_w, anon), dict(stub=True)

    monkeypatch.setattr(RB, "train_rating_model", train_rating_model)
    options = dict(tiers="btp", btp_eps=ALPHA_EPS, attack_edges=1, pool_eps=ALPHA_EPS, arms=G.ARMS,
                   methods=G.METHODS, hold_test_snapshot=True)
    outs, calls = {}, {}
    for backbone in ("seq_scem", "rating_scem"):
        stubs.clear()
        outs[backbone] = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / f"{backbone}.json"),
                                           protocol=G.RunProtocol(**options, backbone=backbone))
        calls[backbone] = list(stubs)
    assert calls["rating_scem"] == calls["seq_scem"] and len(rating_calls) == len(calls["seq_scem"]) == 10

    seq, rating = outs["seq_scem"], outs["rating_scem"]
    backbone = rating["config"].pop("backbone")
    for out in (seq, rating):
        out.pop("runtime_s")
    assert rating == seq and "backbone" not in seq["config"]       # every other output equals the default backbone's
    assert backbone["name"] == "rating_scem" and backbone["settings"] == dict(stub=True)
    assert set(backbone["mb"]["arms"]) == {"base", "whitewash", "badmouth", "control"}
    assert all((a["eps"], a["edge_cap_units"], a["neg_from"]) == (ALPHA_EPS, 2.0, "trusted")
               for a in backbone["mb"]["arms"].values())

    bg = background()
    base_weights = level_weights(E.stream_levels(bg.Tr, bg.Tr, ALPHA_EPS, bg.tt))
    arm_of = [m["arm"] for m in backbone["models"]]
    assert arm_of == ["base"] * 4 + ["whitewash"] * 2 + ["badmouth"] * 2 + ["control"] * 2
    first = {}
    for call, rec in zip(rating_calls, backbone["models"]):
        n_arm, w = rec["n_nodes"] - rec["n_added"], call["weights"]
        assert w.shape == (bg.tt, call["Tr"].n) == (bg.tt, len(call["cred"])) and (w[:T_SEED + 1] == 1).all()
        assert np.array_equal(w[:, :n_arm], first.setdefault(rec["arm"], w[:, :n_arm]))   # one arm, one set of levels
        if rec["arm"] in ("base", "control"):
            assert np.array_equal(w[:, :n_arm], base_weights)
        added = W[call["cred"][n_arm:]]                          # pool nodes: their pool's tier (CPPR), tier 0 else
        for w_idx, a_id in enumerate([] if call["anon"] is None else call["anon"]):
            if w_idx in call["pool_w"]:
                added[a_id - n_arm] = call["pool_w"][w_idx]      # Plain VPE: its pool's weight, as the gate
        assert np.array_equal(w[T_SEED + 1:, n_arm:], np.tile(added, (bg.tt - T_SEED - 1, 1)))
    cppr_models = [r for r, c in zip(backbone["models"], rating_calls) if c["anon"] is None and r["n_added"]]
    assert cppr_models and all(sum(r["added_tier_counts"]) == r["n_added"] for r in cppr_models)
    written = json.loads((tmp_path / "rating_scem.json").read_text())
    assert written["config"]["backbone"]["mb"]["arms"]["base"]["levels_sha256"] == \
        backbone["mb"]["arms"]["base"]["levels_sha256"]
