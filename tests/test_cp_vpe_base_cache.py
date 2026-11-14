"""The base-model cache of the CPPR driver keeper.cppr.cp_vpe_gpu (Task 4.1c, ruling R59), run with stub models on the
CPU (the stubs fixture of tests/conftest.py). D16 to D18 moved verbatim from tests/test_cp_vpe_driver.py (Task 4.1c
fix round 1) so that every test file stays under 800 lines.

  D16 a cache miss then a hit give the uncached run's outputs; the hit builds the base model without training it; the
      result copies the entry's meta (GPU, torch, created utc, commit), and the entry records git_dirty
  D17 a cache entry that does not reproduce stops the run: another key, another identity (GPU, code), altered stored
      outputs or state
  D18 the cache key names the run's tier configuration, the graph, the settings, the code (commit and content hash),
      the GPU model, the torch version and the pipeline mode
Task 4.1c fix round 1:
  D20 --base-cache is refused outside a git checkout, before any work
  D21 another run of the same key stores its entry meanwhile: the same outputs go on (not from the cache), other
      outputs stop the run, and the stored entry is kept

Run from the repository root (CPU is enough):
    CUDA_VISIBLE_DEVICES="" <ps python> -m pytest -q tests/test_cp_vpe_base_cache.py
"""
import dataclasses
import json
import re

import numpy as np
import pytest
import torch

from keeper.analysis.provenance import git_state
from keeper.cppr import base_cache as BC
from keeper.cppr import cp_vpe_gpu as G
from keeper.credibility import tiers
from keeper.laundering import core as C
from tests.cppr_common import ALPHA_EPS, alpha_calibration, background


# ------------------------------------------------------------------ D16-D18. the base-model cache (Task 4.1c)
def _e4_cell(cache=None, **options):
    """An E4-like cell: capped BTP, one ranked max-B(u) attack edge per group, the whitewash arm, CPPR only."""
    return G.RunProtocol(tiers="btp", btp_eps=ALPHA_EPS, pool_eps=ALPHA_EPS, attack_edges=1, endorser_select="max_trust",
                         arms=("whitewash",), methods=("cppr",), hold_test_snapshot=True,
                         base_cache=None if cache is None else str(cache), **options)


def test_a_cache_miss_then_a_hit_give_the_outputs_of_an_uncached_run(stubs, tmp_path, monkeypatch):
    # The first cell of a seed trains the base model and stores it under --base-cache DIR. A later cell with the same
    # key builds the model through train_gated_model without training it (0 epochs), loads the stored state and checks
    # its recomputed z0, r0 and base test metrics bit for bit. Here a model built with 0 epochs is a blank stand-in, so
    # only the loaded state can reproduce the outputs.
    epochs_seen = []
    trained = G.train_gated_model

    def blank_without_epochs(*args, epochs=150, **kwargs):
        epochs_seen.append(epochs)
        model = trained(*args, epochs=epochs, **kwargs)
        if epochs == 0:
            model.z, model.regression_weights = torch.zeros_like(model.z), torch.zeros_like(model.regression_weights)
        return model

    monkeypatch.setattr(G, "train_gated_model", blank_without_epochs)
    cache, runs = tmp_path / "cache", {}
    for name, folder in (("plain", None), ("miss", cache), ("hit", cache)):
        stubs.clear()
        epochs_seen.clear()
        out = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / f"{name}.json"),
                                protocol=_e4_cell(folder, dump_targets=str(tmp_path / name)))
        rows = json.loads((tmp_path / name / f"{name}_targets.json").read_text())["rows"]
        runs[name] = dict(out=out, calls=list(stubs), epochs=list(epochs_seen), rows=rows)
    key = runs["miss"]["out"]["base_model"]["key"]
    assert "base_model" not in runs["plain"]["out"]
    assert [p.name for p in cache.iterdir()] == [f"{key}.pt"]
    # The result copies the entry's meta: the GPU, the torch version, when the entry was stored and the commit. The
    # entry records whether the tree was dirty (fix round 1).
    meta, git = torch.load(cache / f"{key}.pt", weights_only=True)["meta"], git_state()
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", meta["created_utc"]) and meta["git_dirty"] == git["git_dirty"]
    stored = dict(gpu_name="cpu", torch=str(torch.__version__), created_utc=meta["created_utc"],
                  code_commit=git["git_commit"])
    assert runs["miss"]["out"]["base_model"] == dict(from_cache=False, key=key, **stored)
    assert runs["hit"]["out"]["base_model"] == dict(from_cache=True, key=key, **stored)
    bg, cal = background(), alpha_calibration()
    assert key == BC.cache_key(BC.identity("bitcoinalpha", 0, 3, "count", "btp", cal, git["git_commit"]),
                               G.CredibilityTiers("btp", cal)(bg.Tr, bg.D.n), bg.Tr)
    assert runs["plain"]["epochs"] == runs["miss"]["epochs"] == [3, 3] and runs["hit"]["epochs"] == [0, 3]
    assert runs["miss"]["calls"] == runs["plain"]["calls"] and runs["hit"]["calls"][1:] == runs["plain"]["calls"][1:]
    ignored = ("runtime_s", "base_model", "dumps")
    for name in ("miss", "hit"):
        assert {k: v for k, v in runs[name]["out"].items() if k not in ignored} == {
            k: v for k, v in runs["plain"]["out"].items() if k not in ignored}, name
        assert runs[name]["rows"] == runs["plain"]["rows"], name


@pytest.mark.parametrize("alteration,trainings,message", [
    ("key", 0, "the entry holds the key 0{64}, not its own"),
    ("gpu", 0, "the entry was stored under another identity: gpu_name 'NVIDIA RTX A6000', not 'cpu'; delete it"),
    ("code", 0, "the entry was stored under another identity: code_sha256 '0{64}', not '[0-9a-f]{64}'; delete it"),
    ("outputs", 1, r"the base outputs recomputed from its stored state \(z0, r0, base test metrics\) differ"),
    ("state", 1, r"the base outputs recomputed from its stored state \(z0, r0, base test metrics\) differ"),
])
def test_a_cache_entry_that_does_not_reproduce_stops_the_run(stubs, tmp_path, alteration, trainings, message):
    # An entry filed under another key or stored under another identity (GPU model, code, ...; fix round 1), or whose
    # stored state no longer gives its stored outputs, is refused before any model trains: the base model is at most
    # rebuilt, and no result is written.
    cache = tmp_path / "cache"
    key = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "first.json"),
                            protocol=_e4_cell(cache))["base_model"]["key"]
    path = cache / f"{key}.pt"
    entry = torch.load(path, weights_only=True)
    if alteration == "key":
        entry["key"] = "0" * 64
    elif alteration == "gpu":
        entry["meta"]["gpu_name"] = "NVIDIA RTX A6000"
    elif alteration == "code":
        entry["meta"]["code_sha256"] = "0" * 64
    elif alteration == "outputs":
        entry["outputs"]["r0_cohort"][0] += 1e-9
    else:
        entry["state_dict"]["z"] = 2.0 * entry["state_dict"]["z"]
    torch.save(entry, path)
    stubs.clear()
    with pytest.raises(ValueError, match=f"^base-model cache {re.escape(str(path))}: {message}"):
        G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "second.json"), protocol=_e4_cell(cache))
    assert len(stubs) == trainings and not (tmp_path / "second.json").exists()


def test_the_cache_key_names_the_tier_configuration_the_graph_and_the_code(monkeypatch):
    # The key is a SHA-256 over the base model's identity, the cred_levels array, and the training graph's edges,
    # labels and snapshots. The identity holds ds, seed, epochs and gate_norm; the tier method and the run's calibration
    # (eps, K, theta0, edge_cap_units, n_ref); the code commit and, since fix round 1, the code's content hash, the GPU
    # model, the torch version and the memory profile's pipeline mode, read from the process. Any change among them
    # gives another key.
    bg = background()
    n, tr, cal = bg.D.n, bg.Tr, alpha_calibration()
    cred = G.CredibilityTiers("btp", cal)(tr, n)
    monkeypatch.setattr(C, "DEFAULT_STRATEGY", None)
    ident = BC.identity("bitcoinalpha", 0, 150, "count", "btp", cal, "0" * 40)
    assert ident == dict(ds="bitcoinalpha", seed=0, epochs=150, gate_norm="count", tiers="btp",
                         calibration=dict(eps=cal.eps, K=cal.K, theta0=cal.theta0, edge_cap_units=cal.edge_cap_units,
                                          n_ref=cal.n_ref),
                         code_commit="0" * 40, code_sha256=C.CODE_SHA256_AT_IMPORT, gpu_name="cpu",
                         torch=str(torch.__version__), pipeline=None)
    # The environment is read when the identity is taken: a GPU, another torch, the two-GPU pipeline.
    monkeypatch.setattr(C, "DEV", torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA RTX A6000")
    monkeypatch.setattr(torch, "__version__", "2.5.1")
    monkeypatch.setattr(C, "DEFAULT_STRATEGY", {"mode": "pipeline", "gpu_structural": 0, "gpu_s3r": 1})
    elsewhere = BC.identity("bitcoinalpha", 0, 150, "count", "btp", cal, "0" * 40)
    assert {k: v for k, v in elsewhere.items() if v != ident[k]} == dict(gpu_name="NVIDIA RTX A6000", torch="2.5.1",
                                                                         pipeline="pipeline")
    key = BC.cache_key(ident, cred, tr)
    assert re.fullmatch("[0-9a-f]{64}", key) and BC.cache_key(ident, cred, tr.subset(np.arange(len(tr.edges)))) == key

    def graph(**arrays):
        parts = dict(dict(edges=tr.edges, labels=tr.labels, snap=tr.snap), **arrays)
        return C.Graph(parts["edges"], parts["labels"], tr.time, parts["snap"], tr.n, tr.rid)

    flipped, later, moved, promoted = tr.labels.copy(), tr.snap.copy(), tr.edges.copy(), cred.copy()
    flipped[0] = tr.labels[0, ::-1]                           # the first rating's sign
    later[-1] += 1                                            # the last rating's snapshot
    moved[0, 1] = (moved[0, 1] + 1) % n                       # the first rating's receiver
    promoted[int(np.argmin(cred))] += 1                       # one account's tier
    def tiers_of(method, c):
        """The identity of the same run with other tiers."""
        return dict(ident, tiers=method, calibration=None if c is None else dict(
            eps=c.eps, K=c.K, theta0=c.theta0, edge_cap_units=c.edge_cap_units, n_ref=c.n_ref))

    variants = dict(
        ds=(dict(ident, ds="bitcoinotc"), cred, tr), seed=(dict(ident, seed=1), cred, tr),
        epochs=(dict(ident, epochs=149), cred, tr), gate_norm=(dict(ident, gate_norm="snapshot"), cred, tr),
        commit=(dict(ident, code_commit="1" * 40), cred, tr), code=(dict(ident, code_sha256="f" * 64), cred, tr),
        gpu=(dict(ident, gpu_name="NVIDIA RTX A6000"), cred, tr), torch=(dict(ident, torch="2.5.1"), cred, tr),
        pipeline=(dict(ident, pipeline="pipeline"), cred, tr),
        bfs=(tiers_of("bfs", None), G.compute_credibility_levels(tr, n), tr),
        eps=(tiers_of("btp", tiers.calibrate_btp(tr, n, 0.01, edge_cap_units=2.0)), cred, tr),
        cap=(tiers_of("btp", tiers.calibrate_btp(tr, n, ALPHA_EPS, edge_cap_units=4.0)), cred, tr),
        uncapped=(tiers_of("btp", tiers.calibrate_btp(tr, n, ALPHA_EPS)), cred, tr),
        K=(tiers_of("btp", dataclasses.replace(cal, K=cal.K + 1)), cred, tr),
        n_ref=(tiers_of("btp", dataclasses.replace(cal, n_ref=cal.n_ref + 1)), cred, tr),
        cred_levels=(ident, promoted, tr), cred_dtype=(ident, cred.astype(np.int32), tr),
        edges=(ident, cred, graph(edges=moved)), labels=(ident, cred, graph(labels=flipped)),
        snap=(ident, cred, graph(snap=later)), dropped=(ident, cred, tr.subset(np.arange(1, len(tr.edges)))))
    keys = {name: BC.cache_key(*args) for name, args in variants.items()}
    assert key not in keys.values() and len(set(keys.values())) == len(variants)


# ------------------------------------------------------------------ D20-D21. fix round 1
def test_the_cache_is_refused_outside_a_git_checkout(stubs, tmp_path, monkeypatch):
    # The key names the code by its commit; outside a git checkout the commit is None, so the run stops before any work.
    monkeypatch.setattr(G, "git_state", lambda: dict(git_commit=None, git_dirty=False, git_untracked=0, git_status=[]))
    with pytest.raises(ValueError, match=r"^--base-cache needs a git checkout"):
        G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "run.json"),
                          protocol=_e4_cell(tmp_path / "cache"))
    assert stubs == [] and not (tmp_path / "cache").exists() and not (tmp_path / "run.json").exists()


@pytest.mark.parametrize("same", [True, False])
def test_a_run_whose_key_another_run_stores_meanwhile(stubs, tmp_path, monkeypatch, same):
    # A run finds no entry, trains its base model, and another run of the same key stores its entry before this one
    # writes (simulated between load and write). The stored entry is kept. With the same outputs the run goes on, not
    # from the cache; with other outputs it stops.
    first = tmp_path / "first"
    key = G.run_single_seed("bitcoinalpha", 0, epochs=3, out_name=str(tmp_path / "first.json"),
                            protocol=_e4_cell(first))["base_model"]["key"]
    other = torch.load(first / f"{key}.pt", weights_only=True)
    other["meta"]["created_utc"] = "2000-01-01T00:00:00Z"         # marks the other run's entry; not in the identity
    if not same:
        other["outputs"]["r0_trusted"][0] += 1e-9
    real = BC.BaseCache.check_or_store

    def another_run_stores_first(self, entry, *args):
        assert entry is None                              # this run found nothing and trained its own base model
        torch.save(other, self.path)
        return real(self, entry, *args)

    monkeypatch.setattr(BC.BaseCache, "check_or_store", another_run_stores_first)
    cache = tmp_path / "cache"
    run = dict(out_name=str(tmp_path / "second.json"), protocol=_e4_cell(cache))
    if same:
        out = G.run_single_seed("bitcoinalpha", 0, epochs=3, **run)
        assert out["base_model"]["from_cache"] is False and out["base_model"]["key"] == key
    else:
        with pytest.raises(ValueError, match="the base outputs of a model trained anew"):
            G.run_single_seed("bitcoinalpha", 0, epochs=3, **run)
        assert not (tmp_path / "second.json").exists()
    assert torch.load(cache / f"{key}.pt", weights_only=True)["meta"] == other["meta"]   # never replaced
