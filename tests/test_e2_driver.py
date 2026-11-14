"""E2 driver (Task 4.5; keeper.scem.e2_driver) and the SCEM option it adds. CPU only, no training.

1. hazard=True, the default of the new module option, leaves SCEM unchanged bit for bit: parameters, output, auxiliary
   loss and every gradient equal those of the engine with the E2 lines reverted (tests/test_scem_equivalence.E2_EDITS,
   which the source pin also checks), in both modes; the archived LocalGDTE agrees too (forward_check).
2. hazard=False is the forward pass of lambda_max = 0 without its NaN gradients, and it changes the output.
3. stage_b.configured_scem combines the options; with none it is the module.
4. Every E2 model is built with the settings it claims (the constructor swallows unknown keywords).
5. The driver, with stub models: the clean twin and the attack run share the model seed, the plan and the panel; the
   SCEM models get the M-B weights of their own stream; the record carries the protocol's sha256 and the git state;
   an existing output is refused before any work.
6. The task list (keeper.cluster.e2_tasks): the 315 registered runs, in the format of keeper.cluster.cppr_tasks, with
   commands that the driver's own parser accepts.
"""
import hashlib
import json
import shlex
import warnings
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from keeper.scem import stage_b  # first: it sets CUBLAS_WORKSPACE_CONFIG before CUDA starts
from keeper.cluster import e2_tasks
from keeper.laundering import core as C
from keeper.scem import e2_driver as DRV
from keeper.scem import e2_streams as E
from keeper.scem import scem as SC
from keeper.scem.witness_cred import level_weights
from tests import test_e2_streams as TS
from tests import test_scem_equivalence as EQ
from tests import test_scem_phi_fixed as PF
from tests.cppr_common import StubModel


@pytest.fixture(scope="module")
def engine_without_e2(tmp_path_factory):
    """keeper/scem/scem.py with the E2 lines reverted: the module as it was before Task 4.5."""
    source = EQ.ENGINE.read_text()
    for old, new in EQ.E2_EDITS:
        assert source.count(new) == 1 and old not in source
        source = source.replace(new, old)
    assert "self.hazard" not in source
    path = tmp_path_factory.mktemp("e2") / "scem_without_e2.py"
    path.write_text(source)
    return stage_b.load_original_module(str(path))


def run_with_grads(scem_cls, mode, **kw):
    """(output, auxiliary loss, state dict, gradients) of a module built from seed 0, after one backward pass."""
    module = PF.build(scem_cls, **kw)
    z, aux = PF.forward(module, mode)
    (z.sum() + aux).backward()
    grads = {n: None if p.grad is None else p.grad.clone() for n, p in module.named_parameters()}
    return z.detach().clone(), float(aux), module.state_dict(), grads


def same_grads(g0, g1):
    return set(g0) == set(g1) and all((g0[n] is None and g1[n] is None) or torch.equal(g0[n], g1[n]) for n in g0)


# ------------------------------------------------------------------ 1. the default: the module unchanged
@pytest.mark.parametrize("mode", PF.MODES)
def test_hazard_default_leaves_the_module_unchanged(engine_without_e2, mode):
    z0, a0, s0, g0 = run_with_grads(engine_without_e2.SCEM, mode)
    for cls, kw in ((SC.SCEM, {}), (SC.SCEM, {"hazard": True}), (stage_b.configured_scem(), {}),
                    (stage_b.configured_scem(witness_weights=None, phi_fixed=None, hazard=True, N_stages=2), {})):
        z1, a1, s1, g1 = run_with_grads(cls, mode, **kw)
        assert stage_b.same_parameters(s0, s1) and torch.equal(z0, z1) and a0 == a1 and same_grads(g0, g1)


@pytest.mark.filterwarnings(r"ignore:`torch\.(cpu|cuda)\.amp\.autocast:FutureWarning")  # inside torch.utils.checkpoint
def test_archived_localgdte_equals_the_engine_before_e2(engine_without_e2, monkeypatch):
    if C.DEV.type == "cpu":
        monkeypatch.setenv(C.ALLOW_CPU_ENV, "1")
    out = stage_b.forward_check("bitcoinalpha", 0, C.DEV.type, original_scem=engine_without_e2.SCEM)
    assert out["identical_initial_parameters"] and out["embeddings_equal"]
    assert out["max_abs_diff_embedding"] == 0.0 and out["aux_loss_original"] == out["aux_loss_patched"]


# ------------------------------------------------------------------ 2. hazard off
@pytest.mark.parametrize("mode", PF.MODES)
def test_hazard_off_is_lambda_max_zero_without_nan_gradients(mode):
    z_off, a_off, _, g_off = run_with_grads(SC.SCEM, mode, hazard=False)
    z_zero, a_zero, _, g_zero = run_with_grads(SC.SCEM, mode, lambda_max=0.0)
    assert torch.equal(z_off, z_zero) and a_off == a_zero                 # the same forward pass as lambda_max = 0
    assert all(torch.isfinite(g).all() for g in g_off.values() if g is not None)
    assert g_off["gamma_raw"] is None                                     # gamma only scales the hazard
    if mode == "graph_stream":                                            # lambda_max = 0 divides 0 by 0 backwards
        assert not all(torch.isfinite(g).all() for g in g_zero.values() if g is not None)
    z_on = run_with_grads(SC.SCEM, mode)[0]
    assert not torch.equal(z_off, z_on)                                   # the option reaches the computation


def test_hazard_off_returns_a_zero_hazard():
    module = PF.build(SC.SCEM, hazard=False)
    nu, lam = module._compute_hazard(torch.rand(5, module.d_xi), torch.rand(5, 2))
    assert torch.equal(lam, torch.zeros_like(nu)) and (nu > 0).all()


# ------------------------------------------------------------------ 3. the combined factory
def test_configured_scem_combines_every_option():
    x, edges, labels, ends = PF.module_inputs()
    w = torch.rand(PF.K, x.shape[0])
    options = dict(phi_fixed=0.25, N_stages=1, hazard=False)
    combined = PF.build(stage_b.configured_scem(witness_weights=w, **options))
    assert (combined.N, combined.phi_fixed, combined.hazard) == (1, 0.25, False)
    plain = PF.build(SC.SCEM, **options)
    assert torch.equal(combined(x, edges, labels, ends), plain(x, edges, labels, ends, witness_weights=w))
    assert not torch.equal(combined(x, edges, labels, ends), plain(x, edges, labels, ends))


def test_witness_weighted_scem_still_needs_weights():
    with pytest.raises(ValueError, match="needs weights"):
        stage_b.witness_weighted_scem(None)


# ------------------------------------------------------------------ 4. the E2 models
@pytest.fixture(scope="module")
def alpha():
    D = C.load("bitcoinalpha")
    return D, C.train_split(C.full_graph(D), D.tt)


@pytest.mark.parametrize("model", sorted(DRV.SCEM_MODELS))
def test_every_scem_model_is_built_with_its_settings(alpha, model, monkeypatch):
    if C.DEV.type == "cpu":
        monkeypatch.setenv(C.ALLOW_CPU_ENV, "1")
    D, Tr = alpha
    pipeline_class = C.LocalGDTE
    weights = np.full((D.tt, D.n), 0.5)
    m, settings = DRV.build_model(D, Tr, 5, model, weights)
    assert C.LocalGDTE is pipeline_class and isinstance(m.scem, SC.SCEM) and m.s3r is m.scem
    want = DRV.SCEM_MODELS[model][1]
    assert (settings["N"], settings["phi_fixed"], settings["hazard"]) == (want["N"], want["phi_fixed"], want["hazard"])
    assert settings["horizon_A"] == SC.erlang_horizon_A(want["N"], want["horizon_phi"]) == m.scem.horizon_A
    assert m.scem.phi_raw.requires_grad == (want["phi_fixed"] is None)


def test_s3r_is_the_default_backbone(alpha, monkeypatch):
    if C.DEV.type == "cpu":
        monkeypatch.setenv(C.ALLOW_CPU_ENV, "1")
    D, Tr = alpha
    m, settings = DRV.build_model(D, Tr, 5, "s3r")
    assert settings is None and type(m) is C.LocalGDTE and not hasattr(m, "scem")


def test_a_swallowed_option_is_caught():
    default = DRV.scem_settings(PF.build(SC.SCEM))
    DRV.check_settings("scem", default)
    for model in ("scem_nohaz", "scem_n1", "scem_phi05"):
        with pytest.raises(RuntimeError, match=model):
            DRV.check_settings(model, default)


# ------------------------------------------------------------------ 5. the driver with stub models
NONDETERMINISTIC = "index_add_cuda_ does not have a deterministic implementation"


@pytest.fixture
def stub_driver(monkeypatch, tmp_path):
    """run() on a synthetic network named bitcoinalpha; models are StubModel; returns the build calls."""
    G = TS.synthetic_graph()
    D = C.Dataset("bitcoinalpha", G.edges, G.labels, G.time, G.n, TS.T_TRAIN + 1, {})
    calls = []

    def build_model(D_, Tr, seed, model, witness_weights=None):
        calls.append(dict(n=int(Tr.n), seed=seed, model=model, Tr=Tr,
                          weights=None if witness_weights is None else np.asarray(witness_weights)))
        m = StubModel(Tr, np.zeros(Tr.n, dtype=np.int64), seed, model)
        m.scem = SimpleNamespace(phi=torch.tensor(0.3))
        return m, None if model == "s3r" else dict(stub=True)

    monkeypatch.setattr(C, "load", lambda ds: D)
    monkeypatch.setattr(C, "full_graph", lambda D_: G)
    monkeypatch.setattr(C, "apply_memory_profile", lambda D_: {"stub": True})
    monkeypatch.setattr(C, "require_gpu", lambda: None)
    monkeypatch.setattr(C, "set_deterministic", lambda on=True: None)
    def train(m, Tr, tt, epochs=150):
        warnings.warn(NONDETERMINISTIC, UserWarning)
        warnings.warn("an unrelated warning", UserWarning)
        return m

    monkeypatch.setattr(C, "train", train)
    monkeypatch.setattr(C, "embed", lambda m, Tr, tt: m.z)
    monkeypatch.setattr(DRV, "build_model", build_model)
    monkeypatch.delenv(C.OVERWRITE_ENV, raising=False)
    prereg = tmp_path / "prereg.md"
    prereg.write_text("protocol\n")
    return SimpleNamespace(calls=calls, D=D, G=G, prereg=str(prereg), out=tmp_path)


def run_stub(stub, stream, model, seed=5):
    path = stub.out / f"e2_{model}_{stream}_s{seed}.json"
    DRV.run("bitcoinalpha", seed, stream, model, 2, str(path), stub.prereg)
    return json.loads(path.read_text())


def test_clean_twin_and_attack_run_share_the_model_seed_plan_and_panel(stub_driver):
    clean, attack = run_stub(stub_driver, "clean", "scem"), run_stub(stub_driver, "lag1", "scem")
    c0, c1 = stub_driver.calls
    assert c0["seed"] == c1["seed"] == clean["model_seed"] == attack["model_seed"] == 5
    assert (c0["n"], c1["n"]) == (TS.N_NODES, TS.N_NODES + TS.N_SENDERS)
    for field in ("targets", "kind", "direction", "plan", "panel", "epochs"):
        assert clean[field] == attack[field], field
    assert attack["stream_info"]["n_injected"] == 3 * TS.N_SENDERS and clean["stream_info"]["n_injected"] == 0
    assert attack["lag"] == 1 and clean["lag"] is None
    assert clean["proxy"] != attack["proxy"]                      # the stub's scores follow its training graph


def test_scem_models_get_the_mb_weights_of_their_own_stream(stub_driver):
    run_stub(stub_driver, "clean", "scem_nohaz")
    run_stub(stub_driver, "lag3", "scem_nohaz")
    run_stub(stub_driver, "lag3", "s3r")
    clean_call, attack_call, s3r_call = stub_driver.calls
    Tr = C.train_split(stub_driver.G, TS.T_TRAIN)
    assert np.array_equal(clean_call["weights"], level_weights(E.stream_levels(Tr, Tr, 0.02, TS.T_TRAIN)))
    want = level_weights(E.stream_levels(attack_call["Tr"], Tr, 0.02, TS.T_TRAIN))
    assert attack_call["weights"].shape == (TS.T_TRAIN, TS.N_NODES + TS.N_SENDERS)
    assert np.array_equal(attack_call["weights"], want)
    assert s3r_call["weights"] is None


def test_the_record_carries_the_protocol_the_rules_and_the_git_state(stub_driver):
    r = run_stub(stub_driver, "lag1", "scem")
    s3r = run_stub(stub_driver, "lag1", "s3r")
    sha = hashlib.sha256(open(stub_driver.prereg, "rb").read()).hexdigest()
    assert r["prereg"]["sha256"] == sha and "git_commit" in r["provenance"] and "git_dirty" in r["provenance"]
    assert (r["mb"]["eps"], r["mb"]["edge_cap_units"], r["mb"]["neg_from"]) == (0.02, 2.0, "trusted")
    assert (r["plan"]["eps"], r["plan"]["edge_cap_units"], r["plan"]["neg_from"]) == (0.02, 2.0, "trusted")
    assert np.min(r["mb"]["witness_levels"]) >= 2 and r["mb"]["honest_levels_changed"] == 0
    assert r["learned_phi"] == pytest.approx(0.3) and s3r["learned_phi"] is None and s3r["mb"] is None
    assert len(r["percentile"]) == 30 and all(0.0 <= p <= 1.0 for p in r["percentile"])
    assert r["determinism_warnings"] == [NONDETERMINISTIC]


def test_an_existing_output_is_refused_before_any_work(stub_driver, capsys):
    out = stub_driver.out / "taken.json"
    out.write_text("stored")
    with pytest.raises(SystemExit) as stop:
        DRV.main(["--ds", "bitcoinalpha", "--seed", "5", "--stream", "clean", "--model", "scem", "--prereg",
                  stub_driver.prereg, "--out", str(out)])
    assert stop.value.code == 2 and "exists" in capsys.readouterr().err
    assert out.read_text() == "stored" and not stub_driver.calls


# ------------------------------------------------------------------ 6. the task list
RESULTS = "/data/keeper"
LAUNCH = "env -u PYTHONPATH -u VIRTUAL_ENV ~/.conda/envs/ps/bin/python -m keeper.scem.e2_driver "


def test_the_task_list_holds_the_registered_runs_and_commands_the_driver_parses():
    rows = e2_tasks.e2_tasks(RESULTS)
    assert len(rows) == 315 and len({r.id for r in rows}) == 315
    assert [r.id.split("_")[-2] for r in rows][::105] == ["wiki-rfa", "bitcoinotc", "bitcoinalpha"]
    per_seed = Counter((r.id.rsplit("_", 2)[1], int(r.id.rsplit("_s", 1)[1])) for r in rows)
    assert all(per_seed[(ds, s)] == (15 if s < 10 else 6) for ds in ("wiki-rfa", "bitcoinotc", "bitcoinalpha")
               for s in range(5, 15))
    for r in rows:
        assert r.done_path == f"{RESULTS}/results/scem/e2/{r.id}.json"
        assert r.log_path == f"{RESULTS}/results/scem/e2/logs/{r.id}.log"
        assert r.command.startswith(LAUNCH)
        a = DRV.build_parser().parse_args(shlex.split(r.command[len(LAUNCH):]))
        assert e2_tasks.task_id(a.ds, a.seed, a.stream, a.model) == r.id
        assert (a.epochs, a.prereg, a.out, a.force) == (150, "docs/preregistration.md", r.done_path, False)
        if a.seed >= 10:
            assert a.stream in ("clean", "lag1") and a.model in ("s3r", "scem", "scem_nohaz")


def test_the_task_list_refuses_a_relative_or_worktree_root(tmp_path):
    (tmp_path / ".git").write_text("gitdir: elsewhere\n")
    for root in ("relative/root", str(tmp_path)):
        with pytest.raises(ValueError):
            e2_tasks.e2_tasks(root)


def test_the_task_list_cli_refuses_to_overwrite(tmp_path, capsys):
    out = tmp_path / "e2.tsv"
    argv = ["--results-root", RESULTS, "--out", str(out)]
    assert e2_tasks.main(argv) == 0 and len(out.read_text().splitlines()) == 315
    assert all(len(line.split("\t")) == 4 for line in out.read_text().splitlines())
    assert e2_tasks.main(argv) == 1 and "exists" in capsys.readouterr().err
    assert e2_tasks.main(argv + ["--force"]) == 0
