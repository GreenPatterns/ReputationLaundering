"""Task 3.4, upgrade G5 (approved 2026-10-26): the SCEM module with a fixed clock rate phi.

SCEM(phi_fixed=x) uses phi = x, a constant without gradient, and the forced-close age A = erlang_horizon_A(N, x):
13.28 snapshots at N = 2 and x = 0.5, so that records commit inside the 21 training snapshots (experiment E2).
keeper.scem.stage_b.scem_localgdte(stage_b.fixed_phi_scem(0.5)) builds the archived LocalGDTE with it.

1. phi_fixed=None, the default, leaves the module unchanged bit for bit in both modes (graph stream and sequence):
   its parameters, output and auxiliary loss equal those of the engine with the G5 lines reverted
   (tests/test_scem_equivalence.G5_EDITS, which the source pin also checks).
2. With phi_fixed, phi is that constant, phi_raw gets no gradient and A follows phi_fixed; otherwise the module
   computes what the learned module computes at the same phi and A.
3. phi_fixed must lie in the learned range [phi_min, phi_max], bounds included.
4. fixed_phi_scem reaches the archived LocalGDTE (make_model and forward_check).
CPU only.
"""
import math

import pytest
import torch

from keeper.scem import stage_b  # first: it sets CUBLAS_WORKSPACE_CONFIG before CUDA starts
from keeper.laundering import core as C
from keeper.scem import scem as SC
from tests import test_scem_equivalence as EQ

MODES = ("graph_stream", "sequence")
K = 16                       # snapshots: more than A = 13.28 at phi = 0.5, so that records close inside the stream


def module_inputs(n=10, d=16, per_snapshot=8, seed=0):
    """Node features [n, K, d] and a rating stream of `per_snapshot` ratings per snapshot (30% negative)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, K, d, generator=gen)
    u = torch.randint(0, n, (K * per_snapshot,), generator=gen)
    v = (u + torch.randint(1, n, (K * per_snapshot,), generator=gen)) % n
    neg = torch.rand(K * per_snapshot, generator=gen) < 0.3
    labels = torch.stack([(~neg).float(), neg.float()], dim=1)
    ends = [(t + 1) * per_snapshot - 1 for t in range(K)]
    return x, torch.stack([u, v]), labels, ends


def forward(module, mode):
    x, edges, labels, ends = module_inputs()
    z = module(x, edges, labels, ends) if mode == "graph_stream" else module(x)
    return z, module.last_aux_loss


def build(scem_cls, **kw):
    torch.manual_seed(0)
    return scem_cls(input_dim=16, max_seq_len=K, **kw)


def run(scem_cls, mode, **kw):
    """A module built from seed 0, and its forward pass: (output, auxiliary loss, state dict)."""
    module = build(scem_cls, **kw)
    z, aux = forward(module, mode)
    return z.detach().clone(), float(aux), module.state_dict()


@pytest.fixture(scope="module")
def engine_without_g5(tmp_path_factory):
    """keeper/scem/scem.py with the G5 lines reverted: the module as it was before Task 3.4."""
    source = EQ.ENGINE.read_text()
    for old, new in EQ.G5_EDITS:
        assert source.count(new) == 1 and old not in source
        source = source.replace(new, old)
    assert "phi_fixed" not in source
    path = tmp_path_factory.mktemp("g5") / "scem_without_g5.py"
    path.write_text(source)
    module = stage_b.load_original_module(str(path))
    assert module.SCEM is not SC.SCEM
    return module


# ------------------------------------------------------------------ 1. None: the module unchanged
@pytest.mark.parametrize("mode", MODES)
def test_phi_fixed_none_leaves_the_module_unchanged(engine_without_g5, mode):
    z0, a0, s0 = run(engine_without_g5.SCEM, mode)
    for kw in ({}, {"phi_fixed": None}):
        z1, a1, s1 = run(SC.SCEM, mode, **kw)
        assert stage_b.same_parameters(s0, s1) and torch.equal(z0, z1) and a0 == a1


# ------------------------------------------------------------------ 2. a fixed rate
def test_phi_fixed_is_a_constant_rate_and_sets_the_horizon():
    fixed = build(SC.SCEM, phi_fixed=0.5)
    assert fixed.phi_fixed == 0.5 and float(fixed.phi) == 0.5
    assert not fixed.phi.requires_grad and not fixed.phi_raw.requires_grad
    assert fixed.horizon_A == SC.erlang_horizon_A(2, 0.5) and abs(fixed.horizon_A - 13.2767) < 1e-3
    learned = build(SC.SCEM)
    assert learned.phi_fixed is None and learned.phi_raw.requires_grad and learned.phi.requires_grad
    assert learned.horizon_A == SC.erlang_horizon_A(2, learned.phi_min)             # 53.1 snapshots, as before


@pytest.mark.parametrize("mode", MODES)
def test_phi_raw_gets_a_gradient_only_when_phi_is_learned(mode):
    for phi_fixed in (None, 0.5):
        module = build(SC.SCEM, phi_fixed=phi_fixed)
        z, aux = forward(module, mode)
        (z.sum() + aux).backward()
        assert (module.phi_raw.grad is not None) == (phi_fixed is None)
        assert module.M_theta.weight.grad is not None


@pytest.mark.parametrize("mode", MODES)
def test_fixed_phi_computes_the_learned_module_at_that_phi_and_horizon(mode):
    phi = 0.45                       # inside (phi_min, phi_max) = (0.125, 0.5); A = 14.75 < K, so records close
    z_fixed, a_fixed, _ = run(SC.SCEM, mode, phi_fixed=phi)
    learned = build(SC.SCEM)
    share = (phi - learned.phi_min) / (learned.phi_max - learned.phi_min)
    with torch.no_grad():
        learned.phi_raw.fill_(math.log(share / (1.0 - share)))
    learned.horizon_A = SC.erlang_horizon_A(learned.N, phi, q=learned.q_quantile)
    assert abs(float(learned.phi) - phi) < 1e-6 and learned.horizon_A < K
    z, aux = forward(learned, mode)
    assert torch.allclose(z_fixed, z.detach(), rtol=1e-4, atol=1e-5)
    assert math.isclose(a_fixed, float(aux), rel_tol=1e-4, abs_tol=1e-6)
    assert not torch.equal(z_fixed, run(SC.SCEM, mode)[0])        # the option reaches the computation


# ------------------------------------------------------------------ 3. validation: the learned range
@pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf"), 0.1, 0.6])
def test_phi_fixed_must_lie_in_the_learned_range(bad):
    with pytest.raises(ValueError, match=r"learned range \[phi_min, phi_max\] = \[0\.125, 0\.5\]"):
        build(SC.SCEM, phi_fixed=bad)                  # at the module defaults phi_min = 0.125, phi_max = 0.5


def test_phi_fixed_accepts_the_learned_range_bounds_included():
    for phi in (0.125, 0.3, 0.5):                      # phi_min, inside, and phi_max (E2's value)
        assert build(SC.SCEM, phi_fixed=phi).phi_fixed == phi
    three = build(SC.SCEM, N_stages=3, phi_fixed=0.75)             # phi_max = N / (r L_f) = 3 / 4
    assert three.phi_max == 0.75 and three.horizon_A == SC.erlang_horizon_A(3, 0.75)
    with pytest.raises(ValueError, match="learned range"):
        build(SC.SCEM, N_stages=3, phi_fixed=0.8)


# ------------------------------------------------------------------ 4. the E2 model
@pytest.fixture
def pipeline_device(monkeypatch):
    """The device keeper.laundering.core chose; without a GPU the project's override allows the CPU."""
    if C.DEV.type == "cpu":
        monkeypatch.setenv(C.ALLOW_CPU_ENV, "1")
    return C.DEV.type


@pytest.mark.parametrize("mode", MODES)
def test_fixed_phi_scem_none_is_the_module(mode):
    z0, a0, s0 = run(SC.SCEM, mode)
    z1, a1, s1 = run(stage_b.fixed_phi_scem(None), mode)
    assert stage_b.same_parameters(s0, s1) and torch.equal(z0, z1) and a0 == a1


def test_fixed_phi_scem_builds_the_e2_localgdte(pipeline_device, monkeypatch):
    D, _, Tr, _ = stage_b.load_stream("bitcoinalpha")
    monkeypatch.setattr(C, "LocalGDTE", stage_b.scem_localgdte(stage_b.fixed_phi_scem(0.5)))
    m = C.make_model(D, Tr, 0, D.tt)
    assert isinstance(m.scem, SC.SCEM) and m.s3r is m.scem
    assert m.scem.phi_fixed == 0.5 and float(m.scem.phi) == 0.5 and not m.scem.phi_raw.requires_grad
    assert m.scem.horizon_A == SC.erlang_horizon_A(2, 0.5) < D.tt          # records commit inside training


@pytest.mark.filterwarnings(r"ignore:`torch\.(cpu|cuda)\.amp\.autocast:FutureWarning")  # inside torch.utils.checkpoint
def test_fixed_phi_changes_the_archived_localgdte_forward_pass(pipeline_device):
    out = stage_b.forward_check("bitcoinalpha", 0, pipeline_device, original_scem=stage_b.fixed_phi_scem(0.5))
    assert out["identical_initial_parameters"] and not out["embeddings_equal"]
