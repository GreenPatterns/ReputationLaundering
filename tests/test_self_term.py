"""Task 3.4, upgrade G4 (approved 2026-10-26): the self-term ablation flag of LocalGDTE's structure layer.

With self_term=True, each node's incoming aggregation in keeper.backbone.structure_layer.Convolution also holds one
self loop that carries the node's own feature and a zero opinion vector; the mean divides by the in-degree plus one.
The flag reaches every layer from the model arguments (args.self_term), which keeper.laundering.core.make_model(...,
self_term=True) sets. The loops are added to whatever graph the layer runs on, so they act in the inference graph
after U1 as well (experiment E9a).

1. Off (the default, or self_term=False), the layer and a LocalGDTE forward pass equal the original structure layer
   bit for bit (torch.equal on the CPU). The original is thecode/structure_layer.py at tag pre-revision-2026-10-26;
   before Task 3.4 it differed from keeper/backbone/structure_layer.py only in its two import lines.
2. On, the layer's output is the hand-computed one, and edgeless nodes with different features get different
   representations, where the original layer gives every edgeless node relu(bias).
3. A rebuild through make_model inherits the source model's flag: exact retraining (single and batched), VPE
   retraining, and probe4's rebuilds. Every call of the rebuild helpers passes self_term. CPPR's credibility gate
   refuses a model with the flag on.
CPU only.
"""
import ast
import importlib.machinery
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from keeper.backbone import local_model
from keeper.backbone import structure_layer as SL
from keeper.cppr import cp_vpe_gpu as CPV
from keeper.cppr import vpe as V
from keeper.laundering import core as C
from keeper.laundering import probe4 as P4
from keeper.laundering import unlearn as U

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = "pre-revision-2026-10-26:thecode/structure_layer.py"
MOVED_IMPORTS = (("from thecode.robust import *", "from keeper.backbone.robust import *"),   # commit 209d24e
                 ("from thecode.utils import *", "from keeper.backbone.utils import *"))
SEED = 5                                                     # the first of E9a's seeds
N_NODES, DIM, OUT = 7, 8, 6


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    """The original structure layer as a stand-alone module, its imports moved to the keeper package."""
    try:
        source = subprocess.run(["git", "-C", str(ROOT), "show", ORIGINAL],
                                check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):   # no git, a shallow clone or a missing tag
        pytest.fail("tag pre-revision-2026-10-26 not found; fetch tags")
    for old, new in MOVED_IMPORTS:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    path = tmp_path_factory.mktemp("original") / "structure_layer_original.py"
    path.write_text(source)
    loader = importlib.machinery.SourceFileLoader("structure_layer_original", str(path))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    assert module.Convolution is not SL.Convolution
    return module


@pytest.fixture(scope="module")
def alpha():
    D = C.load("bitcoinalpha")
    G = C.full_graph(D)
    return D, G, C.train_split(G, D.tt)


@pytest.fixture
def on_cpu(monkeypatch):
    """keeper.laundering.core builds its models on the CPU, where the forward pass is deterministic."""
    monkeypatch.setattr(C, "DEV", torch.device("cpu"))
    monkeypatch.setenv(C.ALLOW_CPU_ENV, "1")


def graph(empty=False):
    """Ratings among 7 nodes: 0-3 rate each other, 4 rates 0, and 5 and 6 have no edge. empty: a snapshot
    without ratings."""
    if empty:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 2)
    edges = torch.tensor([[0, 1, 2, 3, 1, 4], [1, 2, 0, 0, 3, 0]])
    labels = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [1., 0.], [0., 1.], [1., 0.]])
    return edges, labels


def features():
    return torch.randn(N_NODES, DIM, generator=torch.Generator().manual_seed(1))


def make_layer(module, **kw):
    torch.manual_seed(0)
    return module.Convolution(DIM, OUT, num_labels=2, prune=0.4, device="cpu", **kw)


def same_state(m0, m1):
    s0, s1 = m0.state_dict(), m1.state_dict()
    return bool(s0) and set(s0) == set(s1) and all(torch.equal(s0[k], s1[k]) for k in s0)


def forward_and_grads(layer, x, edges, labels):
    """The output, and the gradients of its sum with respect to the input features and the parameters."""
    x = x.clone().requires_grad_(True)
    out = layer(x, edges, labels)
    out.sum().backward()
    return [out.detach(), x.grad] + [p.grad for _, p in sorted(layer.named_parameters())]


# ------------------------------------------------------------------ 1. off: the original layer
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("kw", [{}, {"self_term": False}])
def test_layer_without_self_term_is_the_original_layer(original, kw, empty):
    old, new = make_layer(original), make_layer(SL, **kw)
    assert same_state(old, new)
    edges, labels = graph(empty)
    got = forward_and_grads(new, features(), edges, labels)
    want = forward_and_grads(old, features(), edges, labels)
    assert len(got) == len(want)
    assert all((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
               for a, b in zip(got, want))


def embedding(D, Tr, **kw):
    m = C.make_model(D, Tr, SEED, D.tt, **kw)
    return m, C.embed(m, Tr, D.tt)


def test_localgdte_without_self_term_is_the_original(original, alpha, on_cpu, monkeypatch):
    D, _, Tr = alpha
    with monkeypatch.context() as patch:                       # the reference arm runs the original layer
        patch.setattr(local_model, "StructuralNetwork", original.StructuralNetwork)
        m0, z0 = embedding(D, Tr)
    assert isinstance(m0.structural_layer, original.StructuralNetwork)
    for kw in ({}, {"self_term": False}):
        m1, z1 = embedding(D, Tr, **kw)
        assert not isinstance(m1.structural_layer, original.StructuralNetwork)
        assert same_state(m0, m1) and torch.equal(z0, z1)


# ------------------------------------------------------------------ 2. on: the self loop
def by_hand(layer, x, edges, labels):
    """The layer's output with the self term, from its definition: loops over the nodes, no scatter."""
    label_mu = layer.transformation_gaussian(labels)
    row, col = edges
    parts = []
    for v in range(x.shape[0]):
        out_e, in_e = (row == v).nonzero().flatten(), (col == v).nonzero().flatten()
        out_mu = x[col[out_e]].mean(0) if len(out_e) else torch.zeros(x.shape[1])
        opinion = label_mu[out_e].mean(0) if len(out_e) else torch.zeros(label_mu.shape[1])
        inn = (x[row[in_e]].sum(0) + x[v]) / (len(in_e) + 1)            # the loop carries x[v] ...
        inn_opinion = label_mu[in_e].sum(0) / (len(in_e) + 1)           # ... and a zero opinion vector
        parts.append(torch.cat([out_mu, opinion, inn, inn_opinion]))
    return torch.relu(torch.stack(parts) @ layer.weight.T + layer.bias)


@pytest.mark.parametrize("empty", [False, True])
def test_self_term_adds_one_self_loop_to_the_incoming_mean(empty):
    layer = make_layer(SL, self_term=True)
    assert layer.self_term is True
    edges, labels = graph(empty)
    x = features()
    with torch.no_grad():
        assert torch.allclose(layer(x, edges, labels), by_hand(layer, x, edges, labels), rtol=0, atol=1e-6)


def test_edgeless_nodes_get_distinct_representations_only_with_the_self_term():
    edges, labels = graph()
    x = features()
    off_layer = make_layer(SL)
    with torch.no_grad():
        off = off_layer(x, edges, labels)
        on = make_layer(SL, self_term=True)(x, edges, labels)
    assert torch.equal(off[5], off[6]) and torch.equal(off[5], torch.relu(off_layer.bias))   # whatever x[5], x[6]
    assert (on[5] - on[6]).abs().max() > 1e-3


def test_make_model_sets_self_term_on_every_structure_layer(alpha, on_cpu):
    D, _, Tr = alpha
    on = C.make_model(D, Tr, SEED, D.tt, self_term=True)
    off = C.make_model(D, Tr, SEED, D.tt)
    assert on.args.self_term is True and off.args.self_term is False
    assert len(on.structural_layer.conv_layers) == len(D.cfg["layers"])
    assert all(conv.self_term for conv in on.structural_layer.conv_layers)
    assert not any(conv.self_term for conv in off.structural_layer.conv_layers)


def edgeless_after_u1(Tr, count=2):
    """The first cohort targets that U1 (all distrust in-edges deleted) leaves without any edge, and that
    forget mask."""
    cohort, neg, pos = C.pick_targets(Tr, max_neg=3, strict=True)
    raters = set(Tr.edges[:, 0].tolist())
    chosen = [int(v) for v in cohort if not pos.get(int(v)) and int(v) not in raters][:count]
    assert len(chosen) == count
    mask = np.zeros(len(Tr.edges), bool)
    for v in chosen:
        mask[neg[v]] = True
    return chosen, mask


def test_self_term_keeps_edgeless_targets_apart_in_the_u1_inference_graph(alpha, on_cpu):
    D, G, Tr = alpha
    (a, b), mask = edgeless_after_u1(Tr)
    z = {}
    for flag in (False, True):
        m = C.make_model(D, Tr, SEED, D.tt, self_term=flag)
        z[flag] = U.unlink(m, G, Tr, mask, D.tt).embed(D.tt)    # U1 without retraining, as E9a runs it
    assert torch.allclose(z[False][a], z[False][b], rtol=0, atol=1e-6)    # the edgeless collapse
    assert (z[True][a] - z[True][b]).abs().max() > 1e-3                   # each target keeps its own feature


# ------------------------------------------------------------------ 3. rebuilds keep the flag
REBUILDS = ("exact_retrain", "batched_exact_retrain", "retrain_on")    # rebuild a model through make_model


def self_terms(m):
    return {conv.self_term for conv in m.structural_layer.conv_layers}


def test_rebuild_helpers_build_with_the_flag_they_are_given(alpha, on_cpu):
    D, G, Tr = alpha
    _, neg, _ = C.pick_targets(Tr, max_neg=3, strict=True)
    (a, b), _ = edgeless_after_u1(Tr)
    masks = {v: np.isin(np.arange(len(Tr.edges)), neg[v]) for v in (a, b)}
    for flag in (False, True):                               # epochs=0: the rebuilt model, untrained
        u = U.exact_retrain(D, G, D.tt, Tr.rid[masks[a]], SEED, 0, self_term=flag)
        assert u.model.args.self_term is flag and self_terms(u.model) == {flag}
        batched = list(U.batched_exact_retrain(D, G, Tr, D.tt, [a, b], masks, SEED, 0, len(D.cfg["layers"]),
                                               self_term=flag))
        assert batched and all(self_terms(ub.model) == {flag} for _, ub in batched)
        assert self_terms(V.retrain_on(SimpleNamespace(D=D, Tr=Tr), SEED, D.tt, 0, self_term=flag)) == {flag}
    assert self_terms(U.exact_retrain(D, G, D.tt, Tr.rid[masks[a]], SEED, 0).model) == {False}   # the default


class Stop(Exception):
    pass


def test_probe4_rebuilds_inherit_the_flag_of_the_source_model(alpha, on_cpu, monkeypatch):
    D, G, Tr = alpha
    _, mask = edgeless_after_u1(Tr)
    cx = SimpleNamespace(a=SimpleNamespace(seed=SEED, epochs=0), D=D, G=G, Tr=Tr, tt=D.tt,
                         cohort=np.array([], dtype=np.int64), hops=len(D.cfg["layers"]))
    seen = {}

    def batched(*args, **kw):                   # stands in for the cohort retrain: records its arguments, then stops
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(U, "batched_exact_retrain", batched)
    for flag in (False, True):
        m0 = C.make_model(D, Tr, SEED, D.tt, self_term=flag)
        assert C.self_term_of(m0) is flag
        assert self_terms(P4._run_operator_once(cx, m0, "exact_retrain", mask).model) == {flag}
        assert self_terms(P4.identical_retrain(cx, m0)) == {flag}                # the same-shape noise floor
        with pytest.raises(Stop):
            P4.full_cohort_h1(cx, m0, base={})
        assert seen.pop("self_term") is flag


def test_every_rebuild_call_in_the_package_passes_self_term():
    missing = []
    for path in sorted((ROOT / "keeper").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if name in REBUILDS and not any(k.arg == "self_term" for k in node.keywords):
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}")
    assert not missing, f"a rebuild drops the source model's self_term: {missing}"


def test_the_credibility_gate_refuses_a_model_with_self_term(alpha, on_cpu):
    D, _, Tr = alpha
    on = C.make_model(D, Tr, SEED, D.tt, self_term=True)
    s3r = on.s3r
    with pytest.raises(ValueError, match="self_term"):
        CPV.attach_scem_and_credibility_gate(on, np.zeros(D.n, dtype=np.int64))
    assert on.s3r is s3r                                               # refused before any change to the model
    off = C.make_model(D, Tr, SEED, D.tt)
    assert isinstance(CPV.attach_scem_and_credibility_gate(off, np.zeros(D.n, dtype=np.int64)).s3r, CPV.SCEM)
