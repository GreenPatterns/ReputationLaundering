"""Regression tests for the laundering core. Run before trusting any number.

Covers the audit's recommended tests:
  T1 preprocessing invariance  - mutating held-out edges cannot change training
                                 features, class weights, or initialisation
  T2 exact-retrain equivalence- deletion retraining == a fresh retained dataset
  T3 snapshot integrity       - no boundary splits a tied timestamp; empty
                                 snapshots stay empty
  T4 head registration        - the output head is optimised and serialised,
                                 and the legacy arm reproduces the bug
  T5 decoder additivity       - log-odds decompose into source and target terms
  T6 alignment                - rating/CSV row-pairing gives 0 sign mismatches
  T7 decision threshold       - distrust predicted when p < tau (not p >= tau)
  T8 NegGrad+                 - a post-process on theta* that forgets F and
                                 preserves retain utility
  T9 indirect channel C3      - candidates never touch an edge incident to T
  T10 panel batch independence- a node's score does not depend on its company
  T12 signed loader           - sign is the label, zeros/self-loops dropped and
                                 counted, legacy Bitcoin labels reproduced exactly
  T13 GPU + provenance        - models refuse CPU; results carry provenance;
                                 T1/T2 utility masks match the framework definition

Run from anywhere:  <ps python> -m laundering.test_core   (or the file path).
It runs in the SAME deterministic configuration as the probes.
"""
import copy
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before torch/CUDA init

import numpy as np  # noqa: E402
import torch  # noqa: E402


from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import signed as S  # noqa: E402
from keeper.laundering.threshold import held_out                    # noqa: E402

C.set_deterministic(True)
# Float32 round-off differs with GEMM shape on some GPUs (5.96e-08 measured on
# L40S). The D6 bug this suite guards against was ~1e-1, so 1e-6 cannot hide it.
F32_TOL = 1e-6
FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILED.append(name)


def synth(n=60, T=6, seed=0):
    """Small synthetic Dataset with tied timestamps."""
    rng = np.random.default_rng(seed)
    e = rng.integers(0, n, size=(300, 2))
    e = e[e[:, 0] != e[:, 1]]
    times = rng.integers(0, 5, size=len(e)) * 3
    lab = np.zeros((len(e), 2), np.float32)
    lab[rng.random(len(e)) < 0.4] = (0, 1)
    cfg = dict(n_heads=2, layers=[8, 8, 8])
    return C.Dataset("synth", e, lab, times, n, T, cfg)


# --------------------------------------------------------------------- T3 snapshots
print("\nT3  snapshot integrity")
ds = synth()
ends = C.snapshot_ends(ds.time, ds.T, tie_avoiding=True)
check("ends are nondecreasing", all(ends[i] <= ends[i + 1] for i in range(len(ends) - 1)),
      str(ends))
splits = sum(1 for i in range(len(ends) - 1) if 0 <= ends[i] < len(ds.time) - 1
             and ds.time[ends[i] + 1] == ds.time[ends[i]])
check("no boundary splits a tied timestamp", splits == 0, f"{splits} splits")

# a leading empty snapshot must yield end == -1, never the old clamp to 0
g_lead = C.Graph(ds.edges[:0], ds.labels[:0], ds.time[:0],
                 np.zeros(0, np.int64), ds.n)
check("empty graph -> all ends -1", C.window(g_lead, 3)[2] == [-1, -1, -1],
      str(C.window(g_lead, 3)[2]))

# deleting a whole leading snapshot must not pull later edges into it
G = C.full_graph(ds)
keep = G.snap != 0
cut = C.train_split(G, ds.tt, keep)
et, lt, wends = C.window(cut, ds.tt)
check("removed leading snapshot -> end -1", wends[0] == -1, f"end[0]={wends[0]}")
check("later snapshots keep their edges",
      len(et[0]) == int((cut.snap < ds.tt).sum()), f"{len(et[0])} edges")

# ----------------------------------------------------------------------- T6 alignment
print("\nT6  data alignment")
for name in ["bitcoinalpha", "bitcoinotc"]:
    try:
        D = C.load(name, verify=True)
        check(f"{name} loads with verified alignment", True,
              f"E={len(D.edges):,} n={D.n} T={D.T}")
    except Exception as e:
        check(f"{name} loads with verified alignment", False, f"{type(e).__name__}: {e}")

# ------------------------------------------------------------ T1 preprocessing invariance
print("\nT1  preprocessing invariance (no test-period leakage)")
D = load_A = C.load("bitcoinalpha")
G = C.full_graph(D)
Tr = C.train_split(G, D.tt)
X1 = C.features(Tr, 0, D.cfg["layers"][-1])
w1 = C.class_weights(Tr)

# corrupt ONLY the held-out snapshot; retained-training preprocessing must not move
G2 = C.Graph(D.edges.copy(), D.labels.copy(), D.time.copy(), G.snap.copy(), D.n)
held = G2.snap == D.tt
G2.labels[held] = G2.labels[held][:, ::-1].copy()     # flip every held-out sign
G2.edges[held] = (G2.edges[held] + 7) % D.n
Tr2 = C.train_split(G2, D.tt)
X2 = C.features(Tr2, 0, D.cfg["layers"][-1])
w2 = C.class_weights(Tr2)
check("held-out label flip leaves features unchanged",
      bool(torch.equal(X1, X2)), f"max|dx|={float((X1 - X2).abs().max()):.3e}")
check("held-out label flip leaves class weights unchanged",
      bool(torch.equal(w1, w2)), f"{w1.tolist()} vs {w2.tolist()}")

# a full-graph build WOULD have differed -- confirm the leak was real
Xfull = C.features(G, 0, D.cfg["layers"][-1])
check("full-graph features differ from train-only (leak was real)",
      not bool(torch.equal(X1, Xfull)),
      f"max|dx|={float((X1 - Xfull).abs().max()):.3e}")

# ----------------------------------------------------------------- T2 exact-retrain equivalence
print("\nT2  exact-retraining equivalence")
neg, pos = C.in_edges(Tr)
v = int(np.array(sorted(neg))[0])
keep = np.ones(len(Tr.edges), bool)
keep[neg[v]] = False
cut = Tr.subset(keep)
fresh = C.Dataset(D.name, cut.edges, cut.labels, cut.time, cut.n, D.T, D.cfg)

# a retrain that regenerates everything from the retained data
seed_all = C.seed_all
C.seed_all(0)
Xa = C.features(cut, 0, D.cfg["layers"][-1])
wa = C.class_weights(cut)
m1 = C.make_model(D, cut, 0, D.tt, legacy_head=True)
sa = m1.state_dict()
C.seed_all(0)
Xb = C.features(fresh, 0, D.cfg["layers"][-1])
wb = C.class_weights(fresh)
m2 = C.make_model(D, fresh, 0, D.tt, legacy_head=True)
sb = m2.state_dict()
check("features identical for retained data", bool(torch.equal(Xa, Xb)))
check("class weights identical", bool(torch.equal(wa, wb)))
shared = set(sa) & set(sb)
check("initial parameters identical", all(torch.equal(sa[k], sb[k]) for k in shared),
      f"{len(shared)} tensors")
check("head absent from state_dict in both (D1 reproduced)",
      not any("regress" in k for k in sa))

# --------------------------------------------------------------------- T4 head registration
print("\nT4  head registration")
m_fix = C.make_model(D, Tr, 0, D.tt, legacy_head=False)
check("fixed head is a registered Parameter",
      isinstance(m_fix.regression_weights, torch.nn.Parameter))
check("fixed head in named_parameters()",
      "regression_weights" in dict(m_fix.named_parameters()))
check("fixed head in state_dict()",
      "regression_weights" in m_fix.state_dict())
m_leg = C.make_model(D, Tr, 0, D.tt, legacy_head=True)
check("legacy head is a plain Tensor (the bug)",
      not isinstance(m_leg.regression_weights, torch.nn.Parameter))

# the head must actually move under training
m_small = C.make_model(D, Tr, 0, D.tt, legacy_head=False)
before = m_small.regression_weights.detach().clone()
m_small = C.train(m_small, Tr, D.tt, epochs=2)
check("head changes after 2 epochs",
      float((m_small.regression_weights.detach() - before).norm()) > 0,
      f"|dW|={float((m_small.regression_weights.detach() - before).norm()):.4e}")

# the legacy head must NOT move
m_leg2 = C.make_model(D, Tr, 0, D.tt, legacy_head=True)
b2 = m_leg2.regression_weights.detach().clone()
m_leg2 = C.train(m_leg2, Tr, D.tt, epochs=2)
check("legacy head stays frozen after training",
      float((m_leg2.regression_weights.detach() - b2).norm()) == 0.0,
      f"|dW|={float((m_leg2.regression_weights.detach() - b2).norm()):.4e}")

# --------------------------------------------------------------------- T5 decoder additivity
print("\nT5  decoder additivity and panel isolation")
m = m_fix
z = C.embed(m, Tr, D.tt)
W = m.regression_weights.to(C.DEV)
rng = np.random.default_rng(0)
s = rng.integers(0, D.n, 200)
d = rng.integers(0, D.n, 200)
lg = C.logit_trust(m, z, s, d)
dim = z.shape[1]
a_vec, b_vec = W[:dim, 0] - W[:dim, 1], W[dim:, 0] - W[dim:, 1]
manual = (z[torch.as_tensor(s, device=C.DEV)] @ a_vec
          + z[torch.as_tensor(d, device=C.DEV)] @ b_vec)
check("log-odds = source term + target term",
      float((lg - manual).abs().max()) < 1e-4,
      f"max|err|={float((lg - manual).abs().max()):.3e}")

# moving only the sources must move the score -> the confound is real
z2 = z.clone()
z2[:] = z2 + 0.5
shifted = C.logit_trust(m, z2, s, d)
check("source-side shift alone changes the score (confound quantified)",
      float((shifted - lg).abs().max()) > 0.1,
      f"max|delta|={float((shifted - lg).abs().max()):.4f}")

panel = C.Panel(Tr, n_probe=32)
nodes = np.array([v, int(np.array(sorted(neg))[1])])
ps, pd = panel.pairs(nodes)
check("self-pairs excluded from panel", bool((ps != pd).all()))
check("panel is frozen (deterministic)", np.array_equal(panel.sources, C.Panel(Tr, 32).sources))
check("pair count == grid size minus self-pairs",
      len(ps) == int(panel._mask(nodes).sum()), f"{len(ps)}")

# a target inside the panel must average over 255, not 256, and stay aligned
in_panel = np.array([panel.sources[3], panel.sources[7]])
rp = panel.reputation(m, z, in_panel)
k = len(panel.sources)
s2, d2 = panel.pairs(in_panel)
manual2 = torch.sigmoid(C.logit_trust(m, z, s2, d2)).cpu().numpy()
check("reputation matches a manual masked mean",
      abs(rp[0] - manual2[:k - 1].mean()) < 1e-6
      and abs(rp[1] - manual2[k - 1:2 * k - 2].mean()) < 1e-6,
      f"{rp.tolist()}")

# out-of-panel targets average over the full panel
outp = np.array([nodes[0], nodes[1]])
ro = panel.reputation(m, z, outp)
s3, d3 = panel.pairs(outp)
manual3 = torch.sigmoid(C.logit_trust(m, z, s3, d3)).cpu().numpy()
check("out-of-panel target uses all k sources",
      abs(ro[0] - manual3[:k].mean()) < 1e-6)

# decomposition must sum to the mean log-odds of the same queries
sa, ta = panel.decompose(m, z, nodes)
s4, d4 = panel.pairs(nodes)
lg = C.logit_trust(m, z, s4, d4)
keep4 = panel._mask(nodes).ravel()
k4 = len(panel.sources)
# rebuild the per-node mean of the log-odds the same way the panel does
col_means = []
cursor = 0
for j in range(len(nodes)):
    c = int(keep4.reshape(k4, len(nodes)).sum(0)[j])
    col_means.append(lg[cursor:cursor + c].mean().item())
    cursor += c
check("source + target terms == per-node mean log-odds",
      np.allclose(sa + ta, np.array(col_means), atol=1e-4),
      f"max|err|={np.abs(sa + ta - np.array(col_means)).max():.2e}")

# ------------------------------------------------------------------- target selection
print("\n  target selection")
tgts, ng, ps_ = C.pick_targets(Tr, 3, strict=True)
ties = sum(1 for x in tgts if len(ng[x]) == len(ps_.get(x, [])))
check("strict selection excludes ties", ties == 0, f"{len(tgts)} targets")
tg_l, _, _ = C.pick_targets(Tr, 3, strict=False)
check("loose selection is a superset", set(tgts) <= set(tg_l), f"{len(tg_l)} loose")

# ------------------------------------------------- T10 panel batch independence
# The panel must return the SAME value for a node regardless of which other nodes
# share the call. This is a regression test for a source-major / node-major pair
# ordering mismatch that silently averaged the wrong sources.
print("\nT10  panel batch independence")
pan = C.Panel(Tr, n_probe=16, seed=1234)
z10 = C.embed(m_fix, Tr, D.tt)
probe_nodes = np.array([int(x) for x in tgts[:6]], dtype=np.int64)
alone = pan.reputation(m_fix, z10, probe_nodes)
for extra in (0, 1, 3):
    batch = np.concatenate(
        [probe_nodes, np.array([int(x) for x in tgts[6:6 + extra]], dtype=np.int64)])
    got = pan.reputation(m_fix, z10, batch)[:len(probe_nodes)]
    check(f"same value when queried alongside {extra} other node(s)",
          np.abs(got - alone).max() < F32_TOL,
          f"max|diff|={np.abs(got - alone).max():.2e}")
# node-major ordering, stated directly
s10, d10 = pan._pairs(probe_nodes)
k10 = len(pan.sources)
blocks = [d10[j * k10:(j + 1) * k10] for j in range(len(probe_nodes))]
check("each node's block contains only that node's destinations",
      all(bool((b == probe_nodes[j]).all()) for j, b in enumerate(blocks)),
      f"{[len(b) for b in blocks]}")
check("reputation equals a hand-built per-node mean",
      np.allclose(alone,
                  [torch.sigmoid(C.logit_trust(m_fix, z10, pan.sources,
                                               np.full(k10, int(v)))).mean().item()
                   for v in probe_nodes], atol=1e-6))
# self-pair exclusion still holds after the reordering
in_p = np.array([pan.sources[0], pan.sources[1], int(tgts[0])])
sp, dp = pan._pairs(in_p)
check("self-pairs still excluded", bool((sp != dp).all()), f"{len(sp)} pairs")

# ------------------------------------------------------------------- T7 threshold
print("\nT7  decision-threshold convention")
from keeper.laundering.threshold import optimal_threshold as T7opt          # noqa: E402

# p = P(trust); y == 1 means DISTRUST, so distrust is predicted when p < tau.
p_s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
y_s = np.array([0, 0, 0, 1, 1])
tau_sep = T7opt(p_s, y_s)
check("separating threshold lies between the classes", 0.1 < tau_sep <= 0.7,
      f"tau={tau_sep:.4f}")
check("separating threshold is near 0.5", abs(tau_sep - 0.5) < 0.35,
      f"tau={tau_sep:.4f}")
# the bug this guards: searching with (p >= tau) inverts MCC and returns tau=0
inv = T7opt(p_s, 1 - y_s)
check("inverted label convention gives a low threshold", inv < 0.2,
      f"tau={inv:.4f} (must NOT be 0.0 for correct labels)")
check("optimal threshold is never 0 for a sane classifier",
      0.0 < T7opt(p_s, y_s), f"tau={tau_sep:.4f}")
# real held-out data: tau* must be a real interior value, not degenerate
_eh, _yh = held_out(G, D)
_pp = C.p_trust(m_fix, z, _eh[:, 0], _eh[:, 1]).cpu().numpy()
tau_real = T7opt(_pp, _yh)
check("Alpha held-out tau* is interior (not the degenerate 0.0)",
      0.0 < tau_real < 1.0, f"tau*={tau_real:.4f} on n={len(_yh)}")

# ------------------------------------------------------------------ T8 NegGrad+
print("\nT8  NegGrad+ operator")
base_params = dict(C.make_model(D, Tr, 0, D.tt).named_parameters())
m_trained = C.make_model(D, Tr, 0, D.tt)
w_before = m_trained.regression_weights.detach().clone()
forget_mask = np.zeros(len(Tr.edges), bool)
forget_mask[neg[v]] = True
retained = Tr.subset(~forget_mask)

# NegGrad+ must be a CREDIBLE comparator, not a broken baseline: it has to forget
# AND leave the model intact. Measured sweep (Alpha, 150 ep) put the feasible
# region at lr=3e-4; at lr=3e-3 the operator destroys the model instead. So the
# test uses the real post-process settings and asserts BOTH axes.
m_theta = C.train(C.make_model(D, Tr, 0, D.tt), Tr, D.tt, epochs=60)
mcc_base = C.test_metrics(m_theta, C.embed(m_theta, Tr, D.tt), D, G)["mcc"]

# It must be a POST-PROCESS on theta*: starting from a fresh init would make it
# training, not unlearning. So perturb theta* first and check the operator keeps
# that starting point.
pert = copy.deepcopy(m_theta)
with torch.no_grad():
    for p in pert.parameters():
        p.add_(torch.randn_like(p) * 0.01)
m_ng = C.neggrad_unlearn(pert, retained, Tr, D.tt, forget_mask, steps=4)
check("neggrad keeps the head registered",
      "regression_weights" in dict(m_ng.named_parameters()))
# snapshot the caller's head BEFORE the call, so the mutation check is real
_w_before = pert.regression_weights.detach().clone()
m_ng_probe = C.neggrad_unlearn(pert, retained, Tr, D.tt, forget_mask, steps=1)
check("neggrad does not mutate the caller's model",
      torch.equal(pert.regression_weights.detach(), _w_before))
check("neggrad moves the head",
      float((m_ng.regression_weights.detach() - _w_before).norm()) > 0,
      f"|dW|={float((m_ng.regression_weights.detach() - _w_before).norm()):.4e}")

# It must FORGET: CE on F rises, measured on the full graph for every operator.
fl_before = C.forget_loss(m_theta, Tr, D.tt, forget_mask)
fl_after = C.forget_loss(m_ng, Tr, D.tt, forget_mask)
check("neggrad raises the forget-set loss", fl_after > fl_before,
      f"{fl_before:.4f} -> {fl_after:.4f}")

# It must PRESERVE utility at the probe's learning rate (lr=3e-4 default).
m_ng2 = C.neggrad_unlearn(m_theta, retained, Tr, D.tt, forget_mask, steps=4)
mcc_ng = C.test_metrics(m_ng2, C.embed(m_ng2, retained, D.tt), D, G)["mcc"]
check("neggrad preserves retain utility at lr=3e-4 (MCC drop < 0.05)",
      mcc_base - mcc_ng < 0.05, f"{mcc_base:.4f} -> {mcc_ng:.4f}")
# and the collapsed regime must be visibly worse, proving the bound has teeth
m_bad = C.neggrad_unlearn(m_theta, retained, Tr, D.tt, forget_mask,
                          steps=4, lr=3e-3)
mcc_bad = C.test_metrics(m_bad, C.embed(m_bad, retained, D.tt), D, G)["mcc"]
check("lr=3e-3 is measurably worse than lr=3e-4 (bound is not vacuous)",
      mcc_bad < mcc_ng, f"lr=3e-3 {mcc_bad:.4f} < lr=3e-4 {mcc_ng:.4f}")

# forget_loss must read the edges the mask names, not an offset-sliced view.
# Use the SAME trained model on both sides, or the comparison is meaningless.
_idx = np.where(forget_mask)[0]
_et, _lt, _ = C.window(Tr, D.tt)
_zref = C.embed(m_theta, Tr, D.tt)
_pair = torch.cat((_zref[torch.as_tensor(Tr.edges[_idx, 0], device=C.DEV)],
                   _zref[torch.as_tensor(Tr.edges[_idx, 1], device=C.DEV)]), 1)
_ref = float(torch.nn.functional.cross_entropy(
    torch.mm(_pair, m_theta.regression_weights), _lt[:, S.NEG_COL].long()[_idx],
    reduction="mean"))
check("forget_loss matches a hand-computed masked mean (same model)",
      abs(fl_before - _ref) < 1e-4, f"{fl_before:.6f} vs {_ref:.6f}")
# a TRUST edge in the forget set must be rejected
tr_edge = pos[int(v)][0] if pos.get(int(v)) else None
if tr_edge is not None:
    bad = np.zeros(len(Tr.edges), bool)
    bad[tr_edge] = True
    try:
        C.neggrad_unlearn(m_trained, Tr.subset(~bad), Tr, D.tt, bad, steps=1)
        check("neggrad rejects a non-distrust forget edge", False, "no AssertionError")
    except AssertionError as e:
        check("neggrad rejects a non-distrust forget edge", True, str(e)[:52])
# the forget edges must NOT be in the graph the operator runs on
check("neggrad runs on the retained graph (forget edges absent)",
      len(retained.edges) == len(Tr.edges) - len(neg[v]),
      f"{len(retained.edges)} vs {len(Tr.edges) - len(neg[v])}")

# ------------------------------------------------------------- T9 indirect C3
print("\nT9  indirect channel C3")
cands = C.indirect_candidates(Tr, int(v), budget=5)
check("indirect candidates exclude edges incident to the target",
      all(int(Tr.edges[i, 1]) != int(v) and int(Tr.edges[i, 0]) != int(v)
          for i in cands), f"{len(cands)} candidates")
check("indirect candidates are all distrust",
      bool((Tr.labels[cands, 1].astype(int) == 1).all()) if cands else True)
check("indirect candidates are sorted and deduplicated",
      cands == sorted(set(cands)), f"{len(cands)}")
check("indirect candidates are deterministic",
      cands == C.indirect_candidates(Tr, int(v), budget=5))
all_c = C.indirect_candidates(Tr, int(v), budget=0)
check("budget=0 returns every candidate", all_c[:5] == cands, f"{len(all_c)} total")
keep_g, chosen = C.greedy_erase(m_fix, Tr, D.tt, int(v), all_c,
                                 panel=panel, budget=2)
check("greedy respects the budget", len(chosen) <= 2, f"chose {len(chosen)}")
check("greedy keep-mask drops exactly the chosen edges",
      int((~keep_g).sum()) == len(chosen), f"{(~keep_g).sum()} dropped")
check("greedy is deterministic",
      C.greedy_erase(m_fix, Tr, D.tt, int(v), all_c, panel=panel, budget=2)[1]
      == chosen)

# ------------------------------------------------------- T11 snapshot_mode
# Epinions' timestamps are 69% one crawl instant, which breaks equal-count
# snapshotting. T11 pins both modes: 'count' must stay byte-identical to the
# Bitcoin behaviour, and 'quantile' must produce no empty snapshots and never
# split a tied timestamp.
print("\nT11  snapshot_mode: count vs quantile")
ds_q = synth()
ds_q.cfg = dict(ds_q.cfg, fl={"snapshot_mode": "quantile"})
ends_q = C.ends_for(ds_q)
check("quantile ends are nondecreasing",
      all(ends_q[i] <= ends_q[i + 1] for i in range(len(ends_q) - 1)), str(ends_q))
check("quantile last end covers every edge", ends_q[-1] == len(ds_q.time) - 1)
splits_q = sum(1 for i in range(len(ends_q) - 1) if 0 <= ends_q[i] < len(ds_q.time) - 1
               and ds_q.time[ends_q[i] + 1] == ds_q.time[ends_q[i]])
check("quantile never splits a tied timestamp", splits_q == 0, f"{splits_q} splits")
check("count mode is unchanged by the dispatcher (no cfg key -> count)",
      C.ends_for(synth()) == C.snapshot_ends(synth().time, 6, True))
try:
    C.ends_for(C.Dataset("x", np.zeros((1, 2), int), np.zeros((1, 2), np.float32),
                         np.zeros(1, int), 2, 2,
                         dict(fl={"snapshot_mode": "bogus"})))
    check("unknown snapshot_mode is rejected", False, "no ValueError")
except ValueError as e:
    check("unknown snapshot_mode is rejected", True, str(e)[:44])

# The real motivation, on the real data: a heavy tie must not empty a snapshot.
_t = np.array([0] * 90 + list(range(1, 11)), np.int64)   # 90% tied at t0
ends_c = C.snapshot_ends(_t, 10, tie_avoiding=True)
ends_q = C.quantile_ends(_t, 10)
_sizes_c = []
_prev = -1
for _e in ends_c:
    _sizes_c.append(_e - _prev)
    _prev = _e
_sizes_q = []
_prev = -1
for _e in ends_q:
    _sizes_q.append(_e - _prev)
    _prev = _e
check("count mode empties snapshots on a heavy tie (the bug being fixed)",
      sum(1 for s in _sizes_c if s == 0) > 0, f"{sum(1 for s in _sizes_c if s == 0)} empty")
check("quantile mode leaves no empty snapshot on the same input",
      all(s > 0 for s in _sizes_q), f"sizes={_sizes_q}")

# ------------------------------------------------------------- T12 signed loader
print("\nT12  signed loader (sign is the label; zero/self-loop policy)")
import json as _json  # noqa: E402
import tempfile as _tf  # noqa: E402

with _tf.TemporaryDirectory() as _d:
    _csv = os.path.join(_d, "toy.csv")
    # raw ids 100..104, ratings with magnitudes, one zero, one self-loop, unsorted times
    with open(_csv, "w") as f:
        f.write("source,target,rating,time\n"
                "100,101,5,30\n101,102,-3,10\n102,103,0,20\n103,103,2,25\n"
                "104,100,-10,5\n100,102,1,10\n")
    _cfg = {"dataset": "toy", "data_path": _csv}
    se = S.load_signed(_cfg, _d)
    check("zero rating dropped and counted", se.meta["n_zero_dropped"] == 1,
          str(se.meta))
    check("self-loop dropped and counted", se.meta["n_self_loops_dropped"] == 1)
    check("rows sorted by time (stable)", list(se.time) == [5, 10, 10, 30],
          str(list(se.time)))
    check("sign comes from the rating, magnitude kept",
          list(se.sign) == [-1, -1, 1, 1] and list(se.rating) == [-10, -3, 1, 5],
          f"{list(se.sign)} {list(se.rating)}")
    check("labels: distrust column set exactly for negative ratings",
          np.array_equal(S.neg_mask(se.labels), se.sign < 0))
    check("raw ids re-indexed to 0..n-1 without a rating file",
          se.meta["id_source"] == "reindexed" and se.edges.max() <= 4)
    try:
        S.load_signed(dict(_cfg, signed={"zero_policy": "error"}), _d)
        check("zero_policy=error refuses a zero rating", False, "no error")
    except ValueError:
        check("zero_policy=error refuses a zero rating", True)
    try:
        S.one_hot_from_sign(np.array([1, 0, -1]))
        check("one-hot refuses a zero sign", False, "no error")
    except ValueError:
        check("one-hot refuses a zero sign", True)

for _name in ("bitcoinalpha", "bitcoinotc"):
    _D = C.load(_name)
    _raw = np.loadtxt(os.path.join(C.ROOT, _D.cfg["edge_path"]))
    check(f"{_name}: loader reproduces the legacy rating file exactly",
          np.array_equal(_D.edges, _raw[:, :2].astype(np.int64))
          and np.array_equal(_D.labels, _raw[:, 2:4].astype(np.float32)))
    check(f"{_name}: label sign agrees with raw rating sign",
          np.array_equal(S.neg_mask(_D.labels), _D.rating < 0))

# ------------------------------------------------------- T13 GPU and provenance
print("\nT13  GPU enforcement, provenance, T1/T2 masks")
_real = torch.cuda.is_available
_env = os.environ.pop(C.ALLOW_CPU_ENV, None)
try:
    torch.cuda.is_available = lambda: False
    try:
        C.require_gpu()
        check("require_gpu raises without CUDA", False, "no error")
    except RuntimeError:
        check("require_gpu raises without CUDA", True)
    os.environ[C.ALLOW_CPU_ENV] = "1"
    C.require_gpu()
    check("FEDUNL_ALLOW_CPU=1 is the only override", True)
finally:
    torch.cuda.is_available = _real
    os.environ.pop(C.ALLOW_CPU_ENV, None)
    if _env is not None:
        os.environ[C.ALLOW_CPU_ENV] = _env
check("this suite runs on a GPU", C.DEV.type == "cuda", str(C.DEV))

_pv = C.provenance(D)
check("provenance names the GPU and code hash",
      bool(_pv["gpu_name"]) and len(_pv["code_sha256"]) == 64, str(_pv["gpu_name"]))
check("provenance carries data md5 and signed meta",
      "data_path" in _pv["data_md5"] and "n_zero_dropped" in _pv["signed_meta"])
with _tf.TemporaryDirectory() as _d:
    _p = C.dump({"x": float("nan"), "y": np.float32(1.5), "z": [np.int64(3)]},
                os.path.join(_d, "r.json"), D)
    _back = _json.load(open(_p))
    check("dump writes provenance, NaN -> null, numpy -> JSON",
          "provenance" in _back and _back["x"] is None and _back["y"] == 1.5
          and _back["z"] == [3])

# task masks against a direct port of basic_client2._select_eval_edges_both
_rows, _t1, _t2 = C.task_masks(G, D.tt)
_tr = [G.edges[G.snap == s] for s in range(D.tt)]
_train_set = set(np.concatenate(_tr).ravel().tolist())
_pre = set(np.concatenate(_tr[:-1]).ravel().tolist())
_last = set(_tr[-1].ravel().tolist())
_ref1, _ref2 = [], []
for a_, b_ in G.edges[_rows]:
    a_, b_ = int(a_), int(b_)
    _ref1.append(a_ in _train_set and b_ in _train_set)
    _ref2.append((a_ in _last and a_ not in _pre and b_ not in _last and b_ in _pre)
                 or (b_ in _last and b_ not in _pre and a_ not in _last and a_ in _pre)
                 or (a_ in _last and b_ in _last and a_ not in _pre and b_ not in _pre))
check("T1 mask equals the framework's Task 1", np.array_equal(_t1, np.array(_ref1)),
      f"{int(_t1.sum())} edges")
check("T2 mask equals the framework's Task 2", np.array_equal(_t2, np.array(_ref2)),
      f"{int(_t2.sum())} edges")
_tm = C.test_metrics(m_fix, z, D, G)
check("test_metrics reports Task 1 and Task 2",
      "t1" in _tm and "t2" in _tm and _tm["mcc"] == _tm["t1"]["mcc"], str(_tm))

# ------------------------------------------------------- T14 unlearning operators
print("\nT14  unlearning operators (laundering/unlearn.py)")
from keeper.laundering import unlearn as U  # noqa: E402
from keeper.laundering import attack as A  # noqa: E402
from keeper.laundering import metrics as M  # noqa: E402

_F = np.zeros(len(Tr.edges), bool)
_F[neg[v]] = True
_sd0 = {k_: t_.detach().clone() for k_, t_ in m_theta.state_dict().items()}


def _unchanged(model, ref):
    sd = model.state_dict()
    return all(torch.equal(sd[k_], ref[k_]) for k_ in ref)


_u = U.unlink(m_theta, G, Tr, _F, D.tt)
check("unlink: inference graph = train minus F",
      len(_u.Tr.edges) == len(Tr.edges) - int(_F.sum()))
check("unlink: features recomputed from the erased graph",
      torch.equal(_u.model.structural_layer.X_raw.cpu(),
                  C.features(Tr.subset(~_F), int(m_theta.args.seed),
                             int(m_theta.args.layers[-1])).cpu()))
check("unlink: trained parameters unchanged",
      all(torch.equal(a_, b_) for (n_, a_), (_, b_) in
          zip(_u.model.named_parameters(), m_theta.named_parameters())))
check("unlink: caller's model untouched", _unchanged(m_theta, _sd0))

_ex = U.exact_retrain(D, G, D.tt, Tr.rid[_F], 0, epochs=3)
check("exact retrain: training rows == train minus F",
      set(_ex.Tr.rid.tolist()) == set(Tr.rid[~_F].tolist()))
check("exact retrain: held-out snapshot rows unchanged",
      set(_ex.G.rid[_ex.G.snap == D.tt].tolist()) == set(G.rid[G.snap == D.tt].tolist()),
      f"moves={_ex.info['n_train_test_boundary_moves']}")
_ex2 = U.exact_retrain(D, G, D.tt, Tr.rid[_F], 0, epochs=3)
check("exact retrain: deterministic",
      all(torch.equal(a_, b_) for a_, b_ in zip(_ex.model.parameters(), _ex2.model.parameters())))

_seen = []
_orig_window = C.window


def _spy(g_, tt_):
    _seen.append(len(g_.edges))
    return _orig_window(g_, tt_)


C.window = _spy
try:
    _ng = U.neggrad_plus(m_theta, G, Tr, _F, D.tt, beta=0.95, epochs=3, lr=5e-3)
finally:
    C.window = _orig_window
check("NegGrad+: message passing runs on the retained graph (instrumented)",
      _seen and _seen[0] == len(Tr.edges) - int(_F.sum()), f"window sizes {_seen[:2]}")
check("NegGrad+: caller's model untouched", _unchanged(m_theta, _sd0))
check("NegGrad+: forget-set CE rises",
      U.forget_ce(_ng, Tr, _F, D.tt) > U.forget_ce(m_theta, Tr, _F, D.tt))
check("NegGrad+: uses the training criterion (class-weighted)",
      _ng.info["criterion"].startswith("training"))

_in = U.influence_unlearn(m_theta, G, Tr, _F, D.tt, damping=1.0, cg_iters=10)
check("influence: never raises the retained objective (line search)",
      _in.info["retain_loss_after"] <= _in.info["retain_loss_before"] + 1e-9,
      f"{_in.info['retain_loss_before']:.4f} -> {_in.info['retain_loss_after']:.4f} "
      f"({_in.info['line_search']}, cg {_in.info['cg_stop']})")
check("influence: caller's model untouched", _unchanged(m_theta, _sd0))

_coh, _neg, _ = C.pick_targets(Tr, 3, True)
_masks = {int(t): np.isin(np.arange(len(Tr.edges)), _neg[int(t)]) for t in _coh}
_batches = U.partition_disjoint(list(_masks), _masks, Tr, len(D.cfg["layers"]))
_adj = U._snapshot_adjacency(Tr)
_rf = {t: U.receptive_field(Tr, _masks[t], len(D.cfg["layers"]), _adj) for t in _masks}
check("batched retrain: no target inside a batch-mate's receptive field",
      all(u_ not in _rf[w_] for b_ in _batches for u_ in b_ for w_ in b_ if u_ != w_),
      f"{len(_batches)} batches for {len(_masks)} targets")
check("batched retrain: every target in exactly one batch",
      sorted(t for b_ in _batches for t in b_) == sorted(_masks))

# ------------------------------------------------------------------ T15 attack
print("\nT15  attack (laundering/attack.py)")
_bad = []
for t in list(_masks)[:40]:
    rows_ = A.indirect_rows(Tr, t)
    if np.any((Tr.edges[rows_, 0] == t) | (Tr.edges[rows_, 1] == t)):
        _bad.append(t)
check("C3 candidates never touch an edge incident to T (40 targets)", not _bad)
check("C3 candidates are all distrust edges",
      bool(S.neg_mask(Tr.labels[A.indirect_rows(Tr, int(v))]).all()))
_old_style = set(C.indirect_candidates(Tr, int(v), budget=0))
check("C3 candidates include in- AND out-neighbours (superset of raters-only)",
      _old_style <= set(A.indirect_rows(Tr, int(v)).tolist()),
      f"{len(_old_style)} -> {len(A.indirect_rows(Tr, int(v)))}")

_pan = C.Panel(Tr, n_probe=32, seed=1234)
_sc = A.Scorer(m_theta, Tr, D.tt, _pan, int(v))
_p3 = A.plan_indirect(_sc, Tr, int(v), 3)
_scores = [s_["cumulative_score"] for s_ in _p3.trace if "unit" in s_]
check("greedy never takes a non-positive step",
      all(b_ > a_ for a_, b_ in zip([0.0] + _scores, _scores)), str(_scores))
_pd = A.plan_direct_full(Tr, int(v))
check("C1 (subject-side) bribes nobody; C2 counts distinct raters",
      _pd.cost["raters_bribed"] == 0
      and A.cost_of(Tr, np.array(neg[v]), "C2-rater-edge")["raters_bribed"]
      == len(set(Tr.edges[neg[v], 0].tolist())))
_pa = A.plan_account(_sc, Tr, int(v), r_budget=2)
check("account-level cost counts account holders, not every erased edge's source",
      _pa.cost["raters_bribed"] == _pa.cost["accounts"], str(_pa.cost))
_sd_before = {k_: t_.detach().clone() for k_, t_ in m_theta.state_dict().items()}
_sm, _sp = A.surrogate(D, Tr, D.tt, 0, epochs=2)
_ssc = A.Scorer(_sm, Tr, D.tt, _sp, int(v))
A.plan_indirect(_ssc, Tr, int(v), 2)
check("grey-box surrogate differs from the victim and never mutates it",
      _unchanged(m_theta, _sd_before) and not all(
          torch.equal(a_, b_) for a_, b_ in zip(_sm.parameters(), m_theta.parameters()))
      and not np.array_equal(_sp.sources, _pan.sources))
_pl = A.placebo_rows(Tr, int(v), 3, np.random.default_rng(0))
_und = np.concatenate([Tr.edges, Tr.edges[:, ::-1]])
_near = {int(v)}
for _ in range(3):
    _near |= set(_und[np.isin(_und[:, 0], list(_near)), 1].tolist())
check("placebo edges have no endpoint within 3 hops of T",
      len(_pl) == 3 and not np.isin(Tr.edges[_pl].ravel(), list(_near)).any())

# ------------------------------------------------------------------ T16 metrics
print("\nT16  metrics (laundering/metrics.py)")
check("Laplace counting score", np.isclose(M.counting_score(1, 2, "laplace"), 0.4)
      and np.isclose(M.counting_score(0, 0, "laplace"), 0.5))
check("raw counting keeps the old 0/0 -> 1.0 convention",
      float(M.counting_score(0, 0, "raw")) == 1.0)
check("Laplace counting gain for n_pos=0, k=2 erased",
      np.isclose(M.counting_gain(0, 2, 2, "laplace"), 0.5 - 0.25))
_fr = M.flip_rates([0.2, 0.3, 0.8], [0.6, 0.4, 0.9])
check("flip rate: literal over all targets, conditional over those below",
      _fr["literal"] == 1 / 3 and _fr["conditional"] == 1 / 2, str(_fr))
check("flip rate of an empty set is None, not 0",
      M.flip_rates([], [])["literal"] is None and M.flip_rates([0.9], [0.95])["conditional"] is None)
check("post-stratified mean weights strata by cohort size",
      np.isclose(M.post_stratified_mean([1.0, 0.0], ["pos", "nopos"], {"pos": 1, "nopos": 3}), 0.25))
_ps = M.post_stratified_flip([0.2, 0.2, 0.2, 0.9], [0.6, 0.3, 0.6, 0.95],
                             ["pos", "pos", "nopos", "nopos"], {"pos": 10, "nopos": 30})
check("post-stratified flip estimate", np.isclose(_ps["estimate"], 0.25 * 0.5 + 0.75 * 0.5),
      str(_ps["estimate"]))
check("platform percentile ranks among all nodes",
      M.platform_percentile(0.5, np.array([0.1, 0.2, 0.9, 0.5])) == 0.5)
from keeper.laundering import aggregate4 as G4  # noqa: E402
check("decision rule: one seed gives no verdict",
      G4.threshold_verdict([0.5], 0.30)["verdict"].startswith("NO VERDICT"))
check("decision rule: mean - sd - 0.02 >= c -> SUPPORTED",
      G4.threshold_verdict([0.50, 0.52, 0.51, 0.49, 0.50], 0.30)["verdict"] == "SUPPORTED")
check("decision rule: mean + sd + 0.02 < c -> NOT SUPPORTED",
      G4.threshold_verdict([0.10, 0.12, 0.11, 0.09, 0.10], 0.30)["verdict"] == "NOT SUPPORTED")
check("decision rule: band straddling c -> INCONCLUSIVE",
      G4.threshold_verdict([0.31, 0.29, 0.30, 0.32, 0.28], 0.30)["verdict"] == "INCONCLUSIVE")
check("decision rule: fewer than 5 seeds is flagged",
      "spec" in G4.threshold_verdict([0.5, 0.5, 0.51], 0.3)["verdict"])
check("decision rule: 'x < c' claims (H6)",
      G4.below_verdict([0.1, 0.12, 0.11, 0.1, 0.09], 0.5)["verdict"] == "SUPPORTED"
      and G4.below_verdict([0.9, 0.92, 0.91, 0.9, 0.89], 0.5)["verdict"] == "NOT SUPPORTED")

# --------------------------------------------- T17 VPE, linkage, verification, run.py
print("\nT17  VPE, linkage, verification, baselines, restored run.py")
from keeper.cppr import vpe as V  # noqa: E402
from keeper.cppr import linkage as L  # noqa: E402
from keeper.laundering import baselines as B  # noqa: E402

_rows = np.asarray(neg[v], dtype=np.int64)
_acc = int(Tr.edges[_rows[0], 0])
_reqs = [V.Request(rows=_rows, tag="launder_edge", target=int(v)),
         V.Request(account=_acc, tag="launder_account")]
_res = V.apply_vpe(D, G, D.tt, _reqs, V.VPEConfig(1, float("inf"), 0), np.random.default_rng(0))
_e = _res.Tr.edges
check("VPE: erased rater->target edges are gone from the served graph",
      not any(((_e[:, 0] == int(Tr.edges[r, 0])) & (_e[:, 1] == int(v))).any()
              for r in _rows if int(Tr.edges[r, 0]) != _acc))
_anon_in = S.neg_mask(_res.Tr.labels[np.isin(_e[:, 0], _res.anon_nodes) & (_e[:, 1] == int(v))])
check("VPE: every erased verdict on T survives on a pooled node (same sign)",
      int(_anon_in.sum()) == len(_rows), f"{int(_anon_in.sum())} pooled distrust vs {len(_rows)}")
check("VPE: an erased account keeps no edge at all",
      not ((_e[:, 0] == _acc) | (_e[:, 1] == _acc)).any())
check("VPE: held-out snapshot untouched",
      set(_res.G.rid[_res.G.snap == D.tt].tolist()) == set(G.rid[G.snap == D.tt].tolist()))
check("VPE: pooled verdicts sit in training snapshots only",
      bool((_res.G.snap[np.isin(_res.G.edges[:, 0], _res.anon_nodes)] < D.tt).all()))
_resn = V.apply_vpe(D, G, D.tt, _reqs, V.VPEConfig(1, 0.5, 10 ** 9), np.random.default_rng(0))
check("VPE: noisy counts are non-negative integers and recorded",
      all(p["released"] >= 0 and p["noised"] for p in _resn.pooled), f"{len(_resn.pooled)} cells")
check("linkage: ties ranked by average (no id-order bias)",
      L._rank_pct({5: 1.0, 2: 1.0, 9: 0.0}) == {5: 0.75, 2: 0.75, 9: 0.0})
check("linkage: AUC of true vs rest counts ties as 1/2",
      L._auc_true_vs_rest({1: 0.5, 2: 0.5, 3: 0.1}, 1) == 0.75)

from keeper.laundering.verification import head_fingerprints  # noqa: E402
_fp = head_fingerprints(m_theta, Tr, D.tt)
_z = C.embed(m_theta, Tr, D.tt)
_k = int(np.argmax(_fp))
_W = m_theta.regression_weights.detach().clone().requires_grad_(True)
_f = torch.cat((_z[int(Tr.edges[_k, 0])], _z[int(Tr.edges[_k, 1])]))[None]
_l = torch.nn.functional.cross_entropy(_f @ _W, torch.as_tensor([int(S.class_ids(Tr.labels[_k:_k + 1])[0])],
                                                              device=_z.device))
_gw = torch.autograd.grad(_l, _W)[0]
check("verification: closed-form head fingerprint == autograd gradient norm",
      abs(float(_gw.norm()) - float(_fp[_k])) < 1e-4 * max(1.0, float(_fp[_k])),
      f"{float(_gw.norm()):.5f} vs {float(_fp[_k]):.5f}")

_sg = B.SGCN(D.n, 0)
_x1, _x2 = _sg.features(Tr), _sg.features(Tr)
check("SGCN spectral features: GPU, deterministic, orthonormal",
      _x1.is_cuda and torch.equal(_x1, _x2)
      and torch.allclose(_x1.T @ _x1, torch.eye(_x1.shape[1], device=_x1.device), atol=1e-4))
_t = B.eigentrust(Tr, D.n)
check("EigenTrust: a probability distribution over nodes", abs(_t.sum() - 1) < 1e-6 and (_t >= 0).all())

import keeper.federated.run as _run  # noqa: E402
_tup = _run.setup_models_and_utils()
check("run.py: restored API returns the 6 framework callables", len(_tup) == 6 and all(map(callable, _tup)))
check("run.py: aliases resolve", _run.load_config("alpha")["dataset"] == "bitcoinalpha")
_idx = _tup[4](22, data_path=os.path.join(C.ROOT, D.cfg["data_path"]))
check("run.py: snapshot index covers exactly the loader's edges",
      _idx[-1] == len(C.load("bitcoinalpha").edges) - 1 and len(_idx) == 22)

# Finite-difference HVPs are NOT used: S3R's top-k attention makes the loss
# piecewise and their error was erratic in the step size (1.3%..136%, measured),
# so the automatic mode must always be exact.
check("HVP: automatic mode is exact (finite differences never auto-selected)",
      U.hvp_mode_for(Tr) == "exact" and U.hvp_mode_for(C.Graph(
          np.zeros((10 ** 6, 2), np.int64), np.zeros((10 ** 6, 2), np.float32),
          np.zeros(10 ** 6, np.int64), np.zeros(10 ** 6, np.int64), 2)) == "exact")
_hn = U.head_newton(m_theta, G, Tr, _F, D.tt, damping=1e-2)
check("head_newton: lowers its convex head objective",
      _hn.info["objective_after"] < _hn.info["objective_before"],
      f"{_hn.info['objective_before']:.4f} -> {_hn.info['objective_after']:.4f}")
check("head_newton: only the head moves; encoder and caller untouched",
      _unchanged(m_theta, _sd0) and all(
          torch.equal(a_, b_) for (n_, a_), (_, b_) in
          zip(_hn.model.named_parameters(), m_theta.named_parameters())
          if n_ != "regression_weights"))
_hn2 = U.head_newton(m_theta, G, Tr, _F, D.tt, damping=1e-2)
check("head_newton: deterministic",
      torch.equal(_hn.model.regression_weights, _hn2.model.regression_weights))

print("\n" + "=" * 60)
if FAILED:
    print(f"FAILED: {len(FAILED)} -> {FAILED}")
    sys.exit(1)
print("all checks passed")
