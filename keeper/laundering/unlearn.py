"""Unlearning operators for the reputation-laundering study (proposal §6).

Every operator answers the same question: given the trained model ``m0`` and a
set of erased training edges F (original row ids), what model, and what
inference graph, does the platform serve afterwards?

* :func:`unlink` -- Unlink-to-Unlearn (Tan et al., WWW'24): parameters
  unchanged, F removed from the inference graph. Structure-derived node features
  are recomputed from the erased graph by default; ``recompute_features=False``
  keeps the frozen pre-erasure features (ablation, audit OP-3).
* :func:`exact_retrain` -- the gold standard: the FULL pipeline re-run on the
  erased data, snapshot boundaries included (audit OP-4). ``resnap=False`` keeps
  the pre-erasure calendar (ablation).
* :func:`batched_exact_retrain` -- §7: one exact retrain per batch of targets
  whose per-snapshot receptive fields are disjoint.
* :func:`neggrad_plus` -- NegGrad+ (Kurmanji, Triantafillou, Hayes and
  Triantafillou, "Towards Unbounded Machine Unlearning", NeurIPS 2023,
  arXiv 2302.09880): minimise ``beta*L_retain - (1-beta)*L_forget`` from theta*,
  with the TRAINING criterion (class-weighted CE) and gradient clipping.
* :func:`influence_unlearn` -- one damped Newton step on the retained-data
  objective, in the style of certified edge unlearning (Wu et al., KDD'23) and
  certified removal (Guo et al., ICML'20). Optional Gaussian noise gives the
  certified-removal construction; its guarantee holds only for strongly convex
  losses, which LocalGDTE's is not, and results say so.

All operators return an :class:`Unlearned` record holding the model and the
graphs it must be evaluated on, so callers cannot pair a model with the wrong
inference graph.
"""
import copy
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch

from keeper.laundering import core as C
from keeper.laundering import signed as S


@dataclass
class Unlearned:
    model: "C.LocalGDTE"
    G: "C.Graph"            # full graph (train + held-out) the model is evaluated on
    Tr: "C.Graph"           # training/inference graph [0, tt)
    operator: str
    info: dict = field(default_factory=dict)

    def embed(self, tt: int) -> torch.Tensor:
        return C.embed(self.model, self.Tr, tt)


def _seed_and_dim(m) -> tuple:
    return int(m.args.seed), int(m.args.layers[-1])


def _rows_to_rids(Tr: "C.Graph", forget_mask: np.ndarray) -> np.ndarray:
    return Tr.rid[np.asarray(forget_mask, bool)]


# ------------------------------------------------------------------- unlink (UtU)
def unlink(m0, G: "C.Graph", Tr: "C.Graph", forget_mask: np.ndarray, tt: int,
           recompute_features: bool = True) -> Unlearned:
    """Remove F from the inference graph; keep the trained parameters.

    LocalGDTE's input features are a function of graph structure
    (``setup_features2``), so a faithful unlink also recomputes them from the
    erased graph. With ``recompute_features=False`` the model is reused as is.
    """
    forget_mask = np.asarray(forget_mask, bool)
    cut = Tr.subset(~forget_mask)
    Gc = G.subset(~G.mask_of(_rows_to_rids(Tr, forget_mask)))
    m = m0
    if recompute_features:
        seed, dim = _seed_and_dim(m0)
        m = copy.deepcopy(m0)
        X = C.features(cut, seed, dim)
        m.structural_layer.X_raw = X.to(m.structural_layer.X_raw.device)
    return Unlearned(m, Gc, cut, "unlink" if recompute_features else "unlink_frozen_x",
                     dict(recompute_features=bool(recompute_features),
                          n_forget=int(forget_mask.sum())))


# ---------------------------------------------------------------- exact retraining
def exact_retrain(D: "C.Dataset", G: "C.Graph", tt: int, erase_rids, seed: int,
                  epochs: int, lr: float = 0.005, wd: float = 1e-5,
                  resnap: bool = True, self_term: bool = False) -> Unlearned:
    """Retrain from scratch on the erased data, with the same seed and budget.

    With ``resnap`` the snapshot boundaries are re-cut from the erased edge list,
    exactly as the pipeline would on a fresh copy of the data. The held-out
    snapshot is the evaluation protocol, not training data, so its membership is
    pinned to the original; any row the re-cut would move across the train/test
    boundary is counted in ``info`` (measured 0 on Bitcoin).

    ``self_term`` is the G4 flag of the model being unlearned, passed to
    make_model; callers pass ``C.self_term_of(m0)`` so that it is never dropped.
    """
    erase_rids = np.asarray(erase_rids, dtype=np.int64)
    if resnap:
        Dp, rid = C.erased_dataset(D, erase_rids)
        Gp = C.full_graph(Dp, rid=rid)
        test_orig = np.isin(Gp.rid, G.rid[G.snap == tt])
        moved_in = (Gp.snap == tt) & ~test_orig
        moved_out = (Gp.snap != tt) & test_orig
        Gp.snap = Gp.snap.copy()
        Gp.snap[moved_in] = tt - 1
        Gp.snap[moved_out] = tt
        n_moves = int(moved_in.sum() + moved_out.sum())
        train_rids_old = G.rid[(G.snap < tt) & ~np.isin(G.rid, erase_rids)]
        old_snap = dict(zip(G.rid.tolist(), G.snap.tolist()))
        Trp = C.train_split(Gp, tt)
        n_resnapped = int(sum(1 for r, s in zip(Trp.rid.tolist(), Trp.snap.tolist())
                              if old_snap[r] != s))
        assert set(Trp.rid.tolist()) == set(train_rids_old.tolist()), \
            "exact retrain: training rows differ from (train - erased)"
    else:
        Dp = D
        Gp = G.subset(~G.mask_of(erase_rids))
        Trp = C.train_split(Gp, tt)
        n_moves, n_resnapped = 0, 0
    m = C.train(C.make_model(Dp, Trp, seed, tt, self_term=self_term), Trp, tt, epochs=epochs, lr=lr, wd=wd)
    return Unlearned(m, Gp, Trp, "exact_retrain" if resnap else "exact_retrain_fixed_calendar",
                     dict(resnap=bool(resnap), n_erased=int(len(erase_rids)),
                          n_train_test_boundary_moves=n_moves,
                          n_train_rows_resnapped=n_resnapped, epochs=int(epochs)))


# ----------------------------------------------------------- batched exact retrain
def _snapshot_adjacency(Tr: "C.Graph") -> dict:
    """Undirected, unweighted CSR adjacency per training snapshot."""
    adj = {}
    for s in np.unique(Tr.snap):
        e = Tr.edges[Tr.snap == s]
        a = sp.coo_matrix((np.ones(2 * len(e), np.int8),
                           (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                          shape=(Tr.n, Tr.n)).tocsr()
        a.data[:] = 1
        adj[int(s)] = a
    return adj


def receptive_field(Tr: "C.Graph", forget_mask: np.ndarray, hops: int,
                    adj: dict | None = None) -> set:
    """Nodes whose structural embedding can change when F is unlinked.

    LocalGDTE's structural layer passes messages only WITHIN a snapshot, over
    ``hops`` = number of conv layers, in both edge directions, and S3R mixes only
    along each node's own timeline. So unlinking an edge in snapshot s can only
    reach nodes within ``hops`` of its endpoints in snapshot s's graph.
    """
    adj = adj if adj is not None else _snapshot_adjacency(Tr)
    out = set()
    rows = np.where(np.asarray(forget_mask, bool))[0]
    for s in np.unique(Tr.snap[rows]):
        seeds = np.unique(Tr.edges[rows[Tr.snap[rows] == s]].ravel())
        x = np.zeros(Tr.n, bool)
        x[seeds] = True
        frontier = x.copy()
        a = adj[int(s)]
        for _ in range(hops):
            frontier = (a @ frontier.astype(np.int8)) > 0
            frontier &= ~x
            x |= frontier
            if not frontier.any():
                break
        out.update(np.where(x)[0].tolist())
    return out


def partition_disjoint(targets, forget_masks: dict, Tr: "C.Graph", hops: int) -> list:
    """Greedy batches of targets with mutually non-interfering erasures.

    Targets u and v may share a batch only if v is outside u's receptive field
    and u is outside v's, so neither target's embedding is moved through
    message passing by the other's erasure. Interaction through the shared
    parameters (the weight channel) and through panel sources is NOT excluded;
    :func:`batched_exact_retrain` callers validate the batching against
    per-target retraining on a sample.
    """
    adj = _snapshot_adjacency(Tr)
    rf = {int(v): receptive_field(Tr, forget_masks[int(v)], hops, adj) for v in targets}
    batches = []
    for v in sorted(int(t) for t in targets):
        for b in batches:
            if all(v not in rf[u] and u not in rf[v] for u in b):
                b.append(v)
                break
        else:
            batches.append([v])
    return batches


def batched_exact_retrain(D, G, Tr, tt, targets, forget_masks: dict, seed: int,
                          epochs: int, hops: int, lr: float = 0.005,
                          wd: float = 1e-5, self_term: bool = False):
    """Yield ``(batch, Unlearned)`` for each disjoint batch (proposal §7).
    ``self_term``: as in exact_retrain."""
    for batch in partition_disjoint(targets, forget_masks, Tr, hops):
        mask = np.zeros(len(Tr.edges), bool)
        for v in batch:
            mask |= forget_masks[v]
        u = exact_retrain(D, G, tt, Tr.rid[mask], seed, epochs, lr, wd, resnap=True,
                          self_term=self_term)
        u.info["batch"] = list(batch)
        u.operator = "exact_retrain_batched"
        yield batch, u


# --------------------------------------------------------------------- NegGrad+
def neggrad_plus(m0, G: "C.Graph", Tr: "C.Graph", forget_mask: np.ndarray, tt: int,
                 *, beta: float = 0.95, epochs: int = 10, lr: float = 5e-4,
                 clip: float = 5.0) -> Unlearned:
    """NegGrad+ post-process of theta* (Kurmanji et al., NeurIPS 2023, Eq. 4).

    Minimises ``beta * CE_w(retain) - (1 - beta) * CE_w(forget)``, where CE_w is
    the model's training criterion (class-weighted cross-entropy) and both terms
    are means. Message passing runs on the RETAINED graph; the forget-set loss is
    evaluated on the erased edges from retained-graph embeddings (the D7 fix).
    The optimiser is the training optimiser (Adam) with gradient clipping as in
    training. Defaults follow the paper's small-scale beta=0.95 and its epoch
    count of 10; full-batch training makes one epoch one step.
    """
    forget_mask = np.asarray(forget_mask, bool)
    assert forget_mask.any(), "empty forget set"
    m = copy.deepcopy(m0)
    retain = Tr.subset(~forget_mask)
    et, lt, ends = C.window(retain, tt)
    y_r = lt[:, S.NEG_COL].long()
    fe = Tr.edges[forget_mask]
    y_f = torch.as_tensor(S.class_ids(Tr.labels[forget_mask]), device=C.DEV)
    f_src = torch.as_tensor(fe[:, 0], device=C.DEV)
    f_dst = torch.as_tensor(fe[:, 1], device=C.DEV)
    m.index_list, m.args.train_time_slots = ends, tt
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    crit = m.criterion
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        _, z = m(et, y_r, lt, return_loss=False)
        zd = z.device
        W = m.regression_weights.to(zd)
        pr = torch.cat((z[et[0].to(zd)], z[et[1].to(zd)]), 1)
        pf = torch.cat((z[f_src.to(zd)], z[f_dst.to(zd)]), 1)
        loss = (beta * crit(pr @ W, y_r.to(zd))
                - (1.0 - beta) * crit(pf @ W, y_f.to(zd)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        opt.step()
    Gc = G.subset(~G.mask_of(_rows_to_rids(Tr, forget_mask)))
    return Unlearned(m, Gc, retain, "neggrad_plus",
                     dict(beta=beta, epochs=epochs, lr=lr, clip=clip,
                          optimizer="Adam", criterion="training (class-weighted CE)",
                          n_forget=int(forget_mask.sum())))


# ------------------------------------------------- Hessian-vector products
# Above this many training edges the full-parameter exact HVP does not fit a 48 GB
# GPU (measured on Epinions: OOM at 46.5 GiB for every S3R chunk size, and also
# with the structural layer frozen). Finite differences are NOT a substitute:
# S3R's dynamic top-k attention masks make the loss piecewise, and the
# finite-difference HVP error was erratic in the step size (1.3% to 136%,
# measured on Bitcoin-Alpha). Callers use head_newton / head-scoped influence there.
FULL_HVP_EDGE_LIMIT = 200_000


class HVP:
    """Hessian-vector products of ``loss_fn`` at the current parameters.

    ``exact``: double backward through the model (create_graph). Verified against
        finite differences on Bitcoin-Alpha (relative difference 3.7% at eps=1e-3,
        consistent with float32 differencing).
    ``fd``: central finite differences of gradients,
        Hv ~ (g(theta + r v) - g(theta - r v)) / (2 r),  r = eps / ||v||.
        Kept for diagnostics only and never selected automatically: LocalGDTE's
        top-k attention makes the loss piecewise, and the error was erratic in
        eps (1.3% to 136% on Bitcoin-Alpha, measured).
    """

    def __init__(self, loss_fn, params, mode: str, eps: float = 1e-3):
        self.loss_fn, self.params, self.mode, self.eps = loss_fn, params, mode, eps
        if mode == "exact":
            self.loss0 = loss_fn()
            self.grads = torch.autograd.grad(self.loss0, params, create_graph=True)
            self.g0 = [g.detach() for g in self.grads]
        elif mode == "fd":
            self.loss0 = loss_fn()
            self.g0 = [g.detach() for g in torch.autograd.grad(self.loss0, params)]
            self.theta0 = [p.detach().clone() for p in params]
        else:
            raise ValueError(mode)

    def _grad_at(self, shift):
        with torch.no_grad():
            for p, t0, s in zip(self.params, self.theta0, shift):
                p.copy_(t0 + s)
        g = torch.autograd.grad(self.loss_fn(), self.params)
        return [x.detach() for x in g]

    def __call__(self, v):
        if self.mode == "exact":
            dot = sum((g * vi).sum() for g, vi in zip(self.grads, v))
            return [h.detach() for h in torch.autograd.grad(dot, self.params, retain_graph=True)]
        vn = float(sum(float((vi * vi).sum()) for vi in v)) ** 0.5
        r = self.eps / max(vn, 1e-12)
        gp = self._grad_at([r * vi for vi in v])
        gm = self._grad_at([-r * vi for vi in v])
        with torch.no_grad():
            for p, t0 in zip(self.params, self.theta0):
                p.copy_(t0)
        return [(a - b) / (2 * r) for a, b in zip(gp, gm)]

    def close(self):
        self.grads = None


def full_hvp_feasible(G: "C.Graph") -> bool:
    return len(G.edges) <= FULL_HVP_EDGE_LIMIT


def hvp_mode_for(G: "C.Graph", requested: str = "auto") -> str:
    """Always exact unless explicitly requested (finite differences are unreliable here)."""
    return "exact" if requested == "auto" else requested


def damped_cg(hvp, b, damping: float, iters: int, tol: float):
    """CG on (H + damping I) x = b; stops on negative curvature at the best iterate."""
    def dot(a, c):
        return sum(float((ai * ci).sum()) for ai, ci in zip(a, c))
    x = [torch.zeros_like(bi) for bi in b]
    r = [bi.clone() for bi in b]
    p = [ri.clone() for ri in r]
    rs = dot(r, r)
    bn = rs ** 0.5
    best, best_res, stop, it = [xi.clone() for xi in x], 1.0, "max_iters", 0
    for it in range(1, iters + 1):
        hp = [h + damping * pi for h, pi in zip(hvp(p), p)]
        curv = dot(p, hp)
        if curv <= 0:
            stop = "negative_curvature"
            break
        a = rs / curv
        x = [xi + a * pi for xi, pi in zip(x, p)]
        r = [ri - a * hi for ri, hi in zip(r, hp)]
        rn = dot(r, r)
        rel = rn ** 0.5 / max(bn, 1e-12)
        if rel < best_res:
            best, best_res = [xi.clone() for xi in x], rel
        if rel <= tol:
            stop = "converged"
            break
        p = [ri + (rn / rs) * pi for ri, pi in zip(r, p)]
        rs = rn
    return best, dict(cg_stop=stop, cg_iters_run=it, cg_rel_residual=float(best_res))


# ------------------------------------------------- influence (one Newton step)
def _retain_loss(m, et, lt, y, crit):
    _, z = m(et, y, lt, return_loss=False)
    zd = z.device
    W = m.regression_weights.to(zd)
    return crit(torch.cat((z[et[0].to(zd)], z[et[1].to(zd)]), 1) @ W, y.to(zd))


def influence_unlearn(m0, G: "C.Graph", Tr: "C.Graph", forget_mask: np.ndarray,
                      tt: int, *, damping: float = 0.1, cg_iters: int = 40,
                      cg_tol: float = 1e-3, max_step_ratio: float = 0.1,
                      noise_std: float = 0.0, noise_seed: int = 0,
                      hvp_mode: str = "auto") -> Unlearned:
    """theta <- theta* - (H' + damping I)^-1 grad L'(theta*), by conjugate gradient.

    L' is the exact-retraining objective: class-weighted CE over the RETAINED
    edges, class weights recomputed from them, message passing on the retained
    graph. H' is its Hessian at theta*, applied through Hessian-vector products
    (exact double backward, or finite differences on graphs too large for it;
    see :class:`HVP`; the mode is recorded in ``info``).
    CG stops at ``cg_iters``, at relative residual ``cg_tol``, or on negative
    curvature (the loss is non-convex), and the best iterate by residual is
    kept. The step is capped at ``max_step_ratio * ||theta*||`` and then
    backtracked (halving, up to 5 times) until it lowers L'; if none does, no
    step is taken. Every stop reason is recorded in ``info``.
    ``noise_std > 0`` adds Gaussian noise (certified-removal construction); the
    certificate needs strong convexity, which this model does not have.
    """
    forget_mask = np.asarray(forget_mask, bool)
    assert forget_mask.any(), "empty forget set"
    m = copy.deepcopy(m0)
    retain = Tr.subset(~forget_mask)
    et, lt, ends = C.window(retain, tt)
    y = lt[:, S.NEG_COL].long()
    m.index_list, m.args.train_time_slots = ends, tt
    crit = torch.nn.CrossEntropyLoss(weight=C.class_weights(retain).to(
        m.regression_weights.device))
    params = [p for p in m.parameters() if p.requires_grad]
    m.train()
    mode = hvp_mode_for(retain, hvp_mode)
    H = HVP(lambda: _retain_loss(m, et, lt, y, crit), params, mode)
    loss0 = H.loss0.detach()
    x, cg = damped_cg(H, H.g0, damping, cg_iters, cg_tol)
    H.close()
    del H
    best_res, stop, it = cg["cg_rel_residual"], cg["cg_stop"], cg["cg_iters_run"]

    def dot(a, c):
        return sum(float((ai * ci).sum()) for ai, ci in zip(a, c))
    step_norm = dot(x, x) ** 0.5
    theta_norm = sum(float((p.detach() ** 2).sum()) for p in params) ** 0.5
    cap = min(1.0, max_step_ratio * theta_norm / max(step_norm, 1e-12))
    theta0 = [p.detach().clone() for p in params]

    # Backtracking: a Newton step must decrease the retained objective it targets.
    scale, loss1 = 0.0, float(loss0)
    with torch.no_grad():
        m.eval()
        for s in (cap * 2.0 ** -k for k in range(6)):
            for p, t0_, xi in zip(params, theta0, x):
                p.copy_(t0_ - s * xi)
            trial = float(_retain_loss(m, et, lt, y, crit))
            if trial < float(loss0):
                scale, loss1 = s, trial
                break
        for p, t0_, xi in zip(params, theta0, x):
            p.copy_(t0_ - scale * xi)
        if noise_std > 0:
            gen = torch.Generator(device=params[0].device).manual_seed(int(noise_seed))
            for p in params:
                p.add_(noise_std * torch.randn(p.shape, generator=gen,
                                               device=p.device, dtype=p.dtype))
            loss1 = float(_retain_loss(m, et, lt, y, crit))
    Gc = G.subset(~G.mask_of(_rows_to_rids(Tr, forget_mask)))
    return Unlearned(m, Gc, retain, "influence_newton",
                     dict(damping=damping, hvp_mode=mode, cg_iters_run=it, cg_stop=stop,
                          cg_rel_residual=float(best_res),
                          step_norm=step_norm, step_scale=scale,
                          line_search="accepted" if scale > 0 else "no_descent",
                          theta_norm=theta_norm, noise_std=noise_std,
                          retain_loss_before=float(loss0), retain_loss_after=loss1,
                          certified=False,
                          certification_note="noise_std>0 is the certified-removal "
                          "construction, but its guarantee requires a strongly convex "
                          "loss; LocalGDTE's loss is non-convex",
                          n_forget=int(forget_mask.sum())))


# ------------------------------------- head Newton (Guo et al. certified-removal step)
def head_newton(m0, G: "C.Graph", Tr: "C.Graph", forget_mask: np.ndarray, tt: int,
                *, damping: float = 1e-2) -> Unlearned:
    """One exact Newton removal step on the output head, embeddings fixed.

    With the encoder frozen and embeddings z computed on the RETAINED graph, the
    head's objective
        L(W) = sum_e w_e CE(f_e W, y_e) / sum_e w_e + (damping / 2) ||W||^2,
        f_e = [z_src, z_dst],  e in retained edges,  w = class weights (retained)
    is convex in W (2-class logistic regression, 128 parameters), so its Hessian
    is formed exactly and the step W <- W - H^-1 grad L(W) is the removal update
    of Guo, Goldstein, Hannun and van der Maaten, "Certified Data Removal from
    Machine Learning Models", ICML 2020. Their (eps, delta) certificate further
    requires training-time objective perturbation, which a jointly trained
    LocalGDTE does not have, so this is reported as uncertified, together with
    the post-step gradient-residual norm that the certificate would bound.
    Cheap on every dataset (the full-parameter Newton step does not fit 48 GB on
    Epinions, measured).
    """
    forget_mask = np.asarray(forget_mask, bool)
    assert forget_mask.any(), "empty forget set"
    m = copy.deepcopy(m0)
    retain = Tr.subset(~forget_mask)
    z = C.embed(m, retain, tt).detach()
    zd = z.device
    e = torch.as_tensor(retain.edges, device=zd)
    Fe = torch.cat((z[e[:, 0]], z[e[:, 1]]), 1)
    y = torch.as_tensor(S.class_ids(retain.labels), device=zd)
    w = C.class_weights(retain).to(zd)[y]
    W0 = m.regression_weights.detach().to(zd).clone()
    shape = W0.shape

    def obj(wflat):
        Wm = wflat.view(shape)
        ce = torch.nn.functional.cross_entropy(Fe @ Wm, y, reduction="none")
        return (w * ce).sum() / w.sum() + 0.5 * damping * (wflat * wflat).sum()

    w0 = W0.flatten().requires_grad_(True)
    g0 = torch.autograd.grad(obj(w0), w0)[0].detach()
    Hm = torch.autograd.functional.hessian(obj, W0.flatten()).detach()
    step = torch.linalg.solve(Hm, g0)
    w1 = (W0.flatten() - step)
    l0, l1 = float(obj(W0.flatten())), float(obj(w1))
    w1r = w1.clone().requires_grad_(True)
    resid = float(torch.autograd.grad(obj(w1r), w1r)[0].norm())
    with torch.no_grad():
        m.regression_weights.copy_(w1.view(shape).to(m.regression_weights.device))
    Gc = G.subset(~G.mask_of(_rows_to_rids(Tr, forget_mask)))
    return Unlearned(m, Gc, retain, "head_newton",
                     dict(damping=damping, objective_before=l0, objective_after=l1,
                          grad_residual_norm=resid, step_norm=float(step.norm()),
                          min_hessian_eig=float(torch.linalg.eigvalsh(Hm).min()),
                          certified=False,
                          certification_note="Guo et al. 2020 removal step; certificate "
                          "requires training-time objective perturbation (absent)",
                          n_forget=int(forget_mask.sum())))


# ------------------------------------------------------------------ diagnostics
@torch.no_grad()
def forget_ce(u_or_m, Tr_full: "C.Graph", forget_mask: np.ndarray, tt: int) -> float:
    """Mean CE on F with embeddings from the PRE-erasure training graph.

    Diagnostic only. In LocalGDTE an edge's own label is a message-passing input
    (it enters its target's mean in-edge opinion), so on a graph that still
    contains F every model can read F's signs back: this yardstick reads ~0 even
    for exact retraining (measured) and cannot show forgetting. Use
    :func:`forget_ce_served`.
    """
    m = u_or_m.model if isinstance(u_or_m, Unlearned) else u_or_m
    return C.forget_loss(m, Tr_full, tt, np.asarray(forget_mask, bool))


@torch.no_grad()
def forget_ce_served(u_or_m, served_Tr: "C.Graph", Tr_full: "C.Graph",
                     forget_mask: np.ndarray, tt: int) -> float:
    """Mean CE on the erased verdicts as the platform SERVES them after erasure.

    Embeddings come from the served inference graph (F removed); the scored
    pairs and labels are F's. Exact retraining defines the target level: an
    approximate operator should come CLOSE to it, not exceed it.
    """
    m = u_or_m.model if isinstance(u_or_m, Unlearned) else u_or_m
    forget_mask = np.asarray(forget_mask, bool)
    z = C.embed(m, served_Tr, tt)
    e = Tr_full.edges[forget_mask]
    y = torch.as_tensor(S.class_ids(Tr_full.labels[forget_mask]), device=z.device)
    zd = z.device
    pair = torch.cat((z[torch.as_tensor(e[:, 0], device=zd)],
                      z[torch.as_tensor(e[:, 1], device=zd)]), 1)
    return float(torch.nn.functional.cross_entropy(
        pair @ m.regression_weights.to(zd), y, reduction="mean"))
