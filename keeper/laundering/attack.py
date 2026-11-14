"""Budgeted reputation-laundering attack (proposal §3 threat model, §5a solver).

Goal: raise the platform's predicted trust in a target T by having genuine
ratings erased, while T keeps its identity and its positive history.

Erasure channels (§3), weakest assumption first:
  C1  subject-side: T asks for ratings about T to be erased. No collusion, so
      no rater has to be bribed. The erasable set is T's distrust in-edges.
  C2  rater-side: colluding or bribed raters erase their ratings (edge level),
      or their whole accounts, which removes every edge incident to them.
  C3  indirect: distrust edges on T's neighbours are erased; no edge incident to
      T is touched, which evades per-target request monitoring.
No edge or node is ever added; injection is out of scope.

Solver (§5a): greedy selection over candidate erasures. Each candidate is
scored by
    unlink-and-forward gain   (exact: one forward pass on the graph without it)
  + weight-channel influence  (first-order: (w_e/W) * s_T . grad l_e, with
                               s_T = (H + lambda I)^-1 grad r_T; the per-edge
                               gradient fingerprints of claudecode/unlearn_breach.py)
The greedy stops as soon as no remaining candidate has a positive score, so it
never erases an edge that would lower T's trust (audit M10, TM-7).

Knowledge (§3): the WHITE-BOX attacker scores with the victim model and the
platform's probe panel. The GREY-BOX attacker trains its own surrogate on the
public signed graph (different seed, its own panel) and never touches the
victim; its plan is then executed against the victim. The white-box greedy is a
strong attack, not a guaranteed upper bound (greedy is myopic).
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from keeper.laundering import core as C
from keeper.laundering import signed as S

CHANNELS = ("C1-subject", "C2-rater-edge", "C2-account", "C3-indirect")


@dataclass
class Plan:
    """An erasure request set: original row ids, channel, cost, and its trace."""
    rids: np.ndarray
    channel: str
    knowledge: str                   # "whitebox" | "greybox" | "oracle-free"
    cost: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    n_candidates: int = 0

    def mask(self, G: "C.Graph") -> np.ndarray:
        return G.mask_of(self.rids)


# ---------------------------------------------------------------- candidates
def _is_neg(G):
    return S.neg_mask(G.labels)


def direct_rows(Tr: "C.Graph", v: int) -> np.ndarray:
    """T's distrust in-edges (rows of Tr): the C1/C2 edge-level erasable set."""
    return np.where((Tr.edges[:, 1] == int(v)) & _is_neg(Tr))[0]


def account_rows(Tr: "C.Graph", rater: int) -> np.ndarray:
    """Every edge incident to ``rater``: what erasing that account removes."""
    return np.where((Tr.edges[:, 0] == int(rater)) | (Tr.edges[:, 1] == int(rater)))[0]


def account_candidates(Tr: "C.Graph", v: int) -> list:
    """Raters of T's distrust in-edges, i.e. accounts whose deletion removes a verdict on T."""
    return sorted(set(Tr.edges[direct_rows(Tr, v), 0].tolist()) - {int(v)})


def indirect_rows(Tr: "C.Graph", v: int) -> np.ndarray:
    """C3: distrust in-edges of T's in- AND out-neighbours, never incident to T.

    The structural layer aggregates over both edge directions, so 'T's endorsers
    or neighbours' (§3) means every neighbour, not only T's raters (audit TM-5).
    There is no row-id cap (audit TM-6).
    """
    v = int(v)
    inc = (Tr.edges[:, 0] == v) | (Tr.edges[:, 1] == v)
    neigh = np.setdiff1d(np.unique(Tr.edges[inc].ravel()), [v])
    sel = np.isin(Tr.edges[:, 1], neigh) & _is_neg(Tr) & ~inc
    return np.where(sel)[0]


def placebo_rows(Tr: "C.Graph", v: int, k: int, rng: np.random.Generator,
                 min_dist: int = 4) -> np.ndarray:
    """``k`` random training edges with no endpoint within ``min_dist``-1 hops of T.

    Exact retraining is chaotic: erasing ANY few edges moves a target's score
    (measured +/-0.03-0.09, audit TM-4). Retraining after erasing these far edges
    gives the placebo distribution every retrain gain must be compared with.
    """
    near = {int(v)}
    frontier = {int(v)}
    und = np.concatenate([Tr.edges, Tr.edges[:, ::-1]])
    for _ in range(min_dist - 1):
        nxt = set(und[np.isin(und[:, 0], list(frontier)), 1].tolist()) - near
        near |= nxt
        frontier = nxt
        if not frontier:
            break
    far = np.where(~np.isin(Tr.edges[:, 0], list(near)) &
                   ~np.isin(Tr.edges[:, 1], list(near)))[0]
    if len(far) < k:
        return np.array([], dtype=np.int64)
    return np.sort(rng.choice(far, k, replace=False))


# ---------------------------------------------------------------- scoring
class Scorer:
    """Scores erasures against one model and one probe panel (unlink-and-forward).

    Uses the model's frozen features for speed: the attacker's cheap score.
    The chosen plan is later evaluated under every operator properly.
    """

    def __init__(self, m, Tr: "C.Graph", tt: int, panel: "C.Panel", v: int):
        self.m, self.Tr, self.tt, self.panel, self.v = m, Tr, tt, panel, int(v)
        self.base = self.rep(np.zeros(len(Tr.edges), bool))

    @torch.no_grad()
    def rep(self, removed: np.ndarray) -> float:
        z = C.embed(self.m, self.Tr.subset(~removed), self.tt)
        return float(self.panel.reputation(self.m, z, [self.v])[0])

    def unlink_gain(self, removed: np.ndarray) -> float:
        return self.rep(removed) - self.base


class WeightChannel:
    """First-order weight-channel influence of erasing one training edge on r_T.

    Removing edge e from the class-weighted mean training loss moves the optimum
    by about (w_e / W) H^-1 grad l_e (influence functions), so
        d r_T ~= (w_e / W) * s_T . grad l_e,   s_T = (H + lambda I)^-1 grad r_T.
    s_T is solved once per target by damped conjugate gradient
    (:func:`laundering.unlearn.damped_cg`). Hessian-vector products are exact
    on small graphs and finite-difference on large ones (:class:`unlearn.HVP`).
    """

    def __init__(self, m, Tr: "C.Graph", tt: int, panel: "C.Panel", v: int,
                 damping: float, cg_iters: int = 30, hvp_mode: str = "auto",
                 scope: str = "auto"):
        from keeper.laundering import unlearn as U
        self.m, self.Tr, self.tt = m, Tr, tt
        self.scope = ("all" if U.full_hvp_feasible(Tr) else "head") if scope == "auto" else scope
        if self.scope == "head":
            self._init_head(panel, v, damping)
            return
        self.params = [p for p in m.parameters() if p.requires_grad]
        et, lt, ends = C.window(Tr, tt)
        # window() sorts by snapshot; keep the map from Tr row to sorted position
        self.order = np.argsort(Tr.snap, kind="mergesort")
        self.pos = np.empty(len(self.order), np.int64)
        self.pos[self.order] = np.arange(len(self.order))
        m.index_list, m.args.train_time_slots = ends, tt
        m.train()
        y = lt[:, S.NEG_COL].long()
        cw = m.criterion.weight

        def forward():
            _, z = m(et, y, lt, return_loss=False)
            zd = z.device
            W = m.regression_weights.to(zd)
            logits = torch.cat((z[et[0].to(zd)], z[et[1].to(zd)]), 1) @ W
            w = cw.to(zd)[y.to(zd)]
            per = torch.nn.functional.cross_entropy(logits, y.to(zd), reduction="none")
            return z, W, w, per

        def loss_fn():
            _, _, w, per = forward()
            return (w * per).sum() / w.sum()

        # 1) gradient of the target's panel reputation at theta*
        z, W, _, _ = forward()
        sp_, dp_ = panel._pairs(np.array([int(v)]))
        zd = z.device
        f = torch.cat((z[torch.as_tensor(sp_, device=zd)], z[torch.as_tensor(dp_, device=zd)]), 1) @ W
        r = torch.sigmoid(f[:, 0] - f[:, 1]).mean()
        g_r = [g.detach() for g in torch.autograd.grad(r, self.params)]
        del z, W, f, r
        # 2) s_T = (H + lambda I)^-1 grad r_T. Finite-difference HVPs shift the
        #    parameters in place, so no saved graph may be alive across this step.
        mode = U.hvp_mode_for(Tr, hvp_mode)
        H = U.HVP(loss_fn, self.params, mode)
        self.s_T, cg = U.damped_cg(H, g_r, damping, cg_iters, 1e-3)
        H.close()
        del H
        self.cg = dict(cg, hvp_mode=mode, stop=cg["cg_stop"], rel_residual=cg["cg_rel_residual"],
                       scope="all")
        # 3) per-edge losses at theta*, graph kept for per-candidate gradients
        _, _, w, per = forward()
        self.per_edge, self.w = per, w.detach()
        self.W = float(self.w.sum())

    @torch.no_grad()
    def _head_parts(self):
        z = C.embed(self.m, self.Tr, self.tt)
        zd = z.device
        e = torch.as_tensor(self.Tr.edges, device=zd)
        return z, torch.cat((z[e[:, 0]], z[e[:, 1]]), 1)

    def _init_head(self, panel, v, damping):
        """Head-scoped influence on large graphs: with embeddings fixed the head's
        training loss is convex, so its 128x128 Hessian is formed exactly and
        per-edge gradients are closed form, f_e (x) (softmax(f_e W) - y_e)."""
        z, Fe = self._head_parts()
        zd = z.device
        y = torch.as_tensor(S.class_ids(self.Tr.labels), device=zd)
        w = self.m.criterion.weight.to(zd)[y]
        W0 = self.m.regression_weights.detach().to(zd).clone()
        shape = W0.shape

        def loss(wf):
            ce = torch.nn.functional.cross_entropy(Fe @ wf.view(shape), y, reduction="none")
            return (w * ce).sum() / w.sum()
        Hm = torch.autograd.functional.hessian(loss, W0.flatten()).detach()
        Hm = Hm + damping * torch.eye(Hm.shape[0], device=zd)
        sp_, dp_ = panel._pairs(np.array([int(v)]))
        wf = W0.flatten().requires_grad_(True)
        f = torch.cat((z[torch.as_tensor(sp_, device=zd)], z[torch.as_tensor(dp_, device=zd)]), 1)
        out = f @ wf.view(shape)
        r = torch.sigmoid(out[:, 0] - out[:, 1]).mean()
        g_r = torch.autograd.grad(r, wf)[0]
        self.s_T_head = torch.linalg.solve(Hm, g_r).view(shape)
        with torch.no_grad():
            p = torch.softmax(Fe @ W0, 1)
            self.head_resid = p - torch.nn.functional.one_hot(y, 2).float()
        self.Fe, self.w, self.W = Fe, w, float(w.sum())
        self.cg = dict(hvp_mode="exact_head", stop="direct_solve", rel_residual=0.0,
                       scope="head", min_hessian_eig=float(torch.linalg.eigvalsh(Hm).min()))

    def influence(self, tr_rows) -> np.ndarray:
        """Estimated weight-channel change of r_T from erasing each Tr row."""
        if self.scope == "head":
            rows = torch.as_tensor(np.asarray(tr_rows, dtype=np.int64), device=self.Fe.device)
            if len(rows) == 0:
                return np.zeros(0)
            g = self.Fe[rows][:, :, None] * self.head_resid[rows][:, None, :]   # [k, 2d, 2]
            val = (g * self.s_T_head[None]).sum((1, 2)) * self.w[rows] / self.W
            return val.detach().cpu().numpy()
        out = np.zeros(len(tr_rows))
        for i, row in enumerate(tr_rows):
            k = int(self.pos[int(row)])
            g = torch.autograd.grad(self.per_edge[k], self.params, retain_graph=True)
            out[i] = float(self.w[k]) / self.W * sum(
                float((si * gi).sum()) for si, gi in zip(self.s_T, g))
        return out

    def close(self):
        self.per_edge = None
        self.Fe = None


# ---------------------------------------------------------------- the solver
def greedy(scorer: Scorer, units: list, budget: int, *, weight: dict | None = None,
           channel: str, knowledge: str, Tr: "C.Graph") -> Plan:
    """Greedy over erasure UNITS (an edge, or an account's whole edge set).

    ``units`` is a list of (label, Tr row array). ``weight`` maps a unit label to
    its additive weight-channel score. Each step adds the unit with the largest
    marginal score; the loop stops when that score is not positive.
    """
    removed = np.zeros(len(Tr.edges), bool)
    chosen, trace = [], []
    current = 0.0
    weight = weight or {}
    for _ in range(min(budget, len(units))):
        best = None
        for label, rows in units:
            if label in chosen:
                continue
            trial = removed.copy()
            trial[rows] = True
            g_unlink = scorer.unlink_gain(trial)
            score = g_unlink + sum(weight.get(c, 0.0) for c in chosen) + weight.get(label, 0.0)
            if best is None or score > best[0] + 1e-12:
                best = (score, label, rows, g_unlink)
        if best is None or best[0] - current <= 0:
            trace.append(dict(stop="no_positive_marginal",
                              best_marginal=None if best is None else best[0] - current))
            break
        current = best[0]
        chosen.append(best[1])
        removed[best[2]] = True
        trace.append(dict(unit=best[1], cumulative_score=best[0],
                          unlink_gain=best[3], weight_term=weight.get(best[1], 0.0)))
    rids = Tr.rid[removed]
    return Plan(np.sort(rids), channel, knowledge, trace=trace, n_candidates=len(units))


def cost_of(Tr: "C.Graph", rows: np.ndarray, channel: str, n_accounts: int = 0) -> dict:
    """Erasure cost: edges erased, parties who must cooperate, accounts deleted.

    C1: the subject files the requests itself, so nobody is bribed.
    C2-account: the cooperating parties are the account holders, even though
        deleting an account also removes edges other people issued to it.
    C2-rater-edge / C3: each distinct rater of an erased edge must cooperate.
    """
    if channel == "C1-subject":
        bribed = 0
    elif channel == "C2-account":
        bribed = int(n_accounts)
    else:
        bribed = len(set(Tr.edges[rows, 0].tolist()))
    return dict(edges=int(len(rows)), raters_bribed=int(bribed),
                accounts=int(n_accounts))


def plan_direct_full(Tr: "C.Graph", v: int, channel: str = "C1-subject") -> Plan:
    """§7 / K1 full direct budget: every distrust in-edge of T."""
    rows = direct_rows(Tr, v)
    p = Plan(np.sort(Tr.rid[rows]), channel, "oracle-free", n_candidates=len(rows))
    p.cost = cost_of(Tr, rows, channel)
    return p


def plan_direct_greedy(scorer, Tr, v, k, wc: WeightChannel | None = None,
                       channel: str = "C2-rater-edge", knowledge: str = "whitebox") -> Plan:
    rows = direct_rows(Tr, v)
    units = [(int(r), np.array([r])) for r in rows]
    weight = ({int(r): float(x) for r, x in zip(rows, wc.influence(rows))} if wc else None)
    p = greedy(scorer, units, k, weight=weight, channel=channel, knowledge=knowledge, Tr=Tr)
    p.cost = cost_of(Tr, np.where(p.mask(Tr))[0], channel)
    return p


def plan_account(scorer, Tr, v, r_budget, knowledge: str = "whitebox") -> Plan:
    units = [(int(a), account_rows(Tr, a)) for a in account_candidates(Tr, v)]
    p = greedy(scorer, units, r_budget, channel="C2-account", knowledge=knowledge, Tr=Tr)
    n_acc = sum(1 for t in p.trace if "unit" in t)
    p.cost = cost_of(Tr, np.where(p.mask(Tr))[0], "C2-account", n_acc)
    return p


def plan_indirect(scorer, Tr, v, k, wc: WeightChannel | None = None,
                  knowledge: str = "whitebox", max_candidates: int = 0,
                  rng: np.random.Generator | None = None) -> Plan:
    """C3 greedy, budget-matched to ``k`` edges. ``max_candidates`` > 0 subsamples
    UNIFORMLY at random (never by row id) and records how many were dropped."""
    rows = indirect_rows(Tr, v)
    dropped = 0
    if max_candidates and len(rows) > max_candidates:
        rng = rng or np.random.default_rng(0)
        dropped = len(rows) - max_candidates
        rows = np.sort(rng.choice(rows, max_candidates, replace=False))
    units = [(int(r), np.array([r])) for r in rows]
    weight = ({int(r): float(x) for r, x in zip(rows, wc.influence(rows))}
              if (wc and len(rows)) else None)
    p = greedy(scorer, units, k, weight=weight, channel="C3-indirect",
               knowledge=knowledge, Tr=Tr)
    p.cost = cost_of(Tr, np.where(p.mask(Tr))[0], "C3-indirect")
    p.cost["candidates_dropped"] = int(dropped)
    assert not np.isin(p.mask(Tr).nonzero()[0],
                       np.where((Tr.edges[:, 0] == v) | (Tr.edges[:, 1] == v))[0]).any(), \
        "C3 plan touches an edge incident to T"
    return p


# ---------------------------------------------------------------- knowledge
def surrogate(D: "C.Dataset", Tr: "C.Graph", tt: int, seed: int, epochs: int,
              n_probe: int = 256):
    """Grey-box attacker: its own LocalGDTE and probe panel, trained on the public
    signed graph (the training window), never on or with the victim's weights."""
    m = C.train(C.make_model(D, Tr, seed + 1000, tt), Tr, tt, epochs=epochs)
    return m, C.Panel(Tr, n_probe=n_probe, seed=4321 + seed)
