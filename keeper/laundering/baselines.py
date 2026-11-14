"""Second learned model and non-learned baselines (proposal §6 "Models").

* :class:`SGCN` -- Signed Graph Convolutional Network (Derr, Ma, Tang, ICDM
  2018) via PyG ``SignedGCN``, with PyG's reference settings (spectral input
  features of dim 64, 2 layers, lambda 5, Adam lr 0.01, weight decay 5e-4).
  Unlike LocalGDTE it passes messages over the WHOLE (static) signed graph
  using balance theory, so erasures can propagate beyond one snapshot. That
  makes it the model on which the indirect channel C3 can be non-zero.
* :func:`eigentrust` -- an EigenTrust-style global trust score (Kamvar,
  Schlosser, Garcia-Molina, WWW 2003), signed variant: local trust
  c_ij = max(#pos_ij - #neg_ij, 0), row-normalised, with uniform pre-trust
  and teleport alpha.
* Counting scores are in :mod:`laundering.metrics`.

All learned-model operators mirror LocalGDTE's: unlink (structure-derived
features recomputed, parameters kept) and exact retraining from the same seed.
"""
import numpy as np
import scipy.sparse as sp
import torch

from keeper.laundering import core as C
from keeper.laundering import signed as S


class SGCN:
    def __init__(self, n: int, seed: int, feat_dim: int = 64, hidden: int = 64,
                 layers: int = 2, lamb: float = 5.0, lr: float = 0.01,
                 wd: float = 5e-4, epochs: int = 200):
        from torch_geometric.nn import SignedGCN
        C.require_gpu()
        self.n, self.seed, self.epochs, self.lr, self.wd = n, seed, epochs, lr, wd
        self.feat_dim = feat_dim
        C.seed_all(seed)
        self.model = SignedGCN(feat_dim, hidden, num_layers=layers, lamb=lamb).to(C.DEV)

    @staticmethod
    def _split(G: "C.Graph"):
        neg = S.neg_mask(G.labels)
        pos_ei = torch.as_tensor(G.edges[~neg].T.copy(), dtype=torch.long, device=C.DEV)
        neg_ei = torch.as_tensor(G.edges[neg].T.copy(), dtype=torch.long, device=C.DEV)
        return pos_ei, neg_ei

    def features(self, G: "C.Graph") -> torch.Tensor:
        """Spectral features of the signed graph, computed exactly ON THE GPU.

        Same matrix as PyG's ``SignedGCN.create_spectral_features``: the
        symmetrised signed adjacency, values +2 (pos) / 0 (neg) coalesced by sum,
        then minus 1. PyG approximates its truncated SVD with scikit-learn's
        randomized TruncatedSVD on the CPU, which made every unlink and retrain
        CPU-bound (measured: 8-80 min per target with 12 jobs on 32 cores). The
        matrix is symmetric, so its truncated SVD is exactly the eigenvectors
        with the largest |eigenvalue|: one dense ``eigh`` on the GPU, which is
        deterministic and exact (0.2 s on Alpha, 1.3 s on Wiki-RfA). Randomized
        GPU SVD was rejected: the trailing components never converged on
        Wiki-RfA, and sparse CUDA matmul is non-deterministic.
        """
        if self.n > 20_000:
            raise ValueError("dense spectral features need n <= 20,000 nodes")
        pos_ei, neg_ei = self._split(G)
        ei = torch.cat([pos_ei, neg_ei], 1)
        val = torch.cat([torch.full((pos_ei.shape[1],), 2.0, device=C.DEV),
                         torch.zeros(neg_ei.shape[1], device=C.DEV)])
        ei = torch.cat([ei, ei.flip(0)], 1)
        val = torch.cat([val, val])
        A = torch.sparse_coo_tensor(ei, val, (self.n, self.n)).coalesce()
        A = torch.sparse_coo_tensor(A.indices(), A.values() - 1.0,
                                    (self.n, self.n)).to_dense()
        w, U = torch.linalg.eigh(A)
        top = torch.argsort(w.abs(), descending=True, stable=True)[:self.feat_dim]
        return U[:, top].contiguous()

    def fit(self, G: "C.Graph") -> "SGCN":
        C.seed_all(self.seed)
        pos_ei, neg_ei = self._split(G)
        x = self.features(G)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        for _ in range(self.epochs):
            self.model.train()
            opt.zero_grad()
            z = self.model(x, pos_ei, neg_ei)
            loss = self.model.loss(z, pos_ei, neg_ei)
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def embed(self, G: "C.Graph", recompute_features: bool = True) -> torch.Tensor:
        self.model.eval()
        pos_ei, neg_ei = self._split(G)
        x = self.features(G) if recompute_features else self._x_cache(G)
        return self.model(x, pos_ei, neg_ei)

    def _x_cache(self, G):
        if not hasattr(self, "_x"):
            self._x = self.features(G)
        return self._x

    @torch.no_grad()
    def p_trust(self, z, src, dst) -> torch.Tensor:
        """P(positive | an edge src->dst exists): softmax over the pos/neg logits."""
        ei = torch.as_tensor(np.stack([np.asarray(src), np.asarray(dst)]), device=z.device)
        logp = self.model.discriminate(z, ei)       # log-softmax over [pos, neg, none]
        return torch.softmax(logp[:, :2], dim=1)[:, 0]

    @torch.no_grad()
    def reputation(self, z, panel: "C.Panel", nodes) -> np.ndarray:
        nodes = np.asarray(nodes)
        s, d = panel._pairs(nodes)
        p = self.p_trust(z, s, d).cpu().numpy()
        counts = panel._mask(nodes).sum(0)
        out, pos = np.zeros(len(nodes)), 0
        for j, c in enumerate(counts):
            out[j] = p[pos:pos + int(c)].mean()
            pos += int(c)
        return out


class SGCNScorer:
    """Attack scorer on SGCN (unlink-and-forward gain, frozen features for speed)."""

    def __init__(self, m: SGCN, Tr: "C.Graph", panel: "C.Panel", v: int):
        self.m, self.Tr, self.panel, self.v = m, Tr, panel, int(v)
        self.base = self.rep(np.zeros(len(Tr.edges), bool))

    def rep(self, removed):
        z = self.m.embed(self.Tr.subset(~removed), recompute_features=False)
        return float(self.m.reputation(z, self.panel, [self.v])[0])

    def unlink_gain(self, removed):
        return self.rep(removed) - self.base


def eigentrust(G: "C.Graph", n: int, alpha: float = 0.15, iters: int = 200,
               tol: float = 1e-10) -> np.ndarray:
    """Signed EigenTrust-style global trust over the nodes of G (sums to 1)."""
    neg = S.neg_mask(G.labels)
    w = np.where(neg, -1.0, 1.0)
    M = sp.coo_matrix((w, (G.edges[:, 0], G.edges[:, 1])), shape=(n, n)).tocsr()
    M.sum_duplicates()
    M.data = np.maximum(M.data, 0.0)
    M.eliminate_zeros()
    rs = np.asarray(M.sum(1)).ravel()
    active = np.unique(G.edges.ravel())
    p = np.zeros(n)
    p[active] = 1.0 / len(active)
    Dinv = sp.diags(np.where(rs > 0, 1.0 / np.maximum(rs, 1e-12), 0.0))
    P = (Dinv @ M).T.tocsr()                   # column-stochastic where rows had mass
    dangling = rs == 0
    t = p.copy()
    for _ in range(iters):
        t_new = (1 - alpha) * (P @ t + p * t[dangling].sum()) + alpha * p
        if np.abs(t_new - t).sum() < tol:
            t = t_new
            break
        t = t_new
    return t
