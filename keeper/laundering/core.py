"""Corrected core for the reputation-laundering study.

Separate from ``laundering/common.py``, which is kept as a labelled legacy
reproduction control. This module fixes five defects that the pre-probe audit
found, each of which invalidated the earlier numbers:

  D1  HEAD  thecode/local_model.py:74-76 does
          ``Parameter(torch.Tensor(...)).to(device)``
      and ``Tensor.to()`` returns a plain Tensor, so ``regression_weights`` was
      never in ``named_parameters()`` or ``state_dict()``: the 64x2 output head
      stayed at its xavier init, was never optimised, and never serialised.
      ``make_model(..., legacy_head=True)`` reproduces that behaviour on purpose.

  D2  ALIGNMENT  bitcoinalpha.csv is NOT in time order; pairing it with
      bitcoinalpha-rating.txt row-by-row in file order mis-pairs 2,852 signs.
      The rating file is the model's edge source, so we take edges from the
      rating file and attach timestamps by stable-sorting the CSV and
      row-pairing. Verified: 0 sign mismatches, consistent id bijection.

  D3  LEAKAGE  features and class weights were built from the whole graph
      including the held-out test snapshot. Everything here is a function of the
      retained TRAINING subgraph only, so exact unlearning is genuinely
      "retrain on the erased graph".

  D4  SNAPSHOTS  equal-count boundaries cut through tied timestamps (18/21 on
      Alpha), and ``window()`` clamped an empty leading snapshot's end index to
      0, which pulled a later edge into the first snapshot. Boundaries here are
      tie-avoiding, and empty snapshots stay empty (index -1, never clamped).

  D5  MEASUREMENT  the decoder is additive in source and target
      (concat + one linear head), so a source-panel change alone moves a
      target's score. The probe panel is frozen before any deletion, self-pairs
      are excluded, and the logit change is decomposed into source and target
      terms.
"""
import os
import sys
import copy
import json
import random
import socket
import hashlib
import datetime
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root

from keeper.backbone.local_model import LocalGDTE          # noqa: E402
from keeper.backbone.utils import setup_features2          # noqa: E402
from keeper.backbone.arg_parser import parameter_parser     # noqa: E402
from keeper.laundering import signed as S                  # noqa: E402

ALLOW_CPU_ENV = "FEDUNL_ALLOW_CPU"


def require_gpu() -> None:
    """Refuse to build or run a model without CUDA.

    The project rule is that every model computation runs on a GPU. A silent
    CPU fallback (the old behaviour of ``select_device``) would let a sweep run
    for days on the wrong device and still write results. ``FEDUNL_ALLOW_CPU=1``
    is the only override and exists for unit tests of non-model logic.
    """
    if torch.cuda.is_available():
        return
    if os.environ.get(ALLOW_CPU_ENV) == "1":
        return
    raise RuntimeError(
        "CUDA is not available, and every model must run on a GPU. Check "
        "CUDA_VISIBLE_DEVICES and the interpreter (~/.conda/envs/ps). Set "
        f"{ALLOW_CPU_ENV}=1 only for unit tests.")


def select_device() -> torch.device:
    """Device for this process, choosing among the GPUs CUDA_VISIBLE_DEVICES exposes.

    Why this exists: on a multi-GPU host (the A6000 box) a hardcoded
    ``cuda:0`` -- or fedlab's ``get_best_gpu_index()``, which ranks by *free
    memory* -- makes every concurrent job pick the same GPU, so N jobs OOM
    together instead of spreading across N cards.

    The scheduler (``laundering/run_multigpu.sh``) gives every job a disjoint
    ``CUDA_VISIBLE_DEVICES`` slice, so "pick the first visible device" is
    exactly right and needs no index arithmetic:

        CUDA_VISIBLE_DEVICES=2,3  ->  cuda:0 is physical GPU 2
        CUDA_VISIBLE_DEVICES=3    ->  cuda:0 is physical GPU 3

    ``HERMES_GPU`` is honoured ONLY when CUDA_VISIBLE_DEVICES is unset, where it
    names a device directly. When both are set the visible mask wins, because
    that is what actually constrains the process. An out-of-range
    ``HERMES_GPU`` raises rather than silently falling back to cuda:0, so a typo
    cannot quietly redirect a job onto a GPU that is already busy.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    want = os.environ.get("HERMES_GPU", "").strip()
    if cvd:
        n_visible = len([d for d in cvd.split(",") if d.strip()])
        if want:
            # Mask present: it wins. Validate so a bad pair is loud.
            named = [d.strip() for d in cvd.split(",") if d.strip()]
            if want not in named:
                raise RuntimeError(
                    f"HERMES_GPU={want} is not in CUDA_VISIBLE_DEVICES={cvd!r}; "
                    f"visible physical GPUs are {named}. Fix the scheduler or "
                    f"unset HERMES_GPU.")
        return torch.device("cuda:0")
    if want:
        count = torch.cuda.device_count()
        if not want.isdigit() or not (0 <= int(want) < count):
            raise RuntimeError(
                f"HERMES_GPU={want!r} is not a valid device index (0..{count - 1})")
        return torch.device(f"cuda:{want}")
    return torch.device("cuda:0")


DEV = select_device()
RES = os.path.join(ROOT, "results", "laundering")   # stored results (laundering/results/ before 2026-10-26)
os.makedirs(RES, exist_ok=True)
OVERWRITE_ENV = "KEEPER_OVERWRITE"   # "1": dump() and keeper.scem.stage_b may replace stored results

# Set by apply_memory_profile() when a config or env asks for a 2-GPU split.
# None means every model is single-GPU, which is the historical behaviour.
DEFAULT_STRATEGY = None


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------------------- data
@dataclass
class Dataset:
    """Signed edges from :func:`laundering.signed.load_signed`, sorted by time.

    ``labels`` is the model's one-hot ``[trust, distrust]`` derived from each
    rating's sign; ``rating`` keeps the raw signed rating (sign and magnitude).
    ``meta`` records the zero/self-loop policy and how many edges it dropped.
    """
    name: str
    edges: np.ndarray      # [E, 2] int64
    labels: np.ndarray     # [E, 2] float32 one-hot, column S.NEG_COL == distrust
    time: np.ndarray       # [E] int64
    n: int                 # full node roster (deployment assumption, see notes)
    T: int                 # number of snapshots
    cfg: dict = field(repr=False, default_factory=dict)
    rating: np.ndarray = field(repr=False, default=None)
    meta: dict = field(repr=False, default_factory=dict)

    @property
    def tt(self) -> int:
        """Test snapshot index; snapshots [0, tt) train, snapshot tt is held out."""
        return self.T - 1

    def y_neg(self) -> np.ndarray:
        """Cross-entropy class per edge: 1 for a negative (distrust) rating."""
        return S.class_ids(self.labels)


def snapshot_ends(times: np.ndarray, n_slots: int, tie_avoiding: bool = True) -> list:
    """Inclusive last-edge index of each snapshot, equal-count in edge numbers.

    With ``tie_avoiding`` a cut that lands inside a group of equal timestamps is
    advanced to the end of that group, so one instant is never split across two
    snapshots. A snapshot may end up empty; its end is then the cumulative count
    minus one, which can be -1 for a leading empty snapshot and must NOT be
    clamped to 0 (that was defect D4).
    """
    n = len(times)
    if n == 0:
        return [-1] * n_slots
    base, rem = divmod(n, n_slots)
    ends, cum = [], 0
    for i in range(n_slots):
        end = cum + base + (1 if i < rem else 0) - 1
        if tie_avoiding and 0 <= end < n - 1:
            while end + 1 < n and times[end + 1] == times[end]:
                end += 1
        ends.append(end)
        cum += base + (1 if i < rem else 0)
    ends[-1] = n - 1
    return ends


def quantile_ends(times: np.ndarray, n_slots: int) -> list:
    """Inclusive last-edge index of each snapshot, equal-span in TIME.

    Equal-count cuts (``snapshot_ends``) break on datasets whose timestamps are
    heavily tied: Epinions has 578,996 of 841,372 edges (68.8%) carrying the
    single crawl timestamp t0 = 979081200, so any equal-count split leaves one
    snapshot absorbing that block and every subsequent cut landing inside it
    produces an empty snapshot (measured: T=40 leaves 26 of 40 empty).

    This variant cuts on timestamp quantiles instead, so each snapshot covers a
    comparable span of wall-clock time. The cost is unequal edge counts, and the
    consequence is stated rather than hidden: snapshot 0 is the huge t0 block
    (69% of edges) and the remaining snapshots hold ~21k each, so models see a
    strongly size-imbalanced temporal partition. That is a property of the data,
    not a defect, and any epoch count chosen for the Bitcoin datasets transfers.

    Returns the same shape as ``snapshot_ends``: inclusive end indices, and
    -1 for any snapshot that ends up empty (never clamped, per D4).
    """
    n = len(times)
    if n == 0:
        return [-1] * n_slots
    uniq = np.unique(times)
    # last index (in the time-sorted array) of each unique timestamp
    last_idx = np.searchsorted(times, uniq, side="right") - 1
    # pick T-1 interior cuts on the unique-time axis, always including the last
    cut_pos = np.unique(np.linspace(0, len(uniq) - 1, n_slots).round().astype(int))
    ends = [int(last_idx[p]) for p in cut_pos]
    # fill any skipped slots by repeating the previous end (keeps len == n_slots)
    # -- only happens if n_slots > number of unique timestamps
    while len(ends) < n_slots:
        ends.append(ends[-1])
    ends = ends[:n_slots]
    ends[-1] = n - 1
    return ends


def ends_for(D: "Dataset", tie_avoiding: bool = True) -> list:
    """Dispatch on ``cfg['fl']['snapshot_mode']``: 'count' (default) or 'quantile'."""
    mode = (D.cfg.get("fl", {}).get("snapshot_mode") or "count").lower()
    if mode == "count":
        return snapshot_ends(D.time, D.T, tie_avoiding)
    if mode == "quantile":
        return quantile_ends(D.time, D.T)
    raise ValueError(f"{D.name}: unknown snapshot_mode {mode!r}")


def load(ds: str, tie_avoiding: bool = True, verify: bool = True) -> Dataset:
    """Load a dataset through the sign-aware loader.

    The CSV/rating-file pairing is always verified (sign of every row and the
    endpoint id bijection, see :func:`laundering.signed.load_signed`); ``verify``
    and ``tie_avoiding`` are kept for call-site compatibility. ``tie_avoiding``
    is applied when snapshots are cut (:func:`ends_for`), not here.
    """
    cfg = json.load(open(os.path.join(ROOT, "configs", f"{ds}.json")))
    cfg.setdefault("dataset", ds)
    se = S.load_signed(cfg, ROOT)
    n = int(cfg.get("num_nodes") or se.edges.max() + 1)
    T = int(cfg["fl"]["time_slots"])
    return Dataset(ds, se.edges, se.labels, se.time, n, T, cfg,
                   rating=se.rating, meta=se.meta)


@dataclass
class Graph:
    """A retained edge set over a fixed node roster, with snapshot assignment.

    ``rid`` is each edge's row id in the ORIGINAL dataset. It survives subsetting
    and re-snapshotting, so a forget set can be mapped between the full graph,
    its training split, and a graph rebuilt from erased data (exact retraining).
    """
    edges: np.ndarray
    labels: np.ndarray
    time: np.ndarray
    snap: np.ndarray
    n: int
    rid: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.rid is None:
            self.rid = np.arange(len(self.edges), dtype=np.int64)

    def subset(self, keep: np.ndarray) -> "Graph":
        return Graph(self.edges[keep], self.labels[keep], self.time[keep],
                     self.snap[keep], self.n, self.rid[keep])

    def mask_of(self, rids) -> np.ndarray:
        """Boolean mask over THIS graph's rows for the given original row ids."""
        return np.isin(self.rid, np.asarray(rids, dtype=np.int64))


def full_graph(D: Dataset, tie_avoiding: bool = True,
               rid: np.ndarray | None = None) -> Graph:
    ends = ends_for(D, tie_avoiding)
    snap = np.zeros(len(D.edges), np.int64)
    prev = -1
    for s, e in enumerate(ends):
        snap[prev + 1:e + 1] = s
        prev = e
    return Graph(D.edges, D.labels, D.time, snap, D.n, rid)


def erased_dataset(D: Dataset, erase_rids) -> tuple:
    """The dataset a platform would hold after erasing the given rows.

    Returns ``(D', rid')``: the same signed edges minus ``erase_rids``, in the
    same time order, and the original row ids of what remains. Running the full
    pipeline on ``D'`` (snapshot cutting included) is exact retraining.
    """
    keep = ~np.isin(np.arange(len(D.edges)), np.asarray(erase_rids, dtype=np.int64))
    Dp = Dataset(D.name, D.edges[keep], D.labels[keep], D.time[keep], D.n, D.T,
                 D.cfg, rating=None if D.rating is None else D.rating[keep],
                 meta=dict(D.meta, n_erased=int((~keep).sum())))
    return Dp, np.where(keep)[0].astype(np.int64)


def train_split(G: Graph, tt: int, keep: np.ndarray | None = None):
    """Retained TRAINING edges only: snapshots [0, tt). Never touches snapshot tt."""
    m = G.snap < tt
    if keep is not None:
        m = m & keep
    return G.subset(m)


def apply_memory_profile(D: Dataset) -> dict:
    """Set the S3R memory knobs from the config, and report what was applied.

    Large graphs OOM on the default SSM path: at Epinions scale the [B, L, D, N]
    tensors dA and dBu are 2.64 GiB EACH (B = 131,828 nodes, L = 21 slots,
    D = 64, N = 4) and forward+backward needs several live at once, measured
    21.3 GiB peak then CUDA OOM at 23.46 GiB. ``s3r_frugal_scan`` routes the scan
    through ``laundering/frugal_ssm.py``, which builds those tensors one
    node-chunk at a time. Verified bit-exact against the default path on values
    AND gradients, and on Bitcoin the forward loss is unchanged to all 12
    printed digits, so enabling it for a large graph does not alter results.

    Config keys live under ``fl`` and are recorded in every probe JSON:
        fl.s3r_frugal_scan : bool, default false
        fl.s3r_node_chunk  : int,  default 4096 (measured: 4096 -> 16.5 GiB peak,
                             8192 -> OOM)

    MULTI-GPU (this machine has several). ``fl.pipeline_gpus`` splits ONE job
    across two GPUs: the structural layer runs on the first and the S3R block --
    where the memory actually goes -- on the second. Measured on bitcoinalpha:
    single-GPU peak 1.27 GiB vs pipeline 0.05 / 1.27 GiB, with logits identical to
    5 decimals. Set it to the number of GPUs to use, e.g.
    ``fl.pipeline_gpus = 2``, or via env ``HERMES_PIPELINE_GPUS=2``.

    Pipeline mode cannot be silently wrong: if fewer GPUs are visible than it
    asks for, this raises instead of degrading, because a silent fallback would
    mean running an OOM-prone configuration believing it was split.
    """
    fl = D.cfg.get("fl", {})
    global DEFAULT_STRATEGY
    frugal = bool(fl.get("s3r_frugal_scan", False))
    chunk = int(fl.get("s3r_node_chunk", 4096))
    ngpu = fl.get("pipeline_gpus")
    if os.environ.get("HERMES_PIPELINE_GPUS"):
        ngpu = int(os.environ["HERMES_PIPELINE_GPUS"])
    # An explicit env override still wins, so a sweep can be re-run without
    # editing configs.
    if os.environ.get("S3R_FRUGAL_SCAN"):
        frugal = os.environ["S3R_FRUGAL_SCAN"] == "1"
    if os.environ.get("S3R_NODE_CHUNK"):
        chunk = int(os.environ["S3R_NODE_CHUNK"])
    os.environ["S3R_FRUGAL_SCAN"] = "1" if frugal else "0"
    os.environ["S3R_NODE_CHUNK"] = str(chunk)

    strategy = None
    if ngpu:
        ngpu = int(ngpu)
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible < ngpu:
            raise RuntimeError(
                f"{D.name}: fl.pipeline_gpus={ngpu} but only {visible} GPU(s) "
                f"visible. Either expose more GPUs or set pipeline_gpus=1.")
        strategy = {"mode": "pipeline", "gpu_structural": 0, "gpu_s3r": 1}
    DEFAULT_STRATEGY = strategy
    return {"s3r_frugal_scan": frugal, "s3r_node_chunk": chunk,
            "pipeline_gpus": int(ngpu) if ngpu else 1,
            "visible_gpus": int(torch.cuda.device_count())
            if torch.cuda.is_available() else 0,
            "device": str(DEV), "strategy": strategy}


# ------------------------------------------------------------------- preprocessing
def features(G: Graph, seed: int, dim: int) -> torch.Tensor:
    """D3: features are a pure function of the retained training graph."""
    X = setup_features2({"edges": G.edges, "ncount": G.n, "labels": G.labels},
                        seed=seed, dim=dim)
    return torch.from_numpy(X).float().to(DEV)


def class_weights(G: Graph) -> torch.Tensor:
    """D3: inverse frequency over the retained training labels, not the full graph."""
    y = S.class_ids(G.labels)
    cnt = np.bincount(y, minlength=2).astype(np.float64).clip(min=1.0)
    return torch.FloatTensor(1.0 / cnt * len(y)).to(DEV)


def window(G: Graph, tt: int):
    """Edges of snapshots [0, tt) sorted by snapshot, plus per-snapshot inclusive
    end indices. D4: no clamping, so an empty snapshot yields end == cumulative-1
    (possibly -1) and contributes an empty slice."""
    s = G.snap
    e, l = G.edges, G.labels
    order = np.argsort(s, kind="mergesort")
    e, l, s = e[order], l[order], s[order]
    ends = [int((s <= t).sum()) - 1 for t in range(tt)]
    et = torch.from_numpy(e.T.copy()).long().to(DEV)
    lt = torch.from_numpy(l.copy()).float().to(DEV)
    return et, lt, ends


# ------------------------------------------------------------------------- model
def make_args(D: Dataset, seed: int, tt: int):
    a = parameter_parser(D.name)
    a.dataset = D.name
    a.seed = seed
    a.attention_head = D.cfg["n_heads"]
    a.layers = D.cfg["layers"]
    a.time_slots = D.T
    a.train_time_slots = tt
    a.max_snapshots_inaClient = tt
    a.attack = False
    a.dropout = 0.0
    return a


def set_deterministic(on: bool = True) -> None:
    """Bitwise-reproducible training (measured: same-seed max|dRep| 3.1e-3 -> 0).

    Needs CUBLAS_WORKSPACE_CONFIG set before CUDA initialises; the probe sets it
    at import. LocalGDTE.set_seed() turns cudnn.benchmark back on, so make_model
    re-pins it after construction.
    """
    torch.use_deterministic_algorithms(on, warn_only=True)
    torch.backends.cudnn.deterministic = on
    torch.backends.cudnn.benchmark = False
    # Filling uninitialised memory is a debugging aid of deterministic mode, not
    # part of determinism: turning it off was measured bit-identical on L40S
    # (Epinions epoch 2.86 s -> 2.57 s).
    fd = getattr(getattr(torch.utils, "deterministic", None),
                 "fill_uninitialized_memory", None)
    if fd is not None:
        torch.utils.deterministic.fill_uninitialized_memory = False


def make_model(D: Dataset, G: Graph, seed: int, tt: int,
               legacy_head: bool = False,
               strategy: dict | None = None,
               self_term: bool = False,
               extra_features: np.ndarray | None = None) -> LocalGDTE:
    """Build a model whose every input is derived from retained graph G.

    ``extra_features`` ([G.n, k], Task 4.3): columns appended to the input features, for E5's withdrawal counts
    (keeper.cppr.baselines). The structure layer then projects the wider input with its own linear layer. The
    default None leaves the model unchanged, bit for bit (tests/test_cppr_baselines.py).

    D1 is now fixed at the source (thecode/local_model.py registers the head).
    ``legacy_head=True`` sets ``args.legacy_unregistered_head`` to reproduce the
    old untrained xavier head as a control arm.

    ``self_term=True`` sets ``args.self_term``: every structure layer then adds
    one self loop per node to its incoming aggregation, in training and in any
    inference graph (upgrade G4, the E9a ablation; see
    keeper/backbone/structure_layer.py). The default False leaves the model
    unchanged, bit for bit (tests/test_self_term.py).

    ``strategy`` is forwarded to LocalGDTE. Pass
    ``{"mode": "pipeline", "gpu_structural": 0, "gpu_s3r": 1}`` to run the
    structural layer and the S3R block on DIFFERENT GPUs. This is what makes a
    single job fit on a multi-GPU host: the S3R side is where the memory goes
    (see laundering/frugal_ssm.py), so splitting it halves the peak. Indices are
    into the *visible* device list, i.e. after CUDA_VISIBLE_DEVICES is applied.
    Verified to reproduce the single-GPU result to atol=1e-4 -- see
    laundering/RESUME.md. Default None keeps the single-GPU path byte-identical.

    When ``strategy`` is None, the module-level ``DEFAULT_STRATEGY`` set by
    :func:`apply_memory_profile` is used, so a multi-GPU sweep does not need every
    call site edited. It is None unless a config or env asked for a split.
    """
    require_gpu()
    strategy = strategy if strategy is not None else DEFAULT_STRATEGY
    seed_all(seed)
    a = make_args(D, seed, tt)
    a.legacy_unregistered_head = bool(legacy_head)
    a.self_term = bool(self_term)
    X = features(G, seed, a.layers[-1])
    if extra_features is not None:
        X = torch.cat([X, torch.as_tensor(np.asarray(extra_features), dtype=X.dtype, device=X.device)], dim=1)
    m = LocalGDTE(DEV, a, X, num_labels=2, index_list=[0],
                  class_weight=class_weights(G), strategy=strategy)
    if strategy and strategy.get("mode") == "pipeline":
        # Do NOT .to(DEV) here: that would drag the S3R back onto the primary
        # device and silently undo the split. LocalGDTE._build_model has already
        # placed each sub-module on its own GPU; this only pins the parameters
        # that are not owned by a sub-module (the output head).
        m.regression_weights.data = m.regression_weights.data.to(m.gpu_s3r)
        if getattr(m, "attention_temp", None) is not None:
            m.attention_temp.data = m.attention_temp.data.to(m.gpu_s3r)
    else:
        m = m.to(DEV)
    is_param = isinstance(m.regression_weights, torch.nn.Parameter)
    assert is_param != bool(legacy_head), \
        f"head registration {is_param} does not match legacy_head={legacy_head}"
    torch.backends.cudnn.benchmark = False
    return m


def self_term_of(m: LocalGDTE) -> bool:
    """The G4 flag model m was built with (False for a model built before
    Task 3.4). A rebuild of m through make_model passes it on, so that exact
    retraining and replicates keep m's architecture."""
    return bool(getattr(m.args, "self_term", False))


def head_is_trained(m: LocalGDTE) -> bool:
    return "regression_weights" in dict(m.named_parameters())


def train(m: LocalGDTE, G: Graph, tt: int, epochs: int = 150,
          lr: float = 0.005, wd: float = 1e-5) -> LocalGDTE:
    et, lt, ends = window(G, tt)
    m.index_list, m.args.train_time_slots = ends, tt
    y = lt[:, S.NEG_COL].long()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        loss, _ = m(et, y, lt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
    return m


@torch.no_grad()
def embed(m: LocalGDTE, G: Graph, tt: int) -> torch.Tensor:
    et, lt, ends = window(G, tt)
    m.index_list, m.args.train_time_slots = ends, tt
    m.eval()
    _, z = m(et, lt[:, S.NEG_COL].long(), lt, return_loss=False)
    return z


# ---------------------------------------------------------------------- scoring
def model_device(m: LocalGDTE) -> torch.device:
    """Device the model's OUTPUT lives on.

    Under ``strategy={"mode": "pipeline", ...}`` the embeddings come back on
    ``m.gpu_s3r``, not on the primary device, so scoring helpers must follow the
    model rather than a module-level constant. Returns the primary device when
    the model is single-GPU, which keeps every previously measured number
    identical.
    """
    if getattr(m, "pipeline_mode", False):
        return m.gpu_s3r
    return DEV


@torch.no_grad()
def logit_trust(m: LocalGDTE, z: torch.Tensor, src, dst) -> torch.Tensor:
    """logit(trust) - logit(distrust) = a.z_src + b.z_dst (additive, D5)."""
    dev = z.device if z is not None else model_device(m)
    W = m.regression_weights.to(dev)
    s = torch.as_tensor(np.asarray(src), device=dev)
    d = torch.as_tensor(np.asarray(dst), device=dev)
    f = torch.cat((z[s], z[d]), 1)
    out = torch.empty(len(s), device=dev)
    for i in range(0, len(s), 1 << 20):
        sl = slice(i, i + (1 << 20))
        out[sl] = (f[sl] @ W)[:, 0] - (f[sl] @ W)[:, 1]
    return out


@torch.no_grad()
def p_trust(m: LocalGDTE, z: torch.Tensor, src, dst) -> torch.Tensor:
    dev = z.device if z is not None else model_device(m)
    W = m.regression_weights.to(dev)
    s = torch.as_tensor(np.asarray(src), device=dev)
    d = torch.as_tensor(np.asarray(dst), device=dev)
    out = []
    for i in range(0, len(s), 1 << 20):
        sl = slice(i, i + (1 << 20))
        f = torch.cat((z[s[sl]], z[d[sl]]), 1)
        out.append(F.softmax(f @ W, 1)[:, 0])
    return torch.cat(out)


class Panel:
    """Frozen source panel for reputation queries (D5).

    Chosen once from the PRE-deletion training graph. Self-pairs (a target that
    is itself a panel source) are dropped per target, so the panel and the
    target cohort are identical in every arm.

    ``grid`` evaluates the full [len(sources), len(nodes)] cross product and
    returns per-target means with the self-pair masked out. The mask is
    ``sources[:, None] != nodes[None, :]``; averaging is done column-wise so a
    target in the panel simply averages over 255 instead of 256 sources.
    """

    def __init__(self, G: Graph, n_probe: int = 256, seed: int = 1234):
        active = np.unique(G.edges.ravel())
        rng = np.random.default_rng(seed)
        self.sources = np.sort(rng.choice(active, min(n_probe, len(active)),
                                           replace=False))
        self.active = active

    def _mask(self, nodes: np.ndarray) -> np.ndarray:
        """Boolean [len(sources), len(nodes)] keep-mask, self-pairs False."""
        nodes = np.asarray(nodes)
        return self.sources[:, None] != nodes[None, :]

    def _pairs(self, nodes: np.ndarray):
        """Flat (src, dst) arrays in NODE-major order: one column per node,
        each holding exactly ``len(sources)`` pairs (minus that node's self-pair).

        Order matters: `reputation` and `decompose` slice this flat array as
        ``[j * k : (j + 1) * k]``, so node 0 must occupy the first block. The
        previous (sources-major) construction paired every source with every node
        per block, which silently gave each node the mean of the WRONG sources and
        made `reputation` depend on how many other nodes were queried.
        """
        nodes = np.asarray(nodes)
        k = len(self.sources)
        s = np.tile(self.sources, len(nodes))
        d = np.repeat(nodes, k)
        keep = self._mask(nodes)                       # [k, n] rows=sources
        return s[keep.T.ravel()], d[keep.T.ravel()]

    def pairs(self, nodes: np.ndarray):
        return self._pairs(nodes)

    def reputation(self, m: LocalGDTE, z: torch.Tensor, nodes: np.ndarray) -> np.ndarray:
        nodes = np.asarray(nodes)
        s, d = self._pairs(nodes)
        p = torch.sigmoid(logit_trust(m, z, s, d)).cpu().numpy()
        k = len(self.sources)
        keep = self._mask(nodes).ravel()
        counts = keep.reshape(k, len(nodes)).sum(0)
        out = np.zeros(len(nodes))
        pos = 0
        for j in range(len(nodes)):
            c = int(counts[j])
            out[j] = p[pos:pos + c].mean()
            pos += c
        return out

    @torch.no_grad()
    def decompose(self, m: LocalGDTE, z: torch.Tensor, nodes: np.ndarray):
        """Per-node (source-term, target-term) mean log-odds over the masked
        queries, so a shift can be attributed to probe-source movement rather
        than target rehabilitation. The head is (2*dim, num_labels): the first
        `dim` rows score the source embedding, the last `dim` the target.
        """
        nodes = np.asarray(nodes)
        s, d = self._pairs(nodes)
        dev = z.device if z is not None else model_device(m)
        W = m.regression_weights.to(dev)
        dim = z.shape[1]
        a, b = W[:dim, 0] - W[:dim, 1], W[dim:, 0] - W[dim:, 1]
        src = (z[torch.as_tensor(s, device=dev)] @ a).cpu().numpy()
        tgt = (z[torch.as_tensor(d, device=dev)] @ b).cpu().numpy()
        keep = self._mask(nodes).ravel()
        k = len(self.sources)
        counts = keep.reshape(k, len(nodes)).sum(0)
        so = np.zeros(len(nodes))
        to = np.zeros(len(nodes))
        pos = 0
        for j in range(len(nodes)):
            c = int(counts[j])
            so[j] = src[pos:pos + c].mean()
            to[j] = tgt[pos:pos + c].mean()
            pos += c
        return so, to


def task_masks(G: Graph, tt: int) -> tuple:
    """Held-out edge masks for the framework's two evaluation tasks.

    Same definitions as ``thecode/basic_client2.py:_select_eval_edges``:
      Task 1 (observed nodes): both endpoints appear in training snapshots [0, tt).
      Task 2 (unobserved nodes): endpoints that are NEW in the last training
        snapshot tt-1 (present there, absent from [0, tt-1)), paired as in the
        framework: new-old(inactive in tt-1), old(inactive)-new, or new-new.
    Returns (held-out row ids, t1 mask, t2 mask) over those rows.
    """
    rows = np.where(G.snap == tt)[0]
    e = G.edges[rows]
    size = G.n + 1
    seen = np.zeros(size, bool)
    seen[np.unique(G.edges[G.snap < tt].ravel())] = True
    pre = np.zeros(size, bool)
    pre[np.unique(G.edges[G.snap < tt - 1].ravel())] = True
    last = np.zeros(size, bool)
    last[np.unique(G.edges[G.snap == tt - 1].ravel())] = True
    new = last & ~pre
    old_inactive = pre & ~last
    a, b = e[:, 0], e[:, 1]
    t1 = seen[a] & seen[b]
    t2 = (new[a] & old_inactive[b]) | (new[b] & old_inactive[a]) | (new[a] & new[b])
    return rows, t1, t2


def _cls_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    if len(y) == 0 or len(np.unique(y)) < 2:
        return dict(mcc=None, auc=None, n=int(len(y)))
    return dict(mcc=float(matthews_corrcoef(y, (p < 0.5).astype(int))),
                auc=float(roc_auc_score(y, 1 - p)), n=int(len(y)))


@torch.no_grad()
def test_metrics(m: LocalGDTE, z: torch.Tensor, D: Dataset, G: Graph) -> dict:
    """Collateral-utility metrics on the held-out snapshot (proposal §6).

    Top-level ``mcc``/``auc``/``n`` are Task 1, as before; ``t1`` and ``t2`` hold
    both tasks. Labels come from ``G`` itself, so a graph rebuilt by exact
    retraining is scored on its own rows. An undefined metric is ``None``.
    """
    rows, t1, t2 = task_masks(G, D.tt)
    e, y = G.edges[rows], S.class_ids(G.labels[rows])
    p = (p_trust(m, z, e[:, 0], e[:, 1]).cpu().numpy() if len(e)
         else np.zeros(0))
    r1 = _cls_metrics(y[t1], p[t1])
    r2 = _cls_metrics(y[t2], p[t2])
    return dict(mcc=r1["mcc"], auc=r1["auc"], n=r1["n"], t1=r1, t2=r2)


# ------------------------------------------------------------------------ targets
def in_edges(G: Graph) -> tuple:
    """dict target -> (distrust in-edge row ids, trust in-edge row ids), training only."""
    y = S.class_ids(G.labels)
    neg, pos = {}, {}
    for i in range(len(y)):
        (neg if y[i] else pos).setdefault(int(G.edges[i, 1]), []).append(i)
    return neg, pos


def pick_targets(G: Graph, max_neg: int = 3, strict: bool = True):
    """Distrust-majority accounts resting on at most ``max_neg`` distrust in-edges.

    ``strict`` requires a genuine majority; the earlier probe used >=, which
    admitted 41/176 ties on Alpha.
    """
    neg, pos = in_edges(G)
    ok = (lambda a, b: a > b) if strict else (lambda a, b: a >= b)
    t = [v for v in neg if ok(len(neg[v]), len(pos.get(v, []))) and len(neg[v]) <= max_neg]
    return np.array(sorted(t), dtype=np.int64), neg, pos


# ------------------------------------------------------- unlearning operators
def neggrad_unlearn(m0: LocalGDTE, G_retain: Graph, G_full: Graph, tt: int,
                    forget: np.ndarray, *, steps: int = 4, lr: float = 3e-4,
                    lam_forget: float = 0.7,
                    retain_weight: float = 1.0) -> LocalGDTE:
    """NegGrad+ (Kurmanji et al. 2024, arXiv 2412.02432): joint optimisation with
    gradient DESCENT on the retain set and ASCENT on the forget set.

    Four things this MUST get right, all of which a naive version gets wrong:

    1. It is a POST-PROCESS on the trained model ``m0`` (theta*), not a procedure
       run from a fresh initialisation. ``U(theta*, F, D\\F)`` is what produces the
       unlearned model; starting from random weights would just be training.

    2. The message-passing graph must be the RETAINED graph ``G_retain`` (the
       forget edges are deleted), while the forget-set LOSS is still evaluated --
       on those same deleted edges, using embeddings computed from the retained
       graph. This is exactly the regime an approximate-unlearning operator is in:
       the edges are gone from the forward pass but the operator is told which ones
       they were. Evaluating the forget loss on a graph that still contains the
       forget edges would let the operator "see" them through message passing and
       would not be an unlearning operator at all.

    3. Both loss terms are MEANS, so the <=3-edge forget set is not swamped by the
       ~23,000-edge retain set.

    4. The learning rate must be small enough to preserve utility. This was
       measured, not guessed (Alpha, 150 epochs, 3 targets, mean over targets;
       dMCC is retain test MCC relative to theta*'s 0.3478):

           lr=3e-3, steps=1 -> dCE_F +1.47, dMCC -0.093   (utility lost)
           lr=3e-3, steps=4 -> dCE_F +34.8, dMCC -0.303   (model destroyed)
           lr=3e-4, steps=4 -> dCE_F +0.023, dMCC +0.052 (forgets, utility kept)

       So ``lr=3e-4`` is the default: it is the only setting tested that forgets
       the forget set while leaving the model intact. At lr=3e-3 the operator
       "launders" by destroying the model, which would make H3 a comparison
       against a broken baseline rather than against NegGrad+.

    The objective minimised is
        L = retain_weight * mean_CE(retain) - lam_forget * mean_CE(forget)
    """
    forget = np.asarray(forget, bool)
    assert forget.any(), "empty forget set"
    y_all = S.class_ids(G_full.labels)
    assert (y_all[forget] == 1).all(), \
        "forget set must contain only distrust edges"

    m = copy.deepcopy(m0)
    order = np.argsort(G_full.snap, kind="mergesort")

    # edges as the model sees them: the RETAINED graph, sorted by snapshot
    et_r, lt_r, ends_r = window(G_retain, tt)
    y_r = lt_r[:, 1].long()

    # the forget edges in that same sorted order, taken from the FULL graph so the
    # labels are authoritative
    f_sorted = forget[order]
    fe = G_full.edges[order[f_sorted]]
    fl = torch.as_tensor(G_full.labels[order[f_sorted]][:, 1].astype(np.int64),
                         device=DEV)
    f_src = torch.as_tensor(fe[:, 0], device=DEV)
    f_dst = torch.as_tensor(fe[:, 1], device=DEV)

    m.index_list, m.args.train_time_slots = ends_r, tt
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        m.train()
        opt.zero_grad()
        _, z = m(et_r, y_r, lt_r, return_loss=False)
        # Index with the device the EMBEDDINGS live on, not the module-level
        # constant: under pipeline_mode z comes back on gpu_s3r. et_r/fl are
        # built on DEV, so move them rather than moving z.
        zdev = z.device
        W = m.regression_weights.to(zdev)
        pair_r = torch.cat((z[et_r[0].to(zdev)], z[et_r[1].to(zdev)]), 1)
        pair_f = torch.cat((z[f_src.to(zdev)], z[f_dst.to(zdev)]), 1)
        loss = retain_weight * crit(torch.mm(pair_r, W), y_r.to(zdev)) \
            - lam_forget * crit(torch.mm(pair_f, W), fl.to(zdev))
        loss.backward()
        opt.step()
    return m


def forget_loss(m: LocalGDTE, G: Graph, tt: int, forget: np.ndarray) -> float:
    """Mean CE on the named edges, with embeddings computed from G.

    Used to verify that an operator actually forgot: the loss on F must RISE.
    """
    forget = np.asarray(forget, bool)
    idx = np.where(forget)[0]
    assert len(idx), "empty forget set"
    et, lt, ends = window(G, tt)
    m.index_list, m.args.train_time_slots = ends, tt
    m.eval()
    with torch.no_grad():
        _, z = m(et, lt[:, S.NEG_COL].long(), lt, return_loss=False)
        e = G.edges[idx]
        # follow the embedding device (gpu_s3r under pipeline_mode)
        zdev = z.device
        src = torch.as_tensor(e[:, 0], device=zdev)
        dst = torch.as_tensor(e[:, 1], device=zdev)
        pair = torch.cat((z[src], z[dst]), 1)
        return float(F.cross_entropy(
            torch.mm(pair, m.regression_weights.to(zdev)),
            lt[:, S.NEG_COL].long()[idx].to(zdev), reduction="mean"))


# --------------------------------------------------- indirect channel (C3) + greedy
def indirect_candidates(G: Graph, v: int, budget: int = 0, sign: int = 1) -> list:
    """Distrust in-edges of v's neighbours; never an edge incident to v (channel C3).

    ``budget=0`` returns every candidate. Sorted by row id so the candidate list,
    and therefore greedy selection, is deterministic across runs.
    """
    y = S.class_ids(G.labels)
    neigh = {int(G.edges[i, 0]) for i in np.where(G.edges[:, 1] == int(v))[0]}
    out = set()
    for u in neigh:
        for i in np.where((G.edges[:, 1] == u) & (y == sign))[0]:
            if int(G.edges[i, 1]) != int(v) and int(G.edges[i, 0]) != int(v):
                out.add(int(i))
    srt = sorted(out)
    return srt[:budget] if budget else srt


@torch.no_grad()
def _rep_one(m: LocalGDTE, G: Graph, tt: int, v: int, panel: Panel) -> float:
    return float(panel.reputation(m, embed(m, G, tt), np.array([v]))[0])


def greedy_erase(m: LocalGDTE, G: Graph, tt: int, v: int, cands: list, *,
                 panel: Panel, budget: int) -> tuple:
    """Greedy budgeted erasure, one forward pass per candidate (proposal section 5a)."""
    keep = np.ones(len(G.edges), bool)
    chosen, base = [], _rep_one(m, G, tt, v, panel)
    for _ in range(min(budget, len(cands))):
        best_gain, best_id = -np.inf, None
        for i in cands:
            if not keep[i]:
                continue
            keep[i] = False
            gain = _rep_one(m, G.subset(keep), tt, v, panel) - base
            keep[i] = True
            if gain > best_gain + 1e-12:
                best_gain, best_id = gain, int(i)
        if best_id is None:
            break
        keep[best_id] = False
        chosen.append(best_id)
    return keep, chosen


# (directory under ROOT, file suffix, recursive): keeper/**/*.py and configs/*. Until the 2026-10
# move into the keeper package the set was laundering/*.py, thecode/*.py and configs/*.json.
_CODE_GLOBS = (("keeper", ".py", True), ("configs", "", False))
_MD5_CACHE: dict = {}


def _code_files(d: str, ext: str, recursive: bool) -> list:
    """Sorted paths, relative to ``d``, of the files under ``d`` ending in ``ext``; no __pycache__."""
    if not recursive:
        return sorted(fn for fn in os.listdir(d)
                      if fn.endswith(ext) and os.path.isfile(os.path.join(d, fn)))
    out = []
    for top, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        out += [os.path.relpath(os.path.join(top, fn), d) for fn in files if fn.endswith(ext)]
    return sorted(out)


def code_hash() -> str:
    """sha256 over every source and config file the experiments depend on.

    The repo is not under git, so this hash is what ties a result to a code state.
    """
    h = hashlib.sha256()
    for sub, ext, recursive in _CODE_GLOBS:
        d = os.path.join(ROOT, sub)
        for fn in _code_files(d, ext, recursive):
            h.update(fn.encode())
            with open(os.path.join(d, fn), "rb") as f:
                h.update(f.read())
    return h.hexdigest()


CODE_SHA256_AT_IMPORT = code_hash()   # the code this process actually runs


def _md5(path: str) -> str:
    if path not in _MD5_CACHE:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        _MD5_CACHE[path] = h.hexdigest()
    return _MD5_CACHE[path]


def provenance(D: "Dataset | None" = None) -> dict:
    """Where, on what, and from which code and data a result was produced.

    Seed-0 Bitcoin results from the TITAN host do not reproduce on the L40S host
    (per-target reputation drifts up to ~0.1), so the GPU model and code hash are
    part of every result, not optional metadata. The python version and the
    versions of numpy, scipy, pandas, torch_scatter, torch_geometric and
    scikit-learn are recorded too (Task 4.1b). They are read from the installed
    package metadata, so no package is loaded only for its version (only the
    SignedGCN baseline imports torch_geometric); a package that is not
    installed gives None.
    """
    import platform
    from importlib import metadata

    def installed(package: str):
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    cuda = torch.cuda.is_available()
    out = dict(
        host=socket.gethostname(),
        utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        argv=list(sys.argv),
        device=str(DEV),
        gpu_name=torch.cuda.get_device_name(DEV) if cuda and DEV.type == "cuda" else None,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch=torch.__version__, cuda_runtime=torch.version.cuda,
        python=platform.python_version(),
        **{package: installed(package) for package in ("numpy", "scipy", "pandas", "torch_scatter",
                                                       "torch_geometric", "scikit-learn")},
        deterministic=bool(torch.are_deterministic_algorithms_enabled()),
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        code_sha256=CODE_SHA256_AT_IMPORT,
        code_sha256_at_write=code_hash(),
    )
    if D is not None:
        out["dataset"] = D.name
        out["data_md5"] = {p: _md5(os.path.join(ROOT, D.cfg[p]))
                           for p in ("data_path", "edge_path") if D.cfg.get(p)}
        out["signed_meta"] = dict(D.meta)
    return out


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return float(x)


def _finite(obj):
    """Replace NaN/inf with None so every result file is valid JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def overwrite_allowed(force: bool = False) -> bool:
    """True if an existing result file may be replaced: the caller passed force (the CLI's --force),
    or the environment sets KEEPER_OVERWRITE=1. Stored results are the paper's evidence."""
    return bool(force) or os.environ.get(OVERWRITE_ENV) == "1"


def result_path(name: str, force: bool = False) -> str:
    """Path of the result file ``name``: ``name`` itself if absolute, else RES/name.

    Raises FileExistsError if the file exists and overwrite_allowed(force) is false. The experiment
    CLIs also call it before their work, so that a refused run stops at its start, not after training.
    """
    path = name if os.path.isabs(name) else os.path.join(RES, name)
    if os.path.exists(path) and not overwrite_allowed(force):
        raise FileExistsError(f"{path} exists; pass --force (force=True) or set {OVERWRITE_ENV}=1 "
                              "to overwrite it")
    return path


def write_atomic(path: str, write, replace: bool = False, binary: bool = False) -> None:
    """Write the file ``path`` through ``write(f)``, a function of an open text file (a binary file
    with ``binary``), so that ``path`` is complete or absent, never partial (Task 4.1b fix round).

    The text goes to a hidden temporary file in the same folder, created as open(path, "w") creates
    a file (mode 0o666 under the umask). It is flushed and fsynced, then hard-linked to ``path``. The
    link fails with FileExistsError if ``path`` exists, so the overwrite guard holds even against a
    writer that created ``path`` meanwhile. With ``replace``, os.replace moves the temporary over
    ``path`` instead. The temporary is removed on any error. ``binary`` (Task 4.1c) opens the
    temporary in binary mode, for writers of bytes such as keeper.cppr.base_cache.
    """
    folder, base = os.path.split(path)
    tmp = os.path.join(folder, f".{base}.{os.getpid()}.{os.urandom(6).hex()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb" if binary else "w") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            try:
                os.link(tmp, path)
            except FileExistsError:
                raise FileExistsError(f"{path} exists: another writer created it meanwhile; pass --force "
                                      f"(force=True) or set {OVERWRITE_ENV}=1 to overwrite it") from None
    finally:
        if os.path.lexists(tmp):                     # gone after os.replace; still there after os.link
            os.unlink(tmp)


def dump(obj, name: str, D: "Dataset | None" = None, force: bool = False) -> str:
    """Write a result JSON with provenance. ``name`` may be an absolute path. An existing file is
    replaced only with force (the CLI's --force) or KEEPER_OVERWRITE=1 (result_path). The file is
    written atomically (write_atomic): it appears complete or not at all."""
    path = result_path(name, force)
    obj = dict(obj)
    obj.setdefault("provenance", provenance(D))
    write_atomic(path, lambda f: json.dump(_finite(obj), f, default=_json_default, allow_nan=False),
                 replace=overwrite_allowed(force))
    return path
