"""Sign-aware loading of signed rating datasets.

Every edge carries a signed rating: positive means trust, negative means
distrust. The sign is the edge's label, and the raw rating (with its
magnitude) is kept alongside it. This module is the single place where a raw
rating becomes a label, so a new signed dataset needs only a config entry,
never new code.

Policy for edges that are not signed evidence, recorded in every Dataset's
``meta`` and from there in every result JSON:

* zero/neutral ratings: ``"drop"`` (default) or ``"error"``. A zero is never
  mapped silently to trust or distrust.
* self-loops (a node rating itself): ``"drop"`` (default) or ``"keep"``. A
  self-rating is not evidence from a rater about a target.

Model labels follow the framework's convention: one-hot ``[trust, distrust]``
with the distrust column at index :data:`NEG_COL`, which is also the
cross-entropy class id of a negative edge.
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

POS_COL = 0
NEG_COL = 1
CSV_COLUMNS = ("source", "target", "rating", "time")


def neg_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of negative (distrust) edges from one-hot labels."""
    return np.asarray(labels)[:, NEG_COL] == 1


def pos_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of positive (trust) edges from one-hot labels."""
    return np.asarray(labels)[:, POS_COL] == 1


def class_ids(labels: np.ndarray) -> np.ndarray:
    """Cross-entropy class id per edge: 1 for negative, 0 for positive."""
    return neg_mask(labels).astype(np.int64)


def one_hot_from_sign(sign: np.ndarray) -> np.ndarray:
    """``[trust, distrust]`` one-hot labels from a +1/-1 sign array."""
    sign = np.asarray(sign)
    if not np.isin(sign, (-1, 1)).all():
        raise ValueError("sign must be +1 or -1 for every edge")
    lab = np.zeros((len(sign), 2), np.float32)
    lab[sign > 0, POS_COL] = 1.0
    lab[sign < 0, NEG_COL] = 1.0
    return lab


@dataclass
class SignedEdges:
    """Signed edges in model node ids, sorted by time (stable)."""
    edges: np.ndarray          # [E, 2] int64
    sign: np.ndarray           # [E] int8, +1 trust / -1 distrust
    rating: np.ndarray         # [E] float64, raw rating (sign and magnitude)
    time: np.ndarray           # [E] int64
    meta: dict = field(default_factory=dict)

    @property
    def labels(self) -> np.ndarray:
        return one_hot_from_sign(self.sign)


def _read_csv(path: str, cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = cfg.get("columns")
    if cols:                                   # explicit mapping for new datasets
        df = df.rename(columns={cols[k]: k for k in CSV_COLUMNS if k in cols})
    elif len(df.columns) == 4:
        df.columns = list(CSV_COLUMNS)
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; set cfg['columns']")
    return df[list(CSV_COLUMNS)]


def _check_bijection(raw_ids: np.ndarray, model_ids: np.ndarray, name: str) -> None:
    """Raw-id -> model-id must be a consistent one-to-one map."""
    pairs = np.unique(np.stack([raw_ids, model_ids], 1), axis=0)
    if len(np.unique(pairs[:, 0])) != len(pairs) or len(np.unique(pairs[:, 1])) != len(pairs):
        raise AssertionError(f"{name}: raw ids and rating-file ids are not a bijection")


def load_signed(cfg: dict, root: str) -> SignedEdges:
    """Read a dataset's raw signed ratings and apply the zero/self-loop policy.

    With a legacy one-hot rating file (``cfg['edge_path']``), its node ids are
    kept so results stay comparable with earlier runs. The file is row-paired
    with the time-sorted CSV, and the pairing is verified on BOTH the sign of
    every row and the endpoint id mapping. Without one, node ids are the raw ids
    re-indexed to 0..n-1 in sorted order.
    """
    name = cfg.get("dataset", "?")
    sp = cfg.get("signed", {})
    zero_policy = sp.get("zero_policy", "drop")
    loop_policy = sp.get("self_loops", "drop")
    if zero_policy not in ("drop", "error") or loop_policy not in ("drop", "keep"):
        raise ValueError(f"{name}: bad signed policy {sp}")

    df = _read_csv(os.path.join(root, cfg["data_path"]), cfg)
    order = np.argsort(df["time"].to_numpy(), kind="mergesort")
    df = df.iloc[order].reset_index(drop=True)
    rating = df["rating"].to_numpy(dtype=np.float64)
    time = df["time"].to_numpy(dtype=np.int64)
    raw_src = df["source"].to_numpy()
    raw_dst = df["target"].to_numpy()

    if cfg.get("edge_path"):
        rf = np.loadtxt(os.path.join(root, cfg["edge_path"]))
        if len(rf) != len(df):
            raise AssertionError(f"{name}: rating file has {len(rf)} rows, CSV {len(df)}")
        rf_neg = rf[:, 2 + NEG_COL].astype(int) == 1
        nz = rating != 0                      # zero rows are dropped below anyway
        n_bad = int(((rating[nz] < 0) != rf_neg[nz]).sum())
        if n_bad:
            raise AssertionError(f"{name}: {n_bad} sign mismatches between CSV and "
                                 f"rating file after stable time sort")
        src, dst = rf[:, 0].astype(np.int64), rf[:, 1].astype(np.int64)
        _check_bijection(np.concatenate([raw_src, raw_dst]),
                         np.concatenate([src, dst]), name)
        id_source = "rating_file"
    else:
        uniq, inv = np.unique(np.concatenate([raw_src, raw_dst]), return_inverse=True)
        src, dst = inv[:len(df)].astype(np.int64), inv[len(df):].astype(np.int64)
        id_source = "reindexed"

    zero = rating == 0
    if zero.any() and zero_policy == "error":
        raise ValueError(f"{name}: {int(zero.sum())} zero ratings and zero_policy=error")
    loop = src == dst
    drop = zero | (loop if loop_policy == "drop" else np.zeros_like(loop))
    keep = ~drop
    meta = dict(n_raw=int(len(df)), n_zero_dropped=int(zero.sum()),
                n_self_loops=int(loop.sum()),
                n_self_loops_dropped=int((loop & drop).sum()),
                zero_policy=zero_policy, self_loop_policy=loop_policy,
                id_source=id_source,
                rating_min=float(rating.min()), rating_max=float(rating.max()))
    return SignedEdges(np.stack([src[keep], dst[keep]], 1),
                       np.sign(rating[keep]).astype(np.int8),
                       rating[keep], time[keep], meta)
