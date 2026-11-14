"""The base-model cache of the CPPR driver keeper.cppr.cp_vpe_gpu (Task 4.1c, ruling R59).

Every E4 cell of a seed trains the same gated base model, because the cells differ only in the attack, which the base
model never sees. With --base-cache DIR, the first cell stores the trained model's state_dict and the base outputs it
determines as DIR/<key>.pt: z0, the reputations r0 of the cohort and of the trusted cohort, and the base test
metrics. A later cell with the same key builds the model through train_gated_model without training it, loads the
state and recomputes those outputs, which must equal the stored ones bit for bit; any difference raises ValueError.
By convention DIR is <repo>/.keeper-cache/base, which git ignores, and never a folder under results/.

The key (cache_key) is a SHA-256 over the base model's identity (identity), the cred_levels array, and the training
graph's edges, labels and snapshots. The identity holds the following:
* ds, seed, epochs and gate_norm;
* the tier method and the run's calibration (eps, K, theta0, edge_cap_units, n_ref);
* the code, by its commit and its content hash (keeper.laundering.core.CODE_SHA256_AT_IMPORT);
* the GPU model, the torch version and the memory profile's pipeline mode, because a model's bits reproduce only on
  the same kind of GPU, torch and split (Task 4.1c fix round 1).
An entry stores its identity with git_dirty and the time it was written, and an entry whose stored identity differs
from the run's is refused.

An entry is written atomically (keeper.laundering.core.write_atomic) and never replaced. If a run of the same key
stored its entry first, a run that trained anew checks its own outputs against that entry instead.
"""
import datetime
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from keeper.laundering import core as C

IDENTITY = ("ds", "seed", "epochs", "gate_norm", "tiers", "calibration", "code_commit", "code_sha256", "gpu_name",
            "torch", "pipeline")
RESULT_FIELDS = ("gpu_name", "torch", "created_utc", "code_commit")   # copied into the result's base_model block


def identity(ds, seed, epochs, gate_norm, method, cal, commit) -> dict:
    """The scalars that fix a base model, in the order of IDENTITY: the settings, the tier method and the run's
    calibration ``cal`` (eps, K, theta0, edge_cap_units, n_ref; None under BFS tiers), the code commit and content hash,
    and this process's GPU model ("cpu" without a GPU), torch version and memory-profile pipeline mode
    (keeper.laundering.core.DEFAULT_STRATEGY, which apply_memory_profile sets; None on one GPU)."""
    cuda = torch.cuda.is_available() and C.DEV.type == "cuda"
    strategy = C.DEFAULT_STRATEGY
    calibration = None if cal is None else dict(eps=cal.eps, K=cal.K, theta0=cal.theta0,
                                                edge_cap_units=cal.edge_cap_units, n_ref=cal.n_ref)
    return dict(ds=ds, seed=int(seed), epochs=int(epochs), gate_norm=gate_norm, tiers=method, calibration=calibration,
                code_commit=commit, code_sha256=C.CODE_SHA256_AT_IMPORT,
                gpu_name=torch.cuda.get_device_name(C.DEV) if cuda else "cpu", torch=str(torch.__version__),
                pipeline=None if strategy is None else strategy.get("mode"))


def cache_key(ident: dict, cred_levels, Tr) -> str:
    """The SHA-256 naming a base model: its identity ``ident`` (identity()), the cred_levels array, and the training
    graph Tr's edges, labels and snapshots, each array with its dtype and shape."""
    h = hashlib.sha256(json.dumps(ident, sort_keys=True).encode())
    for name, values in (("cred_levels", cred_levels), ("edges", Tr.edges), ("labels", Tr.labels), ("snap", Tr.snap)):
        array = np.ascontiguousarray(values)
        h.update(f"|{name}|{array.dtype.str}|{array.shape}|".encode())
        h.update(array.tobytes())
    return h.hexdigest()


def outputs(z0, r_cohort, r_trusted, base_tm) -> dict:
    """A base model's outputs as an entry stores them: z0 on the CPU, the reputations of the cohort and of the trusted
    cohort (float64, in cohort order) and the base test metrics."""
    return dict(z0=z0.detach().cpu().clone(), r0_cohort=torch.tensor(np.asarray(r_cohort, dtype=np.float64)),
                r0_trusted=torch.tensor(np.asarray(r_trusted, dtype=np.float64)), base_test_metrics=base_tm)


def identical(a, b) -> bool:
    """Bit-for-bit equality of tensors (dtype, shape and bytes), floats (their bits, so -0.0 differs from 0.0), other
    scalars, and dicts and lists of them."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and a.detach().cpu().numpy().tobytes() == b.detach().cpu().numpy().tobytes())
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                and all(identical(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(identical(x, y) for x, y in zip(a, b)))
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, float) and isinstance(b, float) and a.hex() == b.hex()
    return type(a) is type(b) and a == b


@dataclass(frozen=True)
class BaseCache:
    """One run's view of the cache folder: its key, and the meta stored with the entry it writes (its identity and
    git_dirty)."""
    folder: str
    key: str
    meta: dict

    @classmethod
    def for_run(cls, folder, ident: dict, cred_levels, Tr, git_dirty: bool) -> "BaseCache":
        """The cache of a run whose identity is ``ident`` (identity()); cred_levels and Tr enter the key as well."""
        return cls(str(folder), cache_key(ident, cred_levels, Tr), dict(ident, git_dirty=bool(git_dirty)))

    @property
    def path(self) -> str:
        return os.path.join(self.folder, f"{self.key}.pt")

    def load(self):
        """The stored entry of this run's key, or None. An entry that holds another key, or whose stored identity
        differs from this run's, raises ValueError."""
        if not os.path.exists(self.path):
            return None
        entry = torch.load(self.path, map_location="cpu", weights_only=True)
        if entry.get("key") != self.key:
            raise ValueError(f"base-model cache {self.path}: the entry holds the key {entry.get('key')}, not its own")
        stored = entry.get("meta", {})
        differ = [f"{name} {stored.get(name)!r}, not {self.meta[name]!r}" for name in IDENTITY
                  if stored.get(name) != self.meta[name]]
        if differ:
            raise ValueError(f"base-model cache {self.path}: the entry was stored under another identity: "
                             f"{'; '.join(differ)}; delete it to train the base model anew")
        return entry

    def check_or_store(self, entry, model, z0, r_cohort, r_trusted, base_tm) -> dict:
        """Return the meta of the base model the run used. With a stored ``entry`` (load), the outputs recomputed from
        its state must equal its stored outputs bit for bit, and its meta is returned. Without one, the trained model's
        state and outputs are stored with this run's meta, stamped with the time of writing. If a run of the same key
        stored its entry meanwhile, this run's outputs must equal that entry's instead."""
        fresh = outputs(z0, r_cohort, r_trusted, base_tm)
        if entry is not None:
            self._require(entry, fresh, "recomputed from its stored state")
            return entry["meta"]
        meta = dict(self.meta, created_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        record = dict(key=self.key, meta=meta, outputs=fresh,
                      state_dict={name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
        try:
            C.write_atomic(self.path, lambda f: torch.save(record, f), binary=True)
        except FileExistsError:
            self._require(self.load(), fresh, "of a model trained anew")
        return meta

    def _require(self, entry, fresh, what) -> None:
        if not identical(entry["outputs"], fresh):
            raise ValueError(f"base-model cache {self.path}: the base outputs {what} (z0, r0, base test metrics) differ "
                             f"from the stored ones, written on {entry.get('meta', {}).get('gpu_name')}; delete the "
                             "entry to train the base model anew")
