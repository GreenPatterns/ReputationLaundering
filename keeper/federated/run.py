"""Restored entry-point API of the federated framework.

``claudecode/fed_harness.py`` and ``claudecode/unlearn_breach.py`` import
``load_config``, ``setup_models_and_utils`` and ``get_device`` from a ``run``
module that was missing from this copy of the repository. This module restores
that API on top of the current code:

* ``get_device`` enforces the project's GPU rule (no silent CPU fallback) and
  takes the first GPU that CUDA_VISIBLE_DEVICES exposes, rather than fedlab's
  free-memory picker, whose physical-vs-logical index bug breaks under masks
  such as ``CUDA_VISIBLE_DEVICES=2,3``.
* ``setup_models_and_utils`` returns the 6-tuple the harness unpacks:
  (LocalGDTE, read_graph, setup_features, calculate_auc, get_snapshot_index,
  parameter_parser). ``read_graph`` goes through the sign-aware loader, and
  ``get_snapshot_index`` cuts the SAME edge list that ``read_graph`` returns, so
  the two cannot disagree (they would on Epinions, whose CSV still contains the
  self-loops the signed loader drops).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALIASES = {"alpha": "bitcoinalpha", "otc": "bitcoinotc", "epinions": "epinions-dated"}


def load_config(dataset: str) -> dict:
    ds = ALIASES.get(dataset, dataset)
    with open(os.path.join(ROOT, "configs", f"{ds}.json")) as f:
        cfg = json.load(f)
    cfg.setdefault("dataset", ds)
    return cfg


def get_device():
    import torch
    from keeper.laundering import core as C
    C.require_gpu()
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _snapshot_index_from_loader(time_slots, data_path=None, homogeneous_edges=True):
    """Equal-count snapshot end indices over the signed loader's edge list.

    Same arithmetic as ``thecode.dataset.get_homogeneous_snapshot_index``, but
    over the edges ``read_graph`` actually returns.
    """
    from keeper.laundering import core as C
    ds = next((d for d in ("bitcoinalpha", "bitcoinotc", "epinions-dated")
               if data_path and os.path.basename(data_path).startswith(d)), None)
    if ds is None:
        from keeper.federated.dataset import get_snapshot_index
        return get_snapshot_index(time_slots, data_path, homogeneous_edges)
    n = len(C.load(ds).edges)
    base, rem = divmod(n, time_slots)
    out, cur = [], -1
    for i in range(time_slots):
        cur += base + (1 if i < rem else 0)
        out.append(cur)
    out[-1] = n - 1
    return out


def setup_models_and_utils():
    from keeper.backbone.local_model import LocalGDTE
    from keeper.backbone.utils import read_graph, setup_features2, calculate_auc
    from keeper.backbone.arg_parser import parameter_parser
    return (LocalGDTE, read_graph, setup_features2, calculate_auc,
            _snapshot_index_from_loader, parameter_parser)


if __name__ == "__main__":
    print("run.py restores the API used by claudecode/fed_harness.py and "
          "claudecode/unlearn_breach.py; run those modules directly.")
