import torch
import numpy as np
from texttable import Texttable
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef
import matplotlib.pyplot as plt
import os
import numpy as np
from texttable import Texttable
import warnings
from keeper.third_party.fedlab.functional import get_best_gpu_index, resolve_device
warnings.filterwarnings('ignore')
import networkx as nx
import scipy.sparse

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def read_graph(args):
    """Edges/labels for a dataset.

    Returns edges in the SAME order that ``get_snapshot_index`` assumes, i.e.
    sorted by time. Previously this returned the rating file in file order while
    ``get_snapshot_index`` sorted the CSV by time, so the two disagreed for
    bitcoinalpha (the CSV is not stored in time order) and every snapshot boundary
    was applied to mis-ordered edges -- 2,852 rows land in the wrong snapshot on
    that dataset.

    Labels come from the rating file, which is the authoritative sign source;
    timestamps only determine ORDER here, never signs.
    """
    import os as _os
    import sys as _sys

    from keeper.laundering.core import load as _load

    ds = getattr(args, "dataset", None) or _os.path.splitext(
        _os.path.basename(args.edge_path))[0].replace("-rating", "")
    D = _load(ds, verify=True)
    return {"labels": D.labels, "edges": D.edges.astype(float),
            "ecount": int(len(D.edges)), "ncount": int(D.n)}

def setup_features2(graph_data, seed, dim=16, alpha=5.0):
    rng = np.random.default_rng(seed)
    num_nodes = graph_data['ncount']
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(graph_data['edges'])
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float64)
    A = A + scipy.sparse.eye(num_nodes)
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = scipy.sparse.diags(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    X = rng.normal(0, 1, (num_nodes, dim))
    trajectory = [X]
    steps_per_node = np.maximum(1, np.floor(alpha / (deg + 1)).astype(int))
    max_steps = int(steps_per_node.max())
    X_cur = X.copy()
    active = np.ones(num_nodes, dtype=bool)
    for step in range(max_steps):
        X_next = X_cur.copy()
        X_next[active] = A_hat[active][:, :] @ X_cur
        trajectory.append(X_next)
        steps_per_node[active] -= 1
        active &= (steps_per_node > 0)
        if not active.any():
            break
        X_cur = X_next
    embeddings = np.mean(trajectory, axis=0)
    mu = embeddings.mean(axis=0, keepdims=True)
    sigma = embeddings.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    embeddings = (embeddings - mu) / sigma
    return embeddings.astype(np.float32)

def calculate_auc(scores, label):
    label_vector = [i for line in label for i in range(len(line)) if line[i] == 1]
    prediction_vector = torch.argmax(scores, dim=1)

    acc_balanced = balanced_accuracy_score(label_vector, prediction_vector.cpu())
    mcc = matthews_corrcoef(label_vector, prediction_vector.cpu())

    f1_micro = f1_score(label_vector, prediction_vector.cpu(), average="micro")
    f1_macro = f1_score(label_vector, prediction_vector.cpu(), average="macro")

    prediction_pr = scores[:,1]
    auc = roc_auc_score(label_vector, prediction_pr.cpu().detach().numpy())
    precision = average_precision_score(label_vector, prediction_pr.cpu().detach().numpy())  # average precision
    return mcc, auc, acc_balanced, precision, f1_micro, f1_macro

def getBestGPU():
    return get_best_gpu_index()

def getProjectDevice(device=None, prefer_cuda=True):
    return resolve_device(device=device, prefer_cuda=prefer_cuda)

def best_printer(rows):
    t = Texttable(max_width=0)  # allow columns to expand freely
    t.set_precision(3)
    t.set_cols_align(["l"] + ["c"] * (len(rows[0]) - 1))
    t.add_rows(rows)
    print(t.draw())

def printBest(R):
    if R is None or len(R) < 2:
        return
    data = np.asarray(R[1:], dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    mean = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    maxi = np.max(data, axis=0)
    mini = np.min(data, axis=0)
    mean_pm_std = [f"{m:.3f}±{s:.3f}" for m, s in zip(mean, std)]
    maxi_str = [f"{v:.3f}" for v in maxi]
    mini_str = [f"{v:.3f}" for v in mini]
    known_headers = [
        "MCC",
        "AUC",
        "ACC_Balanced",
        "AP",
        "F1_Micro",
        "F1_Macro",
        "TrainingLoss",
    ]
    num_metrics = len(mean)

    # Dynamically build the header to match the data columns
    header = ["Metric"]  # Start with the label column
    
    for i in range(num_metrics):
        if i < len(known_headers):
            header.append(known_headers[i])
        else:
            # If data has more columns than known headers, generate a generic name
            header.append(f"Metric_{i+1}")

    # Construct the final rows
    rows = [
        header,
        ["Mean±Std"] + mean_pm_std,
        ["Max"] + maxi_str,
        ["Min"] + mini_str,
    ]

    best_printer(rows)
    
#++++++++++++++++++++++++++++++++++= My Addition +++++++++++++++++++++++++++++++++++++++++++++++++
import collections
import numpy as np
import torch
from typing import Dict, Any, List

def inspect_client_data(cdata_dict: Dict[str, List], print_report: bool = True) -> Dict[str, Any]:
    """
    Inspect client's snapshot edge/label statistics and print per-snapshot label distributions.

    Expected cdata_dict keys:
      'client_edges' : list of numpy arrays of shape (N_edges, 2)
      'client_labels': list of numpy arrays or torch tensors where class is column 1 in your code

    Returns a stats dictionary with per-snapshot summaries and overall summaries.
    """

    def extract_classes(lbls):
        """
        Normalize label array -> 1D numpy int array of class labels.
        Handles:
          - 1D arrays of shape (N,)
          - 2D arrays of shape (N, >=1) where class is in column 1 (index 1)
          - torch tensors (converted to numpy)
        """
        if isinstance(lbls, torch.Tensor):
            a = lbls.detach().cpu().numpy()
        else:
            a = np.array(lbls)

        if a.ndim == 0:
            # single scalar
            return np.array([int(a)], dtype=int)
        if a.ndim == 1:
            return a.astype(int)
        # 2D or higher: prefer column 1 if exists (to match your code using labels[:,1])
        if a.shape[1] >= 2:
            return a[:, 1].astype(int)
        # fallback to first column
        return a[:, 0].astype(int)

    edges_list = cdata_dict.get('client_edges', [])
    labels_list = cdata_dict.get('client_labels', [])
    num_snapshots = max(len(edges_list), len(labels_list))

    per_snap = []
    total_edges = 0
    total_nodes_set = set()

    for t in range(num_snapshots):
        e = np.array(edges_list[t]) if t < len(edges_list) else np.array([], dtype=int).reshape(0, 2)
        l = labels_list[t] if t < len(labels_list) else np.array([], dtype=int)

        # defensive shapes
        if e.size == 0:
            n_edges = 0
            nodes = set()
        else:
            # ensure shape (N,2)
            if e.ndim == 1 and e.size == 2:
                e = e.reshape(1, 2)
            n_edges = len(e)
            nodes = set(e.flatten().tolist())

        total_edges += n_edges
        total_nodes_set.update(nodes)

        classes = extract_classes(l)
        # numpy unique
        if classes.size == 0:
            uniq_np = (np.array([]), np.array([]))
        else:
            uniq_vals, uniq_counts = np.unique(classes, return_counts=True)
            uniq_np = (uniq_vals, uniq_counts)

        # torch unique (for parity, if torch present) - convert even empty arrays safely
        if classes.size == 0:
            uniq_torch = None
        else:
            try:
                t_tensor = torch.from_numpy(classes)
                vals, counts = torch.unique(t_tensor, return_counts=True)
                uniq_torch = (vals.tolist(), counts.tolist())
            except Exception:
                uniq_torch = None

        class_counts = dict(zip(uniq_np[0].astype(int).tolist(), uniq_np[1].astype(int).tolist())) if classes.size > 0 else {}
        class_balance = {k: v / n_edges for k, v in class_counts.items()} if n_edges > 0 else {}

        per_snap.append({
            'snapshot': t,
            'n_edges': n_edges,
            'n_nodes': len(nodes),
            'class_counts': class_counts,
            'class_balance': class_balance,
            'numpy_unique': uniq_np,
            'torch_unique': uniq_torch
        })

    # compute overlap counts between consecutive snapshots (by canonical edge key)
    def edge_key(u, v):
        u, v = int(u), int(v)
        return (min(u, v), max(u, v))

    flips = []
    for t in range(num_snapshots - 1):
        if t >= len(edges_list) - 1:
            flips.append({'pair': (t, t+1), 'overlap': 0, 'flip_rate': None})
            continue
        e1 = np.array(edges_list[t])
        e2 = np.array(edges_list[t+1])
        l1 = labels_list[t] if t < len(labels_list) else np.array([])
        l2 = labels_list[t+1] if t+1 < len(labels_list) else np.array([])

        if e1.size == 0 or e2.size == 0:
            flips.append({'pair': (t, t+1), 'overlap': 0, 'flip_rate': None})
            continue

        # build dicts keyed by canonical undirected edge
        d1 = {}
        cls1 = extract_classes(l1)
        for i in range(len(e1)):
            k = edge_key(e1[i, 0], e1[i, 1])
            d1[k] = int(cls1[i]) if i < len(cls1) else None

        d2 = {}
        cls2 = extract_classes(l2)
        for i in range(len(e2)):
            k = edge_key(e2[i, 0], e2[i, 1])
            d2[k] = int(cls2[i]) if i < len(cls2) else None

        overlap_keys = set(d1.keys()).intersection(d2.keys())
        if len(overlap_keys) == 0:
            flips.append({'pair': (t, t+1), 'overlap': 0, 'flip_rate': None})
            continue

        flip_count = sum(1 for k in overlap_keys if (d1.get(k) is not None and d2.get(k) is not None and d1[k] != d2[k]))
        flips.append({'pair': (t, t+1), 'overlap': len(overlap_keys), 'flip_rate': flip_count / len(overlap_keys)})

    stats = {
        'snapshots': per_snap,
        'total_edges': total_edges,
        'total_nodes': len(total_nodes_set),
        'num_snapshots': num_snapshots,
        'label_flips_between_snapshots': flips
    }

    if print_report:
        print(f"Snapshots: {num_snapshots}, total_edges: {total_edges}, total_nodes: {len(total_nodes_set)}")
        for s in per_snap:
            vals, counts = s['numpy_unique']
            if len(vals) == 0:
                print(f" t={s['snapshot']}\tedges={s['n_edges']}\tnodes={s['n_nodes']}\tcounts={{}}\tbalance={{}}")
            else:
                # pretty-print counts dict
                print(f" t={s['snapshot']}\tedges={s['n_edges']}\tnodes={s['n_nodes']}\tcounts={s['class_counts']}\tbalance={s['class_balance']}")
                # show torch.unique parity if available
                if s['torch_unique'] is not None:
                    print(f"    torch.unique -> vals: {s['torch_unique'][0]}, counts: {s['torch_unique'][1]}")
        for f in flips:
            print(f" Overlap {f['pair'][0]}->{f['pair'][1]}: overlap={f['overlap']}\tflip_rate={f['flip_rate']}")

    return stats

#-----------------------------------------------------------------------------------------------------------



def analyze_client_data(client_data, client_id):
    print(f"\n--- Analysis of Client {client_id} Data ---")
    
    # Analyze edge distribution
    all_edges = np.concatenate(client_data['client_edges'], axis=0)
    unique_nodes = np.unique(all_edges.flatten())
    print(f"Total unique nodes: {len(unique_nodes)}")
    print(f"Total edges: {len(all_edges)}")
    
    # Analyze label distribution
    all_labels = np.concatenate(client_data['client_labels'], axis=0)
    if len(all_labels.shape) > 1 and all_labels.shape[1] > 1:
        # If labels are one-hot encoded
        label_counts = np.sum(all_labels, axis=0)
        print(f"Label distribution: {label_counts}")
    else:
        # If labels are scalar
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")
    
    # Analyze temporal patterns
    print(f"Number of snapshots: {len(client_data['client_edges'])}")
    edges_per_snapshot = [len(edges) for edges in client_data['client_edges']]
    print(f"Edges per snapshot: {edges_per_snapshot}")
    
    return {
        'unique_nodes': len(unique_nodes),
        'total_edges': len(all_edges),
        'num_snapshots': len(client_data['client_edges']),
        'edges_per_snapshot': edges_per_snapshot
    }

import os
import pickle
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def save_parameters(
    model: nn.Module,
    client_id: int,
    type_of_params: str = "structured",
    base_save_dir: str = os.path.join(_PROJECT_ROOT, "results", "legacy", "saved_gradients")
):
    structured_params = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
    }
    save_dir = os.path.join(base_save_dir, f"P_{client_id}")
    os.makedirs(save_dir, exist_ok=True)
    
    file_name = f"{type_of_params}_params_{client_id}.pkl"

    file_path = os.path.join(save_dir, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(structured_params, f)
    print(f"{file_name} saved")


def visualize_gradients_and_parameters(
    stacked_gradients,
    stacked_parameters,
    save_dir: str = os.path.join(_PROJECT_ROOT, "results", "legacy", "vis_test"),
    file_name: str = "gradients_parameters_plot.png",
    show: bool = False
):
    """
    Visualizes stacked gradients and parameters in two subplots and saves the figure.
    
    Args:
        stacked_gradients: 2D array-like (Tensor or numpy) of gradients.
        stacked_parameters: 2D array-like (Tensor or numpy) of parameters.
        save_dir: Directory to save the plot.
        file_name: Name of the saved file.
        show: Whether to show the plot.
    """
    # Convert to numpy if tensors
    if isinstance(stacked_gradients, torch.Tensor):
        stacked_gradients = stacked_gradients.detach().cpu().numpy()
    if isinstance(stacked_parameters, torch.Tensor):
        stacked_parameters = stacked_parameters.detach().cpu().numpy()
        
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot Gradients
    # Assuming shape (num_samples, num_dimensions)
    # Plotting each sample as a line: x=Dimension, y=Value
    axes[0].plot(stacked_gradients.T)
    axes[0].set_title('Stacked Gradients')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Value')
    # axes[0].legend([f"Sample {i}" for i in range(stacked_gradients.shape[0])], loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # Plot Parameters
    axes[1].plot(stacked_parameters.T)
    axes[1].set_title('Stacked Parameters')
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Value')
    # axes[1].legend([f"Sample {i}" for i in range(stacked_parameters.shape[0])], loc='upper right', bbox_to_anchor=(1.1, 1))
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


from decimal import Decimal, ROUND_CEILING

def ceil_list_to_4f(values):
    return [
        float(Decimal(str(x)).quantize(Decimal("0.0001"), rounding=ROUND_CEILING))
        for x in values
    ]
        