# ──────────────────────────────────────────────────────────────────────────────
#  CUDA-Accelerated Trust-Attack Graph Utilities
#  Requires: PyTorch ≥ 1.10 with CUDA
# ──────────────────────────────────────────────────────────────────────────────

import random
import os
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"
# NOTE: networkx import removed. All graph operations are now pure PyTorch/CUDA.


# ──────────────────────────────────────────────────────────────────────────────
# Internal GPU helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_one_hot_labels(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Convert any label format to one-hot int64 on the SAME device as `labels`.
    CHANGE: Removed .cpu() coercions — labels stay on their original device.
    """
    if labels is None:
        raise ValueError("labels is None")

    if labels.dim() == 1:
        return F.one_hot(labels.long(), num_classes=num_classes).to(torch.int64)

    if labels.dim() == 2 and labels.shape[1] == 1:
        return F.one_hot(labels.squeeze(1).long(), num_classes=num_classes).to(torch.int64)

    if labels.dim() == 2 and labels.shape[1] == num_classes:
        return F.one_hot(
            torch.argmax(labels, dim=1).long(), num_classes=num_classes
        ).to(torch.int64)

    raise ValueError(f"Unsupported label shape {labels.shape}")


def _deduplicate_edges_and_labels(
    edges: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if edges.numel() == 0:
        return edges, labels

    seen = set()
    keep_indices: List[int] = []
    for idx, (src, dst) in enumerate(edges.t().tolist()):
        edge = (int(src), int(dst))
        if edge not in seen:
            seen.add(edge)
            keep_indices.append(idx)

    if len(keep_indices) == edges.shape[1]:
        return edges, labels

    keep_idx_tensor = torch.tensor(keep_indices, dtype=torch.long, device=edges.device)
    dedup_edges = edges.index_select(1, keep_idx_tensor)
    dedup_labels = labels.index_select(0, keep_idx_tensor.to(labels.device))
    return dedup_edges, dedup_labels


def _compute_node_ratings_gpu(
    edges_long: torch.Tensor,   # (2, E) int64 on CUDA
    labels: torch.Tensor,       # any supported label format, on CUDA
    num_nodes: int,
) -> torch.Tensor:
    """
    Accumulate per-node [trust, distrust] counts using scatter_add_.
    Returns shape (num_nodes, 2) int64, stays on device.

    CHANGE: Extracted from both identify_* functions to avoid code
    duplication and to keep all compute on GPU in one place.
    """
    device = edges_long.device
    labels_one_hot = _ensure_one_hot_labels(labels, num_classes=2)   # (E, 2) on device
    node_ratings = torch.zeros((num_nodes, 2), dtype=torch.int64, device=device)
    dst = edges_long[1]  # destination nodes, shape (E,)
    node_ratings.scatter_add_(
        0,
        dst.unsqueeze(1).expand(-1, 2),
        labels_one_hot.to(dtype=torch.int64),
    )
    return node_ratings


def _build_reverse_sparse_adj(
    edges_long: torch.Tensor,  # (2, E) int64 on CUDA
    num_nodes: int,
) -> torch.Tensor:
    """
    Build the REVERSE (transposed) sparse COO adjacency matrix on CUDA.

    CHANGE: Replaces nx.DiGraph + G.reverse().  The reverse graph is
    equivalent to transposing the adjacency matrix — achieved here by
    swapping row/col indices before building the sparse tensor.

    Using float32 values because torch.sparse.mm requires floating-point.
    .coalesce() deduplicates edges so SpMV is correct.
    """
    device = edges_long.device
    # Orientation chosen so sparse.mm(reverse_adj, frontier) propagates
    # from frontier nodes to their outgoing neighbours in G_reverse.
    rev_indices = torch.stack([edges_long[0], edges_long[1]], dim=0)  # (2, E)
    values = torch.ones(edges_long.shape[1], dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        rev_indices, values, size=(num_nodes, num_nodes), device=device
    ).coalesce()


def _bfs_distances_gpu(
    reverse_adj: torch.Tensor,  # sparse COO (num_nodes, num_nodes) float32
    source: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    GPU BFS via sparse matrix–vector multiplication.

    CHANGE: Replaces nx.shortest_path_length(G_reverse, source=victim).
    Iteratively multiplies the reverse-adjacency matrix by a float frontier
    vector to propagate reachability one hop at a time.  Only newly reached
    nodes are kept in the frontier each iteration, so the loop exits as soon
    as the graph is fully explored (early stopping on empty frontier).

    Returns int64 tensor of shape (num_nodes,):
        distance[i] = shortest-path distance from `source` in G_reverse
                    = distance from node i TO `source` in the original graph
        distance[i] = -1  if node i is unreachable from `source`.

    Complexity per victim: O(diameter × E) sparse ops, all on CUDA.
    """
    distances = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    distances[source] = 0

    # visited / frontier stored as float32 for sparse.mm compatibility
    visited = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    visited[source] = 1.0
    frontier = visited.clone()  # only source is in the initial frontier

    for depth in range(1, num_nodes):
        # Propagate: new_reach[i] > 0  ⟺  node i has a neighbour in frontier
        new_reach = torch.sparse.mm(reverse_adj, frontier.unsqueeze(1)).squeeze(1)

        # New nodes = reachable this round AND not yet visited
        new_mask = (new_reach > 0) & (visited == 0)

        if not new_mask.any():   # frontier exhausted — early exit
            break

        distances[new_mask] = depth
        visited[new_mask] = 1.0
        frontier = new_mask.float()   # only newly discovered nodes propagate

    return distances


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def identify_good_nodes_from_tensors(
    edges: torch.Tensor,
    labels: torch.Tensor,
    num_nodes: Optional[int] = None,
    verbose: bool = False,
) -> List[int]:
    """
    CHANGE: Removed edges.cpu() / labels.cpu().  All scatter_add_ ops now
    execute on the tensors' native device (GPU when called from CUDA context).
    Final .tolist() is the only CPU touch, needed only for the public interface.
    """
    if edges.numel() == 0:
        return []

    device = edges.device
    edges_long = edges.long()

    if num_nodes is None:
        num_nodes = int(edges_long.max().item()) + 1

    if edges_long.min().item() < 0 or edges_long.max().item() >= num_nodes:
        raise ValueError(
            f"Invalid edge indices: min={int(edges_long.min().item())}, "
            f"max={int(edges_long.max().item())}, num_nodes={num_nodes}"
        )

    node_ratings = _compute_node_ratings_gpu(edges_long, labels, num_nodes)

    if verbose:
        print(f"--- Identifying Good Nodes (Inferred {num_nodes} nodes) ---")
        for i in range(num_nodes):
            pos = int(node_ratings[i, 0].item())
            neg = int(node_ratings[i, 1].item())
            print(f"Node {i}: Received {pos} Trust, {neg} Distrust. -> "
                  f"{'Good' if pos > neg else 'Not Good'}")
        print("-" * 55 + "\n")

    return torch.where(node_ratings[:, 0] > node_ratings[:, 1])[0].tolist()


def identify_bad_nodes_from_tensors(
    edges: torch.Tensor,
    labels: torch.Tensor,
    num_nodes: Optional[int] = None,
    verbose: bool = False,
) -> List[int]:
    """
    CHANGE: Same as identify_good_nodes_from_tensors — .cpu() removed,
    computation stays on GPU.
    """
    if edges.numel() == 0:
        return []

    device = edges.device
    edges_long = edges.long()

    if num_nodes is None:
        num_nodes = int(edges_long.max().item()) + 1

    if edges_long.min().item() < 0 or edges_long.max().item() >= num_nodes:
        raise ValueError(
            f"Invalid edge indices: min={int(edges_long.min().item())}, "
            f"max={int(edges_long.max().item())}, num_nodes={num_nodes}"
        )

    node_ratings = _compute_node_ratings_gpu(edges_long, labels, num_nodes)

    if verbose:
        print(f"--- Identifying Bad Nodes (Inferred {num_nodes} nodes) ---")
        for i in range(num_nodes):
            pos = int(node_ratings[i, 0].item())
            neg = int(node_ratings[i, 1].item())
            print(f"Node {i}: Received {pos} Trust, {neg} Distrust. -> "
                  f"{'Bad' if neg > pos else 'Not Bad'}")
        print("-" * 55 + "\n")

    return torch.where(node_ratings[:, 1] > node_ratings[:, 0])[0].tolist()


def initiate_attack(
    original_edges: torch.Tensor,
    original_labels: torch.Tensor,
    attack_type: str = "bad_mouthing",
    attack_percentage: float = 1.0,
    victim_percentage: float = 0.10,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ── Seed ──────────────────────────────────────────────────────────────────
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # ── Validation ────────────────────────────────────────────────────────────
    if not 0.0 <= attack_percentage <= 10.0:
        raise ValueError(f"attack_percentage must be >= 0.0 (you passed {attack_percentage})")
    if not 0.0 <= victim_percentage <= 1.0:
        raise ValueError(f"victim_percentage must be between 0.0 and 1.0, got {victim_percentage}")
    if original_edges.numel() == 0:
        raise ValueError("Empty edge tensor provided")
    if original_edges.shape[0] != 2:
        raise ValueError(f"Expected edges with shape (2, num_edges), got {original_edges.shape}")

    device = original_edges.device
    # CHANGE: orig_edges_cpu / orig_labels_cpu removed — keep on device
    edges_long = original_edges.long()
    edges_long, original_labels = _deduplicate_edges_and_labels(edges_long, original_labels)
    num_nodes = int(edges_long.max().item()) + 1

    # ── Identify victim candidates entirely on GPU ────────────────────────────
    # CHANGE: Replaces identify_good/bad_nodes_from_tensors(orig_edges_cpu, ...).
    # _compute_node_ratings_gpu performs scatter_add_ on CUDA directly.
    node_ratings = _compute_node_ratings_gpu(edges_long, original_labels, num_nodes)

    if attack_type == "bad_mouthing":
        candidate_mask = node_ratings[:, 0] > node_ratings[:, 1]   # good nodes
        # CHANGE: torch.tensor on device instead of Python list literal
        malicious_label_row = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
        attack_name = "Bad-Mouthing"
    elif attack_type == "good_mouthing":
        candidate_mask = node_ratings[:, 1] > node_ratings[:, 0]   # bad nodes
        malicious_label_row = torch.tensor([[1, 0]], dtype=torch.int64, device=device)
        attack_name = "Good-Mouthing"
    else:
        raise ValueError(f"Unknown attack type: {attack_type}. Use 'bad_mouthing' or 'good_mouthing'")

    candidate_victims_tensor = torch.where(candidate_mask)[0]   # GPU int64

    if verbose:
        label = "Good" if attack_type == "bad_mouthing" else "Bad"
        not_label = "Not Good" if attack_type == "bad_mouthing" else "Not Bad"
        print(f"--- Identifying {label} Nodes (Inferred {num_nodes} nodes) ---")
        for i in range(num_nodes):
            p, n = int(node_ratings[i, 0].item()), int(node_ratings[i, 1].item())
            status = label if (p > n if attack_type == "bad_mouthing" else n > p) else not_label
            print(f"Node {i}: Received {p} Trust, {n} Distrust. -> {status}")
        print("-" * 55 + "\n")

    if candidate_victims_tensor.numel() == 0:
        if verbose:
            print(f"No victim candidates identified for {attack_name}. Returning original graph.")
        return edges_long, original_labels

    # ── Sample victims on GPU ─────────────────────────────────────────────────
    # CHANGE: random.sample(candidate_victims, k=...) on a CPU list →
    #         torch.randperm on CUDA, indexing the candidate tensor directly.
    num_to_select = max(1, int(candidate_victims_tensor.numel() * victim_percentage))
    candidate_victims_list = candidate_victims_tensor.tolist()
    victim_nodes_list = random.sample(candidate_victims_list, k=num_to_select)
    victim_nodes_tensor = torch.tensor(victim_nodes_list, dtype=torch.long, device=device)

    # ── Pre-compute degree vectors on GPU ────────────────────────────────────
    # CHANGE: G.in_degree(v) / G.out_degree(v) per victim →
    #         vectorised scatter_add_ once; O(1) lookup per victim.
    src_nodes = edges_long[0]
    dst_nodes = edges_long[1]
    ones_e = torch.ones(edges_long.shape[1], dtype=torch.long, device=device)
    in_degrees = torch.zeros(num_nodes, dtype=torch.long, device=device).scatter_add_(0, dst_nodes, ones_e)
    out_degrees = torch.zeros(num_nodes, dtype=torch.long, device=device).scatter_add_(0, src_nodes, ones_e)
    total_degrees = in_degrees + out_degrees

    # ── Dense adjacency matrix for O(1) vectorised edge-existence checks ─────
    # CHANGE: Python set of edge tuples (existing_edges) →
    #         bool CUDA tensor adj_matrix[src, dst].
    # Memory note: num_nodes² bytes. For >50 k nodes consider a sparse set.
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    adj_matrix[src_nodes, dst_nodes] = True

    # ── Build reverse-graph sparse adjacency once ─────────────────────────────
    # CHANGE: nx.DiGraph + G.reverse() recreated inside the victim loop →
    #         a single sparse COO CUDA tensor built once before the loop.
    # G does NOT change during the victim loop (only adj_matrix tracks new
    # edges), so we build reverse_adj once and reuse it for every BFS call.
    reverse_adj = _build_reverse_sparse_adj(edges_long, num_nodes)

    all_nodes = torch.arange(num_nodes, dtype=torch.long, device=device)
    newly_added_edges_list: List[torch.Tensor] = []
    total_added = 0

    if verbose:
        print(f"\n--- Generating Malicious Edges ({attack_name}) ---")
        print(f"Total candidate nodes: {candidate_victims_tensor.numel()}")
        print(f"Selected victims: {num_to_select} ({victim_percentage * 100:.1f}%)")
        print(f"Attack intensity: {attack_percentage * 100:.1f}% of victim's total-degree")
        print(f"Victim nodes: {victim_nodes_tensor.tolist()}")

    # ── Per-victim loop (sequential: adj_matrix is updated incrementally) ─────
    for victim_idx in range(victim_nodes_tensor.numel()):
        victim = int(victim_nodes_tensor[victim_idx].item())

        v_in   = int(in_degrees[victim].item())
        v_out  = int(out_degrees[victim].item())
        v_tot  = int(total_degrees[victim].item())

        if v_tot == 0:
            if verbose:
                print(f"Skipping victim {victim}: total degree is zero.")
            continue

        attack_volume = max(1, int(v_tot * attack_percentage))

        # ── Potential attackers: vectorised GPU masking ───────────────────────
        # CHANGE: list comprehension over range(num_nodes) on CPU →
        #         boolean mask on all_nodes tensor, fully on CUDA.
        not_self          = all_nodes != victim
        no_existing_edge  = ~adj_matrix[:, victim]      # column lookup in O(1)
        potential_mask    = not_self & no_existing_edge
        potential_att     = all_nodes[potential_mask]

        if potential_att.numel() == 0:
            potential_att = all_nodes[not_self]         # fallback: allow duplicates

        # ── GPU BFS for distances from victim in G_reverse ────────────────────
        # CHANGE: dict(nx.shortest_path_length(G_reverse, source=victim)) →
        #         _bfs_distances_gpu using sparse.mm on CUDA.
        distances = _bfs_distances_gpu(reverse_adj, victim, num_nodes, device)

        # ── Sort: reachable (farthest-first) + shuffled unreachable ───────────
        potential_list = potential_att.tolist()
        cand_with_dist = [(n, int(distances[n].item())) for n in potential_list]
        reachable = [(n, d) for n, d in cand_with_dist if d != -1]
        unreachable = [(n, d) for n, d in cand_with_dist if d == -1]

        if reachable:
            reachable.sort(key=lambda x: x[1], reverse=True)
            random.shuffle(unreachable)
            combined = reachable + unreachable
            attackers_selected_list = [n for n, _ in combined[:attack_volume]]
        else:
            random.shuffle(unreachable)
            attackers_selected_list = [n for n, _ in unreachable[:attack_volume]]

        attackers_selected = torch.tensor(attackers_selected_list, dtype=torch.long, device=device)

        # ── Filter already-existing edges & update adj_matrix on GPU ─────────
        # CHANGE: `if edge not in existing_edges` Python loop →
        #         vectorised boolean index into adj_matrix row on CUDA.
        already_exist = adj_matrix[attackers_selected, victim]
        new_attackers = attackers_selected[~already_exist]
        n_added = new_attackers.numel()

        if n_added > 0:
            adj_matrix[new_attackers, victim] = True   # in-place GPU update
            victim_col = torch.full((n_added,), victim, dtype=torch.long, device=device)
            newly_added_edges_list.append(torch.stack([new_attackers, victim_col], dim=0))
            total_added += n_added

        if verbose:
            print(
                f" -> Victim {victim}: in-degree={v_in}, out-degree={v_out}, "
                f"total-degree={v_tot}, attack_volume={attack_volume}, added={n_added}"
            )

    # ── Assemble output tensors ───────────────────────────────────────────────
    if not newly_added_edges_list:
        if verbose:
            print(f"\nNo new malicious edges were generated for {attack_name}.")
        return edges_long, original_labels

    malicious_edges_tensor = torch.cat(newly_added_edges_list, dim=1)   # (2, M) on device
    num_malicious_edges    = malicious_edges_tensor.shape[1]

    orig_label_dtype = original_labels.dtype
    # CHANGE: torch.tensor([malicious_label] * N) — Python list multiplication →
    #         .expand() which is a zero-copy view broadcast on GPU.
    malicious_onehot = malicious_label_row.expand(num_malicious_edges, -1)  # (M, 2)

    if original_labels.dim() == 1:
        malicious_labels_tensor = torch.argmax(malicious_onehot, dim=1).to(dtype=orig_label_dtype)
    elif original_labels.dim() == 2 and original_labels.shape[1] == 1:
        malicious_labels_tensor = torch.argmax(malicious_onehot, dim=1).unsqueeze(1).to(dtype=orig_label_dtype)
    else:
        malicious_labels_tensor = malicious_onehot.to(dtype=orig_label_dtype)

    # CHANGE: .to(device) calls removed — everything is already on device.
    updated_edges  = torch.cat([edges_long, malicious_edges_tensor], dim=1)
    updated_labels = torch.cat([original_labels, malicious_labels_tensor], dim=0)

    if verbose:
        print(f"\n--- {attack_name} Attack Summary ---")
        print(f"Original edges:      {original_edges.shape[1]}")
        print(f"Malicious edges added: {num_malicious_edges}")
        print(f"New total edges:     {updated_edges.shape[1]}")
        print(f"Graph size increase: {(num_malicious_edges / original_edges.shape[1]) * 100:.2f}%")
        print("-" * 40)

    return updated_edges, updated_labels


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrappers — interface unchanged
# ──────────────────────────────────────────────────────────────────────────────

def initiate_bad_mouthing_attacks(
    original_edges: torch.Tensor,
    original_labels: torch.Tensor,
    attack_percentage: float = 1.0,
    victim_percentage: float = 0.10,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return initiate_attack(
        original_edges, original_labels,
        attack_type="bad_mouthing",
        attack_percentage=attack_percentage,
        victim_percentage=victim_percentage,
        verbose=verbose,
        seed=seed,
    )


def initiate_good_mouthing_attacks(
    original_edges: torch.Tensor,
    original_labels: torch.Tensor,
    attack_percentage: float = 1.0,
    victim_percentage: float = 0.10,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return initiate_attack(
        original_edges, original_labels,
        attack_type="good_mouthing",
        attack_percentage=attack_percentage,
        victim_percentage=victim_percentage,
        verbose=verbose,
        seed=seed,
    )


def calculate_victim_drift(
    current_time: int,
    total_time_steps: int,
    max_victim_percentage: float,
    mode: str = "linear",
) -> float:
    # Pure scalar math — no tensors involved; unchanged.
    if total_time_steps <= 1:
        return max_victim_percentage
    progress = (current_time + 1) / total_time_steps
    if mode == "linear":
        return float(max_victim_percentage * progress)
    elif mode == "exponential":
        return float(max_victim_percentage * (progress ** 2))
    return float(max_victim_percentage)
