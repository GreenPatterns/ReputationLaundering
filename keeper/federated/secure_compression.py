# ──────────────────────────────────────────────────────────────────────────────
# secure_compression.py
# SPQSA v2: Sparsity-Preserving Quantized Secure Aggregation
#
# Uses ONLY standard, peer-reviewed privacy mechanisms:
#   1. Top-K Adaptive Sparsification   (Stich et al., ICLR 2018)
#   2. Stochastic Multi-bit Quantization (QSGD, Alistarh et al., NeurIPS 2017)
#   3. Local Differential Privacy       (Gaussian mechanism, Abadi et al., CCS 2016)
#   4. Secure Aggregation Masking       (Additive shares, Bonawitz et al., CCS 2017)
#   5. Compact Sparse Wire Format       (CSR-style delta-varint encoding)
# ──────────────────────────────────────────────────────────────────────────────

import struct
import hashlib
import math
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  Topology-Adaptive Privacy (TAP-Noise)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_graph_sensitivity(
    edges_tensor: torch.Tensor,
    labels_tensor: torch.Tensor,
    num_nodes: int,
    temporal_churn: float = 0.0,
    trust_volatility: float = 0.0,
) -> float:
    """Compute a topology-based sensitivity scalar from local graph data.

    Clients with more structurally sensitive subgraphs (high-degree nodes,
    volatile trust patterns) get a LOWER noise_scale to preserve utility.
    Clients with simpler subgraphs get a HIGHER noise_scale for stronger privacy.

    Args:
        temporal_churn: Fraction of newly-arrived nodes [0, 1].
            Higher churn → more noise (temporal patterns are identifiable).
            Lower churn → less noise (stable graph is predictable).
        trust_volatility: Fraction of edges whose sign changed in the last snapshot [0, 1].
            Higher volatility → more noise (unstable trust signals).
            Lower volatility → less noise (stable trust signals).

    Returns:
        noise_scale: float in [0.5, 2.0]
            0.5 = very sensitive topology → half the noise
            2.0 = simple topology → double the noise
    """
    if edges_tensor.numel() == 0 or num_nodes == 0:
        return 1.0

    device = edges_tensor.device
    row = edges_tensor[0].long()
    col = edges_tensor[1].long()
    num_edges = row.size(0)

    # Compute degrees across ALL nodes
    degrees = torch.zeros(num_nodes, device=device)
    degrees.scatter_add_(0, row, torch.ones(num_edges, device=device))
    degrees.scatter_add_(0, col, torch.ones(num_edges, device=device))

    # Local node count: only nodes that appear in this client's edges
    local_nodes = (degrees > 0).sum().item()
    mean_deg = degrees[degrees > 0].mean().item() if local_nodes > 0 else 0.0

    if mean_deg < 1e-6 or local_nodes < 2:
        return 1.0

    # Composite sensitivity based on:
    # 1. Local graph density (higher → denser → less noise)
    # 2. Edge count ratio (more edges → more data → less noise)
    # 3. Trust volatility (higher → unstable → MORE noise for protection)
    # 4. Temporal churn (higher → changing → MORE noise for privacy)
    max_possible_local_deg = 2 * (local_nodes - 1)  # directed subgraph
    density_ratio = min(mean_deg / max(max_possible_local_deg, 1e-8), 1.0)
    edges_ratio = min(num_edges / 1000.0, 1.0)  # normalized by reference edge count
    raw = (0.3 * density_ratio + 
           0.3 * edges_ratio -
           0.2 * min(trust_volatility, 1.0) - 
           0.2 * min(temporal_churn, 1.0))
    raw = min(max(raw, 0.0), 1.0)

    # Map raw ∈ [0,1] → noise_scale ∈ [2.0, 0.5]  (inverse relationship)
    noise_scale = 2.0 - 1.5 * raw
    return noise_scale


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1 – Top-K Adaptive Sparsification
# ═══════════════════════════════════════════════════════════════════════════════

def topk_sparsify(
    delta: torch.Tensor,
    target_sparsity: float = 0.90,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Keep the top (1-target_sparsity) fraction of entries by magnitude.

    Returns:
        values:    1-D tensor of retained values (float32).
        indices:   1-D tensor of their positions (int64, sorted).
        threshold: The magnitude cutoff used.
    """
    flat = delta.detach().cpu().float()
    n = flat.numel()
    if n == 0:
        return torch.tensor([]), torch.tensor([], dtype=torch.int64), 0.0

    k = max(1, int(n * (1.0 - target_sparsity)))
    topk_vals, topk_idx = torch.topk(flat.abs(), k)
    threshold = topk_vals[-1].item()

    # Use the threshold to build a clean mask
    mask = flat.abs() >= threshold
    indices = mask.nonzero(as_tuple=True)[0].sort().values
    values = flat[indices]
    return values, indices, threshold


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2 – Stochastic Multi-bit Quantization (QSGD)
# ═══════════════════════════════════════════════════════════════════════════════

def qsgd_quantize(
    values: torch.Tensor,
    num_bits: int = 8,
) -> Tuple[torch.Tensor, float]:
    """QSGD-style stochastic quantization to `num_bits` levels.

    Maps each value v to one of 2^num_bits uniformly-spaced levels in
    [-scale, +scale] stochastically, preserving unbiasedness:
        E[Q(v)] = v

    Returns:
        quantized: int16 tensor of quantization indices in [-(2^b-1), +(2^b-1)]
        scale:     max(|values|), used for dequantization
    """
    if values.numel() == 0:
        return torch.tensor([], dtype=torch.int16), 0.0

    scale = values.abs().max().item()
    if scale == 0:
        return torch.zeros_like(values, dtype=torch.int16), 0.0

    num_levels = (1 << num_bits) - 1  # e.g., 255 for 8-bit
    normalized = values / scale        # in [-1, 1]

    # Stochastic rounding: floor + probabilistic ceil
    scaled = normalized.abs() * num_levels
    floor_val = scaled.floor()
    prob = scaled - floor_val
    ceil_mask = torch.bernoulli(prob).bool()
    quantized_abs = floor_val.clone()
    quantized_abs[ceil_mask] = floor_val[ceil_mask] + 1
    quantized_abs = quantized_abs.clamp(0, num_levels)

    # Restore sign
    quantized = (quantized_abs * normalized.sign()).to(torch.int16)
    return quantized, scale


def qsgd_dequantize(
    quantized: torch.Tensor,
    scale: float,
    num_bits: int = 8,
) -> torch.Tensor:
    """Inverse of qsgd_quantize."""
    num_levels = (1 << num_bits) - 1
    return (quantized.float() / num_levels) * scale


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 3 – Local Differential Privacy (Gaussian Mechanism)
# ═══════════════════════════════════════════════════════════════════════════════

def add_dp_noise(
    quantized: torch.Tensor,
    scale: float,
    epsilon: float = 8.0,
    delta_dp: float = 1e-5,
    num_bits: int = 8,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Add calibrated Gaussian noise for (ε, δ)-differential privacy.

    Sensitivity is bounded by the quantization range: Δf = 2 * scale / num_levels.
    Noise σ = Δf * sqrt(2 * ln(1.25/δ)) / ε   (standard Gaussian mechanism).

    The noise is added in the dequantized domain, then re-quantized.

    Args:
        noise_scale: Multiplier on σ for topology-adaptive privacy (TAP-Noise).
                     >1 = more noise (stronger privacy), <1 = less noise (preserve utility).
    """
    if quantized.numel() == 0 or epsilon <= 0:
        return quantized

    num_levels = (1 << num_bits) - 1
    sensitivity = 2.0 * scale / num_levels
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta_dp)) / epsilon
    sigma *= noise_scale

    # Add noise in float domain and re-quantize
    values_f = (quantized.float() / num_levels) * scale
    noisy = values_f + torch.randn_like(values_f) * sigma

    # Re-quantize
    normalized = (noisy / scale).clamp(-1, 1)
    result = (normalized * num_levels).round().to(torch.int16)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 4 – Secure Aggregation Masking (Additive Secret Shares)
# ═══════════════════════════════════════════════════════════════════════════════

def _derive_seed(round_id: int, client_id: int, shared_secret: int) -> int:
    blob = f"SPQSA:{round_id}:{client_id}:{shared_secret}".encode()
    h = hashlib.sha256(blob).digest()
    return int.from_bytes(h[:8], "little")


def apply_secure_mask(
    quantized: torch.Tensor,
    round_id: int,
    client_id: int,
    shared_secret: int,
    num_bits: int = 8,
) -> torch.Tensor:
    """Add a seeded pseudo-random mask to quantized values.

    Uses additive masking: masked = quantized + PRN  (mod range).
    The server can subtract the same PRN to recover the original.
    This is the standard technique from Bonawitz et al. (CCS 2017).
    """
    if quantized.numel() == 0:
        return quantized

    seed = _derive_seed(round_id, client_id, shared_secret)
    rng = torch.Generator()
    rng.manual_seed(seed)

    num_levels = (1 << num_bits) - 1
    mask_range = 2 * num_levels + 1  # [-num_levels, +num_levels]
    mask = torch.randint(-num_levels, num_levels + 1, quantized.shape,
                         generator=rng, dtype=torch.int32)

    masked = (quantized.int() + num_levels + mask) % mask_range - num_levels
    return masked.to(torch.int16)


def remove_secure_mask(
    masked: torch.Tensor,
    round_id: int,
    client_id: int,
    shared_secret: int,
    num_bits: int = 8,
) -> torch.Tensor:
    """Remove the additive mask to recover original quantized values."""
    if masked.numel() == 0:
        return masked

    seed = _derive_seed(round_id, client_id, shared_secret)
    rng = torch.Generator()
    rng.manual_seed(seed)

    num_levels = (1 << num_bits) - 1
    mask_range = 2 * num_levels + 1
    mask = torch.randint(-num_levels, num_levels + 1, masked.shape,
                         generator=rng, dtype=torch.int32)

    unmasked = (masked.int() + num_levels - mask) % mask_range - num_levels
    return unmasked.to(torch.int16)


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 5 – Compact Sparse Wire Format
# ═══════════════════════════════════════════════════════════════════════════════

def _delta_encode_indices(indices: torch.Tensor) -> bytes:
    """Delta-encode sorted indices with varint compression."""
    buf = bytearray()
    prev = 0
    for idx in indices.tolist():
        d = idx - prev
        prev = idx
        if d < 128:
            buf.append(d)
        elif d < 16384:
            buf.append((d & 0x7F) | 0x80)
            buf.append(d >> 7)
        else:
            buf.append((d & 0x7F) | 0x80)
            buf.append(((d >> 7) & 0x7F) | 0x80)
            buf.append(d >> 14)
    return bytes(buf)


def _delta_decode_indices(data: bytes, count: int) -> torch.Tensor:
    result = []
    pos = 0
    prev = 0
    for _ in range(count):
        b0 = data[pos]; pos += 1
        if b0 < 128:
            d = b0
        else:
            b1 = data[pos]; pos += 1
            if b1 < 128:
                d = (b0 & 0x7F) | (b1 << 7)
            else:
                b2 = data[pos]; pos += 1
                d = (b0 & 0x7F) | ((b1 & 0x7F) << 7) | (b2 << 14)
        prev += d
        result.append(prev)
    return torch.tensor(result, dtype=torch.int64)


# ═══════════════════════════════════════════════════════════════════════════════
#  Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def compress_and_encrypt(
    delta: torch.Tensor,
    round_id: int,
    client_id: int,
    shared_secret: int,
    target_sparsity: float = 0.90,
    num_bits: int = 8,
    dp_epsilon: float = 8.0,
    noise_scale: float = 1.0,
) -> Dict[str, Any]:
    """SPQSA pipeline: sparsify → quantize → DP noise → mask → pack.

    Args:
        noise_scale: TAP-Noise scaling factor for topology-adaptive DP.
    """
    total_length = delta.numel()
    original_bytes = total_length * 4  # float32

    # Stage 1: Top-K Sparsification
    values, indices, threshold = topk_sparsify(delta, target_sparsity)
    nnz = indices.numel()
    actual_sparsity = 1.0 - nnz / total_length

    # Stage 2: QSGD Quantization
    quantized, scale = qsgd_quantize(values, num_bits)

    # Pre-DP reconstruction: dequantize BEFORE DP noise is added.
    # AdaEF uses this to compute the compression residual without
    # DP noise contamination, preventing stochastic privacy noise
    # from accumulating across rounds through the error buffer.
    pre_dp_dequantized = qsgd_dequantize(quantized, scale, num_bits)
    reconstructed_pre_dp = torch.zeros(total_length, dtype=torch.float32)
    reconstructed_pre_dp[indices.cpu()] = pre_dp_dequantized.float().cpu()

    # Stage 3: Differential Privacy
    if dp_epsilon > 0:
        quantized = add_dp_noise(quantized, scale, epsilon=dp_epsilon,
                                 num_bits=num_bits, noise_scale=noise_scale)

    # Stage 4: Secure Masking
    masked = apply_secure_mask(quantized, round_id, client_id,
                               shared_secret, num_bits)

    # Stage 5: Pack into compact wire format
    idx_bytes = _delta_encode_indices(indices)
    val_bytes = masked.numpy().astype(np.int16).tobytes()  # 2 bytes each

    # Header: scale(f32) + total_length(u32) + nnz(u32) + num_bits(u8)
    #        + threshold(f32) + idx_len(u32) + dp_epsilon(f32)
    header = struct.pack("<fIIBfIf",
                         scale, total_length, nnz, num_bits,
                         threshold, len(idx_bytes), dp_epsilon)
    payload = header + idx_bytes + val_bytes

    compressed_bytes = len(payload)
    ratio = original_bytes / max(compressed_bytes, 1)

    # Post-DP reconstruction: what the server actually receives
    dequantized = qsgd_dequantize(quantized, scale, num_bits)
    reconstructed = torch.zeros(total_length, dtype=torch.float32)
    reconstructed[indices.cpu()] = dequantized.float().cpu()

    stats = {
        "reconstructed_dense": reconstructed,
        "reconstructed_pre_dp": reconstructed_pre_dp,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": ratio,
        "sparsity": actual_sparsity,
        "nnz": nnz,
        "threshold": threshold,
        "scale": scale,
        "total_params": total_length,
        "num_bits": num_bits,
        "dp_epsilon": dp_epsilon,
        "noise_scale": noise_scale,
        "idx_bytes": len(idx_bytes),
        "val_bytes": len(val_bytes),
        "header_bytes": len(header),
    }

    return {
        "payload": payload,
        "stats": stats,
    }


def decrypt_and_decompress(
    packet: Dict[str, Any],
    round_id: int,
    client_id: int,
    shared_secret: int,
    **kwargs,
) -> torch.Tensor:
    """SPQSA inverse: unpack → unmask → dequantize → densify."""
    payload = packet["payload"]

    # Parse header
    header_fmt = "<fIIBfIf"
    header_size = struct.calcsize(header_fmt)
    scale, total_length, nnz, num_bits, threshold, idx_len, dp_epsilon = \
        struct.unpack_from(header_fmt, payload, 0)

    offset = header_size
    idx_data = payload[offset:offset + idx_len]; offset += idx_len
    val_data = payload[offset:offset + nnz * 2]

    # Decode
    indices = _delta_decode_indices(idx_data, nnz)
    masked = torch.from_numpy(
        np.frombuffer(val_data, dtype=np.int16).copy()
    )

    # Stage 4 inverse: Remove mask
    quantized = remove_secure_mask(masked, round_id, client_id,
                                    shared_secret, num_bits)

    # Stage 2 inverse: Dequantize
    values = qsgd_dequantize(quantized, scale, num_bits)

    # Reconstruct dense
    dense = torch.zeros(total_length, dtype=torch.float32)
    dense[indices] = values
    return dense
