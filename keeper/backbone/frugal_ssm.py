"""Memory-frugal SSM scan: identical arithmetic, no [B, L, D, N] history tensor.

WHY THIS EXISTS
---------------
``thecode/s3r.py:_vectorized_ssm_scan`` builds the scan by appending every
timestep's state to a Python list and calling ``torch.stack`` at the end:

    hs = []
    for t in range(L):
        h = h * dA[:, t] + dBu[:, t]
        hs.append(h)          # <-- keeps all L states alive
    return torch.stack(hs, dim=1)

That materialises a [B, L, D, N] tensor whose size is independent of L's
contribution to the graph. On Bitcoin-Alpha (L ~ 23k, D=64, N=4) it is ~236 MB
and nobody notices. On Epinions the training window is L = 833,180 edges, so
the same tensor is 833180 x 64 x 4 x 4 B = 8.5 GB, and dA and dBu are live at
the same time. Measured peak: 21.0 GiB, then CUDA OOM at 23.46 GiB capacity.

The recurrence itself is sequential in t, so the state at step t depends only on
h_{t-1} and the current input. Nothing needs the whole history retained to
compute y_t = sum_n h_t * C_t -- y_t is consumed immediately:

    y_c = torch.sum(hs_c * C[s].unsqueeze(-2), dim=-1)

So we fold that contraction into the loop and keep only h alive. The result is
numerically identical (same ops, same order) but peak memory becomes O(B*D*N)
instead of O(B*L*D*N).

WHY THIS LIVES IN ITS OWN FILE
------------------------------
IMPLEMENTATION_PLAN constraint 6 forbids refactoring thecode/ during the probe
phase, because the frozen Bitcoin pipeline depends on it. This module is
opt-in: s3r.py is NOT modified. Callers that need the large-graph path import
from here and pass use_frugal_scan=True.

SAFETY
------
``use_frugal_scan`` defaults to OFF. Small graphs keep the original code path
byte for byte, so every already-measured Bitcoin number remains reproducible.
"""

import torch
import torch.utils.checkpoint


def frugal_ssm_scan(dA: torch.Tensor, dBu: torch.Tensor,
                    C: torch.Tensor) -> torch.Tensor:
    """Return y = sum_n h_t[n] * C_t[n] for every t, without storing h_t.

    dA, dBu : [B, L, D, N]
    C       : [B, L, N]
    returns  : [B, L, D]

    Mirrors ``sum(_vectorized_ssm_scan(dA, dBu) * C.unsqueeze(-2), dim=-1)``
    exactly: the same recurrence, the same multiply-add, the same reduction
    over N, in the same order along t.
    """
    B, L, D, N = dA.shape
    h = torch.zeros(B, D, N, device=dA.device, dtype=dA.dtype)
    out = []
    Cb = C.unsqueeze(-2)                      # [B, L, 1, N]
    for t in range(L):
        h = h * dA[:, t] + dBu[:, t]          # [B, D, N]
        out.append(torch.sum(h * Cb[:, t], dim=-1))   # [B, D]
    return torch.stack(out, dim=1)


def frugal_ssm_scan_blocked(dA: torch.Tensor, dBu: torch.Tensor,
                            C: torch.Tensor, block: int = 4096) -> torch.Tensor:
    """Same as :func:`frugal_ssm_scan` but writes output in L-blocks.

    Only the [B, L, D] OUTPUT is ever materialised, which is 1/N of the full
    history tensor, so peak memory is dominated by the result rather than by the
    scan. ``block`` trades loop overhead against the transient buffer.
    """
    B, L, D, N = dA.shape
    h = torch.zeros(B, D, N, device=dA.device, dtype=dA.dtype)
    out = torch.empty(B, L, D, device=dA.device, dtype=dA.dtype)
    Cb = C.unsqueeze(-2)
    for s in range(0, L, block):
        e = min(s + block, L)
        for t in range(s, e):
            h = h * dA[:, t] + dBu[:, t]
            out[:, t] = torch.sum(h * Cb[:, t], dim=-1)
    return out


def frugal_ssm_scan_batched(dA: torch.Tensor, dBu: torch.Tensor,
                            C: torch.Tensor, node_chunk: int = 16384,
                            checkpoint: bool = True) -> torch.Tensor:
    """Chunk the BATCH (node) axis so only one node-slice of dA/dBu is live.

    Measured on epinions-dated: B = 131,828 nodes, L = 21 slots, D = 64, N = 4,
    so each [B, L, D, N] tensor is 2.64 GiB and forward+backward needs several
    live simultaneously -> OOM at 23.46 GiB. Chunking B keeps only
    [node_chunk, L, D, N] alive at a time (0.33 GiB at the default), which is
    what makes Epinions trainable.

    The scan is sequential in t and independent across batch rows, so splitting
    B is exact -- no approximation, no numerical drift. Gradient checkpointing
    per chunk trades compute for memory and is exact in eval / no-grad too.

    Falls back to the single-block path when B is small, so small graphs keep
    the cheapest code.
    """
    B, L, D, N = dA.shape
    if B <= node_chunk:
        return frugal_ssm_scan_blocked(dA, dBu, C)
    Cb = C.unsqueeze(-2)
    outs = []
    grad_on = torch.is_grad_enabled() and (
        dA.requires_grad or dBu.requires_grad or C.requires_grad)
    for s in range(0, B, node_chunk):
        e = min(s + node_chunk, B)
        da_c, dbu_c, c_c = dA[s:e], dBu[s:e], C[s:e]
        if grad_on and checkpoint:
            y_c = torch.utils.checkpoint.checkpoint(
                frugal_ssm_scan_blocked, da_c, dbu_c, c_c, use_reentrant=False)
        else:
            y_c = frugal_ssm_scan_blocked(da_c, dbu_c, c_c)
        outs.append(y_c)
        del da_c, dbu_c, c_c
    return torch.cat(outs, dim=0)


def _ssm_chunk(delta_c, B_ssm_c, C_c, u_c, A):
    """dA/dBu + scan for ONE node chunk. Separated so checkpoint() can call it."""
    dA = torch.exp(
        torch.einsum("b l d, d n -> b l d n", delta_c, A)).clamp(1e-6, 1.0)
    dBu = torch.einsum("b l d, b l n -> b l d n", delta_c, B_ssm_c) \
        * u_c.unsqueeze(-1)
    y = frugal_ssm_scan_blocked(dA, dBu, C_c)
    del dA, dBu
    return y


def frugal_ssm_forward(delta: torch.Tensor, B_ssm: torch.Tensor,
                       C: torch.Tensor, u: torch.Tensor, A: torch.Tensor,
                       node_chunk: int = 16384) -> torch.Tensor:
    """Whole scan path, chunked over the NODE (batch) axis.

    This is the entry point s3r.forward calls when S3R_FRUGAL_SCAN=1. It exists
    because chunking only the scan is not enough: dA and dBu are each
    2.64 GiB at Epinions scale (B=131,828 nodes, L=21, D=64, N=4), so building
    them at full size and then chunking the scan still OOMs. Here they are
    built per chunk, so only [node_chunk, L, D, N] is ever live.

    Exactly reproduces the non-frugal expression

        dA   = exp(einsum(delta, A)).clamp(1e-6, 1.0)
        dBu  = einsum(delta, B_ssm) * u.unsqueeze(-1)
        y    = sum(_vectorized_ssm_scan(dA, dBu) * C.unsqueeze(-2), dim=-1)

    chunk 16384 -> 0.33 GiB per [chunk, L, D, N] tensor.
    """
    B = delta.shape[0]
    if B <= node_chunk:
        return _ssm_chunk(delta, B_ssm, C, u, A)
    outs = []
    grad_on = torch.is_grad_enabled() and (
        delta.requires_grad or B_ssm.requires_grad or C.requires_grad
        or u.requires_grad or A.requires_grad)
    for s in range(0, B, node_chunk):
        e = min(s + node_chunk, B)
        args = (delta[s:e], B_ssm[s:e], C[s:e], u[s:e], A)
        if grad_on:
            y_c = torch.utils.checkpoint.checkpoint(
                _ssm_chunk, *args, use_reentrant=False)
        else:
            y_c = _ssm_chunk(*args)
        outs.append(y_c)
    return torch.cat(outs, dim=0)