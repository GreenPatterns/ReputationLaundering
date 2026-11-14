import math
import os
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

# Enable memory-efficient attention kernels
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


def _xavier(t: torch.Tensor) -> torch.Tensor:
    nn.init.xavier_uniform_(t)
    return t


def _vectorized_ssm_scan(dA: torch.Tensor, dBu: torch.Tensor) -> torch.Tensor:
    B, L, D, N = dA.shape
    h = torch.zeros(B, D, N, device=dA.device, dtype=dA.dtype)
    hs = []
    for t in range(L):
        h = h * dA[:, t] + dBu[:, t]
        hs.append(h)
    return torch.stack(hs, dim=1)


def _chunked_ssm_scan(
    dA: torch.Tensor,
    dBu: torch.Tensor,
    chunk_size: int = 8192,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """B-axis chunked SSM scan with optional activation checkpointing.

    Chunking: split nodes into chunks along dim 0 (batch), run scan per chunk.
    Checkpointing: PyTorch recomputes intermediate states during backward,
    trading compute for memory. Each scan step produces (B_c, D, N) states
    which would otherwise consume O(B*L*D*N) memory.

    When chunk_size is None or >= B, falls back to plain vectorized scan.
    This preserves exact behaviour for small graphs (Bitcoin).
    """
    B = dA.shape[0]
    if chunk_size is None or B <= chunk_size:
        return _vectorized_ssm_scan(dA, dBu)

    chunks = []
    for b_start in range(0, B, chunk_size):
        b_end = min(b_start + chunk_size, B)
        dA_c = dA[b_start:b_end]
        dBu_c = dBu[b_start:b_end]
        if use_checkpoint and torch.is_grad_enabled():
            chunk_hs = torch.utils.checkpoint.checkpoint(
                _vectorized_ssm_scan, dA_c, dBu_c, use_reentrant=False
            )
        else:
            chunk_hs = _vectorized_ssm_scan(dA_c, dBu_c)
        chunks.append(chunk_hs)
    return torch.cat(chunks, dim=0)


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 4, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        self.W_in  = nn.Parameter(_xavier(torch.empty(d_model, self.d_inner * 2)))
        self.W_x   = nn.Parameter(_xavier(torch.empty(self.d_inner, self.dt_rank + d_state * 2)))
        self.W_dt  = nn.Parameter(_xavier(torch.empty(self.dt_rank, self.d_inner)))
        self.W_out = nn.Parameter(_xavier(torch.empty(self.d_inner, d_model)))

        dt_bias_val = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        self.b_dt = nn.Parameter(torch.log(torch.expm1(dt_bias_val)))

        A = einops.repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            'n -> d n', d=self.d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))

        self.d_conv = d_conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        xz   = x @ self.W_in
        u, z = xz.chunk(2, dim=-1)

        kernel_size = min(self.d_conv, L)
        weight      = self.conv1d.weight[:, :, :kernel_size]
        padding     = kernel_size - 1
        u_conv = F.conv1d(
            u.transpose(1, 2),
            weight,
            bias=None,
            stride=1,
            padding=padding,
            groups=self.d_inner
        ).transpose(1, 2)[:, :L, :]
        u = F.silu(u_conv)

        x_dbl = u @ self.W_x
        delta_r, B_ssm, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(delta_r @ self.W_dt + self.b_dt).clamp(min=1e-5, max=10.0)
        A  = -torch.exp(self.A_log.float())

        chunk_size = int(os.environ.get("S3R_CHUNK_SIZE", "8192"))
        # Opt-in frugal path for large graphs: identical arithmetic, but the
        # [B, L, D, N] tensors are built ONE NODE-CHUNK AT A TIME so they are
        # never all live at once. Required for Epinions (B=131,828 nodes,
        # L=21, D=64, N=4 -> 2.64 GiB per tensor -> OOM at 23.46 GiB); see
        # laundering/frugal_ssm.py. Off by default so the frozen Bitcoin
        # pipeline stays byte-identical.
        if os.environ.get("S3R_FRUGAL_SCAN", "0") == "1":
            from keeper.backbone.frugal_ssm import frugal_ssm_forward
            y = frugal_ssm_forward(
                delta, B_ssm, C, u, A,
                node_chunk=int(os.environ.get("S3R_NODE_CHUNK", "16384")))
        elif chunk_size > 0 and B >= chunk_size // 2:
            # Chunk the entire SSM path to avoid materializing full [B, L, D, N] tensors
            y_chunks = []
            for b_start in range(0, B, chunk_size):
                b_end = min(b_start + chunk_size, B)
                s = slice(b_start, b_end)

                dA_c = torch.exp(
                    einops.einsum(delta[s], A, 'b l d, d n -> b l d n')
                ).clamp(min=1e-6, max=1.0)

                dBu_c = einops.einsum(
                    delta[s], B_ssm[s], 'b l d, b l n -> b l d n'
                ) * u[s].unsqueeze(-1)

                if self.training and torch.is_grad_enabled():
                    hs_c = torch.utils.checkpoint.checkpoint(
                        _vectorized_ssm_scan, dA_c, dBu_c, use_reentrant=False
                    )
                else:
                    hs_c = _vectorized_ssm_scan(dA_c, dBu_c)

                y_c = torch.sum(hs_c * C[s].unsqueeze(-2), dim=-1)
                y_chunks.append(y_c)
            y = torch.cat(y_chunks, dim=0)
        else:
            dA = torch.exp(
                einops.einsum(delta, A, 'b l d, d n -> b l d n')
            ).clamp(min=1e-6, max=1.0)

            dBu = einops.einsum(
                delta, B_ssm, 'b l d, b l n -> b l d n'
            ) * u.unsqueeze(-1)

            hs = _vectorized_ssm_scan(dA, dBu)
            y = torch.sum(hs * C.unsqueeze(-2), dim=-1)

        y = y * F.silu(z)
        y = y + u * self.D
        return y @ self.W_out


class S3R(nn.Module):
    def __init__(
        self,
        input_dim:   int   = 32,
        n_heads:     int   = 8,
        max_seq_len: int   = 23,
        dropout:     float = 0.1,
        top_k_ratio: float = 0.1,
        base_window: int   = 3,
        residual:    bool  = True,
        seed:        int   = 42,
    ):
        super().__init__()

        if input_dim % n_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by n_heads ({n_heads})")

        self.input_dim   = input_dim
        self.n_heads     = n_heads
        self.max_seq_len = max_seq_len
        self.head_dim    = input_dim // n_heads
        self.top_k_ratio = top_k_ratio
        self.base_window = base_window
        self.residual    = residual

        self.position_embeddings = nn.Parameter(torch.Tensor(max_seq_len, input_dim))
        nn.init.xavier_uniform_(self.position_embeddings)

        self.W_q        = nn.Parameter(_xavier(torch.empty(input_dim, input_dim)))
        self.W_k        = nn.Parameter(_xavier(torch.empty(input_dim, input_dim)))
        self.W_v        = nn.Parameter(_xavier(torch.empty(input_dim, input_dim)))
        self.W_o        = nn.Parameter(_xavier(torch.empty(input_dim, input_dim)))
        self.W_gate     = nn.Parameter(_xavier(torch.empty(input_dim, 2 * input_dim)))
        self.W_ssm_bias = nn.Parameter(_xavier(torch.empty(input_dim, n_heads)))

        self.attention_gain = nn.Parameter(torch.ones(1, 1, input_dim).clamp(max=1.0))
        self.ffn_gain       = nn.Parameter(torch.ones(1, 1, input_dim).clamp(max=1.0))
        self.attention_temp = nn.Parameter(
            torch.full((n_heads, 1, 1), 1.0 / math.sqrt(self.head_dim))
        )

        self.dropout   = nn.Dropout(dropout)
        self.ssm_block = SelectiveSSM(d_model=input_dim, d_state=4)
        self._init_sparse_patterns()

        # Attention chunking config
        # Paper default: chunk_size=8192, checkpointing enabled for memory efficiency
        self.attn_chunk_size = int(os.environ.get("S3R_ATTN_CHUNK", "8192"))
        self.use_checkpoint = os.environ.get("S3R_CHECKPOINT", "1") == "1"

        # Pre-compute and cache causal mask
        causal = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer('causal_mask', causal)

    def _init_sparse_patterns(self):
        patterns = []
        for head_idx in range(self.n_heads):
            pattern  = torch.zeros(self.max_seq_len, self.max_seq_len, dtype=torch.bool)
            dilation = 2 ** (head_idx % 4)
            for i in range(self.max_seq_len):
                pattern[i, i] = True
                for w in range(1, self.base_window + 1):
                    j = i - w * dilation
                    if j >= 0:
                        pattern[i, j] = True
                stride = max(1, self.max_seq_len // 4)
                for j in range(0, i + 1, stride):
                    pattern[i, j] = True
                pattern[i, 0] = True
                exp_pos = 1
                while i - exp_pos >= 0:
                    pattern[i, i - exp_pos] = True
                    exp_pos *= 3
            patterns.append(pattern)
        self.register_buffer('sparse_pattern', torch.stack(patterns))

    def _compute_hybrid_sparse_attention(self, Q, K, V, static_patterns, ssm_context, seq_len):
        """Original dense attention - kept for compatibility."""
        B = Q.shape[0]

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.attention_temp.unsqueeze(0)
        ssm_bias = (ssm_context @ self.W_ssm_bias).permute(0, 2, 1).unsqueeze(-2)
        scores   = scores + ssm_bias

        k = max(1, int(seq_len * self.top_k_ratio))
        _, topk_idx = torch.topk(scores, k, dim=-1)
        dyn_mask = torch.zeros_like(scores, dtype=torch.bool)
        dyn_mask.scatter_(-1, topk_idx, True)
        dyn_mask[:, :, :, 0] = True

        stat_mask   = static_patterns[:, :seq_len, :seq_len].unsqueeze(0).expand(
                          B, self.n_heads, seq_len, seq_len)
        causal = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)

        final_mask    = (stat_mask | dyn_mask) & causal
        scores_masked = scores.masked_fill(~final_mask, float('-inf'))
        weights       = F.softmax(scores_masked, dim=-1)
        if self.training:
            weights = self.dropout(weights)

        output = torch.matmul(weights, V)
        return output, weights

    def _compute_chunked_attention(self, Q, K, V, static_patterns, ssm_context, seq_len):
        """Memory-efficient chunked attention: processes nodes in batches to avoid [B, H, T, T] allocation."""
        B, H, L, D = Q.shape
        chunk_size = self.attn_chunk_size
        
        if chunk_size <= 0 or B <= chunk_size:
            return self._compute_hybrid_sparse_attention(Q, K, V, static_patterns, ssm_context, seq_len)

        # Build masks once (they're the same for all chunks)
        k = max(1, int(seq_len * self.top_k_ratio))
        stat_mask = static_patterns[:, :seq_len, :seq_len].unsqueeze(0)  # [1, H, L, L]
        causal_mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        
        out_chunks = []
        weight_chunks = [] if not self.training else None
        
        for b_start in range(0, B, chunk_size):
            b_end = min(b_start + chunk_size, B)
            B_c = b_end - b_start
            
            Q_c = Q[b_start:b_end]
            K_c = K[b_start:b_end]
            V_c = V[b_start:b_end]
            ssm_context_c = ssm_context[b_start:b_end]
            
            # Compute scores for this chunk
            scores_c = torch.matmul(Q_c, K_c.transpose(-2, -1)) * self.attention_temp.unsqueeze(0)
            ssm_bias_c = (ssm_context_c @ self.W_ssm_bias).permute(0, 2, 1).unsqueeze(-2)
            scores_c = scores_c + ssm_bias_c
            
            # Dynamic top-k mask
            _, topk_idx_c = torch.topk(scores_c, k, dim=-1)
            dyn_mask_c = torch.zeros_like(scores_c, dtype=torch.bool)
            dyn_mask_c.scatter_(-1, topk_idx_c, True)
            dyn_mask_c[:, :, :, 0] = True
            
            # Combine masks
            stat_mask_c = stat_mask.expand(B_c, H, L, L)
            causal_mask_c = causal_mask.expand(B_c, H, L, L)
            final_mask_c = (stat_mask_c | dyn_mask_c) & causal_mask_c
            
            scores_masked_c = scores_c.masked_fill(~final_mask_c, float('-inf'))
            weights_c = F.softmax(scores_masked_c, dim=-1)
            
            if self.training:
                weights_c = self.dropout(weights_c)
            
            output_c = torch.matmul(weights_c, V_c)
            out_chunks.append(output_c)
            
            if weight_chunks is not None:
                weight_chunks.append(weights_c)
        
        output = torch.cat(out_chunks, dim=0)
        weights = torch.cat(weight_chunks, dim=0) if weight_chunks is not None else None
        return output, weights

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds maximum {self.max_seq_len}")

        chunk_size = int(os.environ.get("S3R_CHUNK_SIZE", "8192"))
        if chunk_size > 0 and B > chunk_size:
            return self._forward_chunked(x, chunk_size)

        pos_ids = torch.arange(L, device=x.device)
        h       = x + self.position_embeddings[pos_ids]
        ssm_ctx = self.ssm_block(h)

        Q = (h @ self.W_q).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = (h @ self.W_k).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = (h @ self.W_v).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Use chunked attention if enabled and beneficial
        if self.attn_chunk_size > 0 and B > self.attn_chunk_size:
            attn_out, attn_w = self._compute_chunked_attention(
                Q, K, V, self.sparse_pattern, ssm_ctx, L
            )
        else:
            attn_out, attn_w = self._compute_hybrid_sparse_attention(
                Q, K, V, self.sparse_pattern, ssm_ctx, L
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.input_dim)
        attn_out = attn_out @ self.W_o
        h        = h + attn_out * self.attention_gain

        glu = h @ self.W_gate
        val, gate = glu.chunk(2, dim=-1)
        ffn_out = val * torch.sigmoid(gate)
        if self.training:
            ffn_out = self.dropout(ffn_out)

        if self.residual:
            h = h + ffn_out * self.ffn_gain
        else:
            h = ffn_out

        return h

    def _forward_chunked(self, x: torch.Tensor, chunk_size: int):
        """B-axis chunked forward for large-node-count graphs."""
        B, L, _ = x.shape
        pos_ids = torch.arange(L, device=x.device)
        out_chunks = []

        for b_start in range(0, B, chunk_size):
            b_end = min(b_start + chunk_size, B)
            s = slice(b_start, b_end)

            h_c = x[s] + self.position_embeddings[pos_ids]
            ssm_ctx_c = self.ssm_block(h_c)

            Q_c = (h_c @ self.W_q).view(b_end - b_start, L, self.n_heads, self.head_dim).transpose(1, 2)
            K_c = (h_c @ self.W_k).view(b_end - b_start, L, self.n_heads, self.head_dim).transpose(1, 2)
            V_c = (h_c @ self.W_v).view(b_end - b_start, L, self.n_heads, self.head_dim).transpose(1, 2)

            # Use chunked attention within each chunk if needed
            if self.attn_chunk_size > 0 and (b_end - b_start) > self.attn_chunk_size:
                attn_out_c, _ = self._compute_chunked_attention(
                    Q_c, K_c, V_c, self.sparse_pattern, ssm_ctx_c, L
                )
            else:
                attn_out_c, _ = self._compute_hybrid_sparse_attention(
                    Q_c, K_c, V_c, self.sparse_pattern, ssm_ctx_c, L
                )

            attn_out_c = attn_out_c.transpose(1, 2).contiguous().view(b_end - b_start, L, self.input_dim)
            attn_out_c = attn_out_c @ self.W_o
            h_c = h_c + attn_out_c * self.attention_gain

            glu_c = h_c @ self.W_gate
            val_c, gate_c = glu_c.chunk(2, dim=-1)
            ffn_out_c = val_c * torch.sigmoid(gate_c)
            if self.training:
                ffn_out_c = self.dropout(ffn_out_c)

            if self.residual:
                h_c = h_c + ffn_out_c * self.ffn_gain
            else:
                h_c = ffn_out_c

            out_chunks.append(h_c)

        return torch.cat(out_chunks, dim=0)