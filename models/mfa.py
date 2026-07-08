"""MFA: Manifold Feature Alignment — InfoNCE + Learned-Query Cross-Attention.

Key fix for attention degeneracy:
  - OLD: Self-attention on 3 modality tokens → 3x3=9 QK pairs, 8 heads degenerate
  - NEW: 8 learned query tokens attend to 3 modality tokens (K,V) via cross-attention
    → 8 queries × 3 keys = 24 QK pairs per head, rich differentiated patterns

The query tokens serve as "cross-modal probes" — each learns to extract a different
type of cross-modal information (e.g., geometry-weighted-by-spectra, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignProjector(nn.Module):
    """3-layer residual projection head with LayerNorm."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.ln = nn.LayerNorm(out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(x) + self.residual(x))


class ManifoldAlignment(nn.Module):
    """Projects modality embeddings into shared 1024-dim aligned manifold.

    Architecture:
      1. Per-modality AlignProjector + L2 normalize
      2. Pairwise InfoNCE contrastive loss (training signal)
      3. Learned-query cross-attention: 8 query tokens attend to 3 modality tokens
      4. Mean-pool + fusion projection

    The learned queries replace the previous self-attention which degenerated
    to uniform weights with only 3 tokens. Each query token is a learnable
    "cross-modal probe" — 8 queries × 3 modalities = rich, non-uniform attention.
    """

    def __init__(self, in_dim: int = 1024, align_dim: int = 1024,
                 hidden_dim: int = 1024, num_queries: int = 8,
                 initial_temperature: float = 0.07):
        super().__init__()
        self.align_dim = align_dim
        self.num_queries = num_queries

        # One projector per physical modality
        self.proj_mph = AlignProjector(in_dim, hidden_dim, align_dim)
        self.proj_pls = AlignProjector(in_dim, hidden_dim, align_dim)
        self.proj_egnn = AlignProjector(in_dim, hidden_dim, align_dim)

        # Learnable InfoNCE temperature
        self.log_tau = nn.Parameter(torch.tensor(initial_temperature).log())

        # Learnable query tokens — "cross-modal probes"
        # Each query learns to extract a different type of cross-modal information
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, align_dim) * 0.02
        )

        # Cross-attention: Q = learned queries, K,V = modality tokens
        # num_heads=8 matches query count so each head processes one query position,
        # producing 8×3=24 attention values per head — rich, non-degenerate patterns.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=align_dim, num_heads=num_queries,
            batch_first=True, dropout=0.3
        )

        # Post-attention fusion
        self.fusion_out = nn.Sequential(
            nn.Linear(align_dim, align_dim),
            nn.SiLU(),
            nn.LayerNorm(align_dim),
        )

    def forward(self, h_mph: torch.Tensor, h_pls: torch.Tensor,
                h_egnn: torch.Tensor,
                compute_align_loss: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Align modality embeddings and fuse via learned-query cross-attention.

        Args:
            h_mph, h_pls, h_egnn: (B, in_dim) modality embeddings.
            compute_align_loss: If False, return zero align_loss (eval mode).

        Returns:
            z_fusion: (B, align_dim) fused representation.
            align_loss: scalar InfoNCE loss.
        """
        B = h_mph.shape[0]

        # ---- Phase 1: Project & normalize ----
        z_mph = F.normalize(self.proj_mph(h_mph), dim=-1, eps=1e-6)
        z_pls = F.normalize(self.proj_pls(h_pls), dim=-1, eps=1e-6)
        z_egnn = F.normalize(self.proj_egnn(h_egnn), dim=-1, eps=1e-6)

        # ---- Phase 2: Pairwise InfoNCE contrastive loss ----
        embeddings = [z_mph, z_pls, z_egnn]
        tau = self.log_tau.exp()
        align_loss = 0.0
        if compute_align_loss:
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = torch.matmul(embeddings[i], embeddings[j].T) / tau
                    labels = torch.arange(B, device=sim.device)
                    loss_ij = F.cross_entropy(sim, labels)
                    loss_ji = F.cross_entropy(sim.T, labels)
                    align_loss += (loss_ij + loss_ji) / 2.0
            align_loss /= 3.0

        # ---- Phase 3: Learned-query cross-attention ----
        # Q: (B, num_queries, align_dim) — learned probes, broadcast to batch
        # K,V: (B, 3, align_dim) — modality tokens
        queries = self.query_tokens.expand(B, -1, -1)
        kv_tokens = torch.stack([z_mph, z_pls, z_egnn], dim=1)  # (B, 3, align_dim)

        # Cross-attention: each query learns to attend differently to the 3 modalities
        # attn_weights: (B, num_heads=8, num_queries=8, kv_len=3)
        #   → per-head: 8 queries × 3 keys = 24 QK values → RICH, NON-UNIFORM
        attn_out, _attn_weights = self.cross_attn(queries, kv_tokens, kv_tokens)
        # attn_out: (B, num_queries, align_dim)

        # Mean-pool query outputs → fused representation
        z_fusion = self.fusion_out(attn_out.mean(dim=1))  # (B, align_dim)

        return z_fusion, align_loss

    def get_attention_weights(self, h_mph, h_pls, h_egnn):
        """Extract cross-attention weights for interpretability analysis.

        Returns:
            attn_weights: (B, num_heads, num_queries, 3) — attention over {MPH, PLS, EGNN}
        """
        with torch.no_grad():
            B = h_mph.shape[0]
            z_mph = F.normalize(self.proj_mph(h_mph), dim=-1, eps=1e-6)
            z_pls = F.normalize(self.proj_pls(h_pls), dim=-1, eps=1e-6)
            z_egnn = F.normalize(self.proj_egnn(h_egnn), dim=-1, eps=1e-6)
            queries = self.query_tokens.expand(B, -1, -1)
            kv_tokens = torch.stack([z_mph, z_pls, z_egnn], dim=1)
            _, attn_weights = self.cross_attn(queries, kv_tokens, kv_tokens)
        return attn_weights  # (B, num_heads, num_queries, 3)
