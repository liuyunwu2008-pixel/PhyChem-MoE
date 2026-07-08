"""CoT-MoE: Top-level model combining physical encoders, MFA, MoE, CoT, and task heads."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import MPHEncoder, PLSEncoder, EGNNEncoder, RSASEncoder
from .mfa import ManifoldAlignment
from .moe import SparseMoE
from .cot_module import VirtualTokenMapper
from .heads import ClassificationHead, RegressionHead


class CoTMoEModel(nn.Module):
    """Full CoT-MoE molecular property predictor.

    Flow:
      Features → Encoders → MFA (InfoNCE) → MoE (Top-2) → z_fusion
      z_fusion → Virtual Token Mapper → LLM (CoT) → h_CoT
      [z_fusion ⊕ h_CoT] → Task Heads → Predictions

    Supports a no-LLM placeholder mode for initial development.
    """

    def __init__(
        self,
        # Model dimensions
        align_dim: int = 1024,
        num_experts: int = 8,
        top_k: int = 2,
        # Task config
        classification_datasets: Optional[list] = None,
        regression_datasets: Optional[list] = None,
        num_classes: Optional[Dict[str, int]] = None,
        # LLM config
        num_virtual_tokens: int = 8,
        llm_hidden_dim: int = 4096,
        llm_load_mode: str = "8bit",
        # Feature config
        mph_resolution: int = 80,
        mph_num_scales: int = 3,
        pls_top_k: int = 256,
        egnn_hidden: int = 128,
        rsas_embed_dim: int = 64,
        contrastive_temp: float = 0.07,
        mph_dispersion_weight: float = 0.01,
    ):
        super().__init__()

        self.align_dim = align_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.llm_load_mode = llm_load_mode
        self.mph_dispersion_weight = mph_dispersion_weight

        mph_in_dim = mph_resolution * mph_resolution * mph_num_scales

        # --- Physical Encoders ---
        self.mph_encoder = MPHEncoder(
            in_dim=mph_in_dim, out_dim=align_dim
        )
        self.pls_encoder = PLSEncoder(
            in_dim=pls_top_k, out_dim=align_dim
        )
        self.egnn_encoder = EGNNEncoder(
            in_dim=egnn_hidden, out_dim=align_dim
        )
        self.rsas_encoder = RSASEncoder(
            in_dim=rsas_embed_dim, out_dim=align_dim
        )

        # --- Manifold Alignment (MFA) ---
        self.mfa = ManifoldAlignment(dim=align_dim, temperature=contrastive_temp)

        # --- MoE Router ---
        self.moe = SparseMoE(dim=align_dim, num_experts=num_experts, top_k=top_k)

        # --- CoT Virtual Token Mapper ---
        self.token_mapper = VirtualTokenMapper(
            fusion_dim=align_dim,
            num_virtual_tokens=num_virtual_tokens,
            llm_hidden_dim=llm_hidden_dim,
        )

        # --- Fusion Projection: [z_fusion ⊕ h_CoT] → final ---
        # When LLM is disabled, h_CoT is replaced with zeros
        self.final_fusion = nn.Sequential(
            nn.Linear(align_dim * 2, align_dim),
            nn.SiLU(),
            nn.Linear(align_dim, align_dim),
            nn.LayerNorm(align_dim),
        )

        # --- Task Heads ---
        self.classification_datasets = classification_datasets or ["bbbp", "bace", "clintox", "sider", "tox21"]
        self.regression_datasets = regression_datasets or ["freesolv", "esol", "lipo", "qm7"]
        self.num_classes = num_classes or {
            "bbbp": 1, "bace": 1, "clintox": 2, "sider": 27, "tox21": 12,
        }

        self.heads = nn.ModuleDict()
        for ds in self.classification_datasets:
            self.heads[ds] = ClassificationHead(align_dim, self.num_classes.get(ds, 1))
        for ds in self.regression_datasets:
            self.heads[ds] = RegressionHead(align_dim, 1)

        # Placeholder mode flag (controlled externally)
        self._llm_placeholder: bool = True

    @property
    def use_llm(self) -> bool:
        return not self._llm_placeholder

    @use_llm.setter
    def use_llm(self, value: bool):
        self._llm_placeholder = not value

    def _get_h_cot(self, z_fusion: torch.Tensor, smiles: list) -> torch.Tensor:
        """Get CoT features from LLM (or zeros if placeholder mode).

        Args:
            z_fusion: (B, 1024)
            smiles: list of SMILES strings.

        Returns:
            h_cot: (B, 1024) CoT semantic features.
        """
        if self._llm_placeholder:
            # Placeholder: use projected z_fusion as CoT features
            return torch.zeros_like(z_fusion)

        # Full mode: run virtual token mapper + LLM (implemented in cot_generator)
        virtual_tokens = self.token_mapper(z_fusion)  # (B, N_vt, llm_dim)
        h_cot = self.cot_generator(virtual_tokens, smiles) if hasattr(self, "cot_generator") else torch.zeros_like(z_fusion)
        return h_cot

    def forward(
        self,
        features,
        smiles: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: FeatureBatch from FeaturePipeline.
            smiles: list of SMILES strings (for CoT generation).

        Returns:
            Dict mapping task names to raw predictions, plus auxiliary outputs.
        """
        B = features.mph.shape[0]

        # 1. Encode physical features → 1024d each
        h_mph = self.mph_encoder(features.mph)
        h_pls = self.pls_encoder(features.pls)
        h_egnn = self.egnn_encoder(features.egnn)
        h_rsas = self.rsas_encoder(features.rsas, features.atom_mask)

        # 2. MFA alignment
        z_fusion_mfa, align_loss = self.mfa(h_mph, h_pls, h_egnn, h_rsas)

        # 3. MoE routing
        z_fusion, aux_loss = self.moe(z_fusion_mfa)

        # 4. CoT reasoning features
        h_cot = self._get_h_cot(z_fusion, smiles)

        # 5. Final fusion: [z_fusion ⊕ h_cot]
        final_features = torch.cat([z_fusion, h_cot], dim=-1)  # (B, 2048)
        final_features = self.final_fusion(final_features)  # (B, 1024)

        # 6. Task-specific predictions
        predictions = {}
        for ds_name, head in self.heads.items():
            predictions[ds_name] = head(final_features)

        # Auxiliary outputs
        predictions["_align_loss"] = align_loss
        predictions["_aux_loss"] = aux_loss
        predictions["_mph_dispersion_loss"] = self.mph_encoder.dispersion_loss(h_mph)
        predictions["_z_fusion"] = z_fusion
        predictions["_h_cot"] = h_cot

        return predictions

    def get_virtual_tokens(self, z_fusion: torch.Tensor) -> torch.Tensor:
        """Get virtual token embeddings for LLM input."""
        return self.token_mapper(z_fusion)

    def get_task_losses(self, predictions: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute per-task losses.

        Classification: BCEWithLogitsLoss (numerically stable, log-sum-exp).
          SIDER (27 classes) and Tox21 (12 classes) are multi-label with
          severe class imbalance — pos_weight up-weights rare positive labels.
        Regression: SmoothL1Loss (Huber, β=1.0) — less outlier-sensitive
          than MSE, aligns better with MAE evaluation metric.
        """
        losses = {}
        total = 0.0

        for ds_name in self.classification_datasets:
            if ds_name in predictions and ds_name in labels:
                pred = predictions[ds_name]          # raw logits
                target = labels[ds_name]

                # Multi-label imbalance: estimate pos_weight per task
                pos_weight = self._get_pos_weight(ds_name, target)

                loss = F.binary_cross_entropy_with_logits(
                    pred, target, pos_weight=pos_weight)
                losses[ds_name] = loss
                total = total + loss

        for ds_name in self.regression_datasets:
            if ds_name in predictions and ds_name in labels:
                pred = predictions[ds_name]
                target = labels[ds_name]
                # SmoothL1 (Huber): MSE near zero, MAE for large residuals
                loss = F.smooth_l1_loss(
                    pred.squeeze(-1), target.squeeze(-1), beta=1.0)
                losses[ds_name] = loss
                total = total + loss

        # Add alignment and auxiliary losses
        if "_align_loss" in predictions:
            losses["_align"] = predictions["_align_loss"]
            total = total + predictions["_align_loss"]
        if "_aux_loss" in predictions:
            losses["_aux"] = predictions["_aux_loss"] * 0.01
            total = total + predictions["_aux_loss"] * 0.01
        if "_mph_dispersion_loss" in predictions:
            losses["_mph_dispersion"] = predictions["_mph_dispersion_loss"] * self.mph_dispersion_weight
            total = total + predictions["_mph_dispersion_loss"] * self.mph_dispersion_weight

        losses["_total"] = total
        return losses

    @staticmethod
    def _get_pos_weight(ds_name: str, target: torch.Tensor) -> Optional[torch.Tensor]:
        """Per-class positive weight for imbalanced multi-label tasks.

        pos_weight = #negatives / #positives per class.
        Only applied for SIDER (27 classes) and Tox21 (12 classes) where
        class imbalance is severe (some labels <1% positive).
        """
        if ds_name not in ("sider", "tox21"):
            return None

        pos_count = target.sum(dim=0).clamp(min=1)            # (C,)
        neg_count = target.shape[0] - pos_count                # (C,)
        return neg_count / pos_count
