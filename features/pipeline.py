"""Feature pipeline: orchestrates online computation of all physical features."""

from dataclasses import dataclass
from typing import Dict, Optional
import gc
import numpy as np
import torch

from .mph import compute_persistence_image
from .pls import compute_persistent_laplacians
from .egnn import FrozenEGNN
from .rsas import compute_atom_properties, AtomPropertyEmbedding
from ..data.conformer import generate_conformer, get_atom_types, get_bond_edges
from ..data.caching import FeatureCache


@dataclass
class FeatureBatch:
    mph: torch.Tensor       # (B, mph_dim) — multi-scale PI, dimension varies
    pls: torch.Tensor       # (B, 256)
    egnn: torch.Tensor      # (B, 128)
    rsas: torch.Tensor      # (B, N_max, 64)
    coords: torch.Tensor    # (B, N_max, 3) padded
    atom_mask: torch.Tensor  # (B, N_max) bool


class FeaturePipeline:
    """Orchestrates online feature computation with caching support."""

    def __init__(
        self,
        mph_resolution: int = 80,
        mph_sigma_values: Optional[list] = None,
        pls_top_k: int = 256,
        pls_num_filtrations: int = 100,
        egnn_layers: int = 4,
        egnn_hidden: int = 128,
        rsas_embed_dim: int = 64,
        freeze_egnn: bool = True,
        use_gudhi: bool = True,
        cache_dir: Optional[str] = None,
    ):
        self.mph_resolution = mph_resolution
        self.mph_sigma_values = mph_sigma_values if mph_sigma_values is not None else [0.25, 0.5, 1.0]
        self.pls_top_k = pls_top_k
        self.pls_num_filtrations = pls_num_filtrations
        self.egnn_hidden = egnn_hidden
        self.rsas_embed_dim = rsas_embed_dim
        self.use_gudhi = use_gudhi

        # EGNN will be initialized when device is known (lazy init)
        self.egnn: Optional[FrozenEGNN] = None
        self.egnn_layers = egnn_layers
        self.freeze_egnn = freeze_egnn

        self.rsas_embedding = AtomPropertyEmbedding(rsas_embed_dim)
        self.rsas_embedding.init_weights()

        self.cache = FeatureCache(cache_dir) if cache_dir else None

    @property
    def mph_dim(self) -> int:
        """Total dimensionality of multi-scale persistence image."""
        return self.mph_resolution * self.mph_resolution * len(self.mph_sigma_values)

    def init_egnn(self, device: torch.device):
        self.egnn = FrozenEGNN(
            num_layers=self.egnn_layers,
            hidden_dim=self.egnn_hidden,
            freeze_coords=self.freeze_egnn,
        ).to(device)

    def compute_single(self, smiles: str, mol_idx: int = -1) -> Optional[Dict[str, np.ndarray]]:
        """Compute all features for a single molecule. Returns numpy dict."""
        # Check cache
        if self.cache is not None and mol_idx >= 0 and self.cache.has(mol_idx):
            return self.cache.get(mol_idx)

        coords = generate_conformer(smiles, use_mmff=False)
        if coords is None:
            return None

        atom_types = get_atom_types(smiles)
        edge_index = get_bond_edges(smiles)
        atom_props = compute_atom_properties(smiles)

        mph_vec = compute_persistence_image(coords, self.mph_resolution, sigma_values=self.mph_sigma_values, use_gudhi=self.use_gudhi)
        pls_vec = compute_persistent_laplacians(coords, self.pls_top_k, self.pls_num_filtrations)

        result = {
            "mph": mph_vec.astype(np.float32),
            "pls": pls_vec.astype(np.float32),
            "coords": coords.astype(np.float32),
            "atom_types": atom_types.astype(np.int64) if atom_types is not None else np.zeros(0, dtype=np.int64),
            "edge_index": edge_index.astype(np.int64) if edge_index is not None else np.zeros((2, 0), dtype=np.int64),
            "atom_props": atom_props.astype(np.float32) if atom_props is not None else np.zeros((0, 2), dtype=np.float32),
        }

        if self.cache is not None and mol_idx >= 0:
            self.cache.save(mol_idx, result)
            gc.collect()

        return result

    def collate_features(
        self,
        features_list: list,
        device: torch.device,
    ) -> FeatureBatch:
        """Collate per-molecule feature dicts into a padded batch tensor."""
        if not features_list:
            raise ValueError("Empty feature list")

       

        # EGNN forward pass (per molecule, then stack).
        # Use deterministic init seeded by atom-type hash so the same
        # molecule always produces the same EGNN output across batches.
        egnn_vecs = []
        for f in features_list:
            n_atoms = f["atom_types"].shape[0]
            atypes = torch.tensor(f["atom_types"], device=device)
            # Deterministic seed from atom-type composition
            seed = int(atypes.sum().item() * 2654435761) & 0xFFFFFFFF
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            h_init = torch.randn(n_atoms, self.egnn_hidden, device=device, generator=g) * 0.02
            x = torch.tensor(f["coords"], device=device)
            ei = torch.tensor(f["edge_index"], device=device)
            if ei.numel() == 0:
                egnn_vecs.append(torch.zeros(self.egnn_hidden, device=device))
            else:
                with torch.no_grad():
                    h_geo = self.egnn(h_init, x, ei)
                egnn_vecs.append(h_geo)
        egnn_batch = torch.stack(egnn_vecs, dim=0)

        # RSAS embeddings (pad to max atoms)
        max_atoms = max(f["atom_props"].shape[0] for f in features_list)
        rsas_batch = torch.zeros(len(features_list), max_atoms, self.rsas_embed_dim, device=device)
        
        for i, f in enumerate(features_list):
            n = f["atom_props"].shape[0]
            if n > 0:
                props = torch.tensor(f["atom_props"], device=device)
                w = torch.tensor(self.rsas_embedding.proj_weight, device=device)
                rsas_batch[i, :n] = props @ w
                atom_mask[i, :n] = True

        # Pad coordinates
        max_atoms_coords = max(f["coords"].shape[0] for f in features_list)
        coords_batch = torch.zeros(len(features_list), max_atoms_coords, 3, device=device)
        for i, f in enumerate(features_list):
            n = f["coords"].shape[0]
            coords_batch[i, :n] = torch.tensor(f["coords"], device=device)

        return FeatureBatch(
            mph=mph_batch,
            pls=pls_batch,
            egnn=egnn_batch,
            rsas=rsas_batch,
            coords=coords_batch,
            atom_mask=atom_mask,
        )
