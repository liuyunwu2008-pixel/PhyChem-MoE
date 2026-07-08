"""RSAS: Residual Semantic Atom Signature — atom-level property embeddings.

Extracts Gasteiger partial charges and atomic masses, then embeds them
into a continuous 64-dimensional space per atom.
"""

from typing import Optional
import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem


# Atomic masses (most common isotope) for elements up to Xe (Z=54)
ATOMIC_MASSES = {
    1: 1.008, 2: 4.0026, 3: 6.94, 4: 9.0122, 5: 10.81, 6: 12.011,
    7: 14.007, 8: 15.999, 9: 18.998, 10: 20.180, 11: 22.990, 12: 24.305,
    13: 26.982, 14: 28.085, 15: 30.974, 16: 32.06, 17: 35.45, 18: 39.948,
    19: 39.098, 20: 40.078, 21: 44.956, 22: 47.867, 23: 50.942, 24: 51.996,
    25: 54.938, 26: 55.845, 27: 58.933, 28: 58.693, 29: 63.546, 30: 65.38,
    31: 69.723, 32: 72.630, 33: 74.922, 34: 78.971, 35: 79.904, 36: 83.798,
    37: 85.468, 38: 87.62, 39: 88.906, 40: 91.224, 41: 92.906, 42: 95.95,
    43: 97.0, 44: 101.07, 45: 102.91, 46: 106.42, 47: 107.87, 48: 112.41,
    49: 114.82, 50: 118.71, 51: 121.76, 52: 127.60, 53: 126.90, 54: 131.29,
}


def compute_atom_properties(smiles: str) -> Optional[np.ndarray]:
    """Compute atom-level properties: partial charges and atomic mass.

    Args:
        smiles: SMILES string.

    Returns:
        (num_atoms, 2) array with [partial_charge, log_mass] per atom,
        or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()

    # Compute Gasteiger partial charges
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    props = np.zeros((num_atoms, 2), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        z = atom.GetAtomicNum()
        charge = float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0
        mass = ATOMIC_MASSES.get(z, 12.0)
        props[i, 0] = charge
        props[i, 1] = np.log(mass)

    return props


class AtomPropertyEmbedding:
    """Embeds atom-level properties (charge, mass) into a continuous space."""

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        # Single linear projection: 2d → embed_dim (will be wrapped in nn.Module later)
        self.proj_weight: Optional[np.ndarray] = None

    def init_weights(self, seed: int = 42) -> np.ndarray:
        rng = np.random.RandomState(seed)
        w = rng.randn(2, self.embed_dim).astype(np.float32) * 0.02
        self.proj_weight = w
        return w
