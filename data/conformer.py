"""3D conformer generation via RDKit ETKDG v3, with optional MMFF94 minimization."""

from typing import Optional, Tuple
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom


def generate_conformer(
    smiles: str,
    use_mmff: bool = False,
    num_confs: int = 1,
    random_seed: int = 42,
) -> Optional[np.ndarray]:
    """Generate 3D conformer coordinates from SMILES.

    Returns (num_atoms, 3) numpy array, or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 1
    params.pruneRmsThresh = 0.1

    status = rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if status < 0:
        mol = Chem.RemoveHs(mol)
        return None

    if use_mmff:
        for cid in range(mol.GetNumConformers()):
            ff = AllChem.MMFFGetMoleculeForceField(
                Chem.RemoveHs(Chem.Mol(mol)), AllChem.MMFFGetMoleculeProperties(mol), confId=cid
            )
            if ff is not None:
                ff.Minimize()

    conf = mol.GetConformer(0)
    coords = np.array(conf.GetPositions(), dtype=np.float32)

    mol = Chem.RemoveHs(mol)
    return coords


def get_atom_types(smiles: str) -> Optional[np.ndarray]:
    """Extract atomic numbers from SMILES (with hydrogens)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    atomic_nums = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
    return atomic_nums


def get_bond_edges(smiles: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract bond edge index from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    src, dst = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        src.extend([i, j])
        dst.extend([j, i])
    if not src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    edge_index = np.array([src, dst], dtype=np.int64)
    return edge_index
