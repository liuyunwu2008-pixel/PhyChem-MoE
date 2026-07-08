"""Feature caching via numpy.memmap for RAM-resident features after first epoch."""

import gc
import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np


class FeatureCache:
    """Caches molecular features to numpy memmap files on disk.

    First epoch: features computed online → written to memmap.
    Subsequent epochs: loaded directly from memmap, skipping CPU computation.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._arrays: Dict[int, Dict[str, np.ndarray]] = {}
        self._memmaps: Dict[int, Dict[str, np.ndarray]] = {}
        self._in_ram: bool = False

    def has(self, mol_idx: int) -> bool:
        return (self.cache_dir / f"{mol_idx}.npz").exists()

    def save(self, mol_idx: int, features: Dict[str, np.ndarray]) -> None:
        path = self.cache_dir / f"{mol_idx}.npz"
        np.savez_compressed(path, **features)
        gc.collect()

    def load(self, mol_idx: int, pin_memory: bool = False) -> Optional[Dict[str, np.ndarray]]:
        path = self.cache_dir / f"{mol_idx}.npz"
        if not path.exists():
            return None
        data = np.load(path)
        result = {key: data[key] for key in data.files}
        if pin_memory:
            result = {k: np.ascontiguousarray(v) for k, v in result.items()}
        return result

    def preload_all(self, num_molecules: int) -> None:
        """Load all cached features into RAM for fast access."""
        for idx in range(num_molecules):
            if self.has(idx):
                self._arrays[idx] = self.load(idx)
        self._in_ram = True

    def get(self, mol_idx: int) -> Optional[Dict[str, np.ndarray]]:
        if self._in_ram and mol_idx in self._arrays:
            return self._arrays[mol_idx]
        return self.load(mol_idx)

    def clear_ram(self) -> None:
        self._arrays.clear()
        self._memmaps.clear()
        self._in_ram = False
        gc.collect()
